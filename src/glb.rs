use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::container::pak::{
    DecodedMesh, GeometryInventory, PakArchive, PakError, read_texture_payload_bytes,
    texture_format,
};
use crate::skeleton::Skeleton;
use crate::texture::{BcFormat, decode_bc_level};

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);
type AccessorDef = (
    usize,
    u32,
    usize,
    &'static str,
    Option<([f32; 3], [f32; 3])>,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlbExportSummary {
    pub mesh_count: usize,
    pub vertex_count: usize,
    pub triangle_count: usize,
}

struct GlbSource<'a> {
    pak_path: &'a Path,
    inventory: &'a GeometryInventory,
}

struct MeshAccessors {
    position: usize,
    normal: Option<usize>,
    tangent: Option<usize>,
    uv0: Option<usize>,
    uv1: Option<usize>,
    joints: [Option<usize>; 3],
    weights: [Option<usize>; 3],
    indices: usize,
}

struct GlbJsonContext<'a> {
    meshes: &'a [DecodedMesh],
    views: &'a [(usize, usize, u32)],
    accessors: &'a [AccessorDef],
    binary_len: usize,
    mesh_accessors: &'a [MeshAccessors],
    texture_payloads: &'a [(u64, String, Vec<u8>)],
    image_view_indices: Option<&'a [usize]>,
    bin_filename: Option<&'a str>,
    skeleton: Option<&'a Skeleton>,
    inverse_bind_accessor: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextureRole {
    BaseColor,
    Normal,
    Roughness,
    Occlusion,
    Transparency,
    Other,
}

fn texture_role(name: &str) -> TextureRole {
    let name = name.to_lowercase();
    let auxiliary_normal = name.contains("detail")
        || name.contains("ambient")
        || name.contains("micro")
        || name.contains("sinusoidal")
        || name.contains("burn");
    let auxiliary_color = name.contains("burn")
        || name.contains("blood")
        || name.contains("dirt")
        || name.contains("mud")
        || name.contains("snow")
        || name.contains("wet")
        || name.contains("stitches");
    if name.contains("normal") && !auxiliary_normal {
        TextureRole::Normal
    } else if name.contains("roughness") && !name.contains("burn") {
        TextureRole::Roughness
    } else if name.contains("ao01") || name.contains("occlusion") || name.contains("fetchao") {
        TextureRole::Occlusion
    } else if name.contains("transparency") || name.contains("tearmask") {
        TextureRole::Transparency
    } else if name.contains("basecolor")
        || name.contains("diffuse")
        || (name.contains("color") && !name.contains("mask") && !auxiliary_color)
    {
        TextureRole::BaseColor
    } else {
        TextureRole::Other
    }
}

fn texture_index(ctx: &GlbJsonContext<'_>, mesh: &DecodedMesh, role: TextureRole) -> Option<usize> {
    texture_hash(mesh, role).and_then(|hash| {
        ctx.texture_payloads
            .iter()
            .position(|(candidate, _, _)| *candidate == hash)
    })
}

fn texture_hash(mesh: &DecodedMesh, role: TextureRole) -> Option<u64> {
    mesh.report
        .textures
        .iter()
        .find(|texture| texture_role(&texture.name) == role)
        .map(|texture| texture.hash)
}

fn alpha_uses_blending(mesh: &DecodedMesh) -> bool {
    let name = mesh.report.submesh_name.to_lowercase();
    name.contains("hair")
        || name.contains("tear")
        || name.contains("fur")
        || name.contains("eyebrow")
        || name.contains("eyelash")
}

fn alpha_base_color(mesh: &DecodedMesh) -> [u8; 3] {
    let name = mesh.report.submesh_name.to_lowercase();
    if name.contains("eyelash") {
        [8, 6, 5]
    } else if name.contains("tear") {
        [210, 225, 245]
    } else {
        [255, 255, 255]
    }
}

fn remap_helper_influences(meshes: &mut [DecodedMesh], skeleton: Option<&Skeleton>) {
    let Some(skeleton) = skeleton else {
        return;
    };
    let helper_bones = &skeleton.helper_bones;
    if !helper_bones.iter().any(|helper| *helper) {
        return;
    }
    let is_helper = |bone: u16| helper_bones.get(bone as usize).copied().unwrap_or(false);
    let helper_targets: Vec<Option<u16>> = (0..skeleton.bone_count())
        .map(|bone| skeleton.helper_target(bone).map(|target| target as u16))
        .collect();

    let mut scores: BTreeMap<u16, BTreeMap<u16, f64>> = BTreeMap::new();
    let mut centroids: BTreeMap<u16, ([f64; 3], f64)> = BTreeMap::new();
    for mesh in meshes.iter() {
        let Some(skin) = &mesh.skin else {
            continue;
        };
        if skin.influences.len() != mesh.vertices.len() {
            continue;
        }
        for (vertex, influences) in skin.influences.iter().enumerate() {
            let main: Vec<(u16, f32)> = influences
                .iter()
                .copied()
                .filter(|(bone, _)| !is_helper(*bone))
                .collect();
            let position = mesh.vertices[vertex];
            for &(bone, weight) in influences {
                if !is_helper(bone) || weight <= 0.0 {
                    continue;
                }
                let centroid = centroids.entry(bone).or_insert(([0.0; 3], 0.0));
                for (axis, value) in position.iter().enumerate() {
                    centroid.0[axis] += f64::from(*value) * f64::from(weight);
                }
                centroid.1 += f64::from(weight);
                if !main.is_empty() {
                    let score = scores.entry(bone).or_default();
                    for &(main_bone, main_weight) in &main {
                        *score.entry(main_bone).or_insert(0.0) +=
                            f64::from(weight) * f64::from(main_weight);
                    }
                }
            }
        }
    }

    let main_positions: Vec<(u16, [f32; 3])> = (0..skeleton.bone_count())
        .filter(|bone| !helper_bones[*bone])
        .map(|bone| {
            let matrix = skeleton.world_matrices[bone];
            (bone as u16, [matrix[0][3], matrix[1][3], matrix[2][3]])
        })
        .collect();
    let distance = |left: [f64; 3], right: [f32; 3]| {
        (left[0] - f64::from(right[0])).powi(2)
            + (left[1] - f64::from(right[1])).powi(2)
            + (left[2] - f64::from(right[2])).powi(2)
    };
    let mut fallback: BTreeMap<u16, u16> = BTreeMap::new();
    for (&bone, &(sum, weight)) in &centroids {
        let matched = scores.get(&bone).and_then(|score| {
            score
                .iter()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(&main_bone, _)| main_bone)
        });
        let main_bone = matched.unwrap_or_else(|| {
            let centroid = [sum[0] / weight, sum[1] / weight, sum[2] / weight];
            main_positions
                .iter()
                .min_by(|left, right| {
                    distance(centroid, left.1).total_cmp(&distance(centroid, right.1))
                })
                .map(|(bone, _)| *bone)
                .unwrap_or(0)
        });
        fallback.insert(bone, main_bone);
    }

    if std::env::var("ND_DEBUG_HELPERS").is_ok() {
        for (&bone, &main_bone) in &fallback {
            let name = &skeleton.names[bone as usize];
            if ["knee", "elbow", "wrist", "ankle", "upper_leg", "upper_arm", "shoulder", "hip", "twist", "roll"]
                .iter()
                .any(|needle| name.contains(needle))
            {
                eprintln!("HELPER {name} -> {}", skeleton.names[main_bone as usize]);
            }
        }
    }

    for mesh in meshes.iter_mut() {
        let Some(skin) = &mut mesh.skin else {
            continue;
        };
        for influences in &mut skin.influences {
            let mut kept: Vec<(u16, f32)> = Vec::with_capacity(influences.len());
            for &(bone, weight) in influences.iter() {
                if weight <= 0.0 {
                    continue;
                }
                if is_helper(bone) {
                    let name = &skeleton.names[bone as usize];
                    let prefer_target = name.contains("twist")
                        || name.contains("roll")
                        || name.contains("space")
                        || name.contains("mover");
                    let target = helper_targets.get(bone as usize).copied().flatten();
                    if prefer_target {
                        if let Some(target) = target {
                            kept.push((target, weight));
                        } else if let Some(&main_bone) = fallback.get(&bone) {
                            kept.push((main_bone, weight));
                        }
                    } else if let Some(&main_bone) = fallback.get(&bone) {
                        kept.push((main_bone, weight));
                    } else if let Some(target) = target {
                        kept.push((target, weight));
                    }
                } else {
                    kept.push((bone, weight));
                }
            }
            if kept.is_empty() {
                let dominant_helper = influences
                    .iter()
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .map(|(bone, _)| *bone);
                if let Some(main_bone) =
                    dominant_helper.and_then(|bone| fallback.get(&bone).copied())
                {
                    kept.push((main_bone, 1.0));
                } else {
                    kept.push((0, 1.0));
                }
            }
            kept.sort_by_key(|(bone, _)| *bone);
            let mut merged: Vec<(u16, f32)> = Vec::with_capacity(kept.len());
            for (bone, weight) in kept {
                if let Some(last) = merged.last_mut()
                    && last.0 == bone
                {
                    last.1 += weight;
                    continue;
                }
                merged.push((bone, weight));
            }
            merged.sort_by(|left, right| right.1.total_cmp(&left.1));
            merged.truncate(12);
            let sum = merged.iter().map(|(_, weight)| *weight).sum::<f32>();
            if sum > 0.0 {
                for (_, weight) in &mut merged {
                    *weight /= sum;
                }
            } else {
                merged = vec![(0, 1.0)];
            }
            *influences = merged;
        }
    }
}

fn fill_missing_skin_influences(mesh: &mut DecodedMesh) {
    let Some(skin) = &mut mesh.skin else {
        return;
    };
    if skin.influences.len() != mesh.vertices.len() {
        return;
    }
    let vertices = &mesh.vertices;
    let skinned: Vec<usize> = skin
        .influences
        .iter()
        .enumerate()
        .filter_map(|(index, influences)| (!influences.is_empty()).then_some(index))
        .collect();
    if skinned.is_empty() {
        for influences in &mut skin.influences {
            *influences = vec![(0, 1.0)];
        }
        return;
    }
    for index in 0..skin.influences.len() {
        if !skin.influences[index].is_empty() {
            continue;
        }
        let position = vertices[index];
        let nearest = skinned
            .iter()
            .copied()
            .min_by(|left, right| {
                let distance = |other: usize| {
                    let other = vertices[other];
                    (position[0] - other[0]).powi(2)
                        + (position[1] - other[1]).powi(2)
                        + (position[2] - other[2]).powi(2)
                };
                distance(*left).total_cmp(&distance(*right))
            })
            .expect("skinned vertices are not empty");
        skin.influences[index] = skin.influences[nearest].clone();
    }
}

/// Writes decoded geometry, normals, UVs, and referenced textures as a binary GLB (glTF 2.0).
pub fn export_geometry_glb(
    inventory: &GeometryInventory,
    output: impl AsRef<Path>,
) -> Result<GlbExportSummary, PakError> {
    export_comprehensive_glb(
        Path::new(""),
        inventory,
        None::<&Path>,
        None::<&Path>,
        output,
    )
}

/// Writes a static GLB with optional texture dictionary integration.
pub fn export_comprehensive_glb(
    pak_path: impl AsRef<Path>,
    inventory: &GeometryInventory,
    skeleton_path: Option<impl AsRef<Path>>,
    dictionary_path: Option<impl AsRef<Path>>,
    output: impl AsRef<Path>,
) -> Result<GlbExportSummary, PakError> {
    let source = GlbSource {
        pak_path: pak_path.as_ref(),
        inventory,
    };
    export_comprehensive_glb_sources(
        &[source],
        skeleton_path.as_ref().map(|path| path.as_ref()),
        dictionary_path.as_ref().map(|path| path.as_ref()),
        output,
    )
}

/// Writes several PAKs into one GLB, sharing one optional skeleton and texture set.
pub fn export_comprehensive_glb_multi(
    pak_paths: &[std::path::PathBuf],
    skeleton_path: Option<&Path>,
    dictionary_path: Option<&Path>,
    output: impl AsRef<Path>,
) -> Result<GlbExportSummary, PakError> {
    let mut owned = Vec::with_capacity(pak_paths.len());
    for path in pak_paths {
        owned.push((path.clone(), PakArchive::inspect_geometry_path(path)?));
    }
    let sources: Vec<_> = owned
        .iter()
        .map(|(path, inventory)| GlbSource {
            pak_path: path,
            inventory,
        })
        .collect();
    export_comprehensive_glb_sources(&sources, skeleton_path, dictionary_path, output)
}

fn main_skeleton_path(pak_path: &Path) -> Option<PathBuf> {
    let fallback = pak_path.parent()?.join("ellie-skel.pak");
    fallback.is_file().then_some(fallback)
}

fn sibling_skeleton_path(pak_path: &Path) -> Option<PathBuf> {
    let file_name = pak_path.file_name()?.to_string_lossy();
    if file_name.to_ascii_lowercase().ends_with("-skel.pak") {
        return None;
    }
    let stem = file_name.strip_suffix(".pak")?;
    let sibling = pak_path.with_file_name(format!("{stem}-skel.pak"));
    sibling.is_file().then_some(sibling)
}

fn cloth_root_target(root_name: &str) -> &'static str {
    let lower = root_name.to_ascii_lowercase();
    if lower.contains("hair") {
        "headb"
    } else if lower.contains("jacket") || lower.contains("backpack") {
        "spined"
    } else {
        "root"
    }
}

fn combine_skeletons(
    skeletons: &[(PathBuf, Skeleton)],
    main_index: Option<usize>,
) -> Result<Skeleton, PakError> {
    let total = skeletons
        .iter()
        .map(|(_, skeleton)| skeleton.bone_count())
        .sum();
    let mut names = Vec::with_capacity(total);
    let mut parents = Vec::with_capacity(total);
    let mut chain_ids = Vec::with_capacity(total);
    let mut helper_bones = Vec::with_capacity(total);
    let mut translations = Vec::with_capacity(total);
    let mut rotations = Vec::with_capacity(total);
    let mut scales = Vec::with_capacity(total);
    let mut world_matrices = Vec::with_capacity(total);
    let mut inverse_bind_matrices = Vec::with_capacity(total);
    let mut bases = Vec::with_capacity(skeletons.len());
    let mut used_names = BTreeSet::new();
    let mut base = 0usize;
    for (_, skeleton) in skeletons {
        bases.push(base);
        for bone in 0..skeleton.bone_count() {
            let name = &skeleton.names[bone];
            let mut candidate = name.clone();
            let mut suffix = 2usize;
            while !used_names.insert(candidate.clone()) {
                candidate = format!("{name}_{suffix}");
                suffix += 1;
            }
            names.push(candidate);
            let parent = skeleton.parents[bone];
            parents.push(if parent >= 0 {
                parent + base as i32
            } else {
                -1
            });
            let chain = skeleton.chain_ids[bone];
            chain_ids.push(if chain >= 0 { chain + base as i32 } else { -1 });
            helper_bones.push(skeleton.helper_bones[bone]);
            translations.push(skeleton.translations[bone]);
            rotations.push(skeleton.rotations[bone]);
            scales.push(skeleton.scales[bone]);
            world_matrices.push(skeleton.world_matrices[bone]);
            inverse_bind_matrices.push(skeleton.inverse_bind_matrices[bone]);
        }
        base += skeleton.bone_count();
    }
    let mut combined = Skeleton {
        names,
        parents,
        chain_ids,
        helper_bones,
        translations,
        rotations,
        scales,
        world_matrices,
        inverse_bind_matrices,
    };
    if let Some(main_index) = main_index {
        let (_, main) = &skeletons[main_index];
        let main_base = bases[main_index];
        for (index, (_, skeleton)) in skeletons.iter().enumerate() {
            if index == main_index {
                continue;
            }
            for root in skeleton.root_bones() {
                let target = cloth_root_target(&skeleton.names[root]);
                let Some(target_position) = main.names.iter().position(|name| name == target)
                else {
                    continue;
                };
                combined.reparent_bone(bases[index] + root, main_base + target_position)?;
            }
        }
    }
    Ok(combined)
}

fn export_comprehensive_glb_sources(
    sources: &[GlbSource<'_>],
    skeleton_path: Option<&Path>,
    dictionary_path: Option<&Path>,
    output: impl AsRef<Path>,
) -> Result<GlbExportSummary, PakError> {
    let output_path = output.as_ref();

    let base_path = skeleton_path.map(Path::to_owned).or_else(|| {
        sources
            .iter()
            .find_map(|source| main_skeleton_path(source.pak_path))
    });
    let main_index = base_path.as_ref().map(|_| 0usize);
    let mut unique_skeletons: Vec<(PathBuf, Skeleton)> = Vec::new();
    if let Some(path) = &base_path {
        unique_skeletons.push((path.clone(), Skeleton::parse_path(path)?));
    }
    let mut source_skeleton_paths = Vec::with_capacity(sources.len());
    for source in sources {
        let part_path = if skeleton_path.is_some() {
            None
        } else {
            sibling_skeleton_path(source.pak_path)
        };
        let path = if let Some(part_path) = part_path {
            if !unique_skeletons
                .iter()
                .any(|(candidate, _)| candidate == &part_path)
            {
                unique_skeletons.push((part_path.clone(), Skeleton::parse_path(&part_path)?));
            }
            Some(part_path)
        } else {
            base_path.clone()
        };
        source_skeleton_paths.push(path);
    }
    let skeleton_bases: Vec<usize> = unique_skeletons
        .iter()
        .scan(0usize, |base, (_, skeleton)| {
            let current = *base;
            *base += skeleton.bone_count();
            Some(current)
        })
        .collect();
    let skeleton = if unique_skeletons.is_empty() {
        None
    } else {
        Some(combine_skeletons(&unique_skeletons, main_index)?)
    };

    let mut meshes = Vec::new();
    for (source, skeleton_path) in sources.iter().zip(source_skeleton_paths.iter()) {
        let lod0: Vec<_> = source
            .inventory
            .submeshes
            .iter()
            .filter(|submesh| submesh.lod == Some(0))
            .collect();
        if lod0.is_empty() {
            continue;
        }
        let mut source_meshes = Vec::new();
        for submesh in &lod0 {
            if !submesh.supported {
                return Err(invalid(
                    "LOD0 submesh",
                    &format!(
                        "{} is not supported: {}",
                        submesh.index,
                        submesh.issue.as_deref().unwrap_or("unknown reason")
                    ),
                ));
            }
            let mesh = source
                .inventory
                .decoded_meshes
                .iter()
                .find(|mesh| mesh.report.submesh_index == submesh.index)
                .ok_or_else(|| {
                    invalid(
                        "LOD0 submesh",
                        &format!("{} was not decoded", submesh.index),
                    )
                })?;
            source_meshes.push(mesh.clone());
        }

        let skeleton_index = skeleton_path.as_ref().and_then(|path| {
            unique_skeletons
                .iter()
                .position(|(candidate, _)| candidate == path)
        });
        if let Some(index) = skeleton_index {
            remap_helper_influences(&mut source_meshes, Some(&unique_skeletons[index].1));
            let offset = u16::try_from(skeleton_bases[index])
                .map_err(|_| invalid("GLB skin", "joint offset exceeds u16"))?;
            for mesh in &mut source_meshes {
                if let Some(skin) = &mut mesh.skin {
                    for influences in &mut skin.influences {
                        for (joint, _) in influences {
                            *joint = joint
                                .checked_add(offset)
                                .ok_or_else(|| invalid("GLB skin", "joint index exceeds u16"))?;
                        }
                    }
                }
            }
        } else {
            for mesh in &mut source_meshes {
                mesh.skin = None;
            }
        }
        for mesh in &mut source_meshes {
            fill_missing_skin_influences(mesh);
        }
        meshes.extend(source_meshes);
    }
    if meshes.is_empty() {
        return Err(invalid("geometry", "no LOD0 submeshes are identified"));
    }

    for mesh in &mut meshes {
        if mesh.normals.is_empty() {
            mesh.normals = compute_smooth_normals(&mesh.vertices, &mesh.indices);
        }
    }

    let referenced_hashes: BTreeSet<u64> = meshes
        .iter()
        .flat_map(|mesh| mesh.report.textures.iter().map(|texture| texture.hash))
        .collect();

    // Optional texture dictionary inspection & extraction with PNG decoding and external file export.
    let mut texture_payloads: Vec<(u64, String, Vec<u8>)> = Vec::new();
    let parent = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stem = output_path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "model".into());

    if !referenced_hashes.is_empty() {
        let mut alpha_for_base: BTreeMap<u64, u64> = BTreeMap::new();
        let mut alpha_only_colors: BTreeMap<u64, [u8; 3]> = BTreeMap::new();
        for mesh in &meshes {
            let Some(alpha_hash) = texture_hash(mesh, TextureRole::Transparency) else {
                continue;
            };
            if let Some(base_hash) = texture_hash(mesh, TextureRole::BaseColor) {
                alpha_for_base.entry(base_hash).or_insert(alpha_hash);
            } else {
                alpha_only_colors
                    .entry(alpha_hash)
                    .or_insert_with(|| alpha_base_color(mesh));
            }
        }

        let mut decoded: BTreeMap<u64, (u32, u32, Vec<u8>)> = BTreeMap::new();
        for source in sources {
            if source.pak_path.as_os_str().is_empty() {
                continue;
            }
            let texture_inventory =
                PakArchive::inspect_textures_path(source.pak_path, dictionary_path)?;
            for entry in &texture_inventory.descriptors {
                let hash = entry.descriptor.hash;
                if !referenced_hashes.contains(&hash) || decoded.contains_key(&hash) {
                    continue;
                }
                let Some(payload) = &entry.payload else {
                    continue;
                };
                let Ok(bytes) = read_texture_payload_bytes(payload) else {
                    continue;
                };
                let role = meshes
                    .iter()
                    .flat_map(|mesh| mesh.report.textures.iter())
                    .find(|texture| texture.hash == hash)
                    .map(|texture| texture_role(&texture.name))
                    .unwrap_or(TextureRole::Other);
                let Some(rgba8) = decode_texture_rgba(
                    payload.source_format_id,
                    payload.source_width,
                    payload.source_height,
                    &bytes,
                    role,
                ) else {
                    continue;
                };
                decoded.insert(hash, (payload.source_width, payload.source_height, rgba8));
            }
        }

        for (base_hash, alpha_hash) in &alpha_for_base {
            let Some((alpha_width, alpha_height, alpha_rgba)) = decoded.get(alpha_hash).cloned()
            else {
                continue;
            };
            let Some((width, height, rgba8)) = decoded.get_mut(base_hash) else {
                continue;
            };
            copy_alpha_channel(
                rgba8,
                *width,
                *height,
                &alpha_rgba,
                alpha_width,
                alpha_height,
            );
        }
        for (alpha_hash, color) in &alpha_only_colors {
            if let Some((_width, _height, rgba8)) = decoded.get_mut(alpha_hash) {
                for pixel in rgba8.as_chunks_mut::<4>().0 {
                    let alpha = pixel[3];
                    pixel.copy_from_slice(&[color[0], color[1], color[2], alpha]);
                }
            }
        }

        for (hash, (width, height, rgba8)) in decoded {
            let Some(png_bytes) = encode_png(width, height, &rgba8) else {
                continue;
            };
            let name = format!("texture_{:#018x}", hash);
            let filename = format!("{}_{:#018x}.png", stem, hash);
            let tex_file_path = parent.join(&filename);
            if fs::write(&tex_file_path, &png_bytes).is_ok() {
                texture_payloads.push((hash, name, png_bytes));
            }
        }
    }

    let mut binary = Vec::new();
    let mut views = Vec::new();
    let mut accessors: Vec<AccessorDef> = Vec::new();

    // 1. Write texture payloads (encoded as PNG) contiguously to binary chunk as image buffer views for GLB
    let mut image_view_indices = Vec::new();
    for (_hash, _name, png_bytes) in &texture_payloads {
        align4(&mut binary);
        let img_offset = binary.len();
        binary.extend_from_slice(png_bytes);
        let img_len = binary.len() - img_offset;
        views.push((img_offset, img_len, 0)); // target 0 for raw image buffer view
        image_view_indices.push(views.len() - 1);
    }
    align4(&mut binary); // Align binary stream before vertex buffers

    // 2. Write vertex and index buffers per mesh
    let mut mesh_accessors = Vec::with_capacity(meshes.len());
    for mesh in &meshes {
        validate_mesh(mesh)?;
        let v_count = mesh.vertices.len();

        align4(&mut binary);
        let pos_offset = binary.len();
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
                binary.extend_from_slice(&vertex[axis].to_le_bytes());
            }
        }
        let pos_len = binary.len() - pos_offset;
        views.push((pos_offset, pos_len, 34962));
        accessors.push((views.len() - 1, 5126, v_count, "VEC3", Some((min, max))));
        let position = accessors.len() - 1;

        let normal = if mesh.normals.len() == v_count {
            align4(&mut binary);
            let offset = binary.len();
            for normal in &mesh.normals {
                for component in normal {
                    binary.extend_from_slice(&component.to_le_bytes());
                }
            }
            let length = binary.len() - offset;
            views.push((offset, length, 34962));
            accessors.push((views.len() - 1, 5126, v_count, "VEC3", None));
            Some(accessors.len() - 1)
        } else {
            None
        };

        let tangent = if mesh.tangents.len() == v_count {
            align4(&mut binary);
            let offset = binary.len();
            for tangent in &mesh.tangents {
                for component in tangent {
                    binary.extend_from_slice(&component.to_le_bytes());
                }
            }
            let length = binary.len() - offset;
            views.push((offset, length, 34962));
            accessors.push((views.len() - 1, 5126, v_count, "VEC4", None));
            Some(accessors.len() - 1)
        } else {
            None
        };

        let uv0 = if mesh.uv0.len() == v_count {
            align4(&mut binary);
            let offset = binary.len();
            for uv in &mesh.uv0 {
                binary.extend_from_slice(&uv[0].to_le_bytes());
                binary.extend_from_slice(&uv[1].to_le_bytes());
            }
            let length = binary.len() - offset;
            views.push((offset, length, 34962));
            accessors.push((views.len() - 1, 5126, v_count, "VEC2", None));
            Some(accessors.len() - 1)
        } else {
            None
        };

        let uv1 = if mesh.uv1.len() == v_count {
            align4(&mut binary);
            let offset = binary.len();
            for uv in &mesh.uv1 {
                binary.extend_from_slice(&uv[0].to_le_bytes());
                binary.extend_from_slice(&uv[1].to_le_bytes());
            }
            let length = binary.len() - offset;
            views.push((offset, length, 34962));
            accessors.push((views.len() - 1, 5126, v_count, "VEC2", None));
            Some(accessors.len() - 1)
        } else {
            None
        };

        let mut joints = [None; 3];
        let mut weights = [None; 3];
        if skeleton.is_some()
            && let Some(skin) = &mesh.skin
        {
            if skin.influences.len() != v_count {
                return Err(invalid(
                    "GLB skin",
                    &format!(
                        "{} skin has {} vertices, mesh has {v_count}",
                        mesh.report.submesh_name,
                        skin.influences.len()
                    ),
                ));
            }
            let max_influences = skin
                .influences
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0)
                .min(12);
            let set_count = max_influences.div_ceil(4).max(1);
            for set in 0..set_count {
                align4(&mut binary);
                let joint_offset = binary.len();
                for influences in &skin.influences {
                    for lane in 0..4 {
                        let joint = influences
                            .get(set * 4 + lane)
                            .map_or(0, |(joint, _)| *joint);
                        binary.extend_from_slice(&joint.to_le_bytes());
                    }
                }
                let joint_length = binary.len() - joint_offset;
                views.push((joint_offset, joint_length, 34962));
                accessors.push((views.len() - 1, 5123, v_count, "VEC4", None));
                joints[set] = Some(accessors.len() - 1);

                align4(&mut binary);
                let weight_offset = binary.len();
                for influences in &skin.influences {
                    for lane in 0..4 {
                        let weight = influences
                            .get(set * 4 + lane)
                            .map_or(0.0, |(_, weight)| *weight);
                        binary.extend_from_slice(&weight.to_le_bytes());
                    }
                }
                let weight_length = binary.len() - weight_offset;
                views.push((weight_offset, weight_length, 34962));
                accessors.push((views.len() - 1, 5126, v_count, "VEC4", None));
                weights[set] = Some(accessors.len() - 1);
            }
        }

        align4(&mut binary);
        let indices_offset = binary.len();
        for index in &mesh.indices {
            binary.extend_from_slice(&u32::from(*index).to_le_bytes());
        }
        let indices_len = binary.len() - indices_offset;
        views.push((indices_offset, indices_len, 34963));
        accessors.push((views.len() - 1, 5125, mesh.indices.len(), "SCALAR", None));
        let indices = accessors.len() - 1;

        mesh_accessors.push(MeshAccessors {
            position,
            normal,
            tangent,
            uv0,
            uv1,
            joints,
            weights,
            indices,
        });
    }

    align4(&mut binary);
    let inverse_bind_accessor = if let Some(skeleton) = skeleton.as_ref() {
        let offset = binary.len();
        for matrix in &skeleton.inverse_bind_matrices {
            for value in matrix {
                binary.extend_from_slice(&value.to_le_bytes());
            }
        }
        let length = binary.len() - offset;
        views.push((offset, length, 0));
        accessors.push((views.len() - 1, 5126, skeleton.bone_count(), "MAT4", None));
        Some(accessors.len() - 1)
    } else {
        None
    };

    // 3. Write GLB file
    let ctx = GlbJsonContext {
        meshes: &meshes,
        views: &views,
        accessors: &accessors,
        binary_len: binary.len(),
        mesh_accessors: &mesh_accessors,
        texture_payloads: &texture_payloads,
        image_view_indices: Some(&image_view_indices),
        bin_filename: None,
        skeleton: skeleton.as_ref(),
        inverse_bind_accessor,
    };
    let json = make_gltf_json(&ctx);
    let mut json_chunk = json.into_bytes();
    while !json_chunk.len().is_multiple_of(4) {
        json_chunk.push(b' ');
    }
    let total_len = 12usize
        .checked_add(8)
        .and_then(|n| n.checked_add(json_chunk.len()))
        .and_then(|n| n.checked_add(8))
        .and_then(|n| n.checked_add(binary.len()))
        .ok_or_else(|| invalid("GLB length", "size overflow"))?;
    let total_len = u32::try_from(total_len).map_err(|_| invalid("GLB length", "exceeds 4 GiB"))?;
    let json_len =
        u32::try_from(json_chunk.len()).map_err(|_| invalid("GLB JSON", "exceeds 4 GiB"))?;
    let bin_len =
        u32::try_from(binary.len()).map_err(|_| invalid("GLB binary", "exceeds 4 GiB"))?;

    let mut glb = Vec::with_capacity(total_len as usize);
    glb.extend_from_slice(&0x4654_6c67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total_len.to_le_bytes());
    glb.extend_from_slice(&json_len.to_le_bytes());
    glb.extend_from_slice(&0x4e4f_534au32.to_le_bytes());
    glb.extend_from_slice(&json_chunk);
    glb.extend_from_slice(&bin_len.to_le_bytes());
    glb.extend_from_slice(&0x004e_4942u32.to_le_bytes());
    glb.extend_from_slice(&binary);

    write_atomic(output_path, &glb)?;

    // 4. Also write a companion .gltf file + .bin file for foolproof Blender texture import
    if !texture_payloads.is_empty() {
        let gltf_path = output_path.with_extension("gltf");
        let bin_filename = format!("{}.bin", stem);
        let ext_ctx = GlbJsonContext {
            meshes: &meshes,
            views: &views,
            accessors: &accessors,
            binary_len: binary.len(),
            mesh_accessors: &mesh_accessors,
            texture_payloads: &texture_payloads,
            image_view_indices: None,
            bin_filename: Some(&bin_filename),
            skeleton: skeleton.as_ref(),
            inverse_bind_accessor,
        };
        let gltf_json = make_gltf_json(&ext_ctx);
        let _ = fs::write(&gltf_path, gltf_json);
        let bin_path = parent.join(&bin_filename);
        let _ = fs::write(&bin_path, &binary);
    }

    Ok(GlbExportSummary {
        mesh_count: meshes.len(),
        vertex_count: meshes.iter().map(|mesh| mesh.vertices.len()).sum(),
        triangle_count: meshes.iter().map(|mesh| mesh.indices.len() / 3).sum(),
    })
}

fn extract_mip0_blocks(
    format_id: u32,
    width: u32,
    height: u32,
    raw_bytes: &[u8],
) -> Option<Vec<u8>> {
    let (_, block_bytes_opt) = texture_format(format_id);
    let block_bytes = block_bytes_opt?;
    let w = width as usize;
    let h = height as usize;
    let (row_bytes, rows) = if block_bytes == 0 {
        (w.checked_mul(4)?, h)
    } else {
        let blocks_wide = w.div_ceil(4);
        let blocks_high = h.div_ceil(4);
        (blocks_wide.checked_mul(block_bytes as usize)?, blocks_high)
    };
    let tight_size = row_bytes.checked_mul(rows)?;
    if raw_bytes.len() == tight_size {
        return Some(raw_bytes.to_vec());
    }
    let row_pitch = align_u32(row_bytes as u32, 256)? as usize;

    let mut tight_bytes = Vec::with_capacity(tight_size);
    for row in 0..rows {
        let start = row.checked_mul(row_pitch)?;
        let end = start.checked_add(row_bytes)?;
        if end > raw_bytes.len() {
            return None;
        }
        tight_bytes.extend_from_slice(&raw_bytes[start..end]);
    }
    Some(tight_bytes)
}

fn align_u32(value: u32, alignment: u32) -> Option<u32> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn decode_texture_rgba(
    format_id: u32,
    width: u32,
    height: u32,
    raw_bytes: &[u8],
    role: TextureRole,
) -> Option<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return None;
    }

    let tight_bytes = extract_mip0_blocks(format_id, width, height, raw_bytes)?;

    let mut rgba8 = match format_id {
        0x47 | 0x48 => {
            // BC1
            decode_bc_level(BcFormat::Bc1, w, h, &tight_bytes)
                .ok()?
                .rgba8
        }
        0x4d | 0x4e => {
            // BC3
            decode_bc_level(BcFormat::Bc3, w, h, &tight_bytes)
                .ok()?
                .rgba8
        }
        0x50 | 0x51 => {
            // BC4
            decode_bc_level(BcFormat::Bc4, w, h, &tight_bytes)
                .ok()?
                .rgba8
        }
        0x53 | 0x54 => {
            // BC5
            decode_bc_level(BcFormat::Bc5, w, h, &tight_bytes)
                .ok()?
                .rgba8
        }
        0x61..=0x63 => {
            // BC7
            decode_bc_level(BcFormat::Bc7, w, h, &tight_bytes)
                .ok()?
                .rgba8
        }
        0x1c | 0x1d => {
            // R8G8B8A8
            tight_bytes
        }
        _ => return None,
    };

    match role {
        TextureRole::Normal if matches!(format_id, 0x53 | 0x54) => {
            for pixel in rgba8.chunks_mut(4) {
                let x = f32::from(pixel[0]) / 127.5 - 1.0;
                let y = 1.0 - f32::from(pixel[1]) / 127.5;
                let z = (1.0 - x * x - y * y).max(0.0).sqrt();
                pixel[0] = ((x * 0.5 + 0.5) * 255.0).round() as u8;
                pixel[1] = ((y * 0.5 + 0.5) * 255.0).round() as u8;
                pixel[2] = ((z * 0.5 + 0.5) * 255.0).round() as u8;
                pixel[3] = 255;
            }
        }
        TextureRole::Roughness | TextureRole::Occlusion if matches!(format_id, 0x50 | 0x51) => {
            for pixel in rgba8.chunks_mut(4) {
                pixel[1] = pixel[0];
                pixel[2] = pixel[0];
                pixel[3] = 255;
            }
        }
        TextureRole::Transparency if matches!(format_id, 0x50 | 0x51) => {
            for pixel in rgba8.chunks_mut(4) {
                pixel[3] = pixel[0];
            }
        }
        _ => {}
    }

    Some(rgba8)
}

fn copy_alpha_channel(
    base: &mut [u8],
    base_width: u32,
    base_height: u32,
    alpha: &[u8],
    alpha_width: u32,
    alpha_height: u32,
) {
    if base_width == 0 || base_height == 0 || alpha_width == 0 || alpha_height == 0 {
        return;
    }
    for y in 0..base_height {
        let alpha_y = (u64::from(y) * u64::from(alpha_height) / u64::from(base_height))
            .min(u64::from(alpha_height - 1)) as u32;
        for x in 0..base_width {
            let alpha_x = (u64::from(x) * u64::from(alpha_width) / u64::from(base_width))
                .min(u64::from(alpha_width - 1)) as u32;
            let alpha_index =
                ((u64::from(alpha_y) * u64::from(alpha_width) + u64::from(alpha_x)) * 4) as usize;
            let base_index = ((u64::from(y) * u64::from(base_width) + u64::from(x)) * 4) as usize;
            let source_alpha = alpha.get(alpha_index + 3).copied().unwrap_or(255);
            if let Some(pixel_alpha) = base.get_mut(base_index + 3) {
                *pixel_alpha = ((u16::from(*pixel_alpha) * u16::from(source_alpha)) / 255) as u8;
            }
        }
    }
}

fn encode_png(width: u32, height: u32, rgba8: &[u8]) -> Option<Vec<u8>> {
    let mut png_bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png_bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().ok()?;
        writer.write_image_data(rgba8).ok()?;
    }
    Some(png_bytes)
}

fn compute_smooth_normals(vertices: &[[f32; 3]], indices: &[u16]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f64; 3]; vertices.len()];
    for triangle in indices.chunks(3) {
        let [a, b, c] = [
            usize::from(triangle[0]),
            usize::from(triangle[1]),
            usize::from(triangle[2]),
        ];
        let (Some(va), Some(vb), Some(vc)) = (vertices.get(a), vertices.get(b), vertices.get(c))
        else {
            continue;
        };
        let edge1 = [
            f64::from(vb[0] - va[0]),
            f64::from(vb[1] - va[1]),
            f64::from(vb[2] - va[2]),
        ];
        let edge2 = [
            f64::from(vc[0] - va[0]),
            f64::from(vc[1] - va[1]),
            f64::from(vc[2] - va[2]),
        ];
        let face = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        for vertex in [a, b, c] {
            for axis in 0..3 {
                normals[vertex][axis] += face[axis];
            }
        }
    }
    normals
        .into_iter()
        .map(|normal| {
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if length > f64::EPSILON {
                [
                    (normal[0] / length) as f32,
                    (normal[1] / length) as f32,
                    (normal[2] / length) as f32,
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

fn validate_mesh(mesh: &DecodedMesh) -> Result<(), PakError> {
    if mesh.vertices.is_empty() {
        return Err(invalid("GLB positions", "mesh contains no vertices"));
    }
    if mesh.indices.is_empty() || !mesh.indices.len().is_multiple_of(3) {
        return Err(invalid(
            "GLB indices",
            "index count must be a nonzero multiple of three",
        ));
    }
    let vertex_count = mesh.vertices.len();
    for (label, length) in [
        ("GLB normals", mesh.normals.len()),
        ("GLB tangents", mesh.tangents.len()),
        ("GLB UV0", mesh.uv0.len()),
        ("GLB UV1", mesh.uv1.len()),
    ] {
        if length != 0 && length != vertex_count {
            return Err(invalid(
                label,
                &format!("contains {length} entries for {vertex_count} vertices"),
            ));
        }
    }
    for (vertex_index, vertex) in mesh.vertices.iter().enumerate() {
        for (axis, value) in vertex.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(
                    "GLB position",
                    &format!("vertex {vertex_index} axis {axis} is not finite"),
                ));
            }
        }
    }
    for (vertex_index, value) in mesh.normals.iter().enumerate() {
        if value.iter().any(|component| !component.is_finite()) {
            return Err(invalid(
                "GLB normal",
                &format!("vertex {vertex_index} contains a non-finite component"),
            ));
        }
    }
    for (vertex_index, value) in mesh.tangents.iter().enumerate() {
        if value.iter().any(|component| !component.is_finite()) {
            return Err(invalid(
                "GLB tangent",
                &format!("vertex {vertex_index} contains a non-finite component"),
            ));
        }
    }
    for (label, values) in [("GLB UV0", &mesh.uv0), ("GLB UV1", &mesh.uv1)] {
        for (vertex_index, value) in values.iter().enumerate() {
            if value.iter().any(|component| !component.is_finite()) {
                return Err(invalid(
                    label,
                    &format!("vertex {vertex_index} contains a non-finite component"),
                ));
            }
        }
    }
    for (index_position, index) in mesh.indices.iter().enumerate() {
        if usize::from(*index) >= mesh.vertices.len() {
            return Err(invalid(
                "GLB index",
                &format!(
                    "index {index_position} references vertex {index}, but vertex count is {}",
                    mesh.vertices.len()
                ),
            ));
        }
    }
    Ok(())
}

fn make_gltf_json(ctx: &GlbJsonContext<'_>) -> String {
    let mut json = String::new();
    json.push_str("{\"asset\":{\"version\":\"2.0\",\"generator\":\"nd-extractor\"},\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[");

    let mesh_count = ctx.meshes.len();
    let bone_node = |bone: usize| mesh_count + 1 + bone;
    json.push_str("{\"name\":\"SceneRoot\",\"children\":[");
    let mut child_count = 0usize;
    for i in 1..=mesh_count {
        if child_count > 0 {
            json.push(',');
        }
        json.push_str(&i.to_string());
        child_count += 1;
    }
    if let Some(skeleton) = ctx.skeleton {
        for root in skeleton.root_bones() {
            if child_count > 0 {
                json.push(',');
            }
            json.push_str(&bone_node(root).to_string());
            child_count += 1;
        }
    }
    json.push_str("]}");

    for (index, mesh) in ctx.meshes.iter().enumerate() {
        json.push_str(&format!(
            ",{{\"name\":\"{}\",\"mesh\":{index}",
            json_escape(&mesh.report.submesh_name)
        ));
        if ctx.skeleton.is_some() && mesh.skin.is_some() {
            json.push_str(",\"skin\":0");
        }
        json.push('}');
    }
    if let Some(skeleton) = ctx.skeleton {
        let mut children = vec![Vec::new(); skeleton.bone_count()];
        for (bone, parent) in skeleton.parents.iter().enumerate() {
            if *parent >= 0 {
                children[*parent as usize].push(bone);
            }
        }
        for (bone, bone_children) in children.iter().enumerate() {
            json.push_str(&format!(
                ",{{\"name\":\"{}\"",
                json_escape(&skeleton.names[bone])
            ));
            if !bone_children.is_empty() {
                json.push_str(",\"children\":[");
                for (index, child) in bone_children.iter().enumerate() {
                    if index > 0 {
                        json.push(',');
                    }
                    json.push_str(&bone_node(*child).to_string());
                }
                json.push(']');
            }
            let translation = skeleton.translations[bone];
            let rotation = skeleton.rotations[bone];
            let scale = skeleton.scales[bone];
            json.push_str(&format!(
                ",\"translation\":[{},{},{}],\"rotation\":[{},{},{},{}],\"scale\":[{},{},{}]}}",
                translation[0],
                translation[1],
                translation[2],
                rotation[0],
                rotation[1],
                rotation[2],
                rotation[3],
                scale[0],
                scale[1],
                scale[2]
            ));
        }
    }
    json.push_str("],\"meshes\":[");

    for (index, mesh) in ctx.meshes.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        let accessor = &ctx.mesh_accessors[index];
        let mut attributes = format!("\"POSITION\":{}", accessor.position);
        if let Some(normal) = accessor.normal {
            attributes.push_str(&format!(",\"NORMAL\":{normal}"));
        }
        if let Some(tangent) = accessor.tangent {
            attributes.push_str(&format!(",\"TANGENT\":{tangent}"));
        }
        if let Some(uv0) = accessor.uv0 {
            attributes.push_str(&format!(",\"TEXCOORD_0\":{uv0}"));
        }
        if let Some(uv1) = accessor.uv1 {
            attributes.push_str(&format!(",\"TEXCOORD_1\":{uv1}"));
        }
        for set in 0..3 {
            if let (Some(joints), Some(weights)) = (accessor.joints[set], accessor.weights[set]) {
                attributes.push_str(&format!(
                    ",\"JOINTS_{set}\":{joints},\"WEIGHTS_{set}\":{weights}"
                ));
            }
        }
        json.push_str(&format!(
            "{{\"name\":\"{}\",\"primitives\":[{{\"attributes\":{{{attributes}}},\"indices\":{},\"material\":{index},\"mode\":4}}]}}",
            json_escape(&mesh.report.submesh_name),
            accessor.indices
        ));
    }

    // Materials
    json.push_str("],\"materials\":[");
    for (index, mesh) in ctx.meshes.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        let alpha = texture_index(ctx, mesh, TextureRole::Transparency);
        let base_color = texture_index(ctx, mesh, TextureRole::BaseColor)
            .or(alpha)
            .or_else(|| {
                mesh.report.textures.iter().find_map(|texture| {
                    ctx.texture_payloads
                        .iter()
                        .position(|(hash, _, _)| *hash == texture.hash)
                })
            });
        let normal = texture_index(ctx, mesh, TextureRole::Normal);
        let roughness = texture_index(ctx, mesh, TextureRole::Roughness);
        let occlusion = texture_index(ctx, mesh, TextureRole::Occlusion);
        json.push_str("{\"name\":\"Material_");
        json.push_str(&json_escape(&mesh.report.submesh_name));
        json.push_str("\",\"pbrMetallicRoughness\":{\"baseColorFactor\":[1.0,1.0,1.0,1.0],\"metallicFactor\":0.0");
        if roughness.is_some() {
            json.push_str(",\"roughnessFactor\":1.0");
        } else {
            json.push_str(",\"roughnessFactor\":0.8");
        }
        if let Some(tex_idx) = base_color {
            json.push_str(&format!(",\"baseColorTexture\":{{\"index\":{tex_idx}}}"));
        }
        if let Some(tex_idx) = roughness {
            json.push_str(&format!(
                ",\"metallicRoughnessTexture\":{{\"index\":{tex_idx}}}"
            ));
        }
        json.push('}');
        if let Some(tex_idx) = normal {
            json.push_str(&format!(",\"normalTexture\":{{\"index\":{tex_idx}}}"));
        }
        if let Some(tex_idx) = occlusion {
            json.push_str(&format!(",\"occlusionTexture\":{{\"index\":{tex_idx}}}"));
        }
        if alpha.is_some() {
            if alpha_uses_blending(mesh) {
                json.push_str(",\"alphaMode\":\"BLEND\",\"doubleSided\":true");
            } else {
                json.push_str(",\"alphaMode\":\"MASK\",\"alphaCutoff\":0.5,\"doubleSided\":true");
            }
        }
        json.push('}');
    }
    json.push(']');

    if let (Some(skeleton), Some(inverse_bind_accessor)) = (ctx.skeleton, ctx.inverse_bind_accessor)
    {
        json.push_str(",\"skins\":[{\"name\":\"EllieSkeleton\",\"joints\":[");
        for bone in 0..skeleton.bone_count() {
            if bone > 0 {
                json.push(',');
            }
            json.push_str(&bone_node(bone).to_string());
        }
        json.push_str(&format!(
            "],\"inverseBindMatrices\":{inverse_bind_accessor}}}]"
        ));
    }

    // Textures & Images
    if !ctx.texture_payloads.is_empty() {
        json.push_str(",\"textures\":[");
        for (i, _) in ctx.texture_payloads.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("{{\"source\":{i}}}"));
        }
        json.push_str("],\"images\":[");
        for (i, (_hash, name, _)) in ctx.texture_payloads.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            if let Some(view_indices) = ctx.image_view_indices {
                let bv = view_indices[i];
                json.push_str(&format!(
                    "{{\"name\":\"{}\",\"bufferView\":{},\"mimeType\":\"image/png\"}}",
                    json_escape(name),
                    bv
                ));
            } else if let Some(bin_name) = ctx.bin_filename {
                let parent_stem = Path::new(bin_name)
                    .file_stem()
                    .map(|s| s.to_string_lossy())
                    .unwrap_or_else(|| "model".into());
                let png_filename = format!("{}_{:#018x}.png", parent_stem, _hash);
                json.push_str(&format!(
                    "{{\"name\":\"{}\",\"uri\":\"{}\"}}",
                    json_escape(name),
                    json_escape(&png_filename)
                ));
            }
        }
        json.push(']');
    }

    json.push_str(",\"bufferViews\":[");
    for (index, (offset, length, target)) in ctx.views.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        if *target == 0 {
            json.push_str(&format!(
                "{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{length}}}"
            ));
        } else {
            json.push_str(&format!(
                "{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{length},\"target\":{target}}}"
            ));
        }
    }
    json.push_str("],\"accessors\":[");
    for (index, (view, component, count, kind, bounds)) in ctx.accessors.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"bufferView\":{view},\"componentType\":{component},\"count\":{count},\"type\":\"{kind}\""
        ));
        if let Some((min, max)) = bounds {
            json.push_str(&format!(
                ",\"min\":[{},{},{}],\"max\":[{},{},{}]",
                min[0], min[1], min[2], max[0], max[1], max[2]
            ));
        } else if *kind == "SCALAR"
            && let Some(mesh) = ctx
                .mesh_accessors
                .iter()
                .position(|accessor| accessor.indices == index)
                .map(|mesh_index| &ctx.meshes[mesh_index])
        {
            let maximum = mesh.indices.iter().copied().max().unwrap_or(0);
            json.push_str(&format!(",\"min\":[0],\"max\":[{maximum}]"));
        }
        json.push('}');
    }

    if let Some(bin_file) = ctx.bin_filename {
        json.push_str(&format!(
            "],\"buffers\":[{{\"uri\":\"{}\",\"byteLength\":{}}}]}}",
            json_escape(bin_file),
            ctx.binary_len
        ));
    } else {
        json.push_str(&format!(
            "],\"buffers\":[{{\"byteLength\":{}}}]}}",
            ctx.binary_len
        ));
    }
    json
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c <= '\u{1f}' => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

fn align4(bytes: &mut Vec<u8>) {
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
}

fn invalid(field: &'static str, detail: &str) -> PakError {
    PakError::Invalid {
        field,
        offset: 0,
        detail: detail.to_owned(),
    }
}

fn write_atomic(output: &Path, bytes: &[u8]) -> Result<(), PakError> {
    match fs::symlink_metadata(output) {
        Ok(_) => {
            return Err(invalid(
                "GLB output path",
                "already exists; choose a new path",
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(PakError::Io {
                path: output.to_owned(),
                operation: "inspect GLB output path",
                source,
            });
        }
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = output
        .file_name()
        .ok_or_else(|| invalid("GLB output path", "must include a filename"))?;
    let mut temp = None;
    let mut file = None;
    for _ in 0..100 {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            name.to_string_lossy(),
            std::process::id(),
            sequence
        ));
        match File::options()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(created) => {
                temp = Some(candidate);
                file = Some(created);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(PakError::Io {
                    path: candidate,
                    operation: "create temporary GLB",
                    source,
                });
            }
        }
    }
    let temp =
        temp.ok_or_else(|| invalid("GLB output path", "could not allocate temporary file"))?;
    let mut file =
        file.ok_or_else(|| invalid("GLB output file", "temporary file was not created"))?;
    let result = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::rename(&temp, output));
    if let Err(source) = result {
        let _ = fs::remove_file(&temp);
        return Err(PakError::Io {
            path: PathBuf::from(output),
            operation: "write GLB",
            source,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::pak::{MeshBounds, MeshReport, SubmeshInventory};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_inventory() -> GeometryInventory {
        let report = MeshReport {
            pak_path: PathBuf::new(),
            geometry_resource_offset: 0,
            submesh_table_offset: 0,
            submesh_count: 1,
            submesh_table_size: 192,
            submesh_index: 0,
            submesh_name: "head\"mesh".to_owned(),
            submesh_offset: 0,
            material_offset: 0,
            vertex_count: 3,
            vertex_format: "float3".to_owned(),
            index_count: 3,
            triangle_count: 1,
            vertex_buffer_offset: 0,
            vertex_buffer_size: 36,
            vertex_page_index: 0,
            vertex_page_class: 0,
            index_buffer_offset: 36,
            index_buffer_size: 6,
            index_page_index: 0,
            index_page_class: 0,
            bounds: MeshBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 0.0],
            },
            textures: Vec::new(),
            evidence: Vec::new(),
        };
        GeometryInventory {
            geometry_resource_count: 1,
            submeshes: vec![SubmeshInventory {
                index: 0,
                lod: Some(0),
                name: "head\"mesh".to_owned(),
                vertex_count: 3,
                index_count: 3,
                position_type: Some(0),
                position_format: "float3".to_owned(),
                supported: true,
                issue: None,
            }],
            decoded_meshes: vec![DecodedMesh {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                normals: vec![[0.0, 0.0, 1.0]; 3],
                tangents: Vec::new(),
                uv0: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                uv1: Vec::new(),
                indices: vec![0, 1, 2],
                skin: None,
                report,
            }],
        }
    }

    #[test]
    fn fills_empty_skin_influences_from_the_nearest_skinned_vertex() {
        let mut inventory = sample_inventory();
        let mesh = &mut inventory.decoded_meshes[0];
        mesh.skin = Some(crate::container::pak::MeshSkin {
            influences: vec![vec![(3, 1.0)], Vec::new(), vec![(4, 1.0)]],
        });
        fill_missing_skin_influences(mesh);
        let influences = &mesh.skin.as_ref().unwrap().influences;
        assert_eq!(influences[1], vec![(3, 1.0)]);
        assert_eq!(influences[0], vec![(3, 1.0)]);
        assert_eq!(influences[2], vec![(4, 1.0)]);
    }

    #[test]
    fn emits_aligned_glb_with_mesh_and_placeholder_material() {
        let inventory = sample_inventory();
        let temp = std::env::temp_dir().join(format!(
            "nd-extractor-{}.glb",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        export_geometry_glb(&inventory, &temp).unwrap();
        let bytes = fs::read(&temp).unwrap();
        fs::remove_file(&temp).unwrap();
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len()
        );
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(&bytes[16..20], b"JSON");
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        assert!(json.contains("head\\\"mesh"));
        assert!(json.contains("Material_head\\\"mesh"));
        assert!(json.contains("\"componentType\":5125"));
        let bin_header = 20 + json_len;
        let bin_len =
            u32::from_le_bytes(bytes[bin_header..bin_header + 4].try_into().unwrap()) as usize;
        assert_eq!(&bytes[bin_header + 4..bin_header + 8], b"BIN\0");
        assert_eq!(bin_header + 8 + bin_len, bytes.len());
        assert_eq!(bin_len % 4, 0);
    }

    #[test]
    fn rejects_out_of_range_indices() {
        let mut inventory = sample_inventory();
        inventory.decoded_meshes[0].indices[2] = 3;
        assert!(export_geometry_glb(&inventory, "/tmp/should-not-be-written.glb").is_err());
    }

    #[test]
    fn exports_every_lod0_submesh_and_excludes_lod1_to_lod3() {
        let mut inventory = sample_inventory();
        let base_mesh = inventory.decoded_meshes[0].clone();
        inventory.submeshes.clear();
        inventory.decoded_meshes.clear();
        for (index, lod) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)] {
            let name = format!("mesh-{index}-lod{lod}");
            inventory.submeshes.push(SubmeshInventory {
                index,
                lod: Some(lod),
                name: name.clone(),
                vertex_count: 3,
                index_count: 3,
                position_type: Some(0),
                position_format: "float3".to_owned(),
                supported: true,
                issue: None,
            });
            let mut mesh = base_mesh.clone();
            mesh.report.submesh_index = index;
            mesh.report.submesh_name = name;
            inventory.decoded_meshes.push(mesh);
        }
        let temp = std::env::temp_dir().join(format!(
            "nd-extractor-lod0-{}.glb",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let summary = export_geometry_glb(&inventory, &temp).unwrap();
        assert_eq!(
            summary,
            GlbExportSummary {
                mesh_count: 2,
                vertex_count: 6,
                triangle_count: 2
            }
        );
        let bytes = fs::read(&temp).unwrap();
        fs::remove_file(&temp).unwrap();
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        assert!(json.contains("mesh-0-lod0"));
        assert!(json.contains("mesh-1-lod0"));
        for lod in 1..=3 {
            assert!(!json.contains(&format!("lod{lod}")));
        }
        assert_eq!(json.matches("\"primitives\"").count(), 2);
    }

    #[test]
    fn refuses_to_silently_drop_an_unsupported_lod0_submesh() {
        let mut inventory = sample_inventory();
        inventory.submeshes[0].supported = false;
        inventory.submeshes[0].issue = Some("unsupported format".to_owned());
        assert!(export_geometry_glb(&inventory, "/tmp/unsupported-lod0.glb").is_err());
    }

    #[test]
    fn does_not_report_replacement_error() {
        let temp = std::env::temp_dir().join(format!(
            "nd-extractor-existing-{}.glb",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&temp, b"user data").unwrap();
        assert!(export_geometry_glb(&sample_inventory(), &temp).is_err());
        assert_eq!(fs::read(&temp).unwrap(), b"user data");
        fs::remove_file(temp).unwrap();
    }

    #[test]
    fn emits_static_glb_with_real_normals_and_uvs() {
        let inventory = sample_inventory();
        let temp = std::env::temp_dir().join(format!(
            "nd-extractor-static-{}.glb",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let summary = export_comprehensive_glb(
            Path::new("dummy.pak"),
            &inventory,
            None::<&Path>,
            None::<&Path>,
            &temp,
        )
        .unwrap();
        assert_eq!(summary.mesh_count, 1);
        let bytes = fs::read(&temp).unwrap();
        fs::remove_file(&temp).unwrap();
        assert_eq!(&bytes[0..4], b"glTF");
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        assert!(json.contains("\"NORMAL\""));
        assert!(json.contains("\"TEXCOORD_0\""));
        assert!(!json.contains("\"skins\""));
        assert!(!json.contains("\"JOINTS_0\""));
        assert!(!json.contains("\"WEIGHTS_0\""));
    }

    #[test]
    fn textures_reference_images_through_the_gltf_source_field() {
        let inventory = sample_inventory();
        let meshes = inventory.decoded_meshes;
        let views = vec![(0usize, 1usize, 0u32)];
        let accessors = vec![
            (0usize, 5126u32, 3usize, "VEC3", Some(([0.0; 3], [1.0; 3]))),
            (0usize, 5125u32, 3usize, "SCALAR", None),
        ];
        let mesh_accessors = vec![MeshAccessors {
            position: 0,
            normal: None,
            tangent: None,
            uv0: None,
            uv1: None,
            joints: [None; 3],
            weights: [None; 3],
            indices: 1,
        }];
        let texture_payloads = vec![(1u64, "texture".to_owned(), vec![0u8])];
        let image_view_indices = [0usize];
        let ctx = GlbJsonContext {
            meshes: &meshes,
            views: &views,
            accessors: &accessors,
            binary_len: 1,
            mesh_accessors: &mesh_accessors,
            texture_payloads: &texture_payloads,
            image_view_indices: Some(&image_view_indices),
            bin_filename: None,
            skeleton: None,
            inverse_bind_accessor: None,
        };
        let json = make_gltf_json(&ctx);
        assert!(json.contains("\"textures\":[{\"source\":0}]"));
        assert!(json.contains(
            "\"images\":[{\"name\":\"texture\",\"bufferView\":0,\"mimeType\":\"image/png\"}]"
        ));
    }
    #[test]
    fn classifies_transparency_and_ignores_overlay_colors_as_base_color() {
        assert_eq!(
            texture_role("g_tNdFetchTransparency01Map"),
            TextureRole::Transparency
        );
        assert_eq!(
            texture_role("g_tNdFetchTearMaskMap"),
            TextureRole::Transparency
        );
        assert_eq!(
            texture_role("g_tNdFetchFabricBurnColorMap"),
            TextureRole::Other
        );
        assert_eq!(
            texture_role("g_tNdFetchNewBloodColorMap"),
            TextureRole::Other
        );
        assert_eq!(
            texture_role("g_tNdFetchBaseColor01Map"),
            TextureRole::BaseColor
        );
        assert_eq!(
            texture_role("g_tNdFetchIrisColor01Map"),
            TextureRole::BaseColor
        );
    }

    #[test]
    fn copies_opacity_mask_into_base_color_alpha() {
        let mut base = vec![255_u8; 2 * 2 * 4];
        for pixel in base.as_chunks_mut::<4>().0 {
            pixel[0] = 10;
        }
        let alpha = vec![0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 255, 0, 0, 0, 64];
        copy_alpha_channel(&mut base, 2, 2, &alpha, 2, 2);
        assert_eq!(
            base.as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            vec![0, 128, 255, 64]
        );
        assert!(base.as_chunks::<4>().0.iter().all(|pixel| pixel[0] == 10));
    }
}
