use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::container::pak::{PakArchive, PakError};

const JOINT_HIERARCHY: &str = "JOINT_HIERARCHY";
const RESOURCE_PADDING: u64 = 48;

#[derive(Debug, Clone, PartialEq)]
pub struct Skeleton {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    pub chain_ids: Vec<i32>,
    pub helper_bones: Vec<bool>,
    pub translations: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
    pub world_matrices: Vec<[[f32; 4]; 4]>,
    pub inverse_bind_matrices: Vec<[f32; 16]>,
}

impl Skeleton {
    pub fn parse_path(path: impl AsRef<Path>) -> Result<Self, PakError> {
        let path = path.as_ref();
        let archive = PakArchive::parse_path(path)?;
        let resource = archive
            .resources
            .iter()
            .find(|resource| resource.resource_type == JOINT_HIERARCHY)
            .ok_or_else(|| invalid("skeleton", 0, "no JOINT_HIERARCHY resource found"))?;
        let mut file = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "open skeleton PAK",
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read skeleton PAK metadata",
                source,
            })?
            .len();

        let header = resource.file_offset + RESOURCE_PADDING;
        let bone_count = read_u32_at(&mut file, file_size, header + 20, "bone count")? as usize;
        if bone_count == 0 || bone_count > 100_000 {
            return Err(invalid(
                "skeleton bone count",
                header + 20,
                format!("{bone_count} is outside the supported range"),
            ));
        }
        let _flags = read_u64_at(&mut file, file_size, header + 40, "joint flags")?;
        let _unknown = read_u64_at(&mut file, file_size, header + 48, "joint unknown")?;

        let xforms_field = header + 32;
        let xforms_raw = read_i64_at(&mut file, file_size, xforms_field, "transform table")?;
        let xforms_offset = archive
            .resolve_pointer(xforms_field, xforms_raw)?
            .ok_or_else(|| {
                invalid(
                    "transform table pointer",
                    xforms_field,
                    "has no pointer fixup",
                )
            })?;
        let names_field = header + 56;
        let names_raw = read_i64_at(&mut file, file_size, names_field, "bone names")?;
        let names_offset = archive
            .resolve_pointer(names_field, names_raw)?
            .ok_or_else(|| invalid("bone names pointer", names_field, "has no pointer fixup"))?;

        let xforms_header = xforms_offset + 16;
        let node_count = read_u16_at(&mut file, file_size, xforms_header, "node count")? as usize;
        let xform_count =
            read_u16_at(&mut file, file_size, xforms_header + 2, "transform count")? as usize;
        if node_count != bone_count {
            return Err(invalid(
                "skeleton node count",
                xforms_header,
                format!("node table says {node_count}, hierarchy says {bone_count}"),
            ));
        }
        if xform_count == 0 || xform_count > bone_count {
            return Err(invalid(
                "skeleton transform count",
                xforms_header + 2,
                format!("{xform_count} is outside 1..={bone_count}"),
            ));
        }
        let header_size = read_u32_at(
            &mut file,
            file_size,
            xforms_header + 16,
            "transform header size",
        )?;
        let hierarchy_offset =
            read_u32_at(&mut file, file_size, xforms_header + 44, "hierarchy offset")?;
        let transforms_offset = xforms_offset
            .checked_add(u64::from(header_size))
            .ok_or_else(|| invalid("skeleton transforms", xforms_offset, "offset overflow"))?;
        let hierarchy_start = xforms_offset
            .checked_add(u64::from(hierarchy_offset))
            .ok_or_else(|| invalid("skeleton hierarchy", xforms_offset, "offset overflow"))?;
        let hashes_size = read_u32_at(
            &mut file,
            file_size,
            hierarchy_start + 20,
            "hierarchy hashes size",
        )?;
        let parenting_start = hierarchy_start
            .checked_add(u64::from(hashes_size))
            .ok_or_else(|| invalid("skeleton parenting", hierarchy_start, "offset overflow"))?;
        let parenting_bytes = bone_count
            .checked_mul(16)
            .ok_or_else(|| invalid("skeleton parenting", parenting_start, "size overflow"))?;
        ensure_range(&mut file, file_size, parenting_start, parenting_bytes)?;
        let parenting = read_bytes_at(
            &mut file,
            file_size,
            parenting_start,
            parenting_bytes,
            "parent table",
        )?;
        let mut parents = Vec::with_capacity(bone_count);
        let mut chain_ids = Vec::with_capacity(bone_count);
        for entry in parenting.as_chunks::<16>().0 {
            parents.push(i32::from_le_bytes(
                entry[4..8].try_into().expect("four bytes"),
            ));
            chain_ids.push(i32::from_le_bytes(
                entry[12..16].try_into().expect("four bytes"),
            ));
        }

        let names_bytes = bone_count
            .checked_mul(16)
            .ok_or_else(|| invalid("skeleton names", names_offset, "size overflow"))?;
        ensure_range(&mut file, file_size, names_offset, names_bytes)?;
        let name_entries = read_bytes_at(
            &mut file,
            file_size,
            names_offset,
            names_bytes,
            "bone name table",
        )?;
        let mut names = Vec::with_capacity(bone_count);
        for (index, entry) in name_entries.as_chunks::<16>().0.iter().enumerate() {
            let field = names_offset + (index * 16 + 8) as u64;
            let raw = i64::from_le_bytes(entry[8..16].try_into().expect("eight bytes"));
            let address = archive
                .resolve_pointer(field, raw)?
                .ok_or_else(|| invalid("bone name pointer", field, "has no pointer fixup"))?;
            names.push(read_c_string(&mut file, file_size, address, "bone name")?);
        }

        let transform_bytes = xform_count
            .checked_mul(48)
            .ok_or_else(|| invalid("skeleton transforms", transforms_offset, "size overflow"))?;
        ensure_range(&mut file, file_size, transforms_offset, transform_bytes)?;
        let transform_data = read_bytes_at(
            &mut file,
            file_size,
            transforms_offset,
            transform_bytes,
            "local transforms",
        )?;
        let mut parsed_transforms = Vec::with_capacity(xform_count);
        for entry in transform_data.as_chunks::<48>().0 {
            parsed_transforms.push(LocalTransform {
                scale: read_vec3(entry, 0),
                rotation: normalize_quaternion(read_vec4(entry, 16)),
                translation: read_vec3(entry, 32),
            });
        }

        let bone_map = main_bone_map(&parents)?;
        let helper_bones = (0..bone_count)
            .map(|bone| Ok(root_parent(&parents, bone)? != 0))
            .collect::<Result<Vec<_>, PakError>>()?;
        let mut is_main = vec![false; bone_count];
        for &bone in &bone_map {
            is_main[bone] = true;
        }
        let mut translations = vec![[0.0_f32; 3]; bone_count];
        let mut rotations = vec![[0.0_f32, 0.0, 0.0, 1.0]; bone_count];
        let mut scales = vec![[1.0_f32; 3]; bone_count];
        for (transform, &bone) in parsed_transforms.iter().zip(&bone_map) {
            translations[bone] = transform.translation;
            rotations[bone] = transform.rotation;
            scales[bone] = transform.scale;
        }
        for bone in 0..bone_count {
            if is_main[bone] {
                continue;
            }
            let chain_id = chain_ids[bone];
            if chain_id < 0 {
                continue;
            }
            let Ok(position) = bone_map.binary_search(&(chain_id as usize)) else {
                continue;
            };
            let Some(transform) = parsed_transforms.get(position) else {
                continue;
            };
            translations[bone] = transform.translation;
            rotations[bone] = transform.rotation;
            scales[bone] = transform.scale;
        }

        for bone in 0..bone_count {
            if is_main[bone] || parents[bone] >= 0 {
                continue;
            }
            parents[bone] = matched_main_bone(&names[bone], &names, &bone_map)
                .map(|matched| matched as i32)
                .unwrap_or(0);
        }

        let world_matrices = compose_world_matrices(&parents, &translations, &rotations, &scales);
        let inverse_bind_matrices = world_matrices
            .iter()
            .map(|matrix| {
                inverse_affine(matrix)
                    .map(flatten_column_major)
                    .ok_or_else(|| invalid("skeleton inverse bind", 0, "singular bone matrix"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            names,
            parents,
            chain_ids,
            helper_bones,
            translations,
            rotations,
            scales,
            world_matrices,
            inverse_bind_matrices,
        })
    }

    pub fn bone_count(&self) -> usize {
        self.names.len()
    }

    pub fn root_bones(&self) -> Vec<usize> {
        self.parents
            .iter()
            .enumerate()
            .filter_map(|(index, parent)| (*parent < 0).then_some(index))
            .collect()
    }

    pub fn helper_target(&self, bone: usize) -> Option<usize> {
        if !self.helper_bones.get(bone).copied().unwrap_or(false) {
            return None;
        }
        let main_bones: Vec<usize> = (0..self.bone_count())
            .filter(|candidate| !self.helper_bones[*candidate])
            .collect();
        if let Some(target) = direct_main_match(&self.names[bone], &self.names, &main_bones) {
            return Some(target);
        }
        if let Some(target) = prefix_main_match(&self.names[bone], &self.names, &main_bones) {
            return Some(target);
        }
        let mut current = self.parents[bone];
        while current >= 0 {
            let parent = current as usize;
            if !self.helper_bones[parent] {
                return Some(parent);
            }
            if let Some(target) = direct_main_match(&self.names[parent], &self.names, &main_bones)
                .or_else(|| prefix_main_match(&self.names[parent], &self.names, &main_bones))
            {
                return Some(target);
            }
            current = self.parents[parent];
        }
        None
    }

    pub fn reparent_bone(&mut self, bone: usize, new_parent: usize) -> Result<(), PakError> {
        if bone >= self.bone_count() || new_parent >= self.bone_count() {
            return Err(invalid(
                "skeleton reparent",
                0,
                format!("bone {bone} or parent {new_parent} is out of range"),
            ));
        }
        let parent_world = self.world_matrices[new_parent];
        let inverse = inverse_affine(&parent_world)
            .ok_or_else(|| invalid("skeleton reparent", 0, "singular parent matrix"))?;
        let local = multiply(&inverse, &self.world_matrices[bone]);
        let (translation, rotation, scale) = decompose_trs(&local);
        self.parents[bone] = new_parent as i32;
        self.translations[bone] = translation;
        self.rotations[bone] = rotation;
        self.scales[bone] = scale;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalTransform {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

fn main_bone_map(parents: &[i32]) -> Result<Vec<usize>, PakError> {
    let mut result = Vec::new();
    for bone in 0..parents.len() {
        if root_parent(parents, bone)? == 0 {
            result.push(bone);
        }
    }
    Ok(result)
}

fn matched_main_bone(name: &str, names: &[String], bone_map: &[usize]) -> Option<usize> {
    direct_main_match(name, names, bone_map).or_else(|| prefix_main_match(name, names, bone_map))
}

fn direct_main_match(name: &str, names: &[String], bone_map: &[usize]) -> Option<usize> {
    let suffix = name.rsplit('_').next().unwrap_or(name);
    if let Some(bone) = best_main_match(name, names, bone_map) {
        return Some(bone);
    }
    if suffix.is_empty() || suffix.chars().all(|character| character.is_ascii_digit()) {
        return None;
    }
    let stripped = name.replace(&format!("_{suffix}"), "");
    best_main_match(&stripped, names, bone_map)
}

fn prefix_main_match(name: &str, names: &[String], bone_map: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for &bone in bone_map {
        let candidate = &names[bone];
        if candidate.len() >= 4
            && name.starts_with(candidate)
            && best.is_none_or(|(best_length, _)| candidate.len() > best_length)
        {
            best = Some((candidate.len(), bone));
        }
    }
    best.map(|(_, bone)| bone)
}

fn best_main_match(matched_name: &str, names: &[String], bone_map: &[usize]) -> Option<usize> {
    if matched_name.is_empty() {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for &bone in bone_map {
        if names[bone] == matched_name {
            return Some(bone);
        }
        if names[bone].contains(matched_name) {
            let length = names[bone].len();
            if best.is_none_or(|(best_length, _)| length < best_length) {
                best = Some((length, bone));
            }
        }
    }
    best.map(|(_, bone)| bone)
}

fn root_parent(parents: &[i32], bone: usize) -> Result<usize, PakError> {
    let mut current = bone;
    let mut visited = 0usize;
    loop {
        let parent = *parents
            .get(current)
            .ok_or_else(|| invalid("skeleton parent", 0, format!("bone {current} is missing")))?;
        if parent < 0 {
            return Ok(current);
        }
        let parent = parent as usize;
        if parent >= parents.len() || visited > parents.len() {
            return Err(invalid(
                "skeleton parent",
                0,
                format!("cycle or invalid parent for bone {bone}"),
            ));
        }
        current = parent;
        visited += 1;
    }
}

fn compose_world_matrices(
    parents: &[i32],
    translations: &[[f32; 3]],
    rotations: &[[f32; 4]],
    scales: &[[f32; 3]],
) -> Vec<[[f32; 4]; 4]> {
    let count = parents.len();
    let mut worlds = vec![identity(); count];
    let mut resolved = vec![false; count];
    for bone in 0..count {
        resolve_world(
            bone,
            parents,
            translations,
            rotations,
            scales,
            &mut worlds,
            &mut resolved,
        );
    }
    worlds
}

fn resolve_world(
    bone: usize,
    parents: &[i32],
    translations: &[[f32; 3]],
    rotations: &[[f32; 4]],
    scales: &[[f32; 3]],
    worlds: &mut [[[f32; 4]; 4]],
    resolved: &mut [bool],
) -> [[f32; 4]; 4] {
    if resolved[bone] {
        return worlds[bone];
    }
    let local = compose_trs(translations[bone], rotations[bone], scales[bone]);
    let parent = parents[bone];
    let world = if parent < 0 || parent as usize >= parents.len() {
        local
    } else {
        let parent = parent as usize;
        if parent == bone {
            local
        } else {
            multiply(
                &resolve_world(
                    parent,
                    parents,
                    translations,
                    rotations,
                    scales,
                    worlds,
                    resolved,
                ),
                &local,
            )
        }
    };
    worlds[bone] = world;
    resolved[bone] = true;
    world
}

fn compose_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [[f32; 4]; 4] {
    let r = quaternion_matrix(rotation);
    [
        [
            r[0][0] * scale[0],
            r[0][1] * scale[1],
            r[0][2] * scale[2],
            translation[0],
        ],
        [
            r[1][0] * scale[0],
            r[1][1] * scale[1],
            r[1][2] * scale[2],
            translation[1],
        ],
        [
            r[2][0] * scale[0],
            r[2][1] * scale[1],
            r[2][2] * scale[2],
            translation[2],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn decompose_trs(matrix: &[[f32; 4]; 4]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let translation = [matrix[0][3], matrix[1][3], matrix[2][3]];
    let mut scale = [
        (matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0] + matrix[2][0] * matrix[2][0])
            .sqrt(),
        (matrix[0][1] * matrix[0][1] + matrix[1][1] * matrix[1][1] + matrix[2][1] * matrix[2][1])
            .sqrt(),
        (matrix[0][2] * matrix[0][2] + matrix[1][2] * matrix[1][2] + matrix[2][2] * matrix[2][2])
            .sqrt(),
    ];
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if determinant < 0.0 {
        scale[0] = -scale[0];
    }
    let inverse = scale.map(|value| {
        if value.abs() > 1.0e-8 {
            1.0 / value
        } else {
            0.0
        }
    });
    let rotation_matrix = [
        [
            matrix[0][0] * inverse[0],
            matrix[0][1] * inverse[1],
            matrix[0][2] * inverse[2],
        ],
        [
            matrix[1][0] * inverse[0],
            matrix[1][1] * inverse[1],
            matrix[1][2] * inverse[2],
        ],
        [
            matrix[2][0] * inverse[0],
            matrix[2][1] * inverse[1],
            matrix[2][2] * inverse[2],
        ],
    ];
    (translation, quaternion_from_matrix(&rotation_matrix), scale)
}

fn quaternion_from_matrix(matrix: &[[f32; 3]; 3]) -> [f32; 4] {
    let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            (matrix[2][1] - matrix[1][2]) / s,
            (matrix[0][2] - matrix[2][0]) / s,
            (matrix[1][0] - matrix[0][1]) / s,
            0.25 * s,
        )
    } else if matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2] {
        let s = (1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]).sqrt() * 2.0;
        (
            0.25 * s,
            (matrix[0][1] + matrix[1][0]) / s,
            (matrix[0][2] + matrix[2][0]) / s,
            (matrix[2][1] - matrix[1][2]) / s,
        )
    } else if matrix[1][1] > matrix[2][2] {
        let s = (1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]).sqrt() * 2.0;
        (
            (matrix[0][1] + matrix[1][0]) / s,
            0.25 * s,
            (matrix[1][2] + matrix[2][1]) / s,
            (matrix[0][2] - matrix[2][0]) / s,
        )
    } else {
        let s = (1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]).sqrt() * 2.0;
        (
            (matrix[0][2] + matrix[2][0]) / s,
            (matrix[1][2] + matrix[2][1]) / s,
            0.25 * s,
            (matrix[1][0] - matrix[0][1]) / s,
        )
    };
    normalize_quaternion([x, y, z, w])
}

fn quaternion_matrix(q: [f32; 4]) -> [[f32; 4]; 4] {
    let [x, y, z, w] = q;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0_f32; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            result[row][column] = (0..4)
                .map(|k| f64::from(a[row][k]) * f64::from(b[k][column]))
                .sum::<f64>() as f32;
        }
    }
    result
}

fn inverse_affine(matrix: &[[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut work = [[0.0_f64; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            work[row][column] = f64::from(matrix[row][column]);
        }
        work[row][4 + row] = 1.0;
    }
    for pivot in 0..4 {
        let best = (pivot..4).max_by(|left, right| {
            work[*left][pivot]
                .abs()
                .total_cmp(&work[*right][pivot].abs())
        })?;
        if work[best][pivot].abs() < 1.0e-12 {
            return None;
        }
        work.swap(pivot, best);
        let divisor = work[pivot][pivot];
        for value in &mut work[pivot] {
            *value /= divisor;
        }
        for row in 0..4 {
            if row == pivot {
                continue;
            }
            let factor = work[row][pivot];
            let pivot_row = work[pivot];
            for (column, value) in work[row].iter_mut().enumerate() {
                *value -= factor * pivot_row[column];
            }
        }
    }
    let mut result = [[0.0_f32; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            result[row][column] = work[row][4 + column] as f32;
        }
    }
    Some(result)
}

fn flatten_column_major(matrix: [[f32; 4]; 4]) -> [f32; 16] {
    let mut result = [0.0_f32; 16];
    for column in 0..4 {
        for row in 0..4 {
            result[column * 4 + row] = matrix[row][column];
        }
    }
    result
}

fn normalize_quaternion(mut q: [f32; 4]) -> [f32; 4] {
    let length = (q
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>())
    .sqrt();
    if length < 1.0e-8 || !length.is_finite() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    for value in &mut q {
        *value = (f64::from(*value) / length) as f32;
    }
    q
}

fn read_vec3(bytes: &[u8], offset: usize) -> [f32; 3] {
    [
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes")),
        f32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .expect("four bytes"),
        ),
        f32::from_le_bytes(
            bytes[offset + 8..offset + 12]
                .try_into()
                .expect("four bytes"),
        ),
    ]
}

fn read_vec4(bytes: &[u8], offset: usize) -> [f32; 4] {
    [
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes")),
        f32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .expect("four bytes"),
        ),
        f32::from_le_bytes(
            bytes[offset + 8..offset + 12]
                .try_into()
                .expect("four bytes"),
        ),
        f32::from_le_bytes(
            bytes[offset + 12..offset + 16]
                .try_into()
                .expect("four bytes"),
        ),
    ]
}

fn ensure_range(
    file: &mut File,
    file_size: u64,
    offset: u64,
    length: usize,
) -> Result<(), PakError> {
    let end = offset
        .checked_add(length as u64)
        .ok_or_else(|| invalid("skeleton range", offset, "offset overflow"))?;
    if end > file_size {
        return Err(invalid(
            "skeleton range",
            offset,
            format!("{length} bytes extend past the {file_size}-byte file"),
        ));
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| PakError::Io {
            path: PathBuf::new(),
            operation: "seek skeleton PAK",
            source,
        })?;
    Ok(())
}

fn read_bytes_at(
    file: &mut File,
    file_size: u64,
    offset: u64,
    length: usize,
    field: &'static str,
) -> Result<Vec<u8>, PakError> {
    ensure_range(file, file_size, offset, length)?;
    let mut bytes = vec![0_u8; length];
    file.read_exact(&mut bytes).map_err(|source| PakError::Io {
        path: PathBuf::new(),
        operation: field,
        source,
    })?;
    Ok(bytes)
}

fn read_u16_at(
    file: &mut File,
    file_size: u64,
    offset: u64,
    field: &'static str,
) -> Result<u16, PakError> {
    let bytes = read_bytes_at(file, file_size, offset, 2, field)?;
    Ok(u16::from_le_bytes(bytes.try_into().expect("two bytes")))
}

fn read_u32_at(
    file: &mut File,
    file_size: u64,
    offset: u64,
    field: &'static str,
) -> Result<u32, PakError> {
    let bytes = read_bytes_at(file, file_size, offset, 4, field)?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
}

fn read_u64_at(
    file: &mut File,
    file_size: u64,
    offset: u64,
    field: &'static str,
) -> Result<u64, PakError> {
    let bytes = read_bytes_at(file, file_size, offset, 8, field)?;
    Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
}

fn read_i64_at(
    file: &mut File,
    file_size: u64,
    offset: u64,
    field: &'static str,
) -> Result<i64, PakError> {
    let bytes = read_bytes_at(file, file_size, offset, 8, field)?;
    Ok(i64::from_le_bytes(bytes.try_into().expect("eight bytes")))
}

fn read_c_string(
    file: &mut File,
    file_size: u64,
    offset: u64,
    field: &'static str,
) -> Result<String, PakError> {
    if offset >= file_size {
        return Err(invalid(field, offset, "string starts outside the file"));
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| PakError::Io {
            path: PathBuf::new(),
            operation: "seek skeleton string",
            source,
        })?;
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 256];
    loop {
        let read = file.read(&mut buffer).map_err(|source| PakError::Io {
            path: PathBuf::new(),
            operation: field,
            source,
        })?;
        if read == 0 {
            break;
        }
        if let Some(end) = buffer[..read].iter().position(|byte| *byte == 0) {
            bytes.extend_from_slice(&buffer[..end]);
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
        if bytes.len() > 4096 {
            return Err(invalid(field, offset, "string exceeds 4096 bytes"));
        }
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn invalid(field: &'static str, offset: u64, detail: impl Into<String>) -> PakError {
    PakError::Invalid {
        field,
        offset,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_installed_ellie_skeleton() {
        let path = Path::new("/mnt/storage/tlou2-extraction/common/actor97/ellie-skel.pak");
        if !path.exists() {
            return;
        }
        let skeleton = Skeleton::parse_path(path).unwrap();
        assert_eq!(skeleton.bone_count(), 1074);
        assert_eq!(skeleton.names[0], "root");
        assert_eq!(skeleton.parents[0], -1);
        assert!(skeleton.names.iter().any(|name| name == "headb"));
        assert_eq!(skeleton.root_bones(), vec![0]);
        assert_eq!(
            skeleton
                .helper_bones
                .iter()
                .filter(|helper| **helper)
                .count(),
            554
        );
        assert_eq!(skeleton.inverse_bind_matrices.len(), 1074);
        assert!((skeleton.world_matrices[0][1][3] - 0.945).abs() < 1.0e-3);
    }

    #[test]
    fn matches_helper_bones_to_main_targets() {
        let path = Path::new("/mnt/storage/tlou2-extraction/common/actor97/ellie-skel.pak");
        if !path.exists() {
            return;
        }
        let skeleton = Skeleton::parse_path(path).unwrap();
        let helper = skeleton
            .names
            .iter()
            .position(|name| name == "l_middlea_helper")
            .unwrap();
        let target = skeleton
            .names
            .iter()
            .position(|name| name == "l_middlea")
            .unwrap();
        assert_eq!(skeleton.helper_target(helper), Some(target));

        let helper = skeleton
            .names
            .iter()
            .position(|name| name == "l_twist_10")
            .unwrap();
        let target = skeleton
            .names
            .iter()
            .position(|name| name == "l_twist_10_mover_offset")
            .unwrap();
        assert_eq!(skeleton.helper_target(helper), Some(target));

        let helper = skeleton
            .names
            .iter()
            .position(|name| name == "l_twist_04_offset")
            .unwrap();
        let target = skeleton
            .names
            .iter()
            .position(|name| name == "l_twist_04_mover_offset")
            .unwrap();
        assert_eq!(skeleton.helper_target(helper), Some(target));
    }

    #[test]
    fn reparent_preserves_world_transform() {
        let path = Path::new("/mnt/storage/tlou2-extraction/common/actor97/ellie-skel.pak");
        if !path.exists() {
            return;
        }
        let mut skeleton = Skeleton::parse_path(path).unwrap();
        let bone = skeleton
            .names
            .iter()
            .position(|name| name == "headb")
            .unwrap();
        let new_parent = skeleton
            .names
            .iter()
            .position(|name| name == "root")
            .unwrap();
        let before = skeleton.world_matrices[bone];
        skeleton.reparent_bone(bone, new_parent).unwrap();
        assert_eq!(skeleton.parents[bone], new_parent as i32);
        let worlds = compose_world_matrices(
            &skeleton.parents,
            &skeleton.translations,
            &skeleton.rotations,
            &skeleton.scales,
        );
        let after = worlds[bone];
        let delta = (0..3)
            .flat_map(|row| (0..4).map(move |column| (row, column)))
            .map(|(row, column)| (before[row][column] - after[row][column]).abs())
            .fold(0.0_f32, f32::max);
        assert!(delta < 1.0e-4, "world transform changed by {delta}");
    }
}
