use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::container::pak::{DecodedMesh, GeometryInventory, PakError};

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

/// Writes the decoded, validated positions and triangle indices as a static GLB.
/// The source PAK does not currently provide decoded normals, UVs, or skinning here.
pub fn export_geometry_glb(
    inventory: &GeometryInventory,
    output: impl AsRef<Path>,
) -> Result<GlbExportSummary, PakError> {
    let lod0: Vec<_> = inventory
        .submeshes
        .iter()
        .filter(|submesh| submesh.lod == Some(0))
        .collect();
    if lod0.is_empty() {
        return Err(invalid("geometry", "no LOD0 submeshes are identified"));
    }
    let mut meshes = Vec::with_capacity(lod0.len());
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
        let mesh = inventory
            .decoded_meshes
            .iter()
            .find(|mesh| mesh.report.submesh_index == submesh.index)
            .ok_or_else(|| {
                invalid(
                    "LOD0 submesh",
                    &format!("{} was not decoded", submesh.index),
                )
            })?;
        meshes.push(mesh.clone());
    }
    if meshes.len() != lod0.len() {
        return Err(invalid(
            "LOD0 submeshes",
            "not all identified LOD0 entries were selected",
        ));
    }

    let mut binary = Vec::new();
    let mut views = Vec::with_capacity(meshes.len() * 2);
    let mut accessors: Vec<AccessorDef> = Vec::with_capacity(meshes.len() * 2);
    for mesh in &meshes {
        validate_mesh(mesh)?;
        align4(&mut binary);
        let positions_offset = binary.len();
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
                binary.extend_from_slice(&vertex[axis].to_le_bytes());
            }
        }
        let positions_length = binary.len() - positions_offset;
        views.push((positions_offset, positions_length, 34962));
        accessors.push((
            views.len() - 1,
            5126,
            mesh.vertices.len(),
            "VEC3",
            Some((min, max)),
        ));

        align4(&mut binary);
        let indices_offset = binary.len();
        for index in &mesh.indices {
            let index = u32::from(*index);
            binary.extend_from_slice(&index.to_le_bytes());
        }
        let indices_length = binary.len() - indices_offset;
        views.push((indices_offset, indices_length, 34963));
        accessors.push((views.len() - 1, 5125, mesh.indices.len(), "SCALAR", None));
    }
    align4(&mut binary);
    let json = make_json(&meshes, &views, &accessors, binary.len());
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

    write_atomic(output.as_ref(), &glb)?;
    Ok(GlbExportSummary {
        mesh_count: meshes.len(),
        vertex_count: meshes.iter().map(|mesh| mesh.vertices.len()).sum(),
        triangle_count: meshes.iter().map(|mesh| mesh.indices.len() / 3).sum(),
    })
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

fn make_json(
    meshes: &[DecodedMesh],
    views: &[(usize, usize, u32)],
    accessors: &[AccessorDef],
    binary_len: usize,
) -> String {
    let mut json = String::new();
    json.push_str("{\"asset\":{\"version\":\"2.0\",\"generator\":\"nd-extractor\"},\"scene\":0,\"scenes\":[{\"nodes\":[");
    for index in 0..meshes.len() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&index.to_string());
    }
    json.push_str("]}],\"nodes\":[");
    for (index, mesh) in meshes.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"name\":\"{}\",\"mesh\":{index}}}",
            json_escape(&mesh.report.submesh_name)
        ));
    }
    json.push_str("],\"meshes\":[");
    for (index, mesh) in meshes.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&format!("{{\"name\":\"{}\",\"primitives\":[{{\"attributes\":{{\"POSITION\":{} }},\"indices\":{},\"material\":0,\"mode\":4}}]}}", json_escape(&mesh.report.submesh_name), index * 2, index * 2 + 1));
    }
    json.push_str("],\"materials\":[{\"name\":\"Untextured placeholder\",\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.7,0.7,0.7,1.0],\"metallicFactor\":0.0,\"roughnessFactor\":0.8}}],\"bufferViews\":[");
    for (index, (offset, length, target)) in views.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{length},\"target\":{target}}}"
        ));
    }
    json.push_str("],\"accessors\":[");
    for (index, (view, component, count, kind, bounds)) in accessors.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&format!("{{\"bufferView\":{view},\"componentType\":{component},\"count\":{count},\"type\":\"{kind}\""));
        if let Some((min, max)) = bounds {
            json.push_str(&format!(
                ",\"min\":[{},{},{}],\"max\":[{},{},{}]",
                min[0], min[1], min[2], max[0], max[1], max[2]
            ));
        } else {
            let mesh = &meshes[index / 2];
            let maximum = mesh.indices.iter().copied().max().unwrap_or(0);
            json.push_str(&format!(",\"min\":[0],\"max\":[{maximum}]"));
        }
        json.push('}');
    }
    json.push_str(&format!(
        "],\"buffers\":[{{\"byteLength\":{binary_len}}}]}}"
    ));
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
                indices: vec![0, 1, 2],
                report,
            }],
        }
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
        assert!(json.contains("Untextured placeholder"));
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
    fn does_not_replace_an_existing_output() {
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
}
