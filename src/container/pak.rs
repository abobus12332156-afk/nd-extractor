use std::fmt;
use std::fs::{self, File};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::binary::BinaryReader;
use crate::error::BinaryError;

const PAK_MAGICS: [u32; 3] = [0x0000_0a79, 0x0001_0a79, 0x8000_0a79];
const HEADER_SIZE: usize = 44;
const PAGE_ENTRY_SIZE: usize = 12;
const PAGE_HEADER_SIZE: usize = 20;
const RESOURCE_ENTRY_SIZE: usize = 16;
const FIXUP_ENTRY_SIZE: usize = 8;
const TEXTURE_GRAPH_HEADER_SIZE: usize = 80;
const TEXTURE_GRAPH_COUNT_OFFSET: usize = 48;
const TEXTURE_GRAPH_POINTER_OFFSETS: [usize; 3] = [56, 64, 72];
const MAX_TEXTURE_GRAPH_ENTRIES: usize = 10_000;
const MAX_TEXTURE_GRAPH_NODES: usize = 10_000;
const MAX_TEXTURE_GRAPH_DEPTH: usize = 3;
const TEXTURE_GRAPH_PROBE_BYTES: u64 = 128;
static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakHeader {
    pub magic: u32,
    pub header_size: u32,
    pub login_page_index: u32,
    pub login_table_offset: u32,
    pub page_count: u32,
    pub page_table_offset: u32,
    pub fixup_page_count: u32,
    pub fixup_table_offset: u32,
    pub unknown_words: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakPage {
    pub index: usize,
    pub offset: u64,
    pub stored_size: u32,
    pub page_class: u32,
    pub resource_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakResource {
    pub page_index: usize,
    pub index: usize,
    pub directory_name: String,
    pub name: String,
    pub resource_type: String,
    pub offset_in_page: u32,
    pub file_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakArchive {
    pub header: PakHeader,
    pub file_size: usize,
    pub fixup_count: u32,
    pub pages: Vec<PakPage>,
    pub resources: Vec<PakResource>,
    fixups: Vec<PakFixup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PakFixup {
    pub source_page: usize,
    pub target_page: usize,
    pub pointer_offset: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshTextureReference {
    pub name: String,
    pub hash: u64,
    pub path: String,
    pub vram_descriptor_offset: Option<u64>,
    pub vram_descriptor_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub resource_index: usize,
    pub resource_page: usize,
    pub resource_page_class: u32,
    pub resource_offset: u64,
    pub resource_metadata_words: [u32; 10],
    pub hash: u64,
    pub data_offset: u32,
    pub unknown0: u32,
    pub data_size: u32,
    pub texture_dict_id: u32,
    pub unknown1: u32,
    pub texture_type: u32,
    pub format_id: u32,
    pub field_2c: u32,
    pub format_name: String,
    pub mip_count: u32,
    pub width: u32,
    pub height: u32,
    pub field_3c: u32,
    pub stream_flags: u32,
    pub texture_path_pointer: Option<u64>,
    pub texture_path: Option<String>,
    pub expected_mip_bytes: Option<u64>,
    pub expected_rgba_bytes: Option<u64>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatMapping {
    pub format_id: u32,
    pub flags_and_bpp: u32,
    pub block_dim: u32,
    pub pitch_multiplier: u32,
}

pub fn calculate_subresource_footprint(
    width: u32,
    height: u32,
    depth: u32,
    format_mapping: &FormatMapping,
    compressed: bool,
) -> (u32, u32) {
    let block_factor = if compressed { 4 } else { 1 };
    let w = if compressed { (width + 3) & !3 } else { width };
    let h = if compressed { (height + 3) & !3 } else { height };

    let bpp = format_mapping.block_dim.max(1);

    // row_pitch aligned to 256 bytes
    let row_pitch = (((w * bpp * block_factor / 8) + 255) & !255).max(256);

    // slice/extent aligned to 512 bytes
    let slice_pitch = (((row_pitch * h * depth / block_factor) + 511) & !511).max(512);

    (row_pitch, slice_pitch)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TexturePayload {
    pub source_path: PathBuf,
    pub source_descriptor_index: usize,
    pub source_resource_page: usize,
    pub source_data_offset: u64,
    pub source_data_size: u32,
    pub source_format_id: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub source_mip_count: u32,
    pub dictionary_descriptor_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureInventoryEntry {
    pub descriptor: TextureDescriptor,
    pub payload: Option<TexturePayload>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureInventory {
    pub pak_path: PathBuf,
    pub descriptors: Vec<TextureInventoryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextureGraphRootHeader {
    entry_count: usize,
    pointer_values: [(usize, i64); 3],
}

struct TextureVariantRow {
    archive_path: PathBuf,
    descriptor: TextureDescriptor,
    payload_offset: Option<u64>,
}

#[derive(Debug, Default)]
struct ShiftedMipAnalysis {
    exact_matches: Vec<(usize, usize, u64, u32)>,
    shape_candidates: u64,
    prefix_size_matches: u64,
    suffix_mismatches: u64,
    size_mismatches: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshReport {
    pub pak_path: PathBuf,
    pub geometry_resource_offset: u64,
    pub submesh_table_offset: u64,
    pub submesh_count: usize,
    pub submesh_table_size: usize,
    pub submesh_index: usize,
    pub submesh_name: String,
    pub submesh_offset: u64,
    pub material_offset: u64,
    pub vertex_count: usize,
    pub vertex_format: String,
    pub index_count: usize,
    pub triangle_count: usize,
    pub vertex_buffer_offset: u64,
    pub vertex_buffer_size: usize,
    pub vertex_page_index: usize,
    pub vertex_page_class: u32,
    pub index_buffer_offset: u64,
    pub index_buffer_size: usize,
    pub index_page_index: usize,
    pub index_page_class: u32,
    pub bounds: MeshBounds,
    pub textures: Vec<MeshTextureReference>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedMesh {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u16>,
    pub report: MeshReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmeshInventory {
    pub index: usize,
    pub lod: Option<usize>,
    pub name: String,
    pub vertex_count: usize,
    pub index_count: usize,
    pub position_type: Option<u8>,
    pub position_format: String,
    pub supported: bool,
    pub issue: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometryInventory {
    pub geometry_resource_count: usize,
    pub submeshes: Vec<SubmeshInventory>,
    pub decoded_meshes: Vec<DecodedMesh>,
}

impl DecodedMesh {
    pub fn export_obj(&self, output: impl AsRef<Path>) -> Result<(), PakError> {
        validate_obj_mesh(&self.vertices, &self.indices)?;
        let output = output.as_ref();
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let file_name = output
            .file_name()
            .ok_or_else(|| invalid("OBJ output path", 0, "must include a file name"))?;
        let mut temp_path = None;
        let mut file = None;
        for _ in 0..100 {
            let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
            let candidate = parent.join(format!(
                ".{}.{}.{}.tmp",
                file_name.to_string_lossy(),
                std::process::id(),
                sequence
            ));
            match File::options()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(created) => {
                    temp_path = Some(candidate);
                    file = Some(created);
                    break;
                }

                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(PakError::Io {
                        path: candidate,
                        operation: "create temporary OBJ",
                        source,
                    });
                }
            }
        }
        let temp_path = temp_path
            .ok_or_else(|| invalid("OBJ output path", 0, "could not allocate a temporary file"))?;
        let mut file =
            file.ok_or_else(|| invalid("OBJ output file", 0, "temporary file was not created"))?;
        let result = write_obj(&mut file, &self.vertices, &self.indices)
            .and_then(|()| file.sync_all())
            .and_then(|()| fs::rename(&temp_path, output));
        if let Err(source) = result {
            let _ = fs::remove_file(&temp_path);
            return Err(PakError::Io {
                path: output.to_owned(),
                operation: "write validated OBJ",
                source,
            });
        }
        Ok(())
    }
}

impl GeometryInventory {
    pub fn export_supported_obj(&self, output: impl AsRef<Path>) -> Result<(), PakError> {
        if self.decoded_meshes.is_empty() {
            return Err(invalid(
                "supported geometry",
                0,
                "no submeshes passed format and buffer validation",
            ));
        }
        for mesh in &self.decoded_meshes {
            validate_obj_mesh(&mesh.vertices, &mesh.indices)?;
        }
        export_obj_meshes(&self.decoded_meshes, output)
    }
}

impl TextureInventory {
    pub fn summary(&self) -> String {
        let mut format_counts = std::collections::BTreeMap::new();
        let mut payloads = 0usize;
        for entry in &self.descriptors {
            *format_counts
                .entry(entry.descriptor.format_name.as_str())
                .or_insert(0usize) += 1;
            payloads += usize::from(entry.payload.is_some());
        }
        let mut output = format!(
            "Texture inventory: {} VRAM_DESC resource(s), {} payload(s) located, PNG decoding unavailable because TLOU2 payload layout is unverified\n",
            self.descriptors.len(),
            payloads
        );
        for (format, count) in format_counts {
            output.push_str(&format!("  {format}: {count}\n"));
        }
        output
    }

    pub fn detailed_report(&self) -> String {
        let mut output = self.summary();
        for entry in &self.descriptors {
            let descriptor = &entry.descriptor;
            let payload = entry.payload.as_ref();
            output.push_str(&format!(
                "resource={} page={} class={} hash={:#018x} {} {}x{} mips={} layers=unknown stored_bytes={} logical_mip_bytes={} decoded_rgba_mip_bytes={} payload={} status={}\n",
                descriptor.resource_index,
                descriptor.resource_page,
                descriptor.resource_page_class,
                descriptor.hash,
                descriptor.format_name,
                descriptor.width,
                descriptor.height,
                descriptor.mip_count,
                payload.map_or(0, |entry| entry.source_data_size),
                descriptor
                    .expected_mip_bytes
                    .map_or_else(|| "unknown".to_owned(), |size| size.to_string()),
                descriptor
                    .expected_rgba_bytes
                    .map_or_else(|| "unknown".to_owned(), |size| size.to_string()),
                payload.map_or_else(
                    || "unresolved".to_owned(),
                    |entry| format!(
                        "{}@{:#x} (resource {}:{} {}x{} {} {} mips)",
                        entry.source_path.display(),
                        entry.source_data_offset,
                        entry.source_resource_page,
                        entry.source_descriptor_index,
                        texture_format(entry.source_format_id).0,
                        entry.source_width,
                        entry.source_height,
                        entry.source_mip_count
                    )
                ),
                entry.status
            ));
            output.push_str(&format!(
                "  metadata: unknown0={:#010x} texture_dict_id={:#010x} unknown1={:#010x} texture_type={:#010x} format_id={:#010x} field_2c={:#010x} field_3c={:#010x} stream_flags={:#010x} path_pointer={} path={}\n",
                descriptor.unknown0,
                descriptor.texture_dict_id,
                descriptor.unknown1,
                descriptor.texture_type,
                descriptor.format_id,
                descriptor.field_2c,
                descriptor.field_3c,
                descriptor.stream_flags,
                descriptor
                    .texture_path_pointer
                    .map_or_else(|| "no-fixup".to_owned(), |offset| format!("{offset:#x}")),
                descriptor.texture_path.as_deref().unwrap_or("unresolved")
            ));
            output.push_str(&format!(
                "  leading_resource_words: {}\n",
                descriptor
                    .resource_metadata_words
                    .iter()
                    .map(|word| format!("{word:#010x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
        output
    }

    pub fn extract_raw_payloads(&self, output: impl AsRef<Path>) -> Result<usize, PakError> {
        let output = output.as_ref();
        fs::create_dir_all(output).map_err(|source| PakError::Io {
            path: output.to_owned(),
            operation: "create texture output directory",
            source,
        })?;
        let manifest_path = output.join("texture-manifest.csv");
        let mut manifest = File::options()
            .write(true)
            .create_new(true)
            .open(&manifest_path)
            .map_err(|source| PakError::Io {
                path: manifest_path.clone(),
                operation: "create texture manifest without overwriting",
                source,
            })?;
        writeln!(
            manifest,
            "resource_index,resource_page,hash,width,height,format,mip_count,layers,stored_bytes,source_pak,source_offset,payload_resource_page,payload_resource_index,payload_format_id,payload_width,payload_height,payload_mip_count,raw_file,status"
        )
        .map_err(|source| PakError::Io {
            path: manifest_path.clone(),
            operation: "write texture manifest",
            source,
        })?;
        let mut extracted = 0usize;
        for entry in &self.descriptors {
            let descriptor = &entry.descriptor;
            let payload = entry.payload.as_ref();
            let raw_name = payload.map(|_| {
                format!(
                    "{}-page{:03}-resource{:04}-{:016x}.gpu.bin",
                    safe_filename_stem(&self.pak_path),
                    descriptor.resource_page,
                    descriptor.resource_index,
                    descriptor.hash
                )
            });
            let status = if let (Some(payload), Some(raw_name)) = (payload, raw_name.as_ref()) {
                copy_texture_payload(payload, &output.join(raw_name))?;
                extracted += 1;
                if entry.status.starts_with("raw payload located") {
                    "raw_payload_extracted; PNG unsupported: payload layout is unverified"
                        .to_owned()
                } else {
                    format!("raw_payload_extracted; {}", entry.status)
                }
            } else {
                entry.status.clone()
            };
            writeln!(
                manifest,
                "{},{},{:#018x},{},{},{},{},unknown,{},{},{},{},{},{},{},{},{},{},{}",
                descriptor.resource_index,
                descriptor.resource_page,
                descriptor.hash,
                descriptor.width,
                descriptor.height,
                csv_escape(&descriptor.format_name),
                descriptor.mip_count,
                payload.map_or(0, |payload| payload.source_data_size),
                payload.map_or_else(String::new, |payload| csv_escape(
                    &payload.source_path.display().to_string()
                )),
                payload.map_or_else(String::new, |payload| format!(
                    "{:#x}",
                    payload.source_data_offset
                )),
                payload
                    .map(|payload| payload.source_resource_page)
                    .map_or_else(String::new, |page| page.to_string()),
                payload
                    .map(|payload| payload.source_descriptor_index)
                    .map_or_else(String::new, |index| index.to_string()),
                payload
                    .map(|payload| format!("{:#x}", payload.source_format_id))
                    .unwrap_or_default(),
                payload
                    .map(|payload| payload.source_width)
                    .map_or_else(String::new, |width| width.to_string()),
                payload
                    .map(|payload| payload.source_height)
                    .map_or_else(String::new, |height| height.to_string()),
                payload
                    .map(|payload| payload.source_mip_count)
                    .map_or_else(String::new, |mips| mips.to_string()),
                raw_name.as_deref().unwrap_or(""),
                csv_escape(&status)
            )
            .map_err(|source| PakError::Io {
                path: manifest_path.clone(),
                operation: "write texture manifest row",
                source,
            })?;
        }
        manifest.sync_all().map_err(|source| PakError::Io {
            path: manifest_path,
            operation: "flush texture manifest",
            source,
        })?;
        Ok(extracted)
    }
}

#[derive(Debug)]
pub enum PakError {
    Binary(BinaryError),
    UnsupportedMagic(u32),
    Invalid {
        field: &'static str,
        offset: u64,
        detail: String,
    },
    Io {
        path: PathBuf,
        operation: &'static str,
        source: io::Error,
    },
}

impl fmt::Display for PakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary(error) => error.fmt(f),
            Self::UnsupportedMagic(magic) => {
                write!(f, "unsupported PAK magic {magic:#010x}")
            }
            Self::Invalid {
                field,
                offset,
                detail,
            } => write!(f, "invalid PAK {field} at {offset:#x}: {detail}"),
            Self::Io {
                path,
                operation,
                source,
            } => write!(f, "cannot {operation} {}: {source}", path.display()),
        }
    }
}

impl std::error::Error for PakError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Binary(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<BinaryError> for PakError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

impl PakArchive {
    pub fn analyze_texture_payloads_path(
        path: impl AsRef<Path>,
        format_filter: Option<&str>,
        limit: usize,
    ) -> Result<String, PakError> {
        if limit == 0 {
            return Err(invalid("texture analysis limit", 0, "must be non-zero"));
        }
        let path = path.as_ref();
        let inventory = Self::inspect_textures_path(path, None)?;
        let mut output = String::from(
            "archive,resource_page,resource_index,resource_file_offset,data_offset,data_size,hash,format_id,format,width,height,mip_count,unknown0,texture_dict_id,unknown1,texture_type,field_2c,field_3c,stream_flags,opaque_words,path,expected_linear_base_bytes,expected_linear_mip_bytes,size_delta,alignment16,sha256,entropy_bits_per_byte,distinct_byte_values,zero_bytes,longest_zero_run,adjacent_equal_bytes,unique_16_byte_blocks,hash_first_16,hash_first_256,hash_first_4096,hash_linear_base_prefix,hash_linear_size_prefix,hash_extra_tail\n",
        );
        let mut included = 0usize;
        for entry in &inventory.descriptors {
            if format_filter
                .is_some_and(|filter| !entry.descriptor.format_name.eq_ignore_ascii_case(filter))
            {
                continue;
            }
            let Some(payload) = &entry.payload else {
                continue;
            };
            if included == limit {
                break;
            }
            let bytes = read_texture_payload_bytes(payload)?;
            let descriptor = &entry.descriptor;
            let expected = descriptor.expected_mip_bytes;
            let expected_base = expected_texture_bytes(
                descriptor.format_id,
                descriptor.width,
                descriptor.height,
                1,
                texture_format(descriptor.format_id).1,
            );
            let expected_base_len = expected_base
                .and_then(|size| usize::try_from(size).ok())
                .filter(|size| *size <= bytes.len());
            let expected_len = expected
                .and_then(|size| usize::try_from(size).ok())
                .filter(|size| *size <= bytes.len());
            let stats = TextureByteStats::new(&bytes);
            let opaque_words = descriptor
                .resource_metadata_words
                .iter()
                .map(|word| format!("{word:08x}"))
                .collect::<Vec<_>>()
                .join("|");
            let row = [
                csv_escape(&path.display().to_string()),
                payload.source_resource_page.to_string(),
                payload.source_descriptor_index.to_string(),
                format!("{:#x}", descriptor.resource_offset),
                format!("{:#x}", payload.source_data_offset),
                payload.source_data_size.to_string(),
                format!("{:#018x}", descriptor.hash),
                format!("{:#x}", descriptor.format_id),
                csv_escape(&descriptor.format_name),
                descriptor.width.to_string(),
                descriptor.height.to_string(),
                descriptor.mip_count.to_string(),
                format!("{:#x}", descriptor.unknown0),
                format!("{:#x}", descriptor.texture_dict_id),
                format!("{:#x}", descriptor.unknown1),
                format!("{:#x}", descriptor.texture_type),
                format!("{:#x}", descriptor.field_2c),
                format!("{:#x}", descriptor.field_3c),
                format!("{:#x}", descriptor.stream_flags),
                csv_escape(&opaque_words),
                csv_escape(descriptor.texture_path.as_deref().unwrap_or("")),
                expected_base.map_or_else(String::new, |size| size.to_string()),
                expected.map_or_else(String::new, |size| size.to_string()),
                expected.map_or_else(String::new, |size| {
                    (i128::from(payload.source_data_size) - i128::from(size)).to_string()
                }),
                (payload.source_data_offset % 16 == 0).to_string(),
                stats.sha256,
                format!("{:.6}", stats.entropy),
                stats.distinct_bytes.to_string(),
                stats.zero_bytes.to_string(),
                stats.longest_zero_run.to_string(),
                stats.adjacent_equal_bytes.to_string(),
                stats.unique_16_byte_blocks.to_string(),
                sha256_hex(&bytes[..bytes.len().min(16)]),
                sha256_hex(&bytes[..bytes.len().min(256)]),
                sha256_hex(&bytes[..bytes.len().min(4096)]),
                expected_base_len.map_or_else(String::new, |size| sha256_hex(&bytes[..size])),
                expected_len.map_or_else(String::new, |size| sha256_hex(&bytes[..size])),
                expected_len.map_or_else(String::new, |size| sha256_hex(&bytes[size..])),
            ];
            output.push_str(&row.join(","));
            output.push('\n');
            included += 1;
        }
        Ok(output)
    }

    pub fn analyze_texture_pitch_models_path(path: impl AsRef<Path>) -> Result<String, PakError> {
        use std::collections::BTreeMap;

        let path = path.as_ref();
        let inventory = Self::inspect_textures_path(path, None)?;
        let mut results = BTreeMap::<(u32, u32, u32, u32, u32, String), PitchModelStats>::new();
        for entry in &inventory.descriptors {
            let Some(payload) = &entry.payload else {
                continue;
            };
            let descriptor = &entry.descriptor;
            let stored_size = u64::from(payload.source_data_size);
            let mut record = |model: String, predicted: Option<u64>| {
                let Some(predicted) = predicted else {
                    return;
                };
                results
                    .entry((
                        descriptor.format_id,
                        descriptor.texture_type,
                        descriptor.width,
                        descriptor.height,
                        descriptor.mip_count,
                        model,
                    ))
                    .or_default()
                    .observe(i128::from(stored_size) - i128::from(predicted));
            };
            record("tight".to_owned(), descriptor.expected_mip_bytes);
            record(
                "copy_footprint_256_pitch_512_placement_size_alignment".to_owned(),
                d3d12_copyable_footprint_candidate(descriptor, None)
                    .map(|footprint| footprint.allocation_bytes),
            );
            for alignment in [16, 32, 64, 128, 256, 512, 1024] {
                record(
                    format!("row_pitch_bytes_{alignment}"),
                    pitched_texture_mip_bytes(descriptor, alignment, None),
                );
                record(
                    format!("mip_size_alignment_{alignment}"),
                    aligned_mip_chain_bytes(descriptor, u64::from(alignment)),
                );
            }
            if texture_format(descriptor.format_id)
                .1
                .is_some_and(|bytes| bytes > 0)
            {
                for block_rows in [4, 8, 16, 32, 64] {
                    record(
                        format!("block_columns_alignment_{block_rows}"),
                        pitched_texture_mip_bytes(descriptor, 1, Some(block_rows)),
                    );
                }
            }
        }

        let mut output = format!(
            "Texture pitch-model inventory: {} descriptors with payloads in {}\n\
             Models are size predictions only, not evidence of byte addressing. Residual = stored_bytes - predicted_bytes.\n\
             format_id,format,texture_type,width,height,mip_count,model,count,exact,residual_counts\n",
            inventory
                .descriptors
                .iter()
                .filter(|entry| entry.payload.is_some())
                .count(),
            path.display()
        );
        for ((format_id, texture_type, width, height, mip_count, model), stats) in results {
            let residuals = stats
                .residuals
                .iter()
                .map(|(residual, count)| format!("{residual}:{count}"))
                .collect::<Vec<_>>()
                .join("|");
            output.push_str(&format!(
                "{format_id:#x},{},{texture_type:#x},{width},{height},{mip_count},{model},{},{},{residuals}\n",
                csv_escape(&texture_format(format_id).0),
                stats.count,
                stats.exact
            ));
        }
        Ok(output)
    }

    pub fn parse(data: &[u8]) -> Result<Self, PakError> {
        let mut cursor = Cursor::new(data);
        Self::parse_seekable(&mut cursor, data.len() as u64, "<memory>")
    }

    pub fn parse_path(path: impl AsRef<Path>) -> Result<Self, PakError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "open PAK file",
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read PAK file metadata",
                source,
            })?
            .len();
        Self::parse_seekable(&mut file, file_size, &path.display().to_string())
    }

    pub fn inspect_mesh_path(path: impl AsRef<Path>) -> Result<DecodedMesh, PakError> {
        let path = path.as_ref();
        let archive = Self::parse_path(path)?;
        let mut file = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "reopen PAK for mesh decoding",
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read PAK metadata for mesh decoding",
                source,
            })?
            .len();
        decode_geometry_mesh(&archive, &mut file, file_size, path, None)
    }

    pub fn inspect_geometry_path(path: impl AsRef<Path>) -> Result<GeometryInventory, PakError> {
        let path = path.as_ref();
        let archive = Self::parse_path(path)?;
        let mut file = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "reopen PAK for geometry inventory",
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read PAK metadata for geometry inventory",
                source,
            })?
            .len();
        let mut submeshes = scan_geometry_submeshes(&archive, &mut file, file_size, path)?;
        let mut decoded_meshes = Vec::new();
        for submesh in &mut submeshes {
            if !submesh.supported {
                continue;
            }
            match decode_geometry_mesh(&archive, &mut file, file_size, path, Some(submesh.index)) {
                Ok(mesh) => decoded_meshes.push(mesh),
                Err(error) => {
                    submesh.supported = false;
                    submesh.issue = Some(error.to_string());
                }
            }
        }
        Ok(GeometryInventory {
            geometry_resource_count: archive
                .resources
                .iter()
                .filter(|resource| resource.resource_type == "GEOMETRY_1")
                .count(),
            submeshes,
            decoded_meshes,
        })
    }

    pub fn inspect_textures_path(
        path: impl AsRef<Path>,
        dictionary_path: Option<&Path>,
    ) -> Result<TextureInventory, PakError> {
        let path = path.as_ref();
        let archive = Self::parse_path(path)?;
        let mut file = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "reopen PAK for texture inspection",
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read PAK metadata for texture inspection",
                source,
            })?
            .len();
        let descriptors = read_texture_descriptors(&archive, &mut file, file_size, path)?;
        let payload_base = texture_payload_base(&archive)?;

        let dictionary_data = if let Some(dictionary_path) = dictionary_path {
            let dictionary_archive = Self::parse_path(dictionary_path)?;
            let mut dictionary_file =
                File::open(dictionary_path).map_err(|source| PakError::Io {
                    path: dictionary_path.to_owned(),
                    operation: "open texture dictionary",
                    source,
                })?;
            let dictionary_size = dictionary_file
                .metadata()
                .map_err(|source| PakError::Io {
                    path: dictionary_path.to_owned(),
                    operation: "read texture dictionary metadata",
                    source,
                })?
                .len();
            let dictionary_descriptors = read_texture_descriptors(
                &dictionary_archive,
                &mut dictionary_file,
                dictionary_size,
                dictionary_path,
            )?;
            Some((
                dictionary_path.to_owned(),
                dictionary_size,
                texture_payload_base(&dictionary_archive)?,
                dictionary_descriptors,
            ))
        } else {
            None
        };

        let mut entries = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            let (dictionary_match, dictionary_ambiguous) =
                if let Some((_, _, _, dictionary_entries)) = &dictionary_data {
                    let mut matches = dictionary_entries
                        .iter()
                        .filter(|entry| entry.hash == descriptor.hash);
                    let first = matches.next();
                    let second = matches.next();
                    (
                        if second.is_none() { first } else { None },
                        second.is_some(),
                    )
                } else {
                    (None, false)
                };
            let (source_path, source_file_size, source_base, source_descriptor, dictionary_index) =
                if let (Some((dictionary_path, dictionary_size, dictionary_base, _)), Some(entry)) =
                    (&dictionary_data, dictionary_match)
                {
                    (
                        dictionary_path.as_path(),
                        *dictionary_size,
                        *dictionary_base,
                        entry,
                        Some(entry.resource_index),
                    )
                } else {
                    (path, file_size, payload_base, &descriptor, None)
                };
            let source_data_offset =
                source_base.checked_add(u64::from(source_descriptor.data_offset));
            let source_end = source_data_offset
                .and_then(|offset| offset.checked_add(u64::from(source_descriptor.data_size)));
            let payload = match (source_data_offset, source_end) {
                (Some(offset), Some(end))
                    if source_descriptor.data_size > 0 && end <= source_file_size =>
                {
                    Some(TexturePayload {
                        source_path: source_path.to_owned(),
                        source_descriptor_index: source_descriptor.resource_index,
                        source_resource_page: source_descriptor.resource_page,
                        source_data_offset: offset,
                        source_data_size: source_descriptor.data_size,
                        source_format_id: source_descriptor.format_id,
                        source_width: source_descriptor.width,
                        source_height: source_descriptor.height,
                        source_mip_count: source_descriptor.mip_count,
                        dictionary_descriptor_index: dictionary_index,
                    })
                }
                _ => None,
            };
            let mut status = texture_status(&descriptor, payload.as_ref());
            if dictionary_ambiguous {
                status.push_str("; dictionary hash has multiple matching VRAM_DESC resources");
            }
            entries.push(TextureInventoryEntry {
                descriptor,
                payload,
                status,
            });
        }
        Ok(TextureInventory {
            pak_path: path.to_owned(),
            descriptors: entries,
        })
    }

    pub fn inspect_texture_graphs_path(path: impl AsRef<Path>) -> Result<String, PakError> {
        let path = path.as_ref();
        let archive = Self::parse_path(path)?;
        let mut source = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "reopen PAK for texture graph inspection",
            source,
        })?;
        let file_size = source
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read PAK metadata for texture graph inspection",
                source,
            })?
            .len();
        let descriptors = read_texture_descriptors(&archive, &mut source, file_size, path)?;
        render_texture_pointer_graph(&archive, &descriptors, &mut source, file_size, path)
    }

    pub fn inspect_texture_dictionary_records_path(
        path: impl AsRef<Path>,
    ) -> Result<String, PakError> {
        let path = path.as_ref();
        let archive = Self::parse_path(path)?;
        let mut source = File::open(path).map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "reopen PAK for texture dictionary record inspection",
            source,
        })?;
        let file_size = source
            .metadata()
            .map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "read PAK metadata for texture dictionary record inspection",
                source,
            })?
            .len();
        let descriptors = read_texture_descriptors(&archive, &mut source, file_size, path)?;
        render_texture_dictionary_records(&archive, &descriptors, &mut source, file_size, path)
    }

    pub fn compare_texture_variants_paths<I, P>(paths: I) -> Result<String, PakError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        use std::collections::BTreeMap;

        let mut variants = BTreeMap::<String, Vec<TextureVariantRow>>::new();
        for path in paths {
            let path = path.as_ref();
            let archive = Self::parse_path(path)?;
            let mut source = File::open(path).map_err(|source| PakError::Io {
                path: path.to_owned(),
                operation: "open PAK for texture variant comparison",
                source,
            })?;
            let file_size = source
                .metadata()
                .map_err(|source| PakError::Io {
                    path: path.to_owned(),
                    operation: "read PAK metadata for texture variant comparison",
                    source,
                })?
                .len();
            let descriptors = read_texture_descriptors(&archive, &mut source, file_size, path)?;
            let payload_base = texture_payload_base(&archive)?;
            for descriptor in descriptors {
                let Some(texture_path) = descriptor.texture_path.as_deref() else {
                    continue;
                };
                let asset_path = texture_asset_path(texture_path);
                let payload_offset = payload_base.checked_add(u64::from(descriptor.data_offset));
                let payload_valid = payload_offset
                    .and_then(|offset| offset.checked_add(u64::from(descriptor.data_size)))
                    .is_some_and(|end| end <= file_size);
                variants
                    .entry(asset_path)
                    .or_default()
                    .push(TextureVariantRow {
                        archive_path: path.to_owned(),
                        descriptor,
                        payload_offset: payload_offset.filter(|_| payload_valid),
                    });
            }
        }
        render_texture_variant_comparison(variants)
    }

    pub fn extract_pages_path(
        &self,
        source_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<(), PakError> {
        let source_path = source_path.as_ref();
        let output = output.as_ref();
        let mut source = File::open(source_path).map_err(|source| PakError::Io {
            path: source_path.to_owned(),
            operation: "open PAK for page extraction",
            source,
        })?;
        let file_size = source
            .metadata()
            .map_err(|source| PakError::Io {
                path: source_path.to_owned(),
                operation: "read PAK metadata for extraction",
                source,
            })?
            .len();
        if usize::try_from(file_size).ok() != Some(self.file_size) {
            return Err(invalid("source size", 0, "PAK file changed after parsing"));
        }
        fs::create_dir_all(output).map_err(|source| PakError::Io {
            path: output.to_owned(),
            operation: "create output directory",
            source,
        })?;

        let mut buffer = vec![0; 64 * 1024];
        for page in &self.pages {
            let path = output.join(format!(
                "page-{:03}-class-{:02}.bin",
                page.index, page.page_class
            ));
            let mut target = File::create(&path).map_err(|source| PakError::Io {
                path: path.clone(),
                operation: "create extracted page",
                source,
            })?;
            source
                .seek(SeekFrom::Start(page.offset))
                .map_err(|source| PakError::Io {
                    path: source_path.to_owned(),
                    operation: "seek to stored page",
                    source,
                })?;
            let mut remaining = u64::from(page.stored_size);
            while remaining > 0 {
                let read_size = remaining.min(buffer.len() as u64) as usize;
                source
                    .read_exact(&mut buffer[..read_size])
                    .and_then(|()| target.write_all(&buffer[..read_size]))
                    .map_err(|source| PakError::Io {
                        path: path.clone(),
                        operation: "copy stored page bytes",
                        source,
                    })?;
                remaining -= read_size as u64;
            }
        }
        self.write_manifest(output)
    }

    fn parse_seekable<R: Read + Seek>(
        source: &mut R,
        file_size: u64,
        source_name: &str,
    ) -> Result<Self, PakError> {
        let mut header_bytes = [0; HEADER_SIZE];
        read_exact_at(
            source,
            file_size,
            0,
            &mut header_bytes,
            source_name,
            "read PAK header",
        )?;
        let mut reader = BinaryReader::new(&header_bytes);
        let magic = reader.read_u32()?;
        if !PAK_MAGICS.contains(&magic) {
            return Err(PakError::UnsupportedMagic(magic));
        }

        let header = PakHeader {
            magic,
            header_size: reader.read_u32()?,
            login_page_index: reader.read_u32()?,
            login_table_offset: reader.read_u32()?,
            page_count: reader.read_u32()?,
            page_table_offset: reader.read_u32()?,
            fixup_page_count: reader.read_u32()?,
            fixup_table_offset: reader.read_u32()?,
            unknown_words: [reader.read_u32()?, reader.read_u32()?, reader.read_u32()?],
        };
        if u64::from(header.header_size) > file_size {
            return Err(invalid(
                "header size",
                4,
                format!(
                    "declares {} bytes in a {}-byte file",
                    header.header_size, file_size
                ),
            ));
        }
        if header.page_count == 0 {
            return Err(invalid("page count", 16, "must be non-zero"));
        }
        let login_page_index = usize::try_from(header.login_page_index)
            .map_err(|_| invalid("login page index", 8, "does not fit address space"))?;
        if login_page_index >= header.page_count as usize {
            return Err(invalid(
                "login page index",
                8,
                format!("{} is outside the page table", header.login_page_index),
            ));
        }

        let page_table_offset = header.page_table_offset as usize;
        let table_size = (header.page_count as usize)
            .checked_mul(PAGE_ENTRY_SIZE)
            .ok_or_else(|| invalid("page table", page_table_offset as u64, "size overflow"))?;
        let page_table = read_range(
            source,
            file_size,
            page_table_offset as u64,
            table_size,
            source_name,
            "read PAK page table",
        )?;

        let (fixup_count, fixups) = parse_fixups(source, file_size, &header, source_name)?;
        let mut pages = Vec::with_capacity(header.page_count as usize);
        let mut resources = Vec::new();

        for index in 0..header.page_count as usize {
            let entry_offset = index * PAGE_ENTRY_SIZE;
            let mut entry =
                BinaryReader::new(&page_table[entry_offset..entry_offset + PAGE_ENTRY_SIZE]);
            let offset = u64::from(entry.read_u32()?);
            let stored_size = entry.read_u32()?;
            let table_page_class = entry.read_u32()?;
            if offset < u64::from(header.header_size) {
                return Err(invalid(
                    "page offset",
                    offset,
                    format!(
                        "page begins before declared header end {:#x}",
                        header.header_size
                    ),
                ));
            }
            usize::try_from(offset)
                .map_err(|_| invalid("page offset", offset, "does not fit address space"))?;
            if stored_size < PAGE_HEADER_SIZE as u32 {
                return Err(invalid(
                    "page size",
                    offset,
                    "smaller than the 20-byte page header",
                ));
            }
            let page_header = read_range(
                source,
                file_size,
                offset,
                PAGE_HEADER_SIZE,
                source_name,
                "read PAK page header",
            )?;
            let mut page_reader = BinaryReader::new(&page_header);
            let first_marker = page_reader.read_u32()?;
            let second_marker = page_reader.read_u32()?;
            if first_marker != 0xdead_beef || second_marker != 0xdead_beef {
                return Err(invalid(
                    "page marker",
                    offset,
                    format!(
                        "expected two 0xdeadbeef markers, got {first_marker:#x}/{second_marker:#x}"
                    ),
                ));
            }
            page_reader.skip(4)?;
            let embedded_size = page_reader.read_u32()?;
            let page_class = u32::from(page_reader.read_u16()?);
            if embedded_size != stored_size {
                return Err(invalid(
                    "page size",
                    offset + 12,
                    format!("page header says {embedded_size}, table says {stored_size}"),
                ));
            }
            let resource_count = page_reader.read_u16()?;
            if page_class != table_page_class {
                return Err(invalid(
                    "page class",
                    offset + 16,
                    format!("page header says {page_class}, table says {table_page_class}"),
                ));
            }
            let directory_size = usize::from(resource_count)
                .checked_mul(RESOURCE_ENTRY_SIZE)
                .ok_or_else(|| invalid("resource directory", offset, "size overflow"))?;
            checked_page_range(
                stored_size,
                PAGE_HEADER_SIZE,
                directory_size,
                "resource directory",
                offset,
            )?;

            let page = PakPage {
                index,
                offset,
                stored_size,
                page_class,
                resource_count,
            };
            if resource_count > 0 {
                let page_bytes = read_range(
                    source,
                    file_size,
                    offset,
                    stored_size as usize,
                    source_name,
                    "read PAK resource directory page",
                )?;
                resources.extend(parse_resources(&page_bytes, &page)?);
            }
            pages.push(page);
        }

        validate_non_overlapping_pages(&pages)?;
        validate_fixups(&fixups, &pages)?;
        let login_page = &pages[login_page_index];
        let login_offset = usize::try_from(header.login_table_offset)
            .map_err(|_| invalid("login table offset", 12, "does not fit address space"))?;
        if login_offset >= login_page.stored_size as usize {
            return Err(invalid(
                "login table offset",
                login_page.offset + u64::from(header.login_table_offset),
                "outside the selected page",
            ));
        }

        Ok(Self {
            header,
            file_size: usize::try_from(file_size)
                .map_err(|_| invalid("file size", 0, "does not fit address space"))?,
            fixup_count,
            pages,
            resources,
            fixups,
        })
    }

    pub fn resolve_pointer(
        &self,
        field_file_offset: u64,
        raw_value: i64,
    ) -> Result<Option<u64>, PakError> {
        let Some(fixup) = self.fixups.iter().find(|fixup| {
            self.pages[fixup.source_page].offset + fixup.pointer_offset as u64 == field_file_offset
        }) else {
            return Ok(None);
        };
        let target_page = &self.pages[fixup.target_page];
        let address = i128::from(target_page.offset) + i128::from(raw_value);
        if address < i128::from(target_page.offset)
            || address > i128::from(target_page.offset + u64::from(target_page.stored_size))
        {
            return Err(invalid(
                "fixed-up pointer",
                field_file_offset,
                format!(
                    "raw value {raw_value:#x} resolves outside target page {}",
                    fixup.target_page
                ),
            ));
        }
        Ok(Some(address as u64))
    }

    fn write_manifest(&self, output: &Path) -> Result<(), PakError> {
        let manifest_path = output.join("manifest.txt");
        let mut manifest = File::create(&manifest_path).map_err(|source| PakError::Io {
            path: manifest_path.clone(),
            operation: "create extraction manifest",
            source,
        })?;
        writeln!(manifest, "format=PAK magic={:#010x}", self.header.magic).map_err(|source| {
            PakError::Io {
                path: manifest_path.clone(),
                operation: "write extraction manifest",
                source,
            }
        })?;
        writeln!(
            manifest,
            "extraction=stored page segments copied without decompression"
        )
        .and_then(|()| {
            writeln!(
                manifest,
                "resource_payloads=not extracted; resource extents unresolved"
            )
        })
        .map_err(|source| PakError::Io {
            path: manifest_path.clone(),
            operation: "write extraction manifest",
            source,
        })?;
        writeln!(manifest, "page,class,offset,size,resources,file").map_err(|source| {
            PakError::Io {
                path: manifest_path.clone(),
                operation: "write extraction manifest",
                source,
            }
        })?;
        for page in &self.pages {
            writeln!(
                manifest,
                "{},{},{:#x},{},{},page-{:03}-class-{:02}.bin",
                page.index,
                page.page_class,
                page.offset,
                page.stored_size,
                page.resource_count,
                page.index,
                page.page_class
            )
            .map_err(|source| PakError::Io {
                path: manifest_path.clone(),
                operation: "write extraction manifest",
                source,
            })?;
        }
        writeln!(
            manifest,
            "\nresource_index,page,index,type,name,directory_name,offset"
        )
        .map_err(|source| PakError::Io {
            path: manifest_path.clone(),
            operation: "write extraction manifest",
            source,
        })?;
        for (resource_index, resource) in self.resources.iter().enumerate() {
            writeln!(
                manifest,
                "{},{},{},{},{},{},{:#x}",
                resource_index,
                resource.page_index,
                resource.index,
                csv_escape(&resource.resource_type),
                csv_escape(&resource.name),
                csv_escape(&resource.directory_name),
                resource.offset_in_page
            )
            .map_err(|source| PakError::Io {
                path: manifest_path.clone(),
                operation: "write extraction manifest",
                source,
            })?;
        }
        Ok(())
    }

    pub fn render_mesh_report(report: &MeshReport) -> String {
        let mut output = format!(
            "PAK mesh diagnostic: {}\n\
             Geometry resource: file offset {:#x}; size unresolved (resource extents are not stored in page directory)\n\
             Geometry structures: 48-byte resource prefix + 48-byte root header; submesh table {:#x}, {} x 192 = {} bytes\n\
             Submesh {}: {} at {:#x}\n\
             Material reference: {:#x}\n\
             Vertices: {} ({})\nIndices: {}\nTriangles: {}\n\
             Vertex buffer: {:#x}, {} bytes, page {} class {}\n\
             Index buffer: {:#x}, {} bytes, page {} class {}\n\
             Bounds: min=({:.6}, {:.6}, {:.6}) max=({:.6}, {:.6}, {:.6})\n\
             Texture references in this mesh material: {}\n",
            report.pak_path.display(),
            report.geometry_resource_offset,
            report.submesh_table_offset,
            report.submesh_count,
            report.submesh_table_size,
            report.submesh_index,
            report.submesh_name,
            report.submesh_offset,
            report.material_offset,
            report.vertex_count,
            report.vertex_format,
            report.index_count,
            report.triangle_count,
            report.vertex_buffer_offset,
            report.vertex_buffer_size,
            report.vertex_page_index,
            report.vertex_page_class,
            report.index_buffer_offset,
            report.index_buffer_size,
            report.index_page_index,
            report.index_page_class,
            report.bounds.min[0],
            report.bounds.min[1],
            report.bounds.min[2],
            report.bounds.max[0],
            report.bounds.max[1],
            report.bounds.max[2],
            report.textures.len(),
        );
        for texture in &report.textures {
            output.push_str(&format!(
                "  {} hash={:#018x} path={} VRAM_DESC={}\n",
                texture.name,
                texture.hash,
                texture.path,
                texture
                    .vram_descriptor_offset
                    .map_or_else(|| "not found".to_owned(), |offset| format!("{offset:#x}"))
            ));
            if let Some(path) = &texture.vram_descriptor_path {
                output.push_str(&format!("    VRAM path: {path}\n"));
            }
        }
        output.push_str("Evidence:\n");
        for evidence in &report.evidence {
            output.push_str(&format!("  - {evidence}\n"));
        }
        output
    }

    pub fn render_geometry_inventory(inventory: &GeometryInventory) -> String {
        let mut output = format!(
            "PAK geometry inventory: {} GEOMETRY_1 resource(s), {} submesh descriptor(s), {} decoded\n",
            inventory.geometry_resource_count,
            inventory.submeshes.len(),
            inventory.decoded_meshes.len()
        );
        for submesh in &inventory.submeshes {
            output.push_str(&format!(
                "[{}] LOD{} vertices={} triangles={} position={} status={}\n  {}\n",
                submesh.index,
                submesh
                    .lod
                    .map_or_else(|| "unknown".to_owned(), |lod| lod.to_string()),
                submesh.vertex_count,
                submesh.index_count / 3,
                submesh.position_format,
                if submesh.supported {
                    "supported"
                } else {
                    "unsupported"
                },
                submesh.name
            ));
            if let Some(issue) = &submesh.issue {
                output.push_str(&format!("  issue: {issue}\n"));
            }
        }
        output
    }

    fn page_for_range(&self, offset: u64, size: usize) -> Result<&PakPage, PakError> {
        let end = offset
            .checked_add(size as u64)
            .ok_or_else(|| invalid("referenced buffer", offset, "range overflow"))?;
        self.pages
            .iter()
            .find(|page| offset >= page.offset && end <= page.offset + u64::from(page.stored_size))
            .ok_or_else(|| {
                invalid(
                    "referenced buffer",
                    offset,
                    "range is not contained in a PAK page",
                )
            })
    }

    pub fn extract_pages(&self, source: &[u8], output: impl AsRef<Path>) -> Result<(), PakError> {
        if source.len() != self.file_size {
            return Err(invalid(
                "source size",
                0,
                format!(
                    "parsed {} bytes but extraction source has {}",
                    self.file_size,
                    source.len()
                ),
            ));
        }
        let output = output.as_ref();
        fs::create_dir_all(output).map_err(|source| PakError::Io {
            path: output.to_owned(),
            operation: "create output directory",
            source,
        })?;

        for page in &self.pages {
            let start = page.offset as usize;
            let end = start + page.stored_size as usize;
            let path = output.join(format!(
                "page-{:03}-class-{:02}.bin",
                page.index, page.page_class
            ));
            fs::write(&path, &source[start..end]).map_err(|source| PakError::Io {
                path,
                operation: "write stored page",
                source,
            })?;
        }
        self.write_manifest(output)
    }
}

fn read_texture_descriptors<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    path: &Path,
) -> Result<Vec<TextureDescriptor>, PakError> {
    let mut descriptors = Vec::new();
    for resource in archive
        .resources
        .iter()
        .filter(|resource| resource.resource_type == "VRAM_DESC")
    {
        let content_offset = resource
            .file_offset
            .checked_add(16)
            .ok_or_else(|| invalid("VRAM_DESC content offset", resource.file_offset, "overflow"))?;
        archive.page_for_range(content_offset, 112)?;
        let bytes = read_range(
            source,
            file_size,
            content_offset,
            112,
            &path.display().to_string(),
            "read VRAM_DESC texture metadata",
        )?;
        let mut resource_metadata_words = [0u32; 10];
        for (index, word) in resource_metadata_words.iter_mut().enumerate() {
            *word = u32_at(&bytes, index * 4, "VRAM_DESC leading resource metadata")?;
        }
        let data_offset = u32_at(&bytes, 40, "texture data offset")?;
        let unknown0 = u32_at(&bytes, 44, "VRAM_DESC unknown0")?;
        let data_size = u32_at(&bytes, 48, "texture data size")?;
        let texture_dict_id = u32_at(&bytes, 52, "VRAM_DESC texture dictionary id")?;
        let hash = u64_at(&bytes, 56, "texture hash")?;
        let unknown1 = u32_at(&bytes, 64, "VRAM_DESC unknown1")?;
        let texture_type = u32_at(&bytes, 68, "VRAM_DESC texture type")?;
        let format_id = u32_at(&bytes, 72, "texture format")?;
        let field_2c = u32_at(&bytes, 76, "VRAM_DESC field 0x2c")?;
        let mip_count = u32_at(&bytes, 80, "texture mip count")?;
        let width = u32_at(&bytes, 84, "texture width")?;
        let height = u32_at(&bytes, 88, "texture height")?;
        let field_3c = u32_at(&bytes, 92, "VRAM_DESC field 0x3c")?;
        let stream_flags = u32_at(&bytes, 96, "VRAM_DESC stream flags")?;
        let path_pointer_raw = i64_at(&bytes, 104, "VRAM_DESC texture path pointer")?;
        let texture_path_pointer =
            archive.resolve_pointer(content_offset + 104, path_pointer_raw)?;
        let texture_path = texture_path_pointer
            .map(|offset| read_pointer_string(archive, source, file_size, offset, path))
            .transpose()?;
        let (format_name, block_bytes) = texture_format(format_id);
        let expected_mip_bytes =
            expected_texture_bytes(format_id, width, height, mip_count, block_bytes);
        let expected_rgba_bytes = expected_rgba_mip_bytes(width, height, mip_count);
        let page = &archive.pages[resource.page_index];
        descriptors.push(TextureDescriptor {
            resource_index: resource.index,
            resource_page: resource.page_index,
            resource_page_class: page.page_class,
            resource_offset: resource.file_offset,
            resource_metadata_words,
            hash,
            data_offset,
            unknown0,
            data_size,
            texture_dict_id,
            unknown1,
            texture_type,
            format_id,
            field_2c,
            format_name,
            mip_count,
            width,
            height,
            field_3c,
            stream_flags,
            texture_path_pointer,
            texture_path,
            expected_mip_bytes,
            expected_rgba_bytes,
        });
    }
    Ok(descriptors)
}

fn parse_texture_graph_root_header(bytes: &[u8]) -> Result<TextureGraphRootHeader, PakError> {
    if bytes.len() < TEXTURE_GRAPH_HEADER_SIZE {
        return Err(invalid(
            "texture graph root header",
            0,
            format!(
                "need {TEXTURE_GRAPH_HEADER_SIZE} bytes, got {}",
                bytes.len()
            ),
        ));
    }
    let entry_count = usize::try_from(u32_at(
        bytes,
        TEXTURE_GRAPH_COUNT_OFFSET,
        "texture graph entry count",
    )?)
    .map_err(|_| invalid("texture graph entry count", 0, "does not fit address space"))?;
    if entry_count > MAX_TEXTURE_GRAPH_ENTRIES {
        return Err(invalid(
            "texture graph entry count",
            TEXTURE_GRAPH_COUNT_OFFSET as u64,
            format!("{entry_count} exceeds traversal limit {MAX_TEXTURE_GRAPH_ENTRIES}"),
        ));
    }
    Ok(TextureGraphRootHeader {
        entry_count,
        pointer_values: [
            (
                TEXTURE_GRAPH_POINTER_OFFSETS[0],
                i64_at(
                    bytes,
                    TEXTURE_GRAPH_POINTER_OFFSETS[0],
                    "texture graph candidate pointer",
                )?,
            ),
            (
                TEXTURE_GRAPH_POINTER_OFFSETS[1],
                i64_at(
                    bytes,
                    TEXTURE_GRAPH_POINTER_OFFSETS[1],
                    "texture graph candidate pointer",
                )?,
            ),
            (
                TEXTURE_GRAPH_POINTER_OFFSETS[2],
                i64_at(
                    bytes,
                    TEXTURE_GRAPH_POINTER_OFFSETS[2],
                    "texture graph candidate pointer",
                )?,
            ),
        ],
    })
}

fn render_texture_pointer_graph<R: Read + Seek>(
    archive: &PakArchive,
    descriptors: &[TextureDescriptor],
    source: &mut R,
    file_size: u64,
    path: &Path,
) -> Result<String, PakError> {
    use std::collections::{BTreeMap, HashSet, VecDeque};

    let mut output = format!("Texture pointer graph diagnostic: {}\n", path.display());
    let roots: Vec<_> = archive
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.resource_type.as_str(),
                "TEXTURE_DICTIONARY" | "VRAM_DESC_TABLE" | "TEXTURE_TABLE"
            )
        })
        .collect();
    if roots.is_empty() {
        output.push_str(
            "No TEXTURE_DICTIONARY, VRAM_DESC_TABLE, or TEXTURE_TABLE resource roots found.\n",
        );
        return Ok(output);
    }

    let mut descriptor_by_offset = BTreeMap::new();
    for descriptor in descriptors {
        descriptor_by_offset.insert(descriptor.resource_offset, descriptor);
    }
    let fixups_by_field: BTreeMap<_, _> = archive
        .fixups
        .iter()
        .map(|fixup| {
            (
                archive.pages[fixup.source_page].offset + fixup.pointer_offset as u64,
                *fixup,
            )
        })
        .collect();
    let mut pending = VecDeque::new();
    let mut scheduled = HashSet::new();
    let mut visited = HashSet::new();
    let mut total_edges = 0usize;
    let mut repeated_edges = 0usize;
    let mut truncated = false;
    for root in roots {
        archive.page_for_range(root.file_offset, TEXTURE_GRAPH_HEADER_SIZE)?;
        let bytes = read_range(
            source,
            file_size,
            root.file_offset,
            TEXTURE_GRAPH_HEADER_SIZE,
            &path.display().to_string(),
            "read texture graph root",
        )?;
        let header = parse_texture_graph_root_header(&bytes)?;
        output.push_str(&format!(
            "\nRoot {} resource={} page={} file={:#x}: observed_count={} header_bytes={}\n",
            root.resource_type,
            root.index,
            root.page_index,
            root.file_offset,
            header.entry_count,
            TEXTURE_GRAPH_HEADER_SIZE
        ));

        let array_size = header
            .entry_count
            .checked_mul(8)
            .ok_or_else(|| invalid("texture graph array size", root.file_offset, "overflow"))?;
        for (field_offset, raw_value) in header.pointer_values {
            let field_address = root
                .file_offset
                .checked_add(field_offset as u64)
                .ok_or_else(|| {
                    invalid("texture graph pointer field", root.file_offset, "overflow")
                })?;
            let Some(array_address) = archive.resolve_pointer(field_address, raw_value)? else {
                output.push_str(&format!(
                    "  candidate pointer field +{field_offset:#x}: no explicit fixup (raw={raw_value:#x})\n"
                ));
                continue;
            };
            output.push_str(&format!(
                "  candidate pointer field +{field_offset:#x}: raw={raw_value:#x} -> {array_address:#x}\n"
            ));
            let array_page = match archive.page_for_range(array_address, array_size) {
                Ok(page) => page,
                Err(error) => {
                    output.push_str(&format!(
                        "    count*8 candidate range is not wholly contained in a page: {error}\n"
                    ));
                    continue;
                }
            };
            output.push_str(&format!(
                "    candidate span={} bytes, page={} class={}, alignment_mod16={}\n",
                array_size,
                array_page.index,
                array_page.page_class,
                array_address % 16
            ));
            let array_bytes = read_range(
                source,
                file_size,
                array_address,
                array_size,
                &path.display().to_string(),
                "read candidate texture graph array",
            )?;
            let mut fixed_slots = 0usize;
            let mut target_pages = BTreeMap::new();
            let mut examples = Vec::new();
            let array_end = array_address
                .checked_add(array_size as u64)
                .ok_or_else(|| invalid("texture graph array", array_address, "overflow"))?;
            for (&slot_field, fixup) in fixups_by_field.range(array_address..array_end) {
                let slot_offset = usize::try_from(slot_field - array_address)
                    .map_err(|_| invalid("texture graph slot offset", slot_field, "overflow"))?;
                if slot_offset % 8 != 0 {
                    continue;
                }
                let slot = slot_offset / 8;
                let raw = i64_at(&array_bytes, slot_offset, "texture graph array pointer")?;
                let target = resolve_texture_graph_fixup(archive, slot_field, raw, fixup)?;
                fixed_slots += 1;
                let target_page = archive.page_for_range(target, 1)?;
                *target_pages.entry(target_page.index).or_insert(0usize) += 1;
                total_edges += 1;
                if examples.len() < 8 {
                    examples.push(format!(
                        "slot={slot} field={slot_field:#x} -> {}",
                        classify_texture_graph_target(
                            archive,
                            descriptors,
                            &descriptor_by_offset,
                            target
                        )
                    ));
                }
                let chain = format!(
                    "{} resource {} field +{field_offset:#x} slot {slot} -> {target:#x}",
                    root.resource_type, root.index
                );
                if scheduled.insert(target) {
                    if scheduled.len() > MAX_TEXTURE_GRAPH_NODES {
                        truncated = true;
                    } else {
                        pending.push_back((target, 1usize, chain));
                    }
                } else {
                    repeated_edges += 1;
                }
            }
            output.push_str(&format!(
                "    explicit_fixup_slots={fixed_slots}/{} target_pages={target_pages:?}\n",
                header.entry_count
            ));
            for example in examples {
                output.push_str(&format!("      {example}\n"));
            }
        }
    }
    let mut nodes_by_depth = BTreeMap::new();
    let mut outgoing_by_depth = BTreeMap::new();
    let mut target_pages_by_depth = BTreeMap::new();
    let mut descriptor_deltas_by_depth = BTreeMap::new();
    let mut graph_examples = Vec::new();
    let mut descriptor_examples_by_depth = BTreeMap::<usize, Vec<String>>::new();
    while let Some((node, depth, chain)) = pending.pop_front() {
        if !visited.insert(node) {
            repeated_edges += 1;
            continue;
        }
        if visited.len() > MAX_TEXTURE_GRAPH_NODES {
            truncated = true;
            break;
        }
        *nodes_by_depth.entry(depth).or_insert(0usize) += 1;
        let page = archive.page_for_range(node, 1)?;
        *target_pages_by_depth
            .entry((depth, page.index))
            .or_insert(0usize) += 1;
        if let Some((descriptor, delta)) = nearest_texture_descriptor(descriptors, page.index, node)
            && delta <= 256
        {
            *descriptor_deltas_by_depth
                .entry((depth, delta))
                .or_insert(0usize) += 1;
            let examples = descriptor_examples_by_depth.entry(depth).or_default();
            if examples.len() < 4 {
                examples.push(format!(
                    "depth={depth} target={node:#x} -> VRAM_DESC resource={} delta=+{delta:#x} hash={:#018x} {} {}x{} mips={} path={}",
                    descriptor.resource_index,
                    descriptor.hash,
                    descriptor.format_name,
                    descriptor.width,
                    descriptor.height,
                    descriptor.mip_count,
                    descriptor.texture_path.as_deref().unwrap_or("unresolved")
                ));
            }
        }
        if graph_examples.len() < 24 {
            graph_examples.push(format!(
                "depth={depth} chain={chain}; target={node:#x}; {}",
                classify_texture_graph_target(archive, descriptors, &descriptor_by_offset, node)
            ));
        }

        let page_end = page.offset + u64::from(page.stored_size);
        let probe_end = node.saturating_add(TEXTURE_GRAPH_PROBE_BYTES).min(page_end);
        let mut outgoing = 0usize;
        for (&field, fixup) in fixups_by_field.range(node..probe_end) {
            if field.checked_add(8).is_none_or(|end| end > probe_end) {
                continue;
            }
            let pointer_bytes = read_range(
                source,
                file_size,
                field,
                8,
                &path.display().to_string(),
                "read nested texture graph pointer",
            )?;
            let raw = i64_at(&pointer_bytes, 0, "nested texture graph pointer")?;
            let target = resolve_texture_graph_fixup(archive, field, raw, fixup)?;
            outgoing += 1;
            total_edges += 1;
            if depth >= MAX_TEXTURE_GRAPH_DEPTH {
                continue;
            }
            let child_chain = format!("{chain} -> field {field:#x} -> {target:#x}");
            if scheduled.insert(target) {
                if scheduled.len() > MAX_TEXTURE_GRAPH_NODES {
                    truncated = true;
                    break;
                }
                pending.push_back((target, depth + 1, child_chain));
            } else {
                repeated_edges += 1;
            }
        }
        *outgoing_by_depth.entry(depth).or_insert(0usize) += outgoing;
        if truncated {
            break;
        }
    }
    output.push_str(&format!(
        "\nBounded pointer-follow traversal: max_depth={MAX_TEXTURE_GRAPH_DEPTH}, probe_window={TEXTURE_GRAPH_PROBE_BYTES} bytes per target, visited_nodes={}, explicit_edges={total_edges}, repeated_or_cycle_edges={repeated_edges}, truncated={truncated}\n  nodes_by_depth={nodes_by_depth:?} outgoing_fixups_by_depth={outgoing_by_depth:?} target_pages_by_depth={target_pages_by_depth:?} nearby_vram_desc_deltas_by_depth={descriptor_deltas_by_depth:?}\n",
        visited.len()
    ));
    for examples in descriptor_examples_by_depth.values() {
        for example in examples {
            output.push_str(&format!("  {example}\n"));
        }
    }
    for example in graph_examples {
        output.push_str(&format!("  {example}\n"));
    }
    Ok(output)
}

fn render_texture_dictionary_records<R: Read + Seek>(
    archive: &PakArchive,
    descriptors: &[TextureDescriptor],
    source: &mut R,
    file_size: u64,
    path: &Path,
) -> Result<String, PakError> {
    use std::collections::{BTreeMap, BTreeSet};

    const DICTIONARY_POINTER_OFFSET: usize = 64;
    const DESCRIPTOR_POINTER_OFFSET: usize = 72;
    const RECORD_PROBE_BYTES: u64 = 128;
    const RECORD_PREVIEW_BYTES: usize = 32;

    let root = archive
        .resources
        .iter()
        .find(|resource| resource.resource_type == "TEXTURE_DICTIONARY")
        .ok_or_else(|| invalid("texture dictionary root", 0, "resource not found"))?;
    archive.page_for_range(root.file_offset, TEXTURE_GRAPH_HEADER_SIZE)?;
    let root_bytes = read_range(
        source,
        file_size,
        root.file_offset,
        TEXTURE_GRAPH_HEADER_SIZE,
        &path.display().to_string(),
        "read texture dictionary root",
    )?;
    let header = parse_texture_graph_root_header(&root_bytes)?;
    let pointer_fixups: BTreeMap<_, _> = archive
        .fixups
        .iter()
        .map(|fixup| {
            (
                archive.pages[fixup.source_page].offset + fixup.pointer_offset as u64,
                *fixup,
            )
        })
        .collect();
    let dictionary_field = root.file_offset + DICTIONARY_POINTER_OFFSET as u64;
    let dictionary_raw = i64_at(
        &root_bytes,
        DICTIONARY_POINTER_OFFSET,
        "TEXTURE_DICTIONARY +0x40 pointer",
    )?;
    let dictionary_table = archive
        .resolve_pointer(dictionary_field, dictionary_raw)?
        .ok_or_else(|| {
            invalid(
                "TEXTURE_DICTIONARY +0x40 pointer",
                dictionary_field,
                "has no explicit fixup",
            )
        })?;
    let descriptor_field = root.file_offset + DESCRIPTOR_POINTER_OFFSET as u64;
    let descriptor_raw = i64_at(
        &root_bytes,
        DESCRIPTOR_POINTER_OFFSET,
        "TEXTURE_DICTIONARY +0x48 pointer",
    )?;
    let descriptor_table = archive
        .resolve_pointer(descriptor_field, descriptor_raw)?
        .ok_or_else(|| {
            invalid(
                "TEXTURE_DICTIONARY +0x48 pointer",
                descriptor_field,
                "has no explicit fixup",
            )
        })?;
    let array_size = header.entry_count.checked_mul(8).ok_or_else(|| {
        invalid(
            "dictionary pointer array",
            dictionary_table,
            "size overflow",
        )
    })?;
    archive.page_for_range(dictionary_table, array_size)?;
    archive.page_for_range(descriptor_table, array_size)?;
    let dictionary_bytes = read_range(
        source,
        file_size,
        dictionary_table,
        array_size,
        &path.display().to_string(),
        "read texture dictionary candidate array",
    )?;
    let descriptor_bytes = read_range(
        source,
        file_size,
        descriptor_table,
        array_size,
        &path.display().to_string(),
        "read texture descriptor candidate array",
    )?;
    let descriptors_by_offset: BTreeMap<_, _> = descriptors
        .iter()
        .map(|descriptor| (descriptor.resource_offset, descriptor))
        .collect();
    let mut output = format!(
        "Texture dictionary record diagnostic: {}\nroot_resource={} root_page={} root_offset={:#x} entries={}\n+0x40_candidate_array={:#x} +0x48_candidate_array={:#x}\n",
        path.display(),
        root.index,
        root.page_index,
        root.file_offset,
        header.entry_count,
        dictionary_table,
        descriptor_table
    );
    output.push_str(
        "The 128-byte span is only a fixup-discovery probe, not an inferred record boundary.\n",
    );

    let mut edges_by_target_page = BTreeMap::<usize, usize>::new();
    let mut signature_counts = BTreeMap::<(usize, Vec<usize>), usize>::new();
    let mut outgoing_target_deltas = BTreeMap::<(usize, i64), usize>::new();
    let mut target_path_suffix_matches = 0usize;
    let mut target_path_suffix_mismatches = 0usize;
    let mut descriptor_inline_path_fixups = 0usize;
    let mut distinct_target_addresses = BTreeSet::new();
    let mut selected_samples = BTreeSet::new();
    let mut formats_selected = BTreeMap::<u32, Vec<(u32, u32)>>::new();
    let mut page_zero_samples = 0usize;
    let mut root_edges = Vec::with_capacity(header.entry_count);
    let mut missing_root_fixups = 0usize;
    let mut unmatched_descriptors = 0usize;

    for index in 0..header.entry_count {
        let slot_offset = index
            .checked_mul(8)
            .ok_or_else(|| invalid("dictionary array slot", dictionary_table, "overflow"))?;
        let source_address = dictionary_table
            .checked_add(slot_offset as u64)
            .ok_or_else(|| invalid("dictionary array slot", dictionary_table, "overflow"))?;
        let raw_value = i64_at(
            &dictionary_bytes,
            slot_offset,
            "TEXTURE_DICTIONARY +0x40 array pointer",
        )?;
        let Some(fixup) = pointer_fixups.get(&source_address).copied() else {
            missing_root_fixups += 1;
            root_edges.push(None);
            continue;
        };
        let target = resolve_texture_graph_fixup(archive, source_address, raw_value, &fixup)?;
        distinct_target_addresses.insert(target);
        let target_page = archive.page_for_range(target, 1)?;
        *edges_by_target_page.entry(target_page.index).or_default() += 1;

        let descriptor_source = descriptor_table
            .checked_add(slot_offset as u64)
            .ok_or_else(|| invalid("descriptor array slot", descriptor_table, "overflow"))?;
        let descriptor_pointer = pointer_fixups
            .get(&descriptor_source)
            .copied()
            .map(|descriptor_fixup| {
                let raw = i64_at(
                    &descriptor_bytes,
                    slot_offset,
                    "TEXTURE_DICTIONARY +0x48 array pointer",
                )?;
                resolve_texture_graph_fixup(archive, descriptor_source, raw, &descriptor_fixup)
            })
            .transpose()?;
        let paired_descriptor = descriptor_pointer.and_then(|pointer| {
            pointer.checked_sub(48).and_then(|offset| {
                descriptors_by_offset
                    .get(&offset)
                    .copied()
                    .filter(|descriptor| {
                        descriptor.resource_page
                            == archive
                                .pages
                                .iter()
                                .find(|page| {
                                    pointer >= page.offset
                                        && pointer < page.offset + u64::from(page.stored_size)
                                })
                                .map_or(usize::MAX, |page| page.index)
                    })
            })
        });
        if paired_descriptor.is_none() {
            unmatched_descriptors += 1;
        }
        if let Some(descriptor) = paired_descriptor {
            let target_string = read_pointer_string(archive, source, file_size, target, path)?;
            let expected_suffix = descriptor
                .texture_path
                .as_deref()
                .and_then(|value| value.strip_prefix("[BUILD_INTERMEDIATE]common/texture4/"));
            if expected_suffix == Some(target_string.as_str()) {
                target_path_suffix_matches += 1;
            } else {
                target_path_suffix_mismatches += 1;
            }
            let inline_path_target = descriptor.resource_offset.checked_add(128);
            let path_fixup_source = descriptor.resource_offset.checked_add(120);
            if inline_path_target == descriptor.texture_path_pointer
                && path_fixup_source.is_some_and(|source| pointer_fixups.contains_key(&source))
            {
                descriptor_inline_path_fixups += 1;
            }
        }

        let probe_end = target
            .saturating_add(RECORD_PROBE_BYTES)
            .min(target_page.offset + u64::from(target_page.stored_size));
        let outgoing_fixups: Vec<_> = pointer_fixups
            .range(target..probe_end)
            .filter(|(field, _)| field.checked_add(8).is_some_and(|end| end <= probe_end))
            .map(|(&field, &outgoing_fixup)| (field, outgoing_fixup))
            .collect();
        let relative_fixup_fields: Vec<_> = outgoing_fixups
            .iter()
            .map(|(field, _)| (*field - target) as usize)
            .collect();
        *signature_counts
            .entry((target_page.index, relative_fixup_fields))
            .or_default() += 1;
        for (field, outgoing_fixup) in &outgoing_fixups {
            let pointer_bytes = read_range(
                source,
                file_size,
                *field,
                8,
                &path.display().to_string(),
                "read dictionary record outgoing pointer",
            )?;
            let raw = i64_at(&pointer_bytes, 0, "dictionary record outgoing pointer")?;
            let outgoing_target =
                resolve_texture_graph_fixup(archive, *field, raw, outgoing_fixup)?;
            if let Some(descriptor) = paired_descriptor
                .filter(|descriptor| descriptor.resource_page == outgoing_fixup.target_page)
            {
                let delta = i64::try_from(
                    i128::from(outgoing_target) - i128::from(descriptor.resource_offset),
                )
                .map_err(|_| {
                    invalid(
                        "dictionary outgoing target delta",
                        outgoing_target,
                        "does not fit i64",
                    )
                })?;
                *outgoing_target_deltas
                    .entry((outgoing_fixup.target_page, delta))
                    .or_default() += 1;
            }
        }

        if target_page.index == 0 && page_zero_samples < 10 {
            selected_samples.insert(index);
            page_zero_samples += 1;
        } else if let Some(descriptor) = paired_descriptor {
            let sizes = formats_selected.entry(descriptor.format_id).or_default();
            let dimensions = (descriptor.width, descriptor.height);
            if sizes.len() < 2 && !sizes.contains(&dimensions) {
                sizes.push(dimensions);
                selected_samples.insert(index);
            }
        }

        root_edges.push(Some((
            index,
            fixup.source_page,
            fixup.pointer_offset,
            fixup.target_page,
            target,
            descriptor_pointer,
            paired_descriptor,
        )));
    }

    output.push_str(&format!(
        "root +0x40 explicit fixups={} missing_fixups={} target_pages={edges_by_target_page:?} unique_targets={}; paired +0x48 VRAM_DESC matches={} unmatched={}\n",
        header.entry_count - missing_root_fixups,
        missing_root_fixups,
        distinct_target_addresses.len(),
        header.entry_count - unmatched_descriptors,
        unmatched_descriptors
    ));
    output.push_str("root +0x40 fixup tuples: (root_entry, source_page, source_page_offset, target_page, resolved_file_address)\n");
    for edge in root_edges.iter().flatten() {
        output.push_str(&format!(
            "  ({}, {}, {:#x}, {}, {:#x})\n",
            edge.0, edge.1, edge.2, edge.3, edge.4
        ));
    }
    output.push_str("probe fixup-offset signature groups (relative offsets are not record fields until boundaries are established):\n");
    for ((page, offsets), count) in signature_counts.iter().take(24) {
        output.push_str(&format!(
            "  target_page={page} entries={count} explicit_fixup_offsets={offsets:?}\n"
        ));
    }
    output.push_str(&format!(
        "probe outgoing targets relative to same-entry VRAM_DESC resource starts: {outgoing_target_deltas:?}\n"
    ));
    output.push_str(&format!(
        "root +0x40 targets matching paired descriptor path after [BUILD_INTERMEDIATE]common/texture4/ prefix: matches={target_path_suffix_matches} mismatches={target_path_suffix_mismatches}\n"
    ));
    output.push_str(&format!(
        "paired VRAM_DESC records with an explicit +0x78 fixup to inline path at resource+0x80: {descriptor_inline_path_fixups}/{}\n",
        header.entry_count - unmatched_descriptors
    ));

    output.push_str("representative cross-references:\n");
    for edge in root_edges
        .iter()
        .flatten()
        .filter(|edge| selected_samples.contains(&edge.0))
    {
        let (
            index,
            source_page,
            source_offset,
            target_page_index,
            target,
            descriptor_pointer,
            descriptor,
        ) = *edge;
        let target_page = &archive.pages[target_page_index];
        let bytes = read_range(
            source,
            file_size,
            target,
            RECORD_PREVIEW_BYTES,
            &path.display().to_string(),
            "read dictionary record preview",
        )?;
        let descriptor = descriptor.ok_or_else(|| {
            invalid(
                "paired VRAM_DESC",
                descriptor_pointer.unwrap_or_default(),
                "selected sample has no paired descriptor",
            )
        })?;
        let target_string = read_pointer_string(archive, source, file_size, target, path)?;
        let inline_path_address = descriptor.resource_offset.checked_add(128).ok_or_else(|| {
            invalid(
                "VRAM_DESC inline path address",
                descriptor.resource_offset,
                "offset overflow",
            )
        })?;
        let payload_offset = texture_payload_base(archive)?
            .checked_add(u64::from(descriptor.data_offset))
            .ok_or_else(|| {
                invalid(
                    "texture payload offset",
                    descriptor.resource_offset,
                    "overflow",
                )
            })?;
        output.push_str(&format!(
            "  root_entry={index} fixup=({source_page},{source_offset:#x})->({target_page_index},{target:#x}) target_page_offset={:#x} target_bytes={} target_string={}\n    paired +0x48={:#x} -> VRAM_DESC resource={} offset={:#x} hash={:#018x} path_pointer={} path={} {} {}x{} mips={} stream_flags={:#x} payload=({:#x}, {})\n",
            target - target_page.offset,
            bytes.iter().map(|byte| format!("{byte:02x}")).collect::<Vec<_>>().join(" "),
            target_string,
            descriptor_pointer.unwrap_or_default(),
            descriptor.resource_index,
            descriptor.resource_offset,
            descriptor.hash,
            descriptor
                .texture_path_pointer
                .map_or_else(|| "unresolved".to_owned(), |pointer| format!("{pointer:#x}")),
            descriptor.texture_path.as_deref().unwrap_or("unresolved"),
            descriptor.format_name,
            descriptor.width,
            descriptor.height,
            descriptor.mip_count,
            descriptor.stream_flags,
            payload_offset,
            descriptor.data_size
        ));
        let path_field = descriptor.resource_offset.checked_add(120).ok_or_else(|| {
            invalid(
                "VRAM_DESC path fixup field",
                descriptor.resource_offset,
                "offset overflow",
            )
        })?;
        if let Some(path_fixup) = pointer_fixups.get(&path_field) {
            let pointer_bytes = read_range(
                source,
                file_size,
                path_field,
                8,
                &path.display().to_string(),
                "read VRAM_DESC inline path pointer",
            )?;
            let raw = i64_at(&pointer_bytes, 0, "VRAM_DESC inline path pointer")?;
            let resolved = resolve_texture_graph_fixup(archive, path_field, raw, path_fixup)?;
            output.push_str(&format!(
                "    VRAM_DESC path fixup source=({},{:#x}) target=({},{:#x}) inline_path_address={:#x} exact_match={}\n",
                path_fixup.source_page,
                path_fixup.pointer_offset,
                path_fixup.target_page,
                resolved,
                inline_path_address,
                resolved == inline_path_address
            ));
        } else {
            output.push_str("    VRAM_DESC path field at resource+0x78 has no explicit fixup\n");
        }
        let probe_end = target
            .saturating_add(RECORD_PROBE_BYTES)
            .min(target_page.offset + u64::from(target_page.stored_size));
        for (&field, fixup) in pointer_fixups.range(target..probe_end) {
            if field.checked_add(8).is_none_or(|end| end > probe_end) {
                continue;
            }
            let pointer_bytes = read_range(
                source,
                file_size,
                field,
                8,
                &path.display().to_string(),
                "read representative dictionary pointer",
            )?;
            let raw = i64_at(&pointer_bytes, 0, "representative dictionary pointer")?;
            let outgoing_target = resolve_texture_graph_fixup(archive, field, raw, fixup)?;
            output.push_str(&format!(
                "    explicit_fixup source=({},{:#x}) target=({},{:#x}) relative_to_root_target={:#x} paired_desc_delta={}\n",
                fixup.source_page,
                fixup.pointer_offset,
                fixup.target_page,
                outgoing_target,
                field - target,
                i128::from(outgoing_target) - i128::from(descriptor.resource_offset)
            ));
        }
    }

    append_duplicate_path_payload_fingerprints(
        archive,
        descriptors,
        source,
        file_size,
        path,
        &mut output,
    )?;
    Ok(output)
}

fn texture_asset_path(path: &str) -> String {
    path.strip_prefix("[BUILD_INTERMEDIATE]common/texture4/")
        .unwrap_or(path)
        .to_owned()
}

fn render_texture_variant_comparison(
    variants: std::collections::BTreeMap<String, Vec<TextureVariantRow>>,
) -> Result<String, PakError> {
    const VARIANT_HASH_CHUNK_SIZE: usize = 1024 * 1024;
    let mut output = String::from(
        "Same-path texture variant comparison\nPaths are normalized only by removing the observed [BUILD_INTERMEDIATE]common/texture4/ prefix. Payload fingerprints are FNV-1a 64-bit; identical-content classes are assigned only after exact byte comparison.\n",
    );
    let mut duplicate_groups = 0usize;
    let mut differing_descriptors = 0usize;
    let mut non_offset_descriptor_differences = 0usize;
    let mut distinct_payload_groups = 0usize;
    let mut exact_shared_payload_groups = 0usize;
    let mut shifted_mip_analysis = ShiftedMipAnalysis::default();
    let mut payload_bytes_read = 0u64;

    for (asset_path, rows) in variants.iter().filter(|(_, rows)| rows.len() > 1) {
        duplicate_groups += 1;
        let mut fingerprints = Vec::with_capacity(rows.len());
        let mut representatives = Vec::<(usize, u64)>::new();
        let mut row_classes = Vec::with_capacity(rows.len());
        let mut payloads_complete = true;
        for row in rows {
            let Some(payload_offset) = row.payload_offset else {
                payloads_complete = false;
                fingerprints.push(None);
                row_classes.push(None);
                continue;
            };
            let size = u64::from(row.descriptor.data_size);
            let (fingerprint, bytes_read) = texture_payload_fingerprint(
                &row.archive_path,
                payload_offset,
                size,
                VARIANT_HASH_CHUNK_SIZE,
            )?;
            payload_bytes_read = payload_bytes_read.checked_add(bytes_read).ok_or_else(|| {
                invalid(
                    "variant payload bytes",
                    payload_offset,
                    "total size overflow",
                )
            })?;
            let mut byte_class = None;
            for &(representative_index, representative_fingerprint) in &representatives {
                let representative = &rows[representative_index];
                if representative_fingerprint == fingerprint
                    && representative.descriptor.data_size == row.descriptor.data_size
                    && texture_payloads_equal(representative, row, VARIANT_HASH_CHUNK_SIZE)?
                {
                    byte_class = row_classes[representative_index];
                    break;
                }
            }
            let byte_class = byte_class.unwrap_or_else(|| {
                representatives.push((row_classes.len(), fingerprint));
                representatives.len() - 1
            });
            fingerprints.push(Some(fingerprint));
            row_classes.push(Some(byte_class));
        }

        let format_ids = unique_values(rows.iter().map(|row| row.descriptor.format_id));
        let types = unique_values(rows.iter().map(|row| row.descriptor.texture_type));
        let dimensions = unique_values(
            rows.iter()
                .map(|row| (row.descriptor.width, row.descriptor.height)),
        );
        let mip_counts = unique_values(rows.iter().map(|row| row.descriptor.mip_count));
        let sizes = unique_values(rows.iter().map(|row| row.descriptor.data_size));
        let flags = unique_values(rows.iter().map(|row| row.descriptor.stream_flags));
        let hashes = unique_values(rows.iter().map(|row| row.descriptor.hash));
        let mut varying_fields = Vec::new();
        macro_rules! track_varying_field {
            ($field:literal, $values:expr) => {
                if unique_values($values).len() > 1 {
                    varying_fields.push($field);
                }
            };
        }
        track_varying_field!("format_id", rows.iter().map(|row| row.descriptor.format_id));
        track_varying_field!(
            "texture_type",
            rows.iter().map(|row| row.descriptor.texture_type)
        );
        track_varying_field!(
            "texture_dict_id",
            rows.iter().map(|row| row.descriptor.texture_dict_id)
        );
        track_varying_field!("width", rows.iter().map(|row| row.descriptor.width));
        track_varying_field!("height", rows.iter().map(|row| row.descriptor.height));
        track_varying_field!("mip_count", rows.iter().map(|row| row.descriptor.mip_count));
        track_varying_field!(
            "stream_flags",
            rows.iter().map(|row| row.descriptor.stream_flags)
        );
        track_varying_field!("data_size", rows.iter().map(|row| row.descriptor.data_size));
        track_varying_field!(
            "descriptor_hash",
            rows.iter().map(|row| row.descriptor.hash)
        );
        track_varying_field!("field_2c", rows.iter().map(|row| row.descriptor.field_2c));
        track_varying_field!("field_3c", rows.iter().map(|row| row.descriptor.field_3c));
        track_varying_field!("unknown0", rows.iter().map(|row| row.descriptor.unknown0));
        track_varying_field!("unknown1", rows.iter().map(|row| row.descriptor.unknown1));
        track_varying_field!(
            "data_offset",
            rows.iter().map(|row| row.descriptor.data_offset)
        );
        let metadata_words: Vec<_> = (0..10)
            .filter(|index| {
                unique_values(
                    rows.iter()
                        .map(|row| row.descriptor.resource_metadata_words[*index]),
                )
                .len()
                    > 1
            })
            .collect();
        if !metadata_words.is_empty() {
            varying_fields.push("opaque_metadata_words");
        }
        let distinct_payload_count = representatives.len();
        let descriptor_differs = !varying_fields.is_empty();
        if descriptor_differs {
            differing_descriptors += 1;
        }
        if varying_fields.iter().any(|field| *field != "data_offset") {
            non_offset_descriptor_differences += 1;
        }
        if payloads_complete && distinct_payload_count > 1 {
            distinct_payload_groups += 1;
        }
        if payloads_complete && distinct_payload_count == 1 {
            exact_shared_payload_groups += 1;
        }

        let shifted_suffixes = exact_shifted_mip_suffixes(rows, VARIANT_HASH_CHUNK_SIZE)?;
        shifted_mip_analysis.shape_candidates += shifted_suffixes.shape_candidates;
        shifted_mip_analysis.prefix_size_matches += shifted_suffixes.prefix_size_matches;
        shifted_mip_analysis.suffix_mismatches += shifted_suffixes.suffix_mismatches;
        shifted_mip_analysis.size_mismatches += shifted_suffixes.size_mismatches;
        output.push_str(&format!(
            "\nPATH {asset_path}\n  variants={} varying_fields={varying_fields:?} formats={format_ids:?} texture_types={types:?} dimensions={dimensions:?} mips={mip_counts:?} stored_sizes={sizes:?} stream_flags={flags:?} descriptor_hashes={hashes:?} varying_opaque_words={metadata_words:?} exact_payload_classes={} payload_complete={} exact_shifted_mip_suffixes={:?}\n",
            rows.len(),
            distinct_payload_count,
            payloads_complete,
            shifted_suffixes.exact_matches
        ));
        shifted_mip_analysis
            .exact_matches
            .extend(shifted_suffixes.exact_matches);
        for (index, row) in rows.iter().enumerate() {
            let descriptor = &row.descriptor;
            let payload_class = row_classes[index].map_or_else(
                || "unavailable".to_owned(),
                |class| format!("exact-{class}"),
            );
            let fingerprint = fingerprints[index]
                .map_or_else(|| "unavailable".to_owned(), |hash| format!("{hash:016x}"));
            let path_name = row
                .archive_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<non-utf8>");
            let opaque_words = descriptor
                .resource_metadata_words
                .iter()
                .map(|word| format!("{word:08x}"))
                .collect::<Vec<_>>()
                .join(",");
            output.push_str(&format!(
                "  {path_name} resource={} page={} class={} hash={:#018x} format={:#x}/{} texture_type={} dict_id={} dims={}x{} mips={} flags={:#x} field_2c={:#x} field_3c={:#x} unknown0={:#x} unknown1={:#x} opaque=[{}] data_offset={:#x} payload_offset={} size={} fingerprint={} content_class={}\n",
                descriptor.resource_index,
                descriptor.resource_page,
                descriptor.resource_page_class,
                descriptor.hash,
                descriptor.format_id,
                descriptor.format_name,
                descriptor.texture_type,
                descriptor.texture_dict_id,
                descriptor.width,
                descriptor.height,
                descriptor.mip_count,
                descriptor.stream_flags,
                descriptor.field_2c,
                descriptor.field_3c,
                descriptor.unknown0,
                descriptor.unknown1,
                opaque_words,
                descriptor.data_offset,
                row.payload_offset
                    .map_or_else(|| "unavailable".to_owned(), |offset| format!("{offset:#x}")),
                descriptor.data_size,
                fingerprint,
                payload_class
            ));
        }
    }

    output.push_str(&format!(
        "\nSUMMARY duplicate_path_groups={duplicate_groups} groups_with_any_descriptor_field_differences={differing_descriptors} groups_with_non_offset_descriptor_differences={non_offset_descriptor_differences} groups_with_distinct_payload_bytes={distinct_payload_groups} groups_with_exact_shared_payload={exact_shared_payload_groups} shifted_mip_shape_candidates={} predicted_prefix_size_matches={} exact_shifted_mip_suffix_pairs={} suffix_mismatches={} prefix_size_mismatches={} payload_bytes_fingerprinted={payload_bytes_read}\n",
        shifted_mip_analysis.shape_candidates,
        shifted_mip_analysis.prefix_size_matches,
        shifted_mip_analysis.exact_matches.len(),
        shifted_mip_analysis.suffix_mismatches,
        shifted_mip_analysis.size_mismatches
    ));
    Ok(output)
}

fn exact_shifted_mip_suffixes(
    rows: &[TextureVariantRow],
    chunk_size: usize,
) -> Result<ShiftedMipAnalysis, PakError> {
    let mut analysis = ShiftedMipAnalysis::default();
    for (left_index, left) in rows.iter().enumerate() {
        for right in rows.iter().skip(left_index + 1) {
            let (smaller, larger) = if left.descriptor.data_size < right.descriptor.data_size {
                (left, right)
            } else {
                (right, left)
            };
            let small = &smaller.descriptor;
            let large = &larger.descriptor;
            if small.format_id != large.format_id || large.mip_count <= small.mip_count {
                continue;
            }
            let added_mips = large.mip_count - small.mip_count;
            let Some(scale) = 1u32.checked_shl(added_mips) else {
                continue;
            };
            if small.width.checked_mul(scale) != Some(large.width)
                || small.height.checked_mul(scale) != Some(large.height)
            {
                continue;
            }
            analysis.shape_candidates += 1;
            let block_bytes = texture_format(large.format_id).1;
            let Some(prefix_bytes) = expected_texture_bytes(
                large.format_id,
                large.width,
                large.height,
                added_mips,
                block_bytes,
            ) else {
                continue;
            };
            let Some(payload_size_difference) = large.data_size.checked_sub(small.data_size) else {
                analysis.size_mismatches += 1;
                continue;
            };
            if u64::from(payload_size_difference) != prefix_bytes {
                analysis.size_mismatches += 1;
                continue;
            }
            analysis.prefix_size_matches += 1;
            let (Some(small_offset), Some(large_offset)) =
                (smaller.payload_offset, larger.payload_offset)
            else {
                continue;
            };
            let suffix_offset = large_offset.checked_add(prefix_bytes).ok_or_else(|| {
                invalid(
                    "shifted mip suffix offset",
                    large_offset,
                    "address overflow",
                )
            })?;
            if texture_payload_ranges_equal(
                &smaller.archive_path,
                small_offset,
                &larger.archive_path,
                suffix_offset,
                u64::from(small.data_size),
                chunk_size,
            )? {
                analysis.exact_matches.push((
                    small.resource_index,
                    large.resource_index,
                    prefix_bytes,
                    added_mips,
                ));
            } else {
                analysis.suffix_mismatches += 1;
            }
        }
    }
    Ok(analysis)
}

fn unique_values<T: Ord>(values: impl Iterator<Item = T>) -> std::collections::BTreeSet<T> {
    values.collect()
}

#[derive(Default)]
struct PitchModelStats {
    count: usize,
    exact: usize,
    residuals: std::collections::BTreeMap<i128, usize>,
}

impl PitchModelStats {
    fn observe(&mut self, residual: i128) {
        self.count += 1;
        self.exact += usize::from(residual == 0);
        *self.residuals.entry(residual).or_default() += 1;
    }
}

fn pitched_texture_mip_bytes(
    descriptor: &TextureDescriptor,
    row_alignment: u32,
    block_column_alignment: Option<u32>,
) -> Option<u64> {
    if row_alignment == 0 || descriptor.width == 0 || descriptor.height == 0 {
        return None;
    }
    let (_, block_bytes) = texture_format(descriptor.format_id);
    let mut width = descriptor.width;
    let mut height = descriptor.height;
    let mut total = 0u64;
    for _ in 0..descriptor.mip_count {
        let (row_bytes, rows) = match block_bytes {
            Some(bytes) if bytes > 0 => {
                let block_columns = width.div_ceil(4);
                let block_rows = height.div_ceil(4);
                let columns = match block_column_alignment {
                    Some(alignment) if alignment > 0 => align_u32(block_columns, alignment)?,
                    Some(_) => return None,
                    None => block_columns,
                };
                (
                    u64::from(columns).checked_mul(u64::from(bytes))?,
                    u64::from(block_rows),
                )
            }
            Some(0) if block_column_alignment.is_none() => {
                let bytes_per_pixel = match descriptor.format_id {
                    0x1c | 0x1d => 4,
                    0x3d => 1,
                    _ => return None,
                };
                (
                    u64::from(width).checked_mul(bytes_per_pixel)?,
                    u64::from(height),
                )
            }
            _ => return None,
        };
        let pitch = align_u64(row_bytes, u64::from(row_alignment))?;
        total = total.checked_add(pitch.checked_mul(rows)?)?;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    Some(total)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct D3d12CopyFootprintCandidate {
    end_bytes: u64,
    allocation_bytes: u64,
    row_padding_bytes: u64,
    row_padding_zero_bytes: Option<u64>,
}

/// Predict a buffer-copy-style footprint: 256-byte row pitches, 512-byte
/// subresource starts, and 512-byte final allocation size. This is an analysis
/// model only; a size match does not establish the order of blocks in a row.
fn d3d12_copyable_footprint_candidate(
    descriptor: &TextureDescriptor,
    payload: Option<&[u8]>,
) -> Option<D3d12CopyFootprintCandidate> {
    d3d12_copyable_footprint_candidate_with_order(descriptor, payload, false, false)
}

fn d3d12_copyable_footprint_candidate_with_order(
    descriptor: &TextureDescriptor,
    payload: Option<&[u8]>,
    reverse_mip_order: bool,
    mip_major_surfaces: bool,
) -> Option<D3d12CopyFootprintCandidate> {
    if descriptor.width == 0 || descriptor.height == 0 || descriptor.mip_count == 0 {
        return None;
    }
    let (surface_count, initial_depth, is_volume) = match descriptor.texture_type {
        1 => (1, 1, false),
        2 => {
            let surfaces = descriptor.field_2c >> 24;
            if surfaces != 6 {
                return None;
            }
            (surfaces, 1, false)
        }
        3 => {
            let encoded_depth = descriptor.field_2c >> 24;
            if descriptor.field_3c == 0 || descriptor.field_3c != encoded_depth {
                return None;
            }
            (1, descriptor.field_3c, true)
        }
        _ => return None,
    };
    let (_, block_bytes) = texture_format(descriptor.format_id);
    let mut levels = Vec::with_capacity(descriptor.mip_count as usize);
    let mut width = descriptor.width;
    let mut height = descriptor.height;
    let mut depth = initial_depth;
    for _ in 0..descriptor.mip_count {
        levels.push((width, height, depth));
        width = (width / 2).max(1);
        height = (height / 2).max(1);
        if is_volume {
            depth = (depth / 2).max(1);
        }
    }
    if reverse_mip_order {
        levels.reverse();
    }

    let mut subresources = Vec::new();
    if mip_major_surfaces && surface_count > 1 {
        for level in &levels {
            for _surface in 0..surface_count {
                subresources.push(*level);
            }
        }
    } else {
        for _surface in 0..surface_count {
            subresources.extend_from_slice(&levels);
        }
    }

    let mut offset = 0_u64;
    let mut row_padding_bytes = 0_u64;
    let mut row_padding_zero_bytes = 0_u64;
    let zero_count_available = payload.is_some();
    for (width, height, depth) in subresources {
        let (row_bytes, rows) = match block_bytes {
            Some(bytes) if bytes > 0 => (
                u64::from(width.div_ceil(4)).checked_mul(u64::from(bytes))?,
                u64::from(height.div_ceil(4)),
            ),
            Some(0) => {
                let bytes_per_pixel = match descriptor.format_id {
                    0x1c | 0x1d => 4,
                    0x3d => 1,
                    _ => return None,
                };
                (
                    u64::from(width).checked_mul(bytes_per_pixel)?,
                    u64::from(height),
                )
            }
            _ => return None,
        };
        let pitch = align_u64(row_bytes, 256)?;
        offset = align_u64(offset, 512)?;
        let slice_bytes = pitch.checked_mul(rows)?;
        let subresource_bytes = slice_bytes.checked_mul(u64::from(depth))?;
        let row_padding = pitch.checked_sub(row_bytes)?;
        row_padding_bytes = row_padding_bytes.checked_add(
            row_padding
                .checked_mul(rows)?
                .checked_mul(u64::from(depth))?,
        )?;

        if let Some(bytes) = payload {
            for slice in 0..u64::from(depth) {
                for row in 0..rows {
                    let row_start = offset
                        .checked_add(slice.checked_mul(slice_bytes)?)?
                        .checked_add(row.checked_mul(pitch)?)?;
                    let padding_start = row_start.checked_add(row_bytes)?;
                    let padding_end = padding_start.checked_add(row_padding)?;
                    let start = usize::try_from(padding_start).ok()?;
                    let end = usize::try_from(padding_end).ok()?;
                    let padding = bytes.get(start..end)?;
                    row_padding_zero_bytes = row_padding_zero_bytes
                        .checked_add(padding.iter().filter(|byte| **byte == 0).count() as u64)?;
                }
            }
        }
        offset = offset.checked_add(subresource_bytes)?;
    }

    Some(D3d12CopyFootprintCandidate {
        end_bytes: offset,
        allocation_bytes: align_u64(offset, 512)?,
        row_padding_bytes,
        row_padding_zero_bytes: zero_count_available.then_some(row_padding_zero_bytes),
    })
}

fn aligned_mip_chain_bytes(descriptor: &TextureDescriptor, alignment: u64) -> Option<u64> {
    if alignment == 0 || descriptor.width == 0 || descriptor.height == 0 {
        return None;
    }
    let (_, block_bytes) = texture_format(descriptor.format_id);
    let mut width = descriptor.width;
    let mut height = descriptor.height;
    let mut total = 0u64;
    for _ in 0..descriptor.mip_count {
        let level_size = match block_bytes {
            Some(bytes) if bytes > 0 => u64::from(width.div_ceil(4))
                .checked_mul(u64::from(height.div_ceil(4)))?
                .checked_mul(u64::from(bytes))?,
            Some(0) => {
                let bytes_per_pixel = match descriptor.format_id {
                    0x1c | 0x1d => 4,
                    0x3d => 1,
                    _ => return None,
                };
                u64::from(width)
                    .checked_mul(u64::from(height))?
                    .checked_mul(bytes_per_pixel)?
            }
            _ => return None,
        };
        total = total.checked_add(align_u64(level_size, alignment)?)?;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    Some(total)
}

fn align_u32(value: u32, alignment: u32) -> Option<u32> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|sum| sum / alignment * alignment)
}

fn align_u64(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|sum| sum / alignment * alignment)
}

struct TextureByteStats {
    sha256: String,
    entropy: f64,
    distinct_bytes: usize,
    zero_bytes: usize,
    longest_zero_run: usize,
    adjacent_equal_bytes: usize,
    unique_16_byte_blocks: usize,
}

impl TextureByteStats {
    fn new(bytes: &[u8]) -> Self {
        let mut frequencies = [0usize; 256];
        let mut zero_bytes = 0;
        let mut longest_zero_run = 0;
        let mut current_zero_run = 0;
        let mut adjacent_equal_bytes = 0;
        for (index, byte) in bytes.iter().enumerate() {
            frequencies[usize::from(*byte)] += 1;
            if index > 0 && bytes[index - 1] == *byte {
                adjacent_equal_bytes += 1;
            }
            if *byte == 0 {
                zero_bytes += 1;
                current_zero_run += 1;
                longest_zero_run = longest_zero_run.max(current_zero_run);
            } else {
                current_zero_run = 0;
            }
        }
        let length = bytes.len() as f64;
        let entropy = if bytes.is_empty() {
            0.0
        } else {
            frequencies
                .iter()
                .filter(|frequency| **frequency > 0)
                .map(|frequency| {
                    let probability = *frequency as f64 / length;
                    -probability * probability.log2()
                })
                .sum()
        };
        let unique_16_byte_blocks = bytes
            .as_chunks::<16>()
            .0
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        Self {
            sha256: sha256_hex(bytes),
            entropy,
            distinct_bytes: frequencies
                .iter()
                .filter(|frequency| **frequency > 0)
                .count(),
            zero_bytes,
            longest_zero_run,
            adjacent_equal_bytes,
            unique_16_byte_blocks,
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_texture_payload_bytes(payload: &TexturePayload) -> Result<Vec<u8>, PakError> {
    let mut source = File::open(&payload.source_path).map_err(|source| PakError::Io {
        path: payload.source_path.clone(),
        operation: "open texture payload for analysis",
        source,
    })?;
    let file_size = source
        .metadata()
        .map_err(|source| PakError::Io {
            path: payload.source_path.clone(),
            operation: "read texture payload metadata for analysis",
            source,
        })?
        .len();
    read_range(
        &mut source,
        file_size,
        payload.source_data_offset,
        usize::try_from(payload.source_data_size).map_err(|_| {
            invalid(
                "texture payload analysis size",
                payload.source_data_offset,
                "does not fit address space",
            )
        })?,
        &payload.source_path.display().to_string(),
        "read texture payload for analysis",
    )
}

fn texture_payload_fingerprint(
    path: &Path,
    offset: u64,
    size: u64,
    chunk_size: usize,
) -> Result<(u64, u64), PakError> {
    let mut source = File::open(path).map_err(|source| PakError::Io {
        path: path.to_owned(),
        operation: "open texture payload for fingerprinting",
        source,
    })?;
    let file_size = source
        .metadata()
        .map_err(|source| PakError::Io {
            path: path.to_owned(),
            operation: "read texture payload metadata for fingerprinting",
            source,
        })?
        .len();
    let end = offset
        .checked_add(size)
        .ok_or_else(|| invalid("texture payload fingerprint range", offset, "overflow"))?;
    if end > file_size {
        return Err(invalid(
            "texture payload fingerprint range",
            offset,
            format!("{size} bytes exceed {file_size}-byte file"),
        ));
    }
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    let mut processed = 0u64;
    while processed < size {
        let remaining = size - processed;
        let count = usize::try_from(remaining.min(chunk_size as u64))
            .map_err(|_| invalid("texture payload fingerprint chunk", offset, "overflow"))?;
        let chunk_offset = offset
            .checked_add(processed)
            .ok_or_else(|| invalid("texture payload fingerprint range", offset, "overflow"))?;
        let bytes = read_range(
            &mut source,
            file_size,
            chunk_offset,
            count,
            &path.display().to_string(),
            "read texture payload fingerprint chunk",
        )?;
        for byte in bytes {
            fingerprint = (fingerprint ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
        processed += count as u64;
    }
    Ok((fingerprint, processed))
}

fn texture_payloads_equal(
    left: &TextureVariantRow,
    right: &TextureVariantRow,
    chunk_size: usize,
) -> Result<bool, PakError> {
    if left.descriptor.data_size != right.descriptor.data_size {
        return Ok(false);
    }
    let (Some(left_offset), Some(right_offset)) = (left.payload_offset, right.payload_offset)
    else {
        return Ok(false);
    };
    texture_payload_ranges_equal(
        &left.archive_path,
        left_offset,
        &right.archive_path,
        right_offset,
        u64::from(left.descriptor.data_size),
        chunk_size,
    )
}

fn texture_payload_ranges_equal(
    left_path: &Path,
    left_offset: u64,
    right_path: &Path,
    right_offset: u64,
    size: u64,
    chunk_size: usize,
) -> Result<bool, PakError> {
    if chunk_size == 0 {
        return Err(invalid(
            "texture payload comparison chunk",
            left_offset,
            "must be non-zero",
        ));
    }
    let mut left_file = File::open(left_path).map_err(|source| PakError::Io {
        path: left_path.to_owned(),
        operation: "open left texture payload for exact comparison",
        source,
    })?;
    let left_size = left_file
        .metadata()
        .map_err(|source| PakError::Io {
            path: left_path.to_owned(),
            operation: "read left texture payload metadata",
            source,
        })?
        .len();
    let mut right_file = File::open(right_path).map_err(|source| PakError::Io {
        path: right_path.to_owned(),
        operation: "open right texture payload for exact comparison",
        source,
    })?;
    let right_size = right_file
        .metadata()
        .map_err(|source| PakError::Io {
            path: right_path.to_owned(),
            operation: "read right texture payload metadata",
            source,
        })?
        .len();
    let mut processed = 0u64;
    while processed < size {
        let count = usize::try_from((size - processed).min(chunk_size as u64))
            .map_err(|_| invalid("texture payload comparison chunk", left_offset, "overflow"))?;
        let left_chunk_offset = left_offset.checked_add(processed).ok_or_else(|| {
            invalid(
                "texture payload comparison range",
                left_offset,
                "offset overflow",
            )
        })?;
        let right_chunk_offset = right_offset.checked_add(processed).ok_or_else(|| {
            invalid(
                "texture payload comparison range",
                right_offset,
                "offset overflow",
            )
        })?;
        let left_chunk = read_range(
            &mut left_file,
            left_size,
            left_chunk_offset,
            count,
            &left_path.display().to_string(),
            "read left texture payload comparison chunk",
        )?;
        let right_chunk = read_range(
            &mut right_file,
            right_size,
            right_chunk_offset,
            count,
            &right_path.display().to_string(),
            "read right texture payload comparison chunk",
        )?;
        if left_chunk != right_chunk {
            return Ok(false);
        }
        processed += count as u64;
    }
    Ok(true)
}

fn append_duplicate_path_payload_fingerprints<R: Read + Seek>(
    archive: &PakArchive,
    descriptors: &[TextureDescriptor],
    source: &mut R,
    file_size: u64,
    path: &Path,
    output: &mut String,
) -> Result<(), PakError> {
    use std::collections::BTreeMap;

    const MAX_FINGERPRINT_PAYLOAD_BYTES: u32 = 64 * 1024 * 1024;
    const MAX_FINGERPRINT_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
    let payload_base = texture_payload_base(archive)?;
    let mut by_path = BTreeMap::<&str, Vec<&TextureDescriptor>>::new();
    for descriptor in descriptors {
        if let Some(texture_path) = descriptor.texture_path.as_deref() {
            by_path.entry(texture_path).or_default().push(descriptor);
        }
    }
    output.push_str("duplicate asset paths with distinct raw payload fingerprints (FNV-1a 64-bit; non-cryptographic):\n");
    let mut emitted = 0usize;
    let mut fingerprinted_bytes = 0u64;
    for (texture_path, matches) in by_path.iter().filter(|(_, matches)| matches.len() > 1) {
        if emitted >= 16 {
            break;
        }
        if matches
            .iter()
            .any(|descriptor| descriptor.data_size > MAX_FINGERPRINT_PAYLOAD_BYTES)
        {
            emitted += 1;
            output.push_str(&format!(
                "  path={texture_path}: fingerprint skipped; at least one payload exceeds {MAX_FINGERPRINT_PAYLOAD_BYTES} bytes\n"
            ));
            continue;
        }
        let group_size = matches.iter().try_fold(0u64, |total, descriptor| {
            total.checked_add(u64::from(descriptor.data_size))
        });
        let Some(group_size) = group_size else {
            emitted += 1;
            output.push_str(&format!(
                "  path={texture_path}: fingerprint skipped; aggregate payload size overflow\n"
            ));
            continue;
        };
        let Some(next_fingerprinted_bytes) = fingerprinted_bytes.checked_add(group_size) else {
            emitted += 1;
            output.push_str(&format!(
                "  path={texture_path}: fingerprint skipped; aggregate report size overflow\n"
            ));
            continue;
        };
        if next_fingerprinted_bytes > MAX_FINGERPRINT_TOTAL_BYTES {
            emitted += 1;
            output.push_str(&format!(
                "  path={texture_path}: fingerprint skipped; report limit is {MAX_FINGERPRINT_TOTAL_BYTES} payload bytes\n"
            ));
            continue;
        }
        fingerprinted_bytes = next_fingerprinted_bytes;
        let mut fingerprints = BTreeMap::<u64, Vec<&TextureDescriptor>>::new();
        for descriptor in matches {
            let offset = payload_base
                .checked_add(u64::from(descriptor.data_offset))
                .ok_or_else(|| {
                    invalid(
                        "texture payload offset",
                        descriptor.resource_offset,
                        "overflow",
                    )
                })?;
            let bytes = read_range(
                source,
                file_size,
                offset,
                descriptor.data_size as usize,
                &path.display().to_string(),
                "read duplicate-path texture payload",
            )?;
            let fingerprint = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            });
            fingerprints
                .entry(fingerprint)
                .or_default()
                .push(descriptor);
        }
        if fingerprints.len() < 2 {
            continue;
        }
        emitted += 1;
        output.push_str(&format!("  path={texture_path}\n"));
        for (fingerprint, variants) in fingerprints {
            for descriptor in variants {
                output.push_str(&format!(
                    "    resource={} content_fingerprint={fingerprint:#018x} hash={:#018x} {} {}x{} mips={} flags={:#x} data_size={} data_offset={:#x}\n",
                    descriptor.resource_index,
                    descriptor.hash,
                    descriptor.format_name,
                    descriptor.width,
                    descriptor.height,
                    descriptor.mip_count,
                    descriptor.stream_flags,
                    descriptor.data_size,
                    descriptor.data_offset
                ));
            }
        }
    }
    if emitted == 0 {
        output.push_str("  none found among exact duplicate path strings\n");
    }
    Ok(())
}

fn resolve_texture_graph_fixup(
    archive: &PakArchive,
    field_address: u64,
    raw_value: i64,
    fixup: &PakFixup,
) -> Result<u64, PakError> {
    let target_page = &archive.pages[fixup.target_page];
    let address = i128::from(target_page.offset) + i128::from(raw_value);
    let page_end = i128::from(target_page.offset + u64::from(target_page.stored_size));
    if address < i128::from(target_page.offset) || address >= page_end {
        return Err(invalid(
            "texture graph pointer",
            field_address,
            format!(
                "raw value {raw_value:#x} resolves outside target page {}",
                fixup.target_page
            ),
        ));
    }
    Ok(address as u64)
}

fn classify_texture_graph_target(
    archive: &PakArchive,
    descriptors: &[TextureDescriptor],
    descriptor_by_offset: &std::collections::BTreeMap<u64, &TextureDescriptor>,
    target: u64,
) -> String {
    if let Some(descriptor) = descriptor_by_offset.get(&target) {
        return format!(
            "exact VRAM_DESC resource={} hash={:#018x} {} {}x{} mips={} stored={} path={}",
            descriptor.resource_index,
            descriptor.hash,
            descriptor.format_name,
            descriptor.width,
            descriptor.height,
            descriptor.mip_count,
            descriptor.data_size,
            descriptor.texture_path.as_deref().unwrap_or("unresolved")
        );
    }
    let page_index = archive
        .pages
        .iter()
        .find(|page| target >= page.offset && target < page.offset + u64::from(page.stored_size))
        .map(|page| page.index);
    if let Some(page_index) = page_index
        && let Some((descriptor, delta)) =
            nearest_texture_descriptor(descriptors, page_index, target)
        && delta <= 256
    {
        return format!(
            "near VRAM_DESC resource={} delta=+{delta:#x} hash={:#018x} path={}",
            descriptor.resource_index,
            descriptor.hash,
            descriptor.texture_path.as_deref().unwrap_or("unresolved")
        );
    }
    format!(
        "page={} address not classified as a VRAM_DESC start",
        page_index.map_or_else(|| "outside-pages".to_owned(), |page| page.to_string())
    )
}

fn nearest_texture_descriptor(
    descriptors: &[TextureDescriptor],
    page_index: usize,
    target: u64,
) -> Option<(&TextureDescriptor, u64)> {
    descriptors
        .iter()
        .filter(|descriptor| {
            descriptor.resource_page == page_index && descriptor.resource_offset <= target
        })
        .max_by_key(|descriptor| descriptor.resource_offset)
        .map(|descriptor| (descriptor, target - descriptor.resource_offset))
}

fn texture_payload_base(archive: &PakArchive) -> Result<u64, PakError> {
    let last_page = archive
        .pages
        .last()
        .ok_or_else(|| invalid("texture payload base", 0, "PAK has no pages"))?;
    last_page
        .offset
        .checked_add(u64::from(last_page.stored_size))
        .ok_or_else(|| invalid("texture payload base", last_page.offset, "overflow"))
}

fn texture_format(format_id: u32) -> (String, Option<u32>) {
    match format_id {
        0x1c => ("R8G8B8A8_UNORM".to_owned(), Some(0)),
        0x1d => ("R8G8B8A8_UNORM_SRGB".to_owned(), Some(0)),
        0x3d => ("R8_UNORM".to_owned(), Some(0)),
        0x47 => ("BC1_UNORM".to_owned(), Some(8)),
        0x48 => ("BC1_UNORM_SRGB".to_owned(), Some(8)),
        0x4d => ("BC3_UNORM".to_owned(), Some(16)),
        0x4e => ("BC3_UNORM_SRGB".to_owned(), Some(16)),
        0x50 => ("BC4_UNORM".to_owned(), Some(8)),
        0x51 => ("BC4_SNORM".to_owned(), Some(8)),
        0x53 => ("BC5_UNORM".to_owned(), Some(16)),
        0x54 => ("BC5_SNORM".to_owned(), Some(16)),
        0x5f => ("BC6_UF16".to_owned(), Some(16)),
        0x60 => ("BC6_SF16".to_owned(), Some(16)),
        0x61 => ("BC7_TYPELESS".to_owned(), Some(16)),
        0x62 => ("BC7_UNORM".to_owned(), Some(16)),
        0x63 => ("BC7_UNORM_SRGB".to_owned(), Some(16)),
        _ => (format!("unsupported GPU format {format_id:#x}"), None),
    }
}

fn expected_texture_bytes(
    format_id: u32,
    width: u32,
    height: u32,
    mip_count: u32,
    block_bytes: Option<u32>,
) -> Option<u64> {
    if width == 0 || height == 0 || mip_count == 0 || mip_count > 32 {
        return None;
    }
    let mut level_width = width;
    let mut level_height = height;
    let mut total = 0u64;
    for _ in 0..mip_count {
        let level_size = if let Some(block_bytes) = block_bytes {
            if block_bytes == 0 {
                let bytes_per_pixel = match format_id {
                    0x1c | 0x1d => 4,
                    0x3d => 1,
                    _ => return None,
                };
                u64::from(level_width)
                    .checked_mul(u64::from(level_height))?
                    .checked_mul(bytes_per_pixel)?
            } else {
                let blocks_wide = u64::from(level_width).div_ceil(4);
                let blocks_high = u64::from(level_height).div_ceil(4);
                blocks_wide
                    .checked_mul(blocks_high)?
                    .checked_mul(u64::from(block_bytes))?
            }
        } else if (0x1c..=0x1d).contains(&format_id) {
            u64::from(level_width)
                .checked_mul(u64::from(level_height))?
                .checked_mul(4)?
        } else {
            return None;
        };
        total = total.checked_add(level_size)?;
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
    Some(total)
}

fn expected_rgba_mip_bytes(width: u32, height: u32, mip_count: u32) -> Option<u64> {
    if width == 0 || height == 0 || mip_count == 0 || mip_count > 32 {
        return None;
    }
    let mut level_width = width;
    let mut level_height = height;
    let mut total = 0u64;
    for _ in 0..mip_count {
        total = total.checked_add(
            u64::from(level_width)
                .checked_mul(u64::from(level_height))?
                .checked_mul(4)?,
        )?;
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
    Some(total)
}

fn texture_status(descriptor: &TextureDescriptor, payload: Option<&TexturePayload>) -> String {
    if descriptor.width == 0 || descriptor.height == 0 {
        return "invalid zero texture dimension".to_owned();
    }
    if descriptor.mip_count == 0 || descriptor.mip_count > 32 {
        return format!("invalid mip count {}", descriptor.mip_count);
    }
    if descriptor.expected_mip_bytes.is_none() {
        return format!(
            "{}; no verified byte-size calculation",
            descriptor.format_name
        );
    }
    if payload.is_none() {
        return format!(
            "texture payload range is unresolved or outside the PAK: {}",
            descriptor.format_name
        );
    }
    format!(
        "raw payload located; PNG unsupported because TLOU2 payload layout is unverified ({})",
        descriptor.format_name
    )
}

fn safe_filename_stem(path: &Path) -> String {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let safe: String = stem
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if safe.is_empty() {
        "pak".to_owned()
    } else {
        safe
    }
}

fn copy_texture_payload(payload: &TexturePayload, output_path: &Path) -> Result<(), PakError> {
    let mut source = File::open(&payload.source_path).map_err(|source| PakError::Io {
        path: payload.source_path.clone(),
        operation: "open texture payload source",
        source,
    })?;
    source
        .seek(SeekFrom::Start(payload.source_data_offset))
        .map_err(|source| PakError::Io {
            path: payload.source_path.clone(),
            operation: "seek to texture payload",
            source,
        })?;
    let mut target = File::options()
        .write(true)
        .create_new(true)
        .open(output_path)
        .map_err(|source| PakError::Io {
            path: output_path.to_owned(),
            operation: "create texture payload without overwriting",
            source,
        })?;
    let result = io::copy(
        &mut source.take(u64::from(payload.source_data_size)),
        &mut target,
    );
    match result {
        Ok(copied) if copied == u64::from(payload.source_data_size) => {
            target.sync_all().map_err(|source| PakError::Io {
                path: output_path.to_owned(),
                operation: "flush extracted texture payload",
                source,
            })?;
            Ok(())
        }
        Ok(copied) => {
            let _ = fs::remove_file(output_path);
            Err(invalid(
                "texture payload",
                payload.source_data_offset + copied,
                format!(
                    "source ended after {copied} bytes; descriptor requires {}",
                    payload.source_data_size
                ),
            ))
        }
        Err(source) => {
            let _ = fs::remove_file(output_path);
            Err(PakError::Io {
                path: output_path.to_owned(),
                operation: "copy raw texture payload",
                source,
            })
        }
    }
}

fn scan_geometry_submeshes<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    path: &Path,
) -> Result<Vec<SubmeshInventory>, PakError> {
    let geometry = archive
        .resources
        .iter()
        .find(|resource| resource.resource_type == "GEOMETRY_1")
        .ok_or_else(|| invalid("geometry resource", 0, "no GEOMETRY_1 resource found"))?;
    let geometry_data_offset = geometry.file_offset + 48;
    let geometry_header = read_range(
        source,
        file_size,
        geometry.file_offset,
        104,
        &path.display().to_string(),
        "read GEOMETRY_1 header",
    )?;
    let submesh_count = u32_at(&geometry_header, 56, "submesh count")? as usize;
    if submesh_count == 0 || submesh_count > 4096 {
        return Err(invalid(
            "submesh count",
            geometry_data_offset + 8,
            format!("{submesh_count} is outside the supported diagnostic range"),
        ));
    }
    let submesh_table_raw = i64_at(&geometry_header, 88, "submesh table pointer")?;
    let submesh_table = archive
        .resolve_pointer(geometry_data_offset + 40, submesh_table_raw)?
        .ok_or_else(|| {
            invalid(
                "submesh table pointer",
                geometry_data_offset + 40,
                "has no pointer fixup",
            )
        })?;
    archive.page_for_range(submesh_table, submesh_count * 192)?;
    let lods = read_geometry_lod_membership(
        archive,
        source,
        file_size,
        path,
        geometry_data_offset,
        &geometry_header,
        submesh_table,
        submesh_count,
    )?;
    let mut submeshes = Vec::with_capacity(submesh_count);
    for (index, lod) in lods.iter().copied().enumerate() {
        let record_offset = submesh_table + (index * 192) as u64;
        let record = read_range(
            source,
            file_size,
            record_offset,
            192,
            &path.display().to_string(),
            "read geometry submesh record",
        )?;
        let name_raw = i64_at(&record, 32, "submesh name pointer")?;
        let name_offset = archive
            .resolve_pointer(record_offset + 32, name_raw)?
            .ok_or_else(|| {
                invalid(
                    "submesh name pointer",
                    record_offset + 32,
                    "has no pointer fixup",
                )
            })?;
        let name = read_pointer_string(archive, source, file_size, name_offset, path)?;
        let vertex_count = u32_at(&record, 136, "vertex count")? as usize;
        let index_count = u32_at(&record, 140, "index count")? as usize;
        let stream_count = u32_at(&record, 144, "stream count")? as usize;
        let mut position_type = None;
        let mut position_format = "unresolved".to_owned();
        let mut issue = None;
        if stream_count == 0 || stream_count > 256 {
            issue = Some(format!("unsupported stream count {stream_count}"));
        } else {
            let stream_raw = i64_at(&record, 48, "stream descriptor pointer")?;
            match archive.resolve_pointer(record_offset + 48, stream_raw)? {
                Some(stream_address) => {
                    archive.page_for_range(stream_address, 64)?;
                    let descriptor = read_range(
                        source,
                        file_size,
                        stream_address,
                        64,
                        &path.display().to_string(),
                        "read position stream descriptor",
                    )?;
                    let kind = descriptor[20];
                    position_type = Some(kind);
                    let stride = usize::from(descriptor[22] >> 4);
                    match kind {
                        0 if stride == 12 => {
                            position_format = "type 0 float3 XYZ, 12-byte stride".to_owned()
                        }
                        0 => {
                            position_format = format!("type 0 float3, {stride}-byte stride");
                            issue = Some(format!(
                                "unsupported float3 position stride {stride}; expected 12"
                            ));
                        }
                        64 => {
                            let widths = [descriptor[24], descriptor[25], descriptor[26]];
                            position_format = format!(
                                "type 64 bit-packed XYZ ({} / {} / {} bits)",
                                widths[0], widths[1], widths[2]
                            );
                            if widths.contains(&0)
                                || widths
                                    .iter()
                                    .map(|width| usize::from(*width))
                                    .sum::<usize>()
                                    > 64
                            {
                                issue = Some(format!(
                                    "unsupported quantized XYZ bit widths {widths:?}"
                                ));
                            }
                        }
                        other => {
                            position_format = format!("unsupported position type {other}");
                            issue = Some(format!(
                                "unsupported first vertex stream type {other}; supported types are 0 and 64"
                            ));
                        }
                    }
                    let descriptor_count = u32_at(&descriptor, 8, "stream vertex count")? as usize;
                    if descriptor_count != vertex_count {
                        issue = Some(format!(
                            "stream says {descriptor_count} vertices, submesh says {vertex_count}"
                        ));
                    }
                }
                None => {
                    issue = Some("position stream pointer has no pointer fixup".to_owned());
                }
            }
        }
        if vertex_count == 0 {
            issue = Some("vertex count is zero".to_owned());
        }
        if index_count == 0 || !index_count.is_multiple_of(3) {
            issue = Some(format!(
                "index count {index_count} is not a nonzero triangle index count"
            ));
        }
        submeshes.push(SubmeshInventory {
            index,
            lod: Some(lod),
            name,
            vertex_count,
            index_count,
            position_type,
            position_format,
            supported: issue.is_none(),
            issue,
        });
    }
    Ok(submeshes)
}

#[allow(clippy::too_many_arguments)]
fn read_geometry_lod_membership<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    path: &Path,
    geometry_data_offset: u64,
    geometry_header: &[u8],
    submesh_table: u64,
    submesh_count: usize,
) -> Result<Vec<usize>, PakError> {
    let lod_count = u32_at(geometry_header, 60, "LOD group count")? as usize;
    if lod_count == 0 || lod_count > 32 {
        return Err(invalid(
            "LOD group count",
            geometry_data_offset + 12,
            format!("{lod_count} is outside the supported range 1..=32"),
        ));
    }
    let table_raw = i64_at(geometry_header, 96, "LOD group table pointer")?;
    let table = archive
        .resolve_pointer(geometry_data_offset + 48, table_raw)?
        .ok_or_else(|| {
            invalid(
                "LOD group table pointer",
                geometry_data_offset + 48,
                "has no pointer fixup",
            )
        })?;
    let table_bytes = lod_count.checked_mul(8).ok_or_else(|| {
        invalid(
            "LOD group table",
            geometry_data_offset + 48,
            "size overflow",
        )
    })?;
    archive.page_for_range(table, table_bytes)?;
    let mut membership = vec![None; submesh_count];
    for lod in 0..lod_count {
        let pointer_offset = table + (lod * 8) as u64;
        let pointer = read_range(
            source,
            file_size,
            pointer_offset,
            8,
            &path.display().to_string(),
            "read LOD group pointer",
        )?;
        let raw = i64_at(&pointer, 0, "LOD group pointer")?;
        let group = archive
            .resolve_pointer(pointer_offset, raw)?
            .ok_or_else(|| invalid("LOD group pointer", pointer_offset, "has no pointer fixup"))?;
        archive.page_for_range(group, 0x88)?;
        let record = read_range(
            source,
            file_size,
            group,
            0x88,
            &path.display().to_string(),
            "read LOD group",
        )?;
        let count = u32_at(&record, 0x84, "LOD member count")? as usize;
        if count > submesh_count {
            return Err(invalid(
                "LOD member count",
                group + 0x84,
                format!("{count} exceeds {submesh_count} submeshes"),
            ));
        }
        if count == 0 {
            continue;
        }
        let members_raw = i64_at(&record, 0x40, "LOD member list pointer")?;
        let members = archive
            .resolve_pointer(group + 0x40, members_raw)?
            .ok_or_else(|| {
                invalid(
                    "LOD member list pointer",
                    group + 0x40,
                    "has no pointer fixup",
                )
            })?;
        let member_bytes = count
            .checked_mul(16)
            .ok_or_else(|| invalid("LOD member list", group + 0x40, "size overflow"))?;
        archive.page_for_range(members, member_bytes)?;
        for member_index in 0..count {
            let entry_offset = members + (member_index * 16) as u64;
            let entry = read_range(
                source,
                file_size,
                entry_offset,
                16,
                &path.display().to_string(),
                "read LOD member",
            )?;
            let descriptor_raw = i64_at(&entry, 0, "LOD member descriptor pointer")?;
            let descriptor = archive
                .resolve_pointer(entry_offset, descriptor_raw)?
                .ok_or_else(|| {
                    invalid(
                        "LOD member descriptor pointer",
                        entry_offset,
                        "has no pointer fixup",
                    )
                })?;
            let index = u32_at(&entry, 8, "LOD member submesh index")? as usize;
            if index >= submesh_count || descriptor != submesh_table + (index * 192) as u64 {
                return Err(invalid(
                    "LOD member",
                    entry_offset,
                    format!(
                        "index {index} and descriptor pointer {descriptor:#x} do not identify a submesh in the descriptor table"
                    ),
                ));
            }
            if membership[index].replace(lod).is_some() {
                return Err(invalid(
                    "LOD membership",
                    entry_offset,
                    format!("submesh {index} appears in more than one LOD group"),
                ));
            }
        }
    }
    if let Some(index) = membership.iter().position(Option::is_none) {
        return Err(invalid(
            "LOD membership",
            submesh_table + (index * 192) as u64,
            format!("submesh {index} is absent from every LOD group"),
        ));
    }
    Ok(membership.into_iter().map(Option::unwrap).collect())
}

fn decode_geometry_mesh<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    path: &Path,
    selected_index: Option<usize>,
) -> Result<DecodedMesh, PakError> {
    if archive.header.magic != 0x0000_0a79 {
        return Err(invalid(
            "mesh variant",
            0,
            format!(
                "geometry decoding is only verified for PAK magic 0x00000a79, got {:#010x}",
                archive.header.magic
            ),
        ));
    }
    let geometry = archive
        .resources
        .iter()
        .find(|resource| resource.resource_type == "GEOMETRY_1")
        .ok_or_else(|| invalid("geometry resource", 0, "no GEOMETRY_1 resource found"))?;
    let geometry_header = read_range(
        source,
        file_size,
        geometry.file_offset,
        104,
        &path.display().to_string(),
        "read GEOMETRY_1 header",
    )?;
    let geometry_data_offset = geometry.file_offset + 48;
    let submesh_count = u32_at(&geometry_header, 48 + 8, "submesh count")? as usize;
    if submesh_count == 0 || submesh_count > 4096 {
        return Err(invalid(
            "submesh count",
            geometry_data_offset + 8,
            format!("{submesh_count} is outside the supported diagnostic range"),
        ));
    }
    let submesh_table_raw = i64_at(&geometry_header, 48 + 40, "submesh table pointer")?;
    let submesh_table = archive
        .resolve_pointer(geometry_data_offset + 40, submesh_table_raw)?
        .ok_or_else(|| {
            invalid(
                "submesh table pointer",
                geometry_data_offset + 40,
                "has no pointer fixup",
            )
        })?;
    archive.page_for_range(submesh_table, submesh_count * 192)?;
    let lods = read_geometry_lod_membership(
        archive,
        source,
        file_size,
        path,
        geometry_data_offset,
        &geometry_header,
        submesh_table,
        submesh_count,
    )?;
    let mut float_position_mesh = None;
    let mut quantized_position_mesh = None;
    let mut lod0_submesh_count = 0usize;
    let mut observed_position_types = Vec::new();
    for (index, lod) in lods.iter().copied().enumerate() {
        let record_offset = submesh_table + (index * 192) as u64;
        let record = read_range(
            source,
            file_size,
            record_offset,
            192,
            &path.display().to_string(),
            "read geometry submesh record",
        )?;
        let name_raw = i64_at(&record, 32, "submesh name pointer")?;
        let name_address = archive
            .resolve_pointer(record_offset + 32, name_raw)?
            .ok_or_else(|| {
                invalid(
                    "submesh name pointer",
                    record_offset + 32,
                    "has no pointer fixup",
                )
            })?;
        let name = read_pointer_string(archive, source, file_size, name_address, path)?;
        if selected_index.is_some_and(|selected| selected != index) {
            continue;
        }
        if selected_index.is_none() && lod != 0 {
            continue;
        }
        lod0_submesh_count += 1;
        let stream_count = u32_at(&record, 144, "stream count")? as usize;
        if stream_count == 0 || stream_count > 256 {
            continue;
        }
        let stream_raw = i64_at(&record, 48, "stream descriptor pointer")?;
        let Some(stream_address) = archive.resolve_pointer(record_offset + 48, stream_raw)? else {
            continue;
        };
        archive.page_for_range(stream_address, 64)?;
        let descriptor = read_range(
            source,
            file_size,
            stream_address,
            64,
            &path.display().to_string(),
            "read candidate position stream descriptor",
        )?;
        let position_type = descriptor[20];
        observed_position_types.push(position_type);
        let candidate = (index, record_offset, record, name);
        match position_type {
            0 if float_position_mesh.is_none() => float_position_mesh = Some(candidate),
            64 if quantized_position_mesh.is_none() => quantized_position_mesh = Some(candidate),
            _ if selected_index.is_some() => {
                float_position_mesh = Some(candidate);
                break;
            }
            _ => {}
        }
        if selected_index.is_some() {
            break;
        }
    }
    let (submesh_index, submesh_offset, submesh, submesh_name) = float_position_mesh
        .or(quantized_position_mesh)
        .ok_or_else(|| {
            invalid(
                "supported LOD0 mesh",
                submesh_table,
                format!(
                    "no LOD0 submesh has a position stream of type 0 (float3) or 64 (quantized XYZ); found {lod0_submesh_count} LOD0 names, {} descriptor types {:?}, among {submesh_count} submeshes",
                    observed_position_types.len(),
                    observed_position_types
                ),
            )
        })?;
    let vertex_count = u32_at(&submesh, 136, "vertex count")? as usize;
    let index_count = u32_at(&submesh, 140, "index count")? as usize;
    let stream_count = u32_at(&submesh, 144, "stream count")? as usize;
    if vertex_count == 0 || vertex_count > 10_000_000 {
        return Err(invalid(
            "vertex count",
            submesh_offset + 136,
            format!("{vertex_count} is outside the supported range"),
        ));
    }
    if index_count == 0 || !index_count.is_multiple_of(3) || index_count > 60_000_000 {
        return Err(invalid(
            "index count",
            submesh_offset + 140,
            format!("{index_count} is not a supported triangle index count"),
        ));
    }
    if stream_count == 0 || stream_count > 256 {
        return Err(invalid(
            "stream count",
            submesh_offset + 144,
            format!("{stream_count} is outside the supported range"),
        ));
    }
    let stream_raw = i64_at(&submesh, 48, "stream descriptor pointer")?;
    let stream_address = archive
        .resolve_pointer(submesh_offset + 48, stream_raw)?
        .ok_or_else(|| {
            invalid(
                "stream descriptor pointer",
                submesh_offset + 48,
                "has no pointer fixup",
            )
        })?;
    let stream_bytes = stream_count
        .checked_mul(64)
        .ok_or_else(|| invalid("stream descriptor array", stream_address, "size overflow"))?;
    archive.page_for_range(stream_address, stream_bytes)?;
    let descriptor = read_range(
        source,
        file_size,
        stream_address,
        64,
        &path.display().to_string(),
        "read first vertex stream descriptor",
    )?;
    let vertex_type = descriptor[20];
    let descriptor_vertex_count = u32_at(&descriptor, 8, "stream vertex count")? as usize;
    let vertex_buffer_size = u32_at(&descriptor, 16, "vertex buffer size")? as usize;
    let packed_stride = descriptor[22];
    let vertex_stride = usize::from(packed_stride >> 4);
    if vertex_type != 0 && vertex_type != 64 {
        return Err(invalid(
            "vertex format",
            stream_address + 20,
            format!("unsupported first stream type {vertex_type}; supported types are 0 and 64"),
        ));
    }
    if vertex_type == 0 && vertex_stride != 12 {
        return Err(invalid(
            "vertex stride",
            stream_address + 22,
            format!("unsupported position stride {vertex_stride}; expected 12"),
        ));
    }
    if descriptor_vertex_count != vertex_count {
        return Err(invalid(
            "vertex count",
            stream_address + 8,
            format!("stream says {descriptor_vertex_count}, submesh says {vertex_count}"),
        ));
    }
    if vertex_type == 0 {
        let expected_vertex_size = vertex_count
            .checked_mul(12)
            .ok_or_else(|| invalid("vertex buffer size", stream_address + 16, "size overflow"))?;
        if vertex_buffer_size != expected_vertex_size {
            return Err(invalid(
                "vertex buffer size",
                stream_address + 16,
                format!(
                    "descriptor says {vertex_buffer_size} bytes, {vertex_count} float3 vertices require {expected_vertex_size}"
                ),
            ));
        }
    }
    let vertex_pointer_raw = i64_at(&descriptor, 0, "vertex buffer pointer")?;
    let vertex_buffer_offset = archive
        .resolve_pointer(stream_address, vertex_pointer_raw)?
        .ok_or_else(|| {
            invalid(
                "vertex buffer pointer",
                stream_address,
                "has no pointer fixup",
            )
        })?;
    let vertex_page = archive.page_for_range(vertex_buffer_offset, vertex_buffer_size)?;
    let vertex_buffer = read_range(
        source,
        file_size,
        vertex_buffer_offset,
        vertex_buffer_size,
        &path.display().to_string(),
        "read vertex buffer",
    )?;
    let (vertices, bounds, vertex_format) = match vertex_type {
        0 => {
            let (vertices, bounds) =
                decode_float3_positions(&vertex_buffer, vertex_count, vertex_buffer_offset)?;
            (vertices, bounds, "little-endian float3 XYZ".to_owned())
        }
        64 => {
            let sizes = [
                descriptor[24],
                descriptor[25],
                descriptor[26],
                descriptor[27],
            ];
            let scale = [
                f32_at(&descriptor, 32, "quantized X scale")?,
                f32_at(&descriptor, 36, "quantized Y scale")?,
                f32_at(&descriptor, 40, "quantized Z scale")?,
                f32_at(&descriptor, 44, "quantized W scale")?,
            ];
            let offset = [
                f32_at(&descriptor, 48, "quantized X offset")?,
                f32_at(&descriptor, 52, "quantized Y offset")?,
                f32_at(&descriptor, 56, "quantized Z offset")?,
                f32_at(&descriptor, 60, "quantized W offset")?,
            ];
            let (vertices, bounds) = decode_quantized_positions(
                &vertex_buffer,
                vertex_count,
                sizes,
                scale,
                offset,
                vertex_buffer_offset,
            )?;
            validate_quantized_bounds(&submesh, &bounds, scale, submesh_offset)?;
            (
                vertices,
                bounds,
                format!(
                    "type 64 bit-packed XYZ ({}, {}, {} bits)",
                    sizes[0], sizes[1], sizes[2]
                ),
            )
        }
        _ => unreachable!("position type was validated above"),
    };

    let index_raw = i64_at(&submesh, 64, "index buffer pointer")?;
    let index_buffer_offset = archive
        .resolve_pointer(submesh_offset + 64, index_raw)?
        .ok_or_else(|| {
            invalid(
                "index buffer pointer",
                submesh_offset + 64,
                "has no pointer fixup",
            )
        })?;
    let index_buffer_size = index_count
        .checked_mul(2)
        .ok_or_else(|| invalid("index buffer size", index_buffer_offset, "size overflow"))?;
    let index_page = archive.page_for_range(index_buffer_offset, index_buffer_size)?;
    let index_buffer = read_range(
        source,
        file_size,
        index_buffer_offset,
        index_buffer_size,
        &path.display().to_string(),
        "read 16-bit index buffer",
    )?;
    let indices = decode_u16_indices(
        &index_buffer,
        index_count,
        vertex_count,
        index_buffer_offset,
    )?;
    let material_raw = i64_at(&submesh, 72, "material pointer")?;
    let material_offset = archive
        .resolve_pointer(submesh_offset + 72, material_raw)?
        .ok_or_else(|| {
            invalid(
                "material pointer",
                submesh_offset + 72,
                "has no pointer fixup",
            )
        })?;
    let textures = read_material_textures(archive, source, file_size, material_offset, path)?;
    let mut evidence = vec![
        "fmt_nd_pak.py identifies TLOU2 GEOMETRY_1 records as 48-byte resource-prefix followed by geometry fields; submesh records are 192 bytes.".to_owned(),
        "fmt_nd_pak.py reads TLOU2 stream descriptors at 64-byte intervals; type 0 with stride 12 is float3, and type 64 decodes bit widths from sizes[] then applies qScale * value + qOffs.".to_owned(),
        "fmt_nd_pak.py reads each face buffer as 2 * numIndices bytes and commits it as RPGEODATA_USHORT triangle indices.".to_owned(),
        "Observed descriptor vertex count equals submesh count and buffer size equals count * 12; observed index buffer length equals count * 2.".to_owned(),
        "Both referenced buffers resolve through explicit PAK pointer-fixup records and fit wholly inside stored page ranges; page class is reported but not interpreted as a codec.".to_owned(),
        "Material texture descriptors are traced using the reference plugin's TLOU2 48-byte records and hash field; hashes are matched to VRAM_DESC hash fields at resource offset + 16 + 56.".to_owned(),
        "Coordinates are raw little-endian XYZ f32 values; no additional coordinate transform is applied by this diagnostic.".to_owned(),
    ];
    if vertex_type == 64 {
        evidence.push(
            "Quantized XYZ extrema fit the submesh's stored bounding box within one quantization step per axis; packed bytes are little-endian bit order, corroborated against that box."
                .to_owned(),
        );
    }
    let report = MeshReport {
        pak_path: path.to_owned(),
        geometry_resource_offset: geometry.file_offset,
        submesh_table_offset: submesh_table,
        submesh_count,
        submesh_table_size: submesh_count * 192,
        submesh_index,
        submesh_name,
        submesh_offset,
        material_offset,
        vertex_count,
        vertex_format,
        index_count,
        triangle_count: index_count / 3,
        vertex_buffer_offset,
        vertex_buffer_size,
        vertex_page_index: vertex_page.index,
        vertex_page_class: vertex_page.page_class,
        index_buffer_offset,
        index_buffer_size,
        index_page_index: index_page.index,
        index_page_class: index_page.page_class,
        bounds,
        textures,
        evidence,
    };
    Ok(DecodedMesh {
        vertices,
        indices,
        report,
    })
}

fn decode_float3_positions(
    bytes: &[u8],
    vertex_count: usize,
    file_offset: u64,
) -> Result<(Vec<[f32; 3]>, MeshBounds), PakError> {
    let expected_size = vertex_count
        .checked_mul(12)
        .ok_or_else(|| invalid("vertex buffer size", file_offset, "size overflow"))?;
    if bytes.len() != expected_size {
        return Err(invalid(
            "vertex buffer size",
            file_offset,
            format!(
                "got {} bytes for {vertex_count} float3 vertices; expected {expected_size}",
                bytes.len()
            ),
        ));
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut bounds = MeshBounds {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };
    for (index, vertex_bytes) in bytes.as_chunks::<12>().0.iter().enumerate() {
        let vertex = [
            f32_at(vertex_bytes, 0, "vertex X")?,
            f32_at(vertex_bytes, 4, "vertex Y")?,
            f32_at(vertex_bytes, 8, "vertex Z")?,
        ];
        for (axis, value) in vertex.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(
                    "vertex coordinate",
                    file_offset + (index * 12) as u64 + (axis * 4) as u64,
                    format!("non-finite coordinate {value}"),
                ));
            }
            bounds.min[axis] = bounds.min[axis].min(*value);
            bounds.max[axis] = bounds.max[axis].max(*value);
        }
        vertices.push(vertex);
    }
    Ok((vertices, bounds))
}

fn decode_quantized_positions(
    bytes: &[u8],
    vertex_count: usize,
    bit_widths: [u8; 4],
    scales: [f32; 4],
    offsets: [f32; 4],
    file_offset: u64,
) -> Result<(Vec<[f32; 3]>, MeshBounds), PakError> {
    if bit_widths[0..3].contains(&0) || bit_widths[3] != 0 {
        return Err(invalid(
            "quantized position widths",
            file_offset,
            format!("expected non-zero XYZ widths and zero W width, got {bit_widths:?}"),
        ));
    }
    if bit_widths.iter().any(|width| *width > 32) {
        return Err(invalid(
            "quantized position widths",
            file_offset,
            format!("bit width exceeds 32: {bit_widths:?}"),
        ));
    }
    for axis in 0..3 {
        if !scales[axis].is_finite() || !offsets[axis].is_finite() {
            return Err(invalid(
                "quantized position transform",
                file_offset,
                format!(
                    "axis {axis} has non-finite scale or offset ({}, {})",
                    scales[axis], offsets[axis]
                ),
            ));
        }
    }
    let bits_per_vertex = bit_widths[0..3]
        .iter()
        .try_fold(0usize, |sum, width| sum.checked_add(usize::from(*width)))
        .ok_or_else(|| invalid("quantized position bit count", file_offset, "overflow"))?;
    let total_bits = vertex_count
        .checked_mul(bits_per_vertex)
        .ok_or_else(|| invalid("quantized position bit count", file_offset, "overflow"))?;
    let expected_size = total_bits
        .checked_add(7)
        .ok_or_else(|| invalid("quantized position buffer size", file_offset, "overflow"))?
        / 8;
    if bytes.len() != expected_size {
        return Err(invalid(
            "quantized position buffer size",
            file_offset,
            format!(
                "got {} bytes; {vertex_count} vertices at {bits_per_vertex} bits require {expected_size}",
                bytes.len()
            ),
        ));
    }

    let mut bit_offset = 0usize;
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut bounds = MeshBounds {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };
    for vertex_index in 0..vertex_count {
        let mut vertex = [0.0; 3];
        for axis in 0..3 {
            let width = usize::from(bit_widths[axis]);
            let mut quantized = 0u32;
            for bit_index in 0..width {
                let source_bit = bit_offset + bit_index;
                let bit = (bytes[source_bit / 8] >> (source_bit % 8)) & 1;
                quantized |= u32::from(bit) << bit_index;
            }
            bit_offset += width;
            vertex[axis] = quantized as f32 * scales[axis] + offsets[axis];
            if !vertex[axis].is_finite() {
                return Err(invalid(
                    "decoded vertex coordinate",
                    file_offset + (bit_offset / 8) as u64,
                    format!("vertex {vertex_index} axis {axis} is non-finite"),
                ));
            }
            bounds.min[axis] = bounds.min[axis].min(vertex[axis]);
            bounds.max[axis] = bounds.max[axis].max(vertex[axis]);
        }
        vertices.push(vertex);
    }
    if bit_offset != total_bits {
        return Err(invalid(
            "quantized position bit count",
            file_offset,
            format!("consumed {bit_offset} bits; expected {total_bits}"),
        ));
    }
    Ok((vertices, bounds))
}

fn validate_quantized_bounds(
    submesh: &[u8],
    decoded: &MeshBounds,
    scales: [f32; 4],
    submesh_offset: u64,
) -> Result<(), PakError> {
    for (axis, scale) in scales.iter().enumerate().take(3) {
        let expected_min = f32_at(submesh, axis * 4, "submesh bounds minimum")?;
        let expected_max = f32_at(submesh, 16 + axis * 4, "submesh bounds maximum")?;
        let tolerance = scale.abs() + 1e-5;
        if !expected_min.is_finite()
            || !expected_max.is_finite()
            || decoded.min[axis] < expected_min - tolerance
            || decoded.max[axis] > expected_max + tolerance
            || decoded.min[axis] > expected_min + tolerance
            || decoded.max[axis] < expected_max - tolerance
        {
            return Err(invalid(
                "quantized submesh bounds",
                submesh_offset + (axis * 4) as u64,
                format!(
                    "decoded axis {} range [{}, {}] disagrees with stored [{expected_min}, {expected_max}] by more than tolerance {tolerance}",
                    axis, decoded.min[axis], decoded.max[axis]
                ),
            ));
        }
    }
    Ok(())
}

fn decode_u16_indices(
    bytes: &[u8],
    index_count: usize,
    vertex_count: usize,
    file_offset: u64,
) -> Result<Vec<u16>, PakError> {
    let expected_size = index_count
        .checked_mul(2)
        .ok_or_else(|| invalid("index buffer size", file_offset, "size overflow"))?;
    if bytes.len() != expected_size {
        return Err(invalid(
            "index buffer size",
            file_offset,
            format!(
                "got {} bytes for {index_count} 16-bit indices; expected {expected_size}",
                bytes.len()
            ),
        ));
    }
    let mut indices = Vec::with_capacity(index_count);
    for (index, item) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let decoded = u16::from_le_bytes([item[0], item[1]]);
        if usize::from(decoded) >= vertex_count {
            return Err(invalid(
                "mesh index",
                file_offset + (index * 2) as u64,
                format!("{decoded} exceeds vertex count {vertex_count}"),
            ));
        }
        indices.push(decoded);
    }
    Ok(indices)
}

fn validate_obj_mesh(vertices: &[[f32; 3]], indices: &[u16]) -> Result<(), PakError> {
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(invalid(
            "OBJ triangles",
            0,
            format!(
                "index count {} must be a non-zero multiple of three",
                indices.len()
            ),
        ));
    }
    for (index, vertex) in vertices.iter().enumerate() {
        for (axis, coordinate) in vertex.iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(invalid(
                    "OBJ vertex coordinate",
                    (index * 12 + axis * 4) as u64,
                    format!("non-finite coordinate {coordinate}"),
                ));
            }
        }
    }
    for (index, vertex_index) in indices.iter().enumerate() {
        if usize::from(*vertex_index) >= vertices.len() {
            return Err(invalid(
                "OBJ face index",
                (index * 2) as u64,
                format!(
                    "zero-based index {vertex_index} exceeds {} available vertices",
                    vertices.len()
                ),
            ));
        }
    }
    Ok(())
}

fn write_obj<W: Write>(output: &mut W, vertices: &[[f32; 3]], indices: &[u16]) -> io::Result<()> {
    for vertex in vertices {
        writeln!(output, "v {} {} {}", vertex[0], vertex[1], vertex[2])?;
    }
    for triangle in indices.as_chunks::<3>().0 {
        writeln!(
            output,
            "f {} {} {}",
            usize::from(triangle[0]) + 1,
            usize::from(triangle[1]) + 1,
            usize::from(triangle[2]) + 1
        )?;
    }
    Ok(())
}

fn export_obj_meshes(meshes: &[DecodedMesh], output: impl AsRef<Path>) -> Result<(), PakError> {
    let output = output.as_ref();
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = output
        .file_name()
        .ok_or_else(|| invalid("OBJ output path", 0, "must include a file name"))?;
    let mut temp_path = None;
    let mut file = None;
    for _ in 0..100 {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            sequence
        ));
        match File::options()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(created) => {
                temp_path = Some(candidate);
                file = Some(created);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(PakError::Io {
                    path: candidate,
                    operation: "create temporary OBJ",
                    source,
                });
            }
        }
    }
    let temp_path = temp_path
        .ok_or_else(|| invalid("OBJ output path", 0, "could not allocate a temporary file"))?;
    let mut file =
        file.ok_or_else(|| invalid("OBJ output file", 0, "temporary file was not created"))?;
    let result = write_obj_meshes(&mut file, meshes)
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::rename(&temp_path, output));
    if let Err(source) = result {
        let _ = fs::remove_file(&temp_path);
        return Err(PakError::Io {
            path: output.to_owned(),
            operation: "write validated OBJ",
            source,
        });
    }
    Ok(())
}

fn write_obj_meshes<W: Write>(output: &mut W, meshes: &[DecodedMesh]) -> io::Result<()> {
    let mut vertex_offset = 0usize;
    for mesh in meshes {
        let object_name: String = mesh
            .report
            .submesh_name
            .chars()
            .map(|character| {
                if character.is_whitespace() || character == '|' {
                    '_'
                } else {
                    character
                }
            })
            .collect();
        writeln!(output, "o {object_name}")?;
        for vertex in &mesh.vertices {
            writeln!(output, "v {} {} {}", vertex[0], vertex[1], vertex[2])?;
        }
        for triangle in mesh.indices.as_chunks::<3>().0 {
            writeln!(
                output,
                "f {} {} {}",
                vertex_offset + usize::from(triangle[0]) + 1,
                vertex_offset + usize::from(triangle[1]) + 1,
                vertex_offset + usize::from(triangle[2]) + 1
            )?;
        }
        vertex_offset = vertex_offset
            .checked_add(mesh.vertices.len())
            .ok_or_else(|| io::Error::other("OBJ cumulative vertex count overflow"))?;
    }
    Ok(())
}

fn read_material_textures<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    material_offset: u64,
    path: &Path,
) -> Result<Vec<MeshTextureReference>, PakError> {
    let header = read_range(
        source,
        file_size,
        material_offset,
        288,
        &path.display().to_string(),
        "read material texture references",
    )?;
    let texture_count = u32_at(&header, 276, "material texture count")? as usize;
    if texture_count > 4096 {
        return Err(invalid(
            "material texture count",
            material_offset + 276,
            format!("{texture_count} exceeds safety limit"),
        ));
    }
    let list_raw = i64_at(&header, 32, "material texture descriptor pointer")?;
    let list_offset = archive
        .resolve_pointer(material_offset + 32, list_raw)?
        .ok_or_else(|| {
            invalid(
                "material texture descriptor pointer",
                material_offset + 32,
                "has no pointer fixup",
            )
        })?;
    let descriptor_bytes = texture_count.checked_mul(48).ok_or_else(|| {
        invalid(
            "material texture descriptor array",
            list_offset,
            "size overflow",
        )
    })?;
    archive.page_for_range(list_offset, descriptor_bytes)?;
    let vram_by_hash = read_vram_descriptors(archive, source, file_size, path)?;
    let mut textures = Vec::with_capacity(texture_count);
    for index in 0..texture_count {
        let descriptor_offset = list_offset + (index * 48) as u64;
        let descriptor = read_range(
            source,
            file_size,
            descriptor_offset,
            48,
            &path.display().to_string(),
            "read material texture descriptor",
        )?;
        let name_raw = i64_at(&descriptor, 0, "material texture name pointer")?;
        let name_offset = archive
            .resolve_pointer(descriptor_offset, name_raw)?
            .ok_or_else(|| {
                invalid(
                    "material texture name pointer",
                    descriptor_offset,
                    "has no pointer fixup",
                )
            })?;
        let path_struct_raw = i64_at(&descriptor, 24, "material texture path pointer")?;
        let path_struct = archive
            .resolve_pointer(descriptor_offset + 24, path_struct_raw)?
            .ok_or_else(|| {
                invalid(
                    "material texture path pointer",
                    descriptor_offset + 24,
                    "has no pointer fixup",
                )
            })?;
        let path_struct_bytes = read_range(
            source,
            file_size,
            path_struct,
            16,
            &path.display().to_string(),
            "read material texture path record",
        )?;
        let path_raw = i64_at(
            &path_struct_bytes,
            0,
            "material texture path string pointer",
        )?;
        let material_path_offset =
            archive
                .resolve_pointer(path_struct, path_raw)?
                .ok_or_else(|| {
                    invalid(
                        "material texture path string pointer",
                        path_struct,
                        "has no pointer fixup",
                    )
                })?;
        let hash = u64_at(&path_struct_bytes, 8, "material texture hash")?;
        let name = read_pointer_string(archive, source, file_size, name_offset, path)?;
        let material_path =
            read_pointer_string(archive, source, file_size, material_path_offset, path)?;
        let matching_vram = vram_by_hash.get(&hash);
        textures.push(MeshTextureReference {
            name,
            hash,
            path: material_path,
            vram_descriptor_offset: matching_vram.map(|entry| entry.0),
            vram_descriptor_path: matching_vram.map(|entry| entry.1.clone()),
        });
    }
    Ok(textures)
}

fn read_vram_descriptors<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    path: &Path,
) -> Result<std::collections::BTreeMap<u64, (u64, String)>, PakError> {
    let mut entries = std::collections::BTreeMap::new();
    for resource in archive
        .resources
        .iter()
        .filter(|resource| resource.resource_type == "VRAM_DESC")
    {
        let content_offset = resource.file_offset + 16;
        archive.page_for_range(content_offset, 120)?;
        let hash_bytes = read_range(
            source,
            file_size,
            content_offset + 56,
            8,
            &path.display().to_string(),
            "read VRAM descriptor hash",
        )?;
        let hash = u64_at(&hash_bytes, 0, "VRAM descriptor hash")?;
        let texture_path =
            read_pointer_string(archive, source, file_size, content_offset + 112, path)?;
        entries.insert(hash, (resource.file_offset, texture_path));
    }
    Ok(entries)
}

fn read_pointer_string<R: Read + Seek>(
    archive: &PakArchive,
    source: &mut R,
    file_size: u64,
    offset: u64,
    path: &Path,
) -> Result<String, PakError> {
    let page = archive.page_for_range(offset, 1)?;
    let available = page.offset + u64::from(page.stored_size) - offset;
    let size = usize::try_from(available.min(4096))
        .map_err(|_| invalid("string range", offset, "does not fit address space"))?;
    let bytes = read_range(
        source,
        file_size,
        offset,
        size,
        &path.display().to_string(),
        "read referenced string",
    )?;
    let end = bytes.iter().position(|byte| *byte == 0).ok_or_else(|| {
        invalid(
            "referenced string",
            offset,
            "no NUL before page end or 4096-byte limit",
        )
    })?;
    Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

fn u32_at(bytes: &[u8], offset: usize, field: &'static str) -> Result<u32, PakError> {
    let value = checked_slice(bytes, offset, 4, field)?;
    Ok(u32::from_le_bytes(value.try_into().map_err(|_| {
        invalid(field, offset as u64, "could not read four bytes")
    })?))
}

fn u64_at(bytes: &[u8], offset: usize, field: &'static str) -> Result<u64, PakError> {
    let value = checked_slice(bytes, offset, 8, field)?;
    Ok(u64::from_le_bytes(value.try_into().map_err(|_| {
        invalid(field, offset as u64, "could not read eight bytes")
    })?))
}

fn i64_at(bytes: &[u8], offset: usize, field: &'static str) -> Result<i64, PakError> {
    let value = checked_slice(bytes, offset, 8, field)?;
    Ok(i64::from_le_bytes(value.try_into().map_err(|_| {
        invalid(field, offset as u64, "could not read eight bytes")
    })?))
}

fn f32_at(bytes: &[u8], offset: usize, field: &'static str) -> Result<f32, PakError> {
    let value = checked_slice(bytes, offset, 4, field)?;
    Ok(f32::from_le_bytes(value.try_into().map_err(|_| {
        invalid(field, offset as u64, "could not read four bytes")
    })?))
}

fn parse_fixups<R: Read + Seek>(
    source: &mut R,
    file_size: u64,
    header: &PakHeader,
    source_name: &str,
) -> Result<(u32, Vec<PakFixup>), PakError> {
    let table = u64::from(header.fixup_table_offset);
    let table_header = read_range(
        source,
        file_size,
        table,
        12,
        source_name,
        "read PAK fixup table header",
    )?;
    let mut reader = BinaryReader::new(&table_header);
    let _page_entry_number = reader.read_u32()?;
    let records_offset = u64::from(reader.read_u32()?);
    let count = reader.read_u32()?;
    let record_bytes = (count as usize)
        .checked_mul(FIXUP_ENTRY_SIZE)
        .ok_or_else(|| invalid("fixup table", records_offset, "size overflow"))?;
    let records = read_range(
        source,
        file_size,
        records_offset,
        record_bytes,
        source_name,
        "read PAK pointer fixups",
    )?;
    let mut parsed = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let mut record =
            BinaryReader::new(&records[index * FIXUP_ENTRY_SIZE..(index + 1) * FIXUP_ENTRY_SIZE]);
        let source_page = record.read_u16()? as usize;
        let target_page = record.read_u16()? as usize;
        let pointer_offset = record.read_u32()? as usize;
        if source_page >= header.page_count as usize || target_page >= header.page_count as usize {
            return Err(invalid(
                "pointer fixup page index",
                records_offset + (index * FIXUP_ENTRY_SIZE) as u64,
                format!("source {source_page}, target {target_page}"),
            ));
        }
        parsed.push(PakFixup {
            source_page,
            target_page,
            pointer_offset,
        });
    }
    Ok((count, parsed))
}

fn validate_fixups(fixups: &[PakFixup], pages: &[PakPage]) -> Result<(), PakError> {
    for fixup in fixups {
        let source = &pages[fixup.source_page];
        let pointer_end = fixup
            .pointer_offset
            .checked_add(8)
            .ok_or_else(|| invalid("pointer fixup", source.offset, "pointer range overflow"))?;
        if pointer_end > source.stored_size as usize {
            return Err(invalid(
                "pointer fixup",
                source.offset + fixup.pointer_offset as u64,
                "pointer field extends beyond its source page",
            ));
        }
        if fixup.target_page >= pages.len() {
            return Err(invalid(
                "pointer fixup target",
                source.offset + fixup.pointer_offset as u64,
                "target page is outside the page table",
            ));
        }
    }
    Ok(())
}

fn parse_resources(page_bytes: &[u8], page: &PakPage) -> Result<Vec<PakResource>, PakError> {
    let mut resources = Vec::with_capacity(page.resource_count as usize);
    for index in 0..page.resource_count as usize {
        let entry_offset = PAGE_HEADER_SIZE + index * RESOURCE_ENTRY_SIZE;
        let mut entry =
            BinaryReader::new(&page_bytes[entry_offset..entry_offset + RESOURCE_ENTRY_SIZE]);
        let directory_name_offset = usize::try_from(entry.read_u64()?)
            .map_err(|_| invalid("resource directory name", page.offset, "offset overflow"))?;
        let resource_offset = entry.read_u32()?;
        let _unknown = entry.read_u32()?;
        let resource_at = usize::try_from(resource_offset)
            .map_err(|_| invalid("resource offset", page.offset, "offset overflow"))?;
        let item = checked_slice(page_bytes, resource_at, 16, "resource record")?;
        let mut item_reader = BinaryReader::new(item);
        let name_offset = usize::try_from(item_reader.read_u64()?)
            .map_err(|_| invalid("resource name", page.offset, "offset overflow"))?;
        let type_offset = usize::try_from(item_reader.read_u64()?)
            .map_err(|_| invalid("resource type", page.offset, "offset overflow"))?;
        resources.push(PakResource {
            page_index: page.index,
            index,
            directory_name: read_page_string(page_bytes, directory_name_offset, page.offset)?,
            name: read_page_string(page_bytes, name_offset, page.offset)?,
            resource_type: read_page_string(page_bytes, type_offset, page.offset)?,
            offset_in_page: resource_offset,
            file_offset: page.offset + u64::from(resource_offset),
        });
    }
    Ok(resources)
}

fn read_page_string(data: &[u8], offset: usize, page_offset: u64) -> Result<String, PakError> {
    let tail = data.get(offset..).ok_or_else(|| {
        invalid(
            "resource string offset",
            page_offset + offset as u64,
            "outside page",
        )
    })?;
    let end = tail.iter().position(|byte| *byte == 0).ok_or_else(|| {
        invalid(
            "resource string",
            page_offset + offset as u64,
            "missing NUL terminator before page end",
        )
    })?;
    Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
}

fn validate_non_overlapping_pages(pages: &[PakPage]) -> Result<(), PakError> {
    let mut ranges = pages
        .iter()
        .map(|page| {
            let end = page
                .offset
                .checked_add(u64::from(page.stored_size))
                .ok_or_else(|| invalid("page range", page.offset, "address overflow"))?;
            Ok((page.offset, end, page.index))
        })
        .collect::<Result<Vec<_>, PakError>>()?;
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(invalid(
                "page range",
                pair[1].0,
                format!("page {} overlaps page {}", pair[0].2, pair[1].2),
            ));
        }
    }
    Ok(())
}

fn checked_page_range(
    page_size: u32,
    offset: usize,
    size: usize,
    field: &'static str,
    page_file_offset: u64,
) -> Result<(), PakError> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| invalid(field, page_file_offset + offset as u64, "range overflow"))?;
    if end > page_size as usize {
        return Err(invalid(
            field,
            page_file_offset + offset as u64,
            format!("{size} bytes exceed {}-byte page", page_size),
        ));
    }
    Ok(())
}

fn checked_slice<'a>(
    data: &'a [u8],
    offset: usize,
    size: usize,
    field: &'static str,
) -> Result<&'a [u8], PakError> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| invalid(field, offset as u64, "range overflow"))?;
    data.get(offset..end).ok_or_else(|| {
        invalid(
            field,
            offset as u64,
            format!("{size} bytes exceed {}-byte buffer", data.len()),
        )
    })
}

fn read_exact_at<R: Read + Seek>(
    source: &mut R,
    file_size: u64,
    offset: u64,
    buffer: &mut [u8],
    source_name: &str,
    operation: &'static str,
) -> Result<(), PakError> {
    let end = offset
        .checked_add(buffer.len() as u64)
        .ok_or_else(|| invalid("file range", offset, "range overflow"))?;
    if end > file_size {
        return Err(invalid(
            "file range",
            offset,
            format!("{} bytes exceed {file_size}-byte file", buffer.len()),
        ));
    }
    source
        .seek(SeekFrom::Start(offset))
        .and_then(|_| source.read_exact(buffer))
        .map_err(|source| PakError::Io {
            path: PathBuf::from(source_name),
            operation,
            source,
        })
}

fn read_range<R: Read + Seek>(
    source: &mut R,
    file_size: u64,
    offset: u64,
    size: usize,
    source_name: &str,
    operation: &'static str,
) -> Result<Vec<u8>, PakError> {
    let mut bytes = vec![0; size];
    read_exact_at(
        source,
        file_size,
        offset,
        &mut bytes,
        source_name,
        operation,
    )?;
    Ok(bytes)
}

fn invalid(field: &'static str, offset: u64, detail: impl Into<String>) -> PakError {
    PakError::Invalid {
        field,
        offset,
        detail: detail.into(),
    }
}

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture_descriptor_for_layout(
        format_id: u32,
        width: u32,
        height: u32,
        mip_count: u32,
    ) -> TextureDescriptor {
        let (format_name, block_bytes) = texture_format(format_id);
        TextureDescriptor {
            resource_index: 0,
            resource_page: 0,
            resource_page_class: 0,
            resource_offset: 0,
            resource_metadata_words: [0; 10],
            hash: 0,
            data_offset: 0,
            unknown0: 0,
            data_size: 0,
            texture_dict_id: 0,
            unknown1: 0,
            texture_type: 1,
            format_id,
            field_2c: 0,
            format_name,
            mip_count,
            width,
            height,
            field_3c: 0,
            stream_flags: 0,
            texture_path_pointer: None,
            texture_path: None,
            expected_mip_bytes: expected_texture_bytes(
                format_id,
                width,
                height,
                mip_count,
                block_bytes,
            ),
            expected_rgba_bytes: expected_rgba_mip_bytes(width, height, mip_count),
        }
    }

    #[test]
    fn sha256_helper_matches_standard_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(TextureByteStats::new(&[]).entropy, 0.0);
    }

    #[test]
    fn copy_footprint_candidate_covers_2d_cubemap_and_volume_arithmetic() {
        let bc7 = texture_descriptor_for_layout(0x62, 64, 64, 7);
        assert_eq!(
            d3d12_copyable_footprint_candidate(&bc7, None)
                .map(|footprint| footprint.allocation_bytes),
            Some(9_216)
        );

        let mut cube = texture_descriptor_for_layout(0x5f, 256, 256, 7);
        cube.texture_type = 2;
        cube.field_2c = 0x0600_0001;
        assert_eq!(
            d3d12_copyable_footprint_candidate(&cube, None)
                .map(|footprint| footprint.allocation_bytes),
            Some(540_672)
        );

        let mut volume = texture_descriptor_for_layout(0x50, 128, 128, 8);
        volume.texture_type = 3;
        volume.field_2c = 0x8000_0001;
        volume.field_3c = 128;
        assert_eq!(
            d3d12_copyable_footprint_candidate(&volume, None)
                .map(|footprint| footprint.allocation_bytes),
            Some(1_398_784)
        );
        volume.field_3c = 64;
        assert!(d3d12_copyable_footprint_candidate(&volume, None).is_none());
    }

    #[test]
    fn pitch_models_validate_arithmetic_without_asserting_game_layout() {
        let bc7 = texture_descriptor_for_layout(0x62, 64, 64, 7);
        assert_eq!(bc7.expected_mip_bytes, Some(5_488));
        assert_eq!(pitched_texture_mip_bytes(&bc7, 256, None), Some(8_448));
        assert_eq!(pitched_texture_mip_bytes(&bc7, 1, Some(16)), Some(8_448));

        let bc1 = texture_descriptor_for_layout(0x47, 64, 64, 7);
        assert_eq!(pitched_texture_mip_bytes(&bc1, 256, None), Some(8_448));
        assert_eq!(pitched_texture_mip_bytes(&bc1, 1, Some(32)), Some(8_448));

        let rgba = texture_descriptor_for_layout(0x1c, 64, 64, 1);
        assert_eq!(pitched_texture_mip_bytes(&rgba, 256, None), Some(16_384));
        assert_eq!(aligned_mip_chain_bytes(&rgba, 256), Some(16_384));
    }

    fn test_mesh(name: &str, vertices: Vec<[f32; 3]>, indices: Vec<u16>) -> DecodedMesh {
        DecodedMesh {
            vertices,
            indices,
            report: MeshReport {
                pak_path: PathBuf::new(),
                geometry_resource_offset: 0,
                submesh_table_offset: 0,
                submesh_count: 0,
                submesh_table_size: 0,
                submesh_index: 0,
                submesh_name: name.to_owned(),
                submesh_offset: 0,
                material_offset: 0,
                vertex_count: 0,
                vertex_format: String::new(),
                index_count: 0,
                triangle_count: 0,
                vertex_buffer_offset: 0,
                vertex_buffer_size: 0,
                vertex_page_index: 0,
                vertex_page_class: 0,
                index_buffer_offset: 0,
                index_buffer_size: 0,
                index_page_index: 0,
                index_page_class: 0,
                bounds: MeshBounds {
                    min: [0.0; 3],
                    max: [0.0; 3],
                },
                textures: Vec::new(),
                evidence: Vec::new(),
            },
        }
    }

    fn sample_pak() -> Vec<u8> {
        let mut data = vec![0; 180];
        fn put_u16(data: &mut [u8], offset: usize, value: u16) {
            data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        fn put_u32(data: &mut [u8], offset: usize, value: u32) {
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        fn put_u64(data: &mut [u8], offset: usize, value: u64) {
            data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        put_u32(&mut data, 0, 0x0000_0a79);
        put_u32(&mut data, 4, 80);
        put_u32(&mut data, 8, 0);
        put_u32(&mut data, 12, 32);
        put_u32(&mut data, 16, 1);
        put_u32(&mut data, 20, 44);
        put_u32(&mut data, 24, 0);
        put_u32(&mut data, 28, 56);
        put_u32(&mut data, 44, 80);
        put_u32(&mut data, 48, 100);
        put_u32(&mut data, 52, 0);
        put_u32(&mut data, 56, 0);
        put_u32(&mut data, 60, 68);
        put_u32(&mut data, 64, 0);
        put_u32(&mut data, 80, 0xdead_beef);
        put_u32(&mut data, 84, 0xdead_beef);
        put_u32(&mut data, 88, 0);
        put_u32(&mut data, 92, 100);
        put_u16(&mut data, 96, 0);
        put_u16(&mut data, 98, 1);
        put_u64(&mut data, 100, 0);
        put_u32(&mut data, 108, 40);
        put_u32(&mut data, 112, 0);
        put_u64(&mut data, 120, 68);
        put_u64(&mut data, 128, 77);
        data[148..157].copy_from_slice(b"testname\0");
        data[157..168].copy_from_slice(b"GEOMETRY_1\0");
        data
    }

    #[test]
    fn parses_page_and_resource_directory() {
        let bytes = sample_pak();
        let parsed = PakArchive::parse(&bytes).unwrap();
        assert_eq!(parsed.pages.len(), 1);
        assert_eq!(parsed.pages[0].page_class, 0);
        assert_eq!(parsed.resources.len(), 1);
        assert_eq!(parsed.resources[0].name, "testname");
        assert_eq!(parsed.resources[0].resource_type, "GEOMETRY_1");
    }

    #[test]
    fn accepts_the_observed_texture_dictionary_magic_variant() {
        let mut bytes = sample_pak();
        bytes[0..4].copy_from_slice(&0x0001_0a79u32.to_le_bytes());
        assert_eq!(PakArchive::parse(&bytes).unwrap().header.magic, 0x0001_0a79);
    }

    #[test]
    fn rejects_truncated_page_and_invalid_resource_string() {
        let mut bytes = sample_pak();
        bytes[48..52].copy_from_slice(&u32::to_le_bytes(200));
        assert!(PakArchive::parse(&bytes).is_err());

        let mut bytes = sample_pak();
        bytes[128..136].copy_from_slice(&u64::to_le_bytes(200));
        assert!(PakArchive::parse(&bytes).is_err());
    }

    #[test]
    fn decodes_float3_and_u16_buffers_and_rejects_malformed_data() {
        let mut positions = Vec::new();
        for value in [1.0_f32, -2.0, 3.5] {
            positions.extend_from_slice(&value.to_le_bytes());
        }
        let (vertices, bounds) = decode_float3_positions(&positions, 1, 100).unwrap();
        assert_eq!(vertices, vec![[1.0, -2.0, 3.5]]);
        assert_eq!(bounds.min, [1.0, -2.0, 3.5]);
        assert_eq!(bounds.max, [1.0, -2.0, 3.5]);
        assert!(decode_float3_positions(&positions[..11], 1, 100).is_err());

        let mut non_finite = positions.clone();
        non_finite[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_float3_positions(&non_finite, 1, 100).is_err());

        let indices = [0_u16, 1, 2];
        let bytes = indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(decode_u16_indices(&bytes, 3, 3, 200).unwrap(), indices);
        assert!(decode_u16_indices(&bytes[..5], 3, 3, 200).is_err());
        assert!(decode_u16_indices(&bytes, 3, 2, 200).is_err());
    }

    #[test]
    fn serializes_obj_vertices_and_one_based_triangle_indices() {
        let vertices = [[0.0, 1.0, 2.0], [-1.5, 0.25, 3.0], [4.0, 5.0, 6.0]];
        let mut output = Vec::new();
        validate_obj_mesh(&vertices, &[0, 2, 1]).unwrap();
        write_obj(&mut output, &vertices, &[0, 2, 1]).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "v 0 1 2\nv -1.5 0.25 3\nv 4 5 6\nf 1 3 2\n"
        );
    }

    #[test]
    fn serializes_separate_obj_objects_with_global_one_based_indices() {
        let meshes = [
            test_mesh(
                "partition|first",
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![0, 1, 2],
            ),
            test_mesh(
                "partition second",
                vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                vec![2, 1, 0],
            ),
        ];
        let mut output = Vec::new();
        write_obj_meshes(&mut output, &meshes).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("o partition_first\n"));
        assert!(output.contains("o partition_second\n"));
        assert!(output.contains("f 1 2 3\n"));
        assert!(output.contains("f 6 5 4\n"));
        assert_eq!(
            output.lines().filter(|line| line.starts_with("v ")).count(),
            6
        );
        assert_eq!(
            output.lines().filter(|line| line.starts_with("f ")).count(),
            2
        );
    }

    #[test]
    fn rejects_invalid_obj_indices_non_finite_positions_and_incomplete_triangles() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(validate_obj_mesh(&vertices, &[0, 1, 3]).is_err());
        assert!(validate_obj_mesh(&vertices, &[0, 1]).is_err());
        assert!(validate_obj_mesh(&vertices, &[]).is_err());

        let non_finite = [[f32::INFINITY, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(validate_obj_mesh(&non_finite, &[0, 1, 2]).is_err());
        let nan = [[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(validate_obj_mesh(&nan, &[0, 1, 2]).is_err());
    }

    #[test]
    fn decodes_little_endian_bitpacked_positions_and_rejects_bad_layouts() {
        let vertices = [1u8, 2, 3];
        let (decoded, bounds) = decode_quantized_positions(
            &vertices,
            1,
            [8, 8, 8, 0],
            [0.5, 0.25, 2.0, 0.0],
            [-1.0, 1.0, -2.0, 0.0],
            100,
        )
        .unwrap();
        assert_eq!(decoded, vec![[-0.5, 1.5, 4.0]]);
        assert_eq!(bounds.min, [-0.5, 1.5, 4.0]);
        assert_eq!(bounds.max, [-0.5, 1.5, 4.0]);
        assert!(
            decode_quantized_positions(&[1, 2], 1, [8, 8, 8, 0], [1.0; 4], [0.0; 4], 0).is_err()
        );
        assert!(
            decode_quantized_positions(&[1, 2, 3], 1, [8, 8, 8, 8], [1.0; 4], [0.0; 4], 0).is_err()
        );
    }

    #[test]
    fn calculates_texture_mip_sizes_for_reference_formats() {
        assert_eq!(expected_texture_bytes(0x47, 4, 4, 1, Some(8)), Some(8));
        assert_eq!(expected_texture_bytes(0x4d, 5, 7, 1, Some(16)), Some(64));
        assert_eq!(expected_texture_bytes(0x1c, 2, 2, 2, Some(0)), Some(20));
        assert_eq!(expected_texture_bytes(0x3d, 4, 4, 1, Some(0)), Some(16));
        assert_eq!(expected_texture_bytes(0x47, 0, 4, 1, Some(8)), None);
        assert_eq!(expected_texture_bytes(0x47, 4, 4, 0, Some(8)), None);
        assert_eq!(texture_format(0x63).0, "BC7_UNORM_SRGB");
        assert!(expected_texture_bytes(0xdead, 4, 4, 1, None).is_none());
        assert_eq!(expected_rgba_mip_bytes(2, 2, 2), Some(20));
        let malformed = TextureDescriptor {
            resource_index: 0,
            resource_page: 0,
            resource_page_class: 0,
            resource_offset: 0,
            resource_metadata_words: [0; 10],
            hash: 0,
            data_offset: 0,
            unknown0: 0,
            data_size: 0,
            texture_dict_id: 0,
            unknown1: 0,
            texture_type: 0,
            format_id: 0x47,
            field_2c: 0,
            format_name: "BC1_UNORM".to_owned(),
            mip_count: 1,
            width: 0,
            height: 4,
            field_3c: 0,
            stream_flags: 0,
            texture_path_pointer: None,
            texture_path: None,
            expected_mip_bytes: None,
            expected_rgba_bytes: None,
        };
        assert_eq!(
            texture_status(&malformed, None),
            "invalid zero texture dimension"
        );
    }

    #[test]
    fn raw_texture_extraction_preserves_bytes_and_rejects_truncated_source() {
        let source_path = std::env::temp_dir().join(format!(
            "nd-extractor-texture-source-{}.bin",
            std::process::id()
        ));
        let output_path = std::env::temp_dir().join(format!(
            "nd-extractor-texture-output-{}.bin",
            std::process::id()
        ));
        fs::write(&source_path, [0, 1, 2, 3, 4, 5]).unwrap();
        let payload = TexturePayload {
            source_path: source_path.clone(),
            source_descriptor_index: 0,
            source_resource_page: 0,
            source_data_offset: 1,
            source_data_size: 4,
            source_format_id: 0x47,
            source_width: 4,
            source_height: 4,
            source_mip_count: 1,
            dictionary_descriptor_index: None,
        };
        copy_texture_payload(&payload, &output_path).unwrap();
        assert_eq!(fs::read(&output_path).unwrap(), [1, 2, 3, 4]);
        fs::remove_file(&output_path).unwrap();
        fs::write(&output_path, b"keep existing").unwrap();
        assert!(copy_texture_payload(&payload, &output_path).is_err());
        assert_eq!(fs::read(&output_path).unwrap(), b"keep existing");
        fs::remove_file(&output_path).unwrap();

        let malformed = TexturePayload {
            source_data_offset: 4,
            source_data_size: 4,
            ..payload
        };
        assert!(copy_texture_payload(&malformed, &output_path).is_err());
        assert!(!output_path.exists());
        fs::remove_file(source_path).unwrap();
    }

    #[test]
    fn obj_export_does_not_replace_output_when_validation_fails() {
        let mesh = DecodedMesh {
            vertices: vec![[0.0, 0.0, 0.0]],
            indices: vec![0, 0, 1],
            report: MeshReport {
                pak_path: PathBuf::new(),
                geometry_resource_offset: 0,
                submesh_table_offset: 0,
                submesh_count: 0,
                submesh_table_size: 0,
                submesh_index: 0,
                submesh_name: String::new(),
                submesh_offset: 0,
                material_offset: 0,
                vertex_count: 0,
                vertex_format: String::new(),
                index_count: 0,
                triangle_count: 0,
                vertex_buffer_offset: 0,
                vertex_buffer_size: 0,
                vertex_page_index: 0,
                vertex_page_class: 0,
                index_buffer_offset: 0,
                index_buffer_size: 0,
                index_page_index: 0,
                index_page_class: 0,
                bounds: MeshBounds {
                    min: [0.0; 3],
                    max: [0.0; 3],
                },
                textures: Vec::new(),
                evidence: Vec::new(),
            },
        };
        let output = std::env::temp_dir().join(format!(
            "nd-extractor-invalid-mesh-{}.obj",
            std::process::id()
        ));
        fs::write(&output, "keep existing output").unwrap();
        assert!(mesh.export_obj(&output).is_err());
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep existing output");
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn reports_actual_installed_ellie_packages_when_available() {
        let root = Path::new("/mnt/storage/tlou2-extraction/common/actor97");
        let head = root.join("ellie-head.pak");
        if !head.exists() {
            return;
        }
        let archive = PakArchive::parse_path(&head).unwrap();
        assert!(
            archive
                .resources
                .iter()
                .any(|resource| resource.resource_type == "GEOMETRY_1")
        );

        let textures = PakArchive::inspect_textures_path(&head, None).unwrap();
        assert_eq!(textures.descriptors.len(), 192);
        assert!(textures.descriptors.iter().all(|entry| {
            entry.payload.is_some()
                && entry.descriptor.width > 0
                && entry.descriptor.height > 0
                && entry.descriptor.mip_count > 0
        }));
        let format_counts = textures.descriptors.iter().fold(
            std::collections::BTreeMap::new(),
            |mut counts, entry| {
                *counts.entry(entry.descriptor.format_id).or_insert(0usize) += 1;
                counts
            },
        );
        assert_eq!(format_counts.get(&0x47), Some(&63));
        assert_eq!(format_counts.get(&0x4d), Some(&7));
        assert_eq!(format_counts.get(&0x50), Some(&61));
        assert_eq!(format_counts.get(&0x53), Some(&31));
        assert_eq!(format_counts.get(&0x62), Some(&28));
        assert_eq!(format_counts.get(&0x48), Some(&1));
        assert_eq!(format_counts.get(&0x1c), Some(&1));

        let local_texture = textures
            .descriptors
            .iter()
            .find(|entry| entry.descriptor.resource_index == 41)
            .unwrap();
        assert_eq!(local_texture.descriptor.hash, 0x6799_596d_19d8_1aab);
        assert_eq!(
            (
                local_texture.descriptor.format_id,
                local_texture.descriptor.width,
                local_texture.descriptor.height,
                local_texture.descriptor.mip_count,
                local_texture.descriptor.data_size,
            ),
            (0x62, 64, 64, 7, 9_216)
        );
        assert_eq!(local_texture.descriptor.expected_mip_bytes, Some(5_488));
        let local_payload = local_texture.payload.as_ref().unwrap();
        assert_eq!(local_payload.source_path, head);
        assert_eq!(local_payload.source_data_offset, 0xf3_6530);
        assert_eq!(local_payload.source_data_size, 9_216);
        let payload_prefix = read_range(
            &mut File::open(&head).unwrap(),
            fs::metadata(&head).unwrap().len(),
            local_payload.source_data_offset,
            16,
            &head.display().to_string(),
            "read real BC7 payload prefix",
        )
        .unwrap();
        assert_eq!(
            payload_prefix,
            [
                0x20, 0x00, 0xc0, 0x0f, 0xf8, 0xff, 0xff, 0xfd, 0xad, 0xaa, 0xaa, 0xaa, 0x00, 0x00,
                0x00, 0x00
            ]
        );
        let real_payload = read_range(
            &mut File::open(&head).unwrap(),
            fs::metadata(&head).unwrap().len(),
            local_payload.source_data_offset,
            local_payload.source_data_size as usize,
            &head.display().to_string(),
            "read complete real BC7 payload",
        )
        .unwrap();
        let payload_fingerprint = real_payload
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            });
        assert_eq!(payload_fingerprint, 0xbf97_f8a1_54cd_7052);
        assert_eq!(
            sha256_hex(&real_payload),
            "9c7f466d5f84408aba2e2b8bae8d5122d1badae1b409150d21e7a39a10d9f6ae"
        );
        let diagnostic =
            PakArchive::analyze_texture_payloads_path(&head, Some("BC7_UNORM"), 100).unwrap();
        assert!(
            diagnostic.contains("9c7f466d5f84408aba2e2b8bae8d5122d1badae1b409150d21e7a39a10d9f6ae")
        );
        assert!(diagnostic.contains(
            ",0,41,0x10da70,0xf36530,9216,0x6799596d19d81aab,0x62,\"BC7_UNORM\",64,64,7,"
        ));
        let pitch_models = PakArchive::analyze_texture_pitch_models_path(&head).unwrap();
        assert!(
            pitch_models.contains("0x62,\"BC7_UNORM\",0x1,64,64,7,row_pitch_bytes_256,19,0,768:19")
        );
        assert_eq!(local_texture.descriptor.resource_metadata_words[0], 0xec);
        assert_eq!(
            &local_texture.descriptor.resource_metadata_words[1..8],
            &[0x12345; 7]
        );
        assert_eq!(local_texture.descriptor.unknown0, 0);
        assert_eq!(local_texture.descriptor.texture_dict_id, 3);
        assert_eq!(local_texture.descriptor.unknown1, 0);
        assert_eq!(local_texture.descriptor.texture_type, 1);
        assert_eq!(local_texture.descriptor.field_2c, 0x0100_0001);
        assert_eq!(local_texture.descriptor.field_3c, 1);
        assert_eq!(local_texture.descriptor.stream_flags, 2);
        assert_eq!(
            local_texture.descriptor.texture_path_pointer,
            Some(local_texture.descriptor.resource_offset + 128)
        );
        assert_eq!(
            local_texture.descriptor.texture_path.as_deref(),
            Some(
                "[BUILD_INTERMEDIATE]common/texture4/art/objects/characters/wrinkles/ellie-head/generated/ellie-head_brow_raiser_inner_l_wrinkles_0.tga/E17478B7A98858E6.ndb"
            )
        );
        assert!(
            local_payload
                .source_data_offset
                .checked_add(u64::from(local_payload.source_data_size))
                .is_some_and(|end| end <= fs::metadata(&head).unwrap().len())
        );

        let common_dictionary =
            Path::new("/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak");
        if common_dictionary.exists() {
            let resolved =
                PakArchive::inspect_textures_path(&head, Some(common_dictionary)).unwrap();
            let dictionary_matches = resolved
                .descriptors
                .iter()
                .filter(|entry| {
                    entry
                        .payload
                        .as_ref()
                        .is_some_and(|payload| payload.dictionary_descriptor_index.is_some())
                })
                .count();
            assert_eq!(dictionary_matches, 132);

            let resolved_texture = resolved
                .descriptors
                .iter()
                .find(|entry| entry.descriptor.resource_index == 41)
                .unwrap();
            let dictionary_payload = resolved_texture.payload.as_ref().unwrap();
            assert_eq!(dictionary_payload.dictionary_descriptor_index, Some(576));
            assert_eq!(dictionary_payload.source_path, common_dictionary);
            assert_eq!(dictionary_payload.source_data_offset, 0x40f8_10e0);
            assert_eq!(dictionary_payload.source_data_size, 25_600);
            assert!(
                dictionary_payload
                    .source_data_offset
                    .checked_add(u64::from(dictionary_payload.source_data_size))
                    .is_some_and(|end| end <= fs::metadata(common_dictionary).unwrap().len())
            );
            assert_eq!(
                (
                    dictionary_payload.source_format_id,
                    dictionary_payload.source_width,
                    dictionary_payload.source_height,
                    dictionary_payload.source_mip_count,
                ),
                (0x62, 128, 128, 8)
            );

            let dictionary_inventory =
                PakArchive::inspect_textures_path(common_dictionary, None).unwrap();
            let dictionary_texture = dictionary_inventory
                .descriptors
                .iter()
                .find(|entry| entry.descriptor.resource_index == 576)
                .unwrap();
            assert_eq!(
                dictionary_texture.descriptor.hash,
                local_texture.descriptor.hash
            );
            assert_eq!(
                (
                    dictionary_texture.descriptor.format_id,
                    dictionary_texture.descriptor.width,
                    dictionary_texture.descriptor.height,
                    dictionary_texture.descriptor.mip_count,
                    dictionary_texture.descriptor.data_size,
                ),
                (0x62, 128, 128, 8, 25_600)
            );
            assert_eq!(
                dictionary_texture
                    .payload
                    .as_ref()
                    .unwrap()
                    .source_data_offset,
                0x40f8_10e0
            );
            assert_eq!(
                dictionary_texture.descriptor.texture_path,
                local_texture.descriptor.texture_path
            );

            let dictionary_archive = PakArchive::parse_path(common_dictionary).unwrap();
            let local_bytes = read_range(
                &mut File::open(&head).unwrap(),
                fs::metadata(&head).unwrap().len(),
                local_payload.source_data_offset,
                local_payload.source_data_size as usize,
                &head.display().to_string(),
                "read local Ellie-head texture payload for fingerprint test",
            )
            .unwrap();
            let dictionary_bytes = read_range(
                &mut File::open(common_dictionary).unwrap(),
                fs::metadata(common_dictionary).unwrap().len(),
                dictionary_payload.source_data_offset,
                dictionary_payload.source_data_size as usize,
                &common_dictionary.display().to_string(),
                "read dictionary texture payload for fingerprint test",
            )
            .unwrap();
            let fingerprint = |bytes: &[u8]| {
                bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
                })
            };
            assert_ne!(fingerprint(&local_bytes), fingerprint(&dictionary_bytes));
            assert_ne!(local_bytes, dictionary_bytes);
            assert_eq!(
                texture_payload_base(&dictionary_archive).unwrap()
                    + u64::from(dictionary_texture.descriptor.data_offset),
                dictionary_payload.source_data_offset
            );
        }

        let mesh = PakArchive::inspect_mesh_path(&head).unwrap();
        assert_eq!(mesh.vertices.len(), 8_415);
        assert_eq!(mesh.indices.len(), 46_242);
        assert_eq!(mesh.indices.len() / 3, 15_414);
        assert!(
            mesh.indices
                .iter()
                .all(|index| usize::from(*index) < mesh.vertices.len())
        );
        assert!(
            mesh.vertices
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert_eq!(mesh.report.textures.len(), 44);
        assert_eq!(mesh.report.submesh_index, 6);
        assert!(
            mesh.report
                .textures
                .iter()
                .all(|texture| texture.vram_descriptor_offset.is_some())
        );

        let inventory = PakArchive::inspect_geometry_path(&head).unwrap();
        assert_eq!(inventory.geometry_resource_count, 1);
        assert_eq!(inventory.submeshes.len(), 30);
        assert_eq!(inventory.decoded_meshes.len(), 30);
        assert!(inventory.submeshes.iter().all(|submesh| submesh.supported));
        assert_eq!(
            (0..4)
                .map(|lod| inventory
                    .submeshes
                    .iter()
                    .filter(|s| s.lod == Some(lod))
                    .count())
                .collect::<Vec<_>>(),
            vec![9, 7, 7, 7]
        );
        assert_eq!(
            inventory
                .submeshes
                .iter()
                .filter(|submesh| submesh.position_type == Some(0))
                .count(),
            5
        );
        assert_eq!(
            inventory
                .submeshes
                .iter()
                .filter(|submesh| submesh.position_type == Some(64))
                .count(),
            25
        );
        assert_eq!(
            inventory
                .decoded_meshes
                .iter()
                .map(|mesh| mesh.vertices.len())
                .sum::<usize>(),
            38_498
        );
        assert_eq!(
            inventory
                .decoded_meshes
                .iter()
                .map(|mesh| mesh.indices.len() / 3)
                .sum::<usize>(),
            58_751
        );

        let skeleton = root.join("ellie-skel.pak");
        if skeleton.exists() {
            let archive = PakArchive::parse_path(skeleton).unwrap();
            assert!(
                archive
                    .resources
                    .iter()
                    .any(|resource| resource.resource_type == "JOINT_HIERARCHY")
            );
        }

        let body = root.join("ellie-body.pak");
        if body.exists() {
            let body_inventory = PakArchive::inspect_geometry_path(&body).unwrap();
            assert_eq!(body_inventory.submeshes.len(), 15);
            assert_eq!(body_inventory.decoded_meshes.len(), 15);
            assert!(
                body_inventory
                    .submeshes
                    .iter()
                    .all(|submesh| submesh.supported)
            );
            assert_eq!(
                (0..4)
                    .map(|lod| body_inventory
                        .submeshes
                        .iter()
                        .filter(|s| s.lod == Some(lod))
                        .count())
                    .collect::<Vec<_>>(),
                vec![4, 4, 4, 3]
            );
            let mesh = PakArchive::inspect_mesh_path(&body).unwrap();
            assert_eq!(mesh.report.submesh_index, 0);
            assert!(
                mesh.report
                    .submesh_name
                    .contains("innershirt_tuck_lod0_LOD0")
            );
            assert_eq!(
                mesh.report.vertex_format,
                "type 64 bit-packed XYZ (8, 10, 8 bits)"
            );
            assert_eq!(mesh.vertices.len(), 2_332);
            assert_eq!(mesh.indices.len(), 12_033);
            assert_eq!(mesh.indices.len() / 3, 4_011);
            assert!(
                mesh.indices
                    .iter()
                    .all(|index| usize::from(*index) < mesh.vertices.len())
            );
            assert!(
                mesh.vertices
                    .iter()
                    .flatten()
                    .all(|coordinate| coordinate.is_finite())
            );
        }
    }

    #[test]
    fn real_texture_payload_ranges_are_aligned_and_contiguous() {
        let archives = [
            "/mnt/storage/tlou2-extraction/common/actor97/ellie-head.pak",
            "/mnt/storage/tlou2-extraction/common/actor97/ellie-body.pak",
            "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
            "/mnt/storage/tlou2-extraction/santa-barbara/texturedict3/world-santa-barbara-dict.pak",
        ];

        for archive_path in archives {
            let path = Path::new(archive_path);
            if !path.exists() {
                continue;
            }
            let archive = PakArchive::parse_path(path).unwrap();
            let inventory = PakArchive::inspect_textures_path(path, None).unwrap();
            let file_size = fs::metadata(path).unwrap().len();
            let payload_base = texture_payload_base(&archive).unwrap();
            let mut ranges: Vec<_> = inventory
                .descriptors
                .iter()
                .map(|entry| {
                    assert_eq!(
                        entry.descriptor.texture_path_pointer,
                        Some(entry.descriptor.resource_offset + 128),
                        "path pointer fixup in {archive_path}"
                    );
                    assert!(
                        entry.descriptor.texture_path.is_some(),
                        "texture path in {archive_path}"
                    );
                    let payload = entry.payload.as_ref().unwrap();
                    (
                        payload.source_data_offset,
                        u64::from(payload.source_data_size),
                    )
                })
                .collect();
            ranges.sort_unstable();

            assert!(!ranges.is_empty(), "no texture payloads in {archive_path}");
            assert_eq!(ranges[0].0, payload_base, "first payload in {archive_path}");
            for (index, (start, size)) in ranges.iter().copied().enumerate() {
                assert_eq!(start % 16, 0, "payload start alignment in {archive_path}");
                assert_eq!(size % 16, 0, "payload size alignment in {archive_path}");
                let end = start.checked_add(size).unwrap();
                assert!(end <= file_size, "payload range in {archive_path}");
                if let Some((next_start, _)) = ranges.get(index + 1) {
                    assert_eq!(end, *next_start, "payload continuity in {archive_path}");
                } else {
                    assert_eq!(end, file_size, "last payload boundary in {archive_path}");
                }
            }
        }
    }

    #[test]
    fn real_texture_mip_size_calculations_match_selected_descriptors() {
        let samples = [
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                3,
                0x47,
                512,
                512,
                10,
                181_248,
                174_776,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                166,
                0x4d,
                128,
                64,
                8,
                13_824,
                10_960,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                2,
                0x50,
                512,
                512,
                10,
                181_248,
                174_776,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                5,
                0x53,
                512,
                512,
                10,
                353_280,
                349_552,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                11,
                0x5f,
                256,
                256,
                7,
                540_672,
                87_376,
            ),
            (
                "/mnt/storage/tlou2-extraction/santa-barbara/texturedict3/world-santa-barbara-dict.pak",
                2,
                0x5f,
                2048,
                2048,
                1,
                4_194_304,
                4_194_304,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                207,
                0x62,
                1024,
                1024,
                11,
                1_401_856,
                1_398_128,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                1222,
                0x1c,
                1024,
                1024,
                11,
                5_603_328,
                5_592_404,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                3164,
                0x3d,
                512,
                512,
                10,
                393_216,
                349_525,
            ),
        ];

        for (
            archive_path,
            resource_index,
            format_id,
            width,
            height,
            mip_count,
            stored_bytes,
            calculated_mip_bytes,
        ) in samples
        {
            let path = Path::new(archive_path);
            if !path.exists() {
                continue;
            }
            let inventory = PakArchive::inspect_textures_path(path, None).unwrap();
            let descriptor = &inventory
                .descriptors
                .iter()
                .find(|entry| entry.descriptor.resource_index == resource_index)
                .unwrap()
                .descriptor;
            assert_eq!(
                (
                    descriptor.format_id,
                    descriptor.width,
                    descriptor.height,
                    descriptor.mip_count,
                    descriptor.data_size,
                    descriptor.expected_mip_bytes,
                ),
                (
                    format_id,
                    width,
                    height,
                    mip_count,
                    stored_bytes,
                    Some(calculated_mip_bytes)
                ),
                "descriptor {resource_index} in {archive_path}"
            );
        }
    }

    #[test]
    fn real_ellie_bc7_mip_ranges_match_independent_offsets_and_variant_suffix() {
        let actor_path = Path::new("/mnt/storage/tlou2-extraction/common/actor97/ellie-head.pak");
        let dictionary_path =
            Path::new("/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak");
        if !actor_path.exists() || !dictionary_path.exists() {
            return;
        }

        let actor_inventory = PakArchive::inspect_textures_path(actor_path, None).unwrap();
        let actor = actor_inventory
            .descriptors
            .iter()
            .find(|entry| entry.descriptor.resource_index == 41)
            .unwrap();
        assert_eq!(actor.descriptor.format_id, 0x62);
        assert_eq!((actor.descriptor.width, actor.descriptor.height), (64, 64));
        assert_eq!(actor.descriptor.mip_count, 7);
        assert_eq!(actor.descriptor.hash, 0x6799_596d_19d8_1aab);
        assert!(
            actor
                .descriptor
                .texture_path
                .as_deref()
                .unwrap()
                .contains("ellie-head_brow_raiser_inner_l_wrinkles_0.tga/E17478B7A98858E6.ndb")
        );
        let payload = actor.payload.as_ref().unwrap();
        assert_eq!(payload.source_data_offset, 0x00f3_6530);
        assert_eq!(payload.source_data_size, 9_216);
        let bytes = read_texture_payload_bytes(payload).unwrap();
        assert_eq!(
            sha256_hex(&bytes),
            "9c7f466d5f84408aba2e2b8bae8d5122d1badae1b409150d21e7a39a10d9f6ae"
        );

        // Independently calculated constants for BC7's 16-byte 4x4 blocks.
        struct ExpectedMip {
            dimensions: (usize, usize),
            blocks: (usize, usize),
            useful_row_bytes: usize,
            row_pitch: usize,
            subresource_start: usize,
            physical_end: usize,
            next_placement_gap: usize,
        }
        let levels = [
            ExpectedMip {
                dimensions: (64, 64),
                blocks: (16, 16),
                useful_row_bytes: 256,
                row_pitch: 256,
                subresource_start: 0,
                physical_end: 4096,
                next_placement_gap: 0,
            },
            ExpectedMip {
                dimensions: (32, 32),
                blocks: (8, 8),
                useful_row_bytes: 128,
                row_pitch: 256,
                subresource_start: 4096,
                physical_end: 6144,
                next_placement_gap: 0,
            },
            ExpectedMip {
                dimensions: (16, 16),
                blocks: (4, 4),
                useful_row_bytes: 64,
                row_pitch: 256,
                subresource_start: 6144,
                physical_end: 7168,
                next_placement_gap: 0,
            },
            ExpectedMip {
                dimensions: (8, 8),
                blocks: (2, 2),
                useful_row_bytes: 32,
                row_pitch: 256,
                subresource_start: 7168,
                physical_end: 7680,
                next_placement_gap: 0,
            },
            ExpectedMip {
                dimensions: (4, 4),
                blocks: (1, 1),
                useful_row_bytes: 16,
                row_pitch: 256,
                subresource_start: 7680,
                physical_end: 7936,
                next_placement_gap: 256,
            },
            ExpectedMip {
                dimensions: (2, 2),
                blocks: (1, 1),
                useful_row_bytes: 16,
                row_pitch: 256,
                subresource_start: 8192,
                physical_end: 8448,
                next_placement_gap: 256,
            },
            ExpectedMip {
                dimensions: (1, 1),
                blocks: (1, 1),
                useful_row_bytes: 16,
                row_pitch: 256,
                subresource_start: 8704,
                physical_end: 8960,
                next_placement_gap: 0,
            },
        ];
        let mut useful_total = 0;
        for level in levels {
            let (width, height) = level.dimensions;
            let (blocks_x, blocks_y) = level.blocks;
            assert_eq!(blocks_x, width.div_ceil(4));
            assert_eq!(blocks_y, height.div_ceil(4));
            assert_eq!(level.useful_row_bytes, blocks_x * 16);
            assert_eq!(level.row_pitch, level.useful_row_bytes.div_ceil(256) * 256);
            assert!(level.physical_end <= bytes.len());
            useful_total += level.useful_row_bytes * blocks_y;
            for row in 0..blocks_y {
                let padding_start =
                    level.subresource_start + row * level.row_pitch + level.useful_row_bytes;
                let padding_end = level.subresource_start + (row + 1) * level.row_pitch;
                assert!(
                    bytes[padding_start..padding_end]
                        .iter()
                        .all(|byte| *byte == 0)
                );
            }
            let gap_start = level.physical_end;
            let gap_end = gap_start + level.next_placement_gap;
            assert!(bytes[gap_start..gap_end].iter().all(|byte| *byte == 0));
        }
        assert_eq!(useful_total, 5_488);
        assert!(bytes[8_960..9_216].iter().all(|byte| *byte == 0));

        let dictionary_inventory =
            PakArchive::inspect_textures_path(dictionary_path, None).unwrap();
        let dictionary = dictionary_inventory
            .descriptors
            .iter()
            .find(|entry| entry.descriptor.resource_index == 576)
            .unwrap();
        assert_eq!(
            dictionary.descriptor.texture_path,
            actor.descriptor.texture_path
        );
        assert_eq!(dictionary.descriptor.format_id, 0x62);
        assert_eq!(
            (dictionary.descriptor.width, dictionary.descriptor.height),
            (128, 128)
        );
        assert_eq!(dictionary.descriptor.mip_count, 8);
        let dictionary_payload = dictionary.payload.as_ref().unwrap();
        assert_eq!(dictionary_payload.source_data_offset, 0x40f8_10e0);
        assert_eq!(dictionary_payload.source_data_size, 25_600);
        let dictionary_bytes = read_texture_payload_bytes(dictionary_payload).unwrap();
        assert_eq!(dictionary_bytes[16_384..], bytes);
    }

    #[test]
    fn real_texture_copy_footprint_candidate_matches_sizes_and_zero_padding() {
        let archives = [
            (
                "/mnt/storage/tlou2-extraction/common/actor97/ellie-head.pak",
                192,
                192,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                3_224,
                111,
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict-1.pak",
                527,
                0,
            ),
            (
                "/mnt/storage/tlou2-extraction/santa-barbara/texturedict3/world-santa-barbara-dict.pak",
                961,
                98,
            ),
        ];
        let mut checked_archives = 0;
        let mut reverse_mip_padding_bytes = 0;
        let mut reverse_mip_zero_bytes = 0;
        let mut reverse_mip_all_zero_resources = 0;
        let mut mip_major_cube_padding_bytes = 0;
        let mut mip_major_cube_zero_bytes = 0;
        let mut mip_major_cube_all_zero_resources = 0;
        for (archive_path, expected_resources, expected_raw_padding_checks) in archives {
            let path = Path::new(archive_path);
            if !path.exists() {
                continue;
            }
            checked_archives += 1;
            let inventory = PakArchive::inspect_textures_path(path, None).unwrap();
            let mut checked_raw_padding = 0;
            for entry in &inventory.descriptors {
                let payload = entry.payload.as_ref().expect("located texture payload");
                let descriptor = &entry.descriptor;
                let expected = d3d12_copyable_footprint_candidate(descriptor, None)
                    .expect("supported observed texture class");
                assert_eq!(
                    expected.allocation_bytes,
                    u64::from(payload.source_data_size),
                    "copy-footprint candidate size for {archive_path} resource {}",
                    descriptor.resource_index
                );

                let check_padding = archive_path.ends_with("ellie-head.pak")
                    || matches!(descriptor.texture_type, 2 | 3);
                if !check_padding {
                    continue;
                }
                checked_raw_padding += 1;
                let bytes = read_texture_payload_bytes(payload).unwrap();
                let measured = d3d12_copyable_footprint_candidate(descriptor, Some(&bytes))
                    .expect("candidate padding ranges fit within payload");
                assert_eq!(
                    measured.row_padding_zero_bytes,
                    Some(measured.row_padding_bytes),
                    "nonzero predicted row padding for {archive_path} resource {}",
                    descriptor.resource_index
                );
                let tail_start = usize::try_from(measured.end_bytes).unwrap();
                let allocation_end = usize::try_from(measured.allocation_bytes).unwrap();
                let tail = bytes.get(tail_start..allocation_end).unwrap();
                assert!(
                    tail.iter().all(|byte| *byte == 0),
                    "nonzero final allocation padding for {archive_path} resource {}",
                    descriptor.resource_index
                );

                if archive_path.ends_with("ellie-head.pak") {
                    let reversed = d3d12_copyable_footprint_candidate_with_order(
                        descriptor,
                        Some(&bytes),
                        true,
                        false,
                    )
                    .expect("reversed mip candidate ranges fit within payload");
                    reverse_mip_padding_bytes += reversed.row_padding_bytes;
                    reverse_mip_zero_bytes += reversed.row_padding_zero_bytes.unwrap();
                    reverse_mip_all_zero_resources += usize::from(
                        reversed.row_padding_zero_bytes == Some(reversed.row_padding_bytes),
                    );
                }
                if descriptor.texture_type == 2 {
                    let mip_major = d3d12_copyable_footprint_candidate_with_order(
                        descriptor,
                        Some(&bytes),
                        false,
                        true,
                    )
                    .expect("mip-major cubemap candidate ranges fit within payload");
                    mip_major_cube_all_zero_resources += usize::from(
                        mip_major.row_padding_zero_bytes == Some(mip_major.row_padding_bytes),
                    );
                    mip_major_cube_padding_bytes += mip_major.row_padding_bytes;
                    mip_major_cube_zero_bytes += mip_major.row_padding_zero_bytes.unwrap();
                }
            }
            assert_eq!(
                inventory.descriptors.len(),
                expected_resources,
                "{archive_path}"
            );
            assert_eq!(
                checked_raw_padding, expected_raw_padding_checks,
                "raw padding checks in {archive_path}"
            );
        }
        if checked_archives == 0 {
            return;
        }
        if Path::new(archives[0].0).exists() {
            assert_eq!(reverse_mip_padding_bytes, 905_588);
            assert_eq!(reverse_mip_zero_bytes, 627_141);
            assert_eq!(reverse_mip_all_zero_resources, 2);
        }
        if Path::new(archives[1].0).exists() || Path::new(archives[3].0).exists() {
            assert_eq!(mip_major_cube_padding_bytes, 3_096_144);
            assert_eq!(mip_major_cube_zero_bytes, 483_979);
            assert_eq!(mip_major_cube_all_zero_resources, 0);
        }
    }

    #[test]
    fn texture_graph_root_header_is_bounded_and_decodes_candidate_fields() {
        let mut bytes = vec![0; TEXTURE_GRAPH_HEADER_SIZE];
        bytes[TEXTURE_GRAPH_COUNT_OFFSET..TEXTURE_GRAPH_COUNT_OFFSET + 4]
            .copy_from_slice(&192_u32.to_le_bytes());
        for (offset, value) in TEXTURE_GRAPH_POINTER_OFFSETS
            .into_iter()
            .zip([11_i64, 22, 33])
        {
            bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }

        let parsed = parse_texture_graph_root_header(&bytes).unwrap();
        assert_eq!(parsed.entry_count, 192);
        assert_eq!(parsed.pointer_values, [(56, 11), (64, 22), (72, 33)]);
        assert!(parse_texture_graph_root_header(&bytes[..TEXTURE_GRAPH_HEADER_SIZE - 1]).is_err());

        bytes[TEXTURE_GRAPH_COUNT_OFFSET..TEXTURE_GRAPH_COUNT_OFFSET + 4]
            .copy_from_slice(&((MAX_TEXTURE_GRAPH_ENTRIES + 1) as u32).to_le_bytes());
        assert!(parse_texture_graph_root_header(&bytes).is_err());
    }

    #[test]
    fn real_texture_pointer_graphs_report_explicit_edges_and_descriptor_relations() {
        let samples = [
            (
                "/mnt/storage/tlou2-extraction/common/actor97/ellie-head.pak",
                "Root TEXTURE_TABLE",
                "observed_count=192",
                "nearby_vram_desc_deltas_by_depth={(1, 48): 192",
            ),
            (
                "/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak",
                "Root TEXTURE_DICTIONARY",
                "observed_count=3224",
                "nearby_vram_desc_deltas_by_depth={(1, 48): 3224",
            ),
        ];
        for (path, expected_root, expected_count, expected_relation) in samples {
            let path = Path::new(path);
            if !path.exists() {
                continue;
            }
            let report = PakArchive::inspect_texture_graphs_path(path).unwrap();
            assert!(report.contains(expected_root), "{path:?}");
            assert!(report.contains(expected_count), "{path:?}");
            assert!(
                report.contains("nearby_vram_desc_deltas_by_depth"),
                "{path:?}"
            );
            assert!(
                report.contains("probe_window=128 bytes per target"),
                "{path:?}"
            );
            assert!(report.contains(expected_relation), "{path:?}");
        }
    }

    #[test]
    fn real_dictionary_targets_are_path_suffixes_with_paired_descriptors() {
        let path = Path::new("/mnt/storage/tlou2-extraction/common/texturedict3/common-dict.pak");
        if !path.exists() {
            return;
        }
        let report = PakArchive::inspect_texture_dictionary_records_path(path).unwrap();
        assert!(report.contains("root +0x40 explicit fixups=3224 missing_fixups=0"));
        assert!(report.contains("paired +0x48 VRAM_DESC matches=3224 unmatched=0"));
        assert!(report.contains("target_pages={0: 10, 1: 3214} unique_targets=3224"));
        assert!(report.contains("matches=3224 mismatches=0"));
        assert!(report.contains("inline path at resource+0x80: 3224/3224"));
        assert!(report.contains("(0, 0, 0x1f880, 1, 0x15a558)"));
        assert!(report.contains("target_page=1 entries=3214 explicit_fixup_offsets=[]"));
        assert!(report.contains(
            "VRAM_DESC path fixup source=(0,0xd3158) target=(0,0xec530) inline_path_address=0xec530 exact_match=true"
        ));
        assert!(report.contains("resource=1257"));
        assert!(report.contains("resource=3164"));
        assert!(report.contains("resource=2915"));
    }

    #[test]
    fn real_actor_and_dictionary_texture_variants_compare_exact_payloads() {
        let root = Path::new("/mnt/storage/tlou2-extraction/common");
        let paths = [
            root.join("actor97/ellie-head.pak"),
            root.join("texturedict3/common-dict.pak"),
            root.join("actor97/window-0o75x1.pak"),
            root.join("actor97/window-1o3x2o0.pak"),
        ];
        if paths.iter().any(|path| !path.exists()) {
            return;
        }
        let report = PakArchive::compare_texture_variants_paths(&paths).unwrap();
        let ellie_path = "PATH art/objects/characters/wrinkles/ellie-head/generated/ellie-head_brow_raiser_inner_l_wrinkles_0.tga/E17478B7A98858E6.ndb";
        let ellie_start = report.find(ellie_path).unwrap();
        let ellie_group = &report[ellie_start..];
        let ellie_end = ellie_group[ellie_path.len()..]
            .find("\nPATH ")
            .map_or(ellie_group.len(), |offset| offset + ellie_path.len());
        let ellie_group = &ellie_group[..ellie_end];
        assert!(ellie_group.contains(
            "varying_fields=[\"width\", \"height\", \"mip_count\", \"data_size\", \"data_offset\"]"
        ));
        assert!(ellie_group.contains("formats={98}"));
        assert!(ellie_group.contains("stream_flags={2}"));
        assert!(ellie_group.contains("exact_payload_classes=2"));
        assert!(ellie_group.contains("exact_shifted_mip_suffixes=[(41, 576, 16384, 1)]"));
        assert!(ellie_group.contains("resource=41"));
        assert!(ellie_group.contains("resource=576"));
        assert!(ellie_group.contains("fingerprint=bf97f8a154cd7052"));
        assert!(ellie_group.contains("fingerprint=b55137fd04e36865"));
        assert!(report.contains("exact_shifted_mip_suffixes=[(56, 740, 32768, 1)]"));
        assert!(report.contains("exact_shifted_mip_suffixes=[(184, 2980, 8192, 1)]"));

        let shared_path = "PATH art/general/textures/layout-textures/gen-flat-black-color.tga/3F6AF6BCAFF546AD.ndb";
        let shared_start = report.find(shared_path).unwrap();
        let shared_group = &report[shared_start..];
        assert!(shared_group.starts_with(shared_path));
        assert!(shared_group.contains("exact_payload_classes=1"));
        assert!(shared_group.contains("content_class=exact-0"));
    }
}
