use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::binary::BinaryReader;
use crate::error::BinaryError;

use super::dsar::{DsarEntryError, DsarMetadataStats, DsarTableEntry, TABLE_ENTRY_SIZE};

const DSAR_MAGIC: &[u8; 4] = b"DSAR";
const DSAR_VERSION: u32 = 0x0001_0003;
const HEADER_SIZE: u64 = 32;
const CHUNK_ENTRY_SIZE: u64 = 32;
const PADDING_MARKER: &[u8; 8] = b"PADDING*";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsarcSummary {
    pub header: PsarcHeader,
    pub archive_size: u64,
    pub chunk_table: Range<u64>,
    pub compressed_payload: Range<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsarcHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub chunk_count: u32,
    pub table_end: u32,
    pub decompressed_size: u64,
    pub padding_marker: [u8; 8],
}

#[derive(Debug)]
pub enum PsarcError {
    Io {
        path: PathBuf,
        operation: &'static str,
        source: io::Error,
    },
    Binary(BinaryError),
    Entry(DsarEntryError),
    UnsupportedMagic {
        source_label: Option<String>,
        magic: [u8; 4],
    },
    UnsupportedVersion {
        source_label: Option<String>,
        version: u32,
    },
    InvalidHeader {
        source_label: Option<String>,
        message: String,
    },
    RangeOverflow {
        source_label: Option<String>,
        field: &'static str,
    },
}

impl fmt::Display for PsarcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                path,
                operation,
                source,
            } => write!(f, "cannot {operation} {}: {source}", path.display()),
            Self::Binary(error) => error.fmt(f),
            Self::Entry(error) => error.fmt(f),
            Self::UnsupportedMagic {
                source_label,
                magic,
            } => {
                write_source_label(f, source_label)?;
                write!(f, "unsupported archive magic {magic:?}")
            }
            Self::UnsupportedVersion {
                source_label,
                version,
            } => {
                write_source_label(f, source_label)?;
                write!(f, "unsupported DSAR version {version:#010x}")
            }
            Self::InvalidHeader {
                source_label,
                message,
            } => {
                write_source_label(f, source_label)?;
                write!(f, "invalid DSAR header: {message}")
            }
            Self::RangeOverflow {
                source_label,
                field,
            } => {
                write_source_label(f, source_label)?;
                write!(f, "DSAR {field} range overflows address arithmetic")
            }
        }
    }
}

impl Error for PsarcError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Binary(error) => Some(error),
            Self::Entry(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BinaryError> for PsarcError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

impl From<DsarEntryError> for PsarcError {
    fn from(error: DsarEntryError) -> Self {
        Self::Entry(error)
    }
}

pub fn inspect_path(path: impl AsRef<Path>) -> Result<PsarcSummary, PsarcError> {
    inspect_path_with_limit(path, 0).map(|(summary, _)| summary)
}

pub fn inspect_path_with_limit(
    path: impl AsRef<Path>,
    limit: usize,
) -> Result<(PsarcSummary, Vec<DsarTableEntry>), PsarcError> {
    inspect_path_internal(path.as_ref(), limit, false)
        .map(|(summary, entries, _)| (summary, entries))
}

pub fn inspect_path_with_metadata_stats(
    path: impl AsRef<Path>,
    limit: usize,
) -> Result<(PsarcSummary, Vec<DsarTableEntry>, DsarMetadataStats), PsarcError> {
    inspect_path_internal(path.as_ref(), limit, true)
}

fn inspect_path_internal(
    path: &Path,
    limit: usize,
    collect_metadata_stats: bool,
) -> Result<(PsarcSummary, Vec<DsarTableEntry>, DsarMetadataStats), PsarcError> {
    let mut file = File::open(path).map_err(|source| PsarcError::Io {
        path: path.to_owned(),
        operation: "open archive",
        source,
    })?;
    let archive_size = file
        .metadata()
        .map_err(|source| PsarcError::Io {
            path: path.to_owned(),
            operation: "read archive metadata for",
            source,
        })?
        .len();
    let mut header = [0; HEADER_SIZE as usize];
    file.read_exact(&mut header)
        .map_err(|source| PsarcError::Io {
            path: path.to_owned(),
            operation: "read 32-byte archive header from",
            source,
        })?;

    let source_label = path.to_string_lossy();
    let summary = inspect_header(&header, archive_size, Some(&source_label))?;
    let entry_count =
        usize::try_from(summary.header.chunk_count).map_err(|_| PsarcError::RangeOverflow {
            source_label: Some(source_label.to_string()),
            field: "chunk count",
        })?;
    let entries_to_read = entry_count.min(limit);
    let mut entries = Vec::with_capacity(entries_to_read);
    let mut stats = DsarMetadataStats::default();
    let entries_to_scan = if collect_metadata_stats {
        entry_count
    } else {
        entries_to_read
    };

    file.seek(SeekFrom::Start(HEADER_SIZE))
        .map_err(|source| PsarcError::Io {
            path: path.to_owned(),
            operation: "seek to chunk table in",
            source,
        })?;

    for index in 0..entries_to_scan {
        let mut entry_bytes = [0; TABLE_ENTRY_SIZE];
        file.read_exact(&mut entry_bytes)
            .map_err(|source| PsarcError::Io {
                path: path.to_owned(),
                operation: "read chunk table entry from",
                source,
            })?;
        let entry = DsarTableEntry::parse(
            &entry_bytes,
            index,
            archive_size,
            summary.compressed_payload.start,
            summary.header.decompressed_size,
            Some(&source_label),
        )?;
        if collect_metadata_stats {
            stats.observe(&entry);
        }
        if index < entries_to_read {
            entries.push(entry);
        }
    }

    Ok((summary, entries, stats))
}

pub fn inspect_header(
    bytes: &[u8],
    archive_size: u64,
    source_label: Option<&str>,
) -> Result<PsarcSummary, PsarcError> {
    let mut reader = match source_label {
        Some(label) => BinaryReader::with_source(bytes, label),
        None => BinaryReader::new(bytes),
    };

    let magic: [u8; 4] = reader.read_bytes(4)?.try_into().expect("four-byte read");
    if &magic != DSAR_MAGIC {
        return Err(PsarcError::UnsupportedMagic {
            source_label: source_label.map(str::to_owned),
            magic,
        });
    }

    let version = reader.read_u32()?;
    if version != DSAR_VERSION {
        return Err(PsarcError::UnsupportedVersion {
            source_label: source_label.map(str::to_owned),
            version,
        });
    }

    let chunk_count = reader.read_u32()?;
    let table_end = reader.read_u32()?;
    let decompressed_size = reader.read_u64()?;
    let padding_marker: [u8; 8] = reader.read_bytes(8)?.try_into().expect("eight-byte read");

    if &padding_marker != PADDING_MARKER {
        return Err(PsarcError::InvalidHeader {
            source_label: source_label.map(str::to_owned),
            message: format!(
                "expected padding marker {PADDING_MARKER:?}, found {padding_marker:?}"
            ),
        });
    }

    let table_size = u64::from(chunk_count)
        .checked_mul(CHUNK_ENTRY_SIZE)
        .ok_or_else(|| PsarcError::RangeOverflow {
            source_label: source_label.map(str::to_owned),
            field: "chunk table",
        })?;
    let expected_table_end =
        HEADER_SIZE
            .checked_add(table_size)
            .ok_or_else(|| PsarcError::RangeOverflow {
                source_label: source_label.map(str::to_owned),
                field: "chunk table",
            })?;

    if u64::from(table_end) != expected_table_end {
        return Err(PsarcError::InvalidHeader {
            source_label: source_label.map(str::to_owned),
            message: format!(
                "declared table end {table_end:#x} does not match header plus \
                 {chunk_count} entries ({expected_table_end:#x})"
            ),
        });
    }
    if expected_table_end > archive_size {
        return Err(PsarcError::InvalidHeader {
            source_label: source_label.map(str::to_owned),
            message: format!(
                "chunk table range {HEADER_SIZE:#x}..{expected_table_end:#x} exceeds \
                 archive length {archive_size:#x}"
            ),
        });
    }

    Ok(PsarcSummary {
        header: PsarcHeader {
            magic,
            version,
            chunk_count,
            table_end,
            decompressed_size,
            padding_marker,
        },
        archive_size,
        chunk_table: HEADER_SIZE..expected_table_end,
        compressed_payload: expected_table_end..archive_size,
    })
}

fn write_source_label(f: &mut fmt::Formatter<'_>, source_label: &Option<String>) -> fmt::Result {
    if let Some(label) = source_label {
        write!(f, "{label}: ")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive_header(chunk_count: u32, table_end: u32) -> [u8; HEADER_SIZE as usize] {
        let mut bytes = [0; HEADER_SIZE as usize];
        bytes[0..4].copy_from_slice(DSAR_MAGIC);
        bytes[4..8].copy_from_slice(&DSAR_VERSION.to_le_bytes());
        bytes[8..12].copy_from_slice(&chunk_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&table_end.to_le_bytes());
        bytes[16..24].copy_from_slice(&0x1234_5678_u64.to_le_bytes());
        bytes[24..32].copy_from_slice(PADDING_MARKER);
        bytes
    }

    #[test]
    fn parses_supported_dsar_header_and_ranges() {
        let header = archive_header(2, 96);
        let summary = inspect_header(&header, 160, Some("sample.psarc")).unwrap();

        assert_eq!(summary.header.magic, *DSAR_MAGIC);
        assert_eq!(summary.header.version, DSAR_VERSION);
        assert_eq!(summary.header.chunk_count, 2);
        assert_eq!(summary.header.table_end, 96);
        assert_eq!(summary.header.decompressed_size, 0x1234_5678);
        assert_eq!(summary.header.padding_marker, *PADDING_MARKER);
        assert_eq!(summary.chunk_table, 32..96);
        assert_eq!(summary.compressed_payload, 96..160);
    }

    #[test]
    fn accepts_minimal_header_with_no_entries_or_payload() {
        let header = archive_header(0, 32);
        let summary = inspect_header(&header, 32, None).unwrap();

        assert_eq!(summary.header.chunk_count, 0);
        assert_eq!(summary.chunk_table, 32..32);
        assert_eq!(summary.compressed_payload, 32..32);
    }

    #[test]
    fn rejects_non_dsar_magic() {
        let mut header = archive_header(0, 32);
        header[..4].copy_from_slice(b"PSAR");

        let error = inspect_header(&header, 32, Some("sample.psarc")).unwrap_err();

        assert!(matches!(
            error,
            PsarcError::UnsupportedMagic {
                source_label: Some(label),
                magic: [b'P', b'S', b'A', b'R']
            } if label == "sample.psarc"
        ));
    }

    #[test]
    fn rejects_unknown_version() {
        let mut header = archive_header(0, 32);
        header[4..8].copy_from_slice(&1_u32.to_le_bytes());

        assert!(matches!(
            inspect_header(&header, 32, None),
            Err(PsarcError::UnsupportedVersion { version: 1, .. })
        ));
    }

    #[test]
    fn rejects_truncated_header() {
        let mut truncated = [0; 12];
        truncated[..4].copy_from_slice(DSAR_MAGIC);
        truncated[4..8].copy_from_slice(&DSAR_VERSION.to_le_bytes());
        let error = inspect_header(&truncated, 12, Some("truncated.psarc")).unwrap_err();

        assert!(matches!(error, PsarcError::Binary(_)));
        assert!(error.to_string().contains("truncated.psarc"));
    }

    #[test]
    fn rejects_inconsistent_table_end() {
        let header = archive_header(2, 95);

        let error = inspect_header(&header, 160, None).unwrap_err();

        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn rejects_table_extending_past_archive_end() {
        let header = archive_header(2, 96);

        let error = inspect_header(&header, 95, None).unwrap_err();

        assert!(error.to_string().contains("exceeds archive length"));
    }

    #[test]
    fn rejects_unknown_padding_marker() {
        let mut header = archive_header(0, 32);
        header[24] = b'X';

        let error = inspect_header(&header, 32, None).unwrap_err();

        assert!(error.to_string().contains("padding marker"));
    }
}
