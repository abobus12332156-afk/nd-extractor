use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::binary::BinaryReader;
use crate::error::BinaryError;

pub const TABLE_ENTRY_SIZE: usize = 32;
const MAX_RAW_SEQUENCE_PATTERNS: usize = 256;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DsarMetadataStats {
    pub entries_validated: u64,
    pub raw_byte_24_counts: BTreeMap<u8, u64>,
    pub raw_bytes_24_to_31_counts: BTreeMap<[u8; 8], u64>,
    pub zero_compressed_size_by_byte_24: BTreeMap<u8, u64>,
    pub untracked_raw_sequence_count: u64,
}

impl DsarMetadataStats {
    pub(crate) fn observe(&mut self, entry: &DsarTableEntry) {
        self.entries_validated += 1;
        *self.raw_byte_24_counts.entry(entry.tag).or_default() += 1;
        let mut raw_field = [0; 8];
        raw_field[0] = entry.tag;
        raw_field[1..].copy_from_slice(&entry.trailing_bytes);
        if let Some(count) = self.raw_bytes_24_to_31_counts.get_mut(&raw_field) {
            *count += 1;
        } else if self.raw_bytes_24_to_31_counts.len() < MAX_RAW_SEQUENCE_PATTERNS {
            self.raw_bytes_24_to_31_counts.insert(raw_field, 1);
        } else {
            self.untracked_raw_sequence_count += 1;
        }
        if entry.compressed_size == 0 {
            *self
                .zero_compressed_size_by_byte_24
                .entry(entry.tag)
                .or_default() += 1;
        }
    }
}

/// One DSAR v0x00010003 table record.
///
/// The offset and size fields match the installed TLOU2 PC archives and the
/// independent ALERT DSAR codec. Other DSAR versions have not been verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsarTableEntry {
    /// Zero-based position in the table.
    pub index: usize,
    /// Offset in the declared decompressed byte stream.
    pub decompressed_offset: u64,
    /// Absolute offset in the archive file.
    pub compressed_offset: u64,
    /// Number of decompressed bytes represented by this record.
    pub decompressed_size: u32,
    /// Number of stored bytes in the archive for this record.
    pub compressed_size: u32,
    /// Raw per-record tag; its algorithmic meaning is not established.
    pub tag: u8,
    /// Seven bytes following the tag, preserved without interpretation.
    pub trailing_bytes: [u8; 7],
}

#[derive(Debug)]
pub enum DsarEntryError {
    Binary(BinaryError),
    InvalidRecordLength {
        source_label: Option<String>,
        index: usize,
        actual: usize,
        expected: usize,
    },
    InvalidRange {
        source_label: Option<String>,
        index: usize,
        field: &'static str,
        offset: u64,
        size: u32,
        limit: u64,
        overflow: bool,
    },
    BeforePayload {
        source_label: Option<String>,
        index: usize,
        offset: u64,
        payload_start: u64,
    },
}

impl fmt::Display for DsarEntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary(error) => error.fmt(f),
            Self::InvalidRecordLength {
                source_label,
                index,
                actual,
                expected,
            } => {
                write_source_label(f, source_label)?;
                write!(
                    f,
                    "DSAR table entry {index} is {actual} bytes; expected {expected}"
                )
            }
            Self::InvalidRange {
                source_label,
                index,
                field,
                offset,
                size,
                limit,
                overflow,
            } => {
                write_source_label(f, source_label)?;
                if *overflow {
                    write!(
                        f,
                        "DSAR table entry {index} {field} range overflows: \
                         offset {offset:#x} + size {size:#x}"
                    )
                } else {
                    write!(
                        f,
                        "DSAR table entry {index} {field} range at {offset:#x} \
                         with size {size:#x} exceeds limit {limit:#x}"
                    )
                }
            }
            Self::BeforePayload {
                source_label,
                index,
                offset,
                payload_start,
            } => {
                write_source_label(f, source_label)?;
                write!(
                    f,
                    "DSAR table entry {index} compressed offset {offset:#x} \
                     precedes payload start {payload_start:#x}"
                )
            }
        }
    }
}

impl Error for DsarEntryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Binary(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BinaryError> for DsarEntryError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

impl DsarTableEntry {
    /// Parse and validate one fixed-size record without interpreting its tag.
    ///
    /// The stored data range must lie after the metadata table and within the
    /// archive, and the decompressed range must fit the header's declared size.
    pub fn parse(
        bytes: &[u8],
        index: usize,
        archive_size: u64,
        payload_start: u64,
        decompressed_size_limit: u64,
        source_label: Option<&str>,
    ) -> Result<Self, DsarEntryError> {
        if bytes.len() != TABLE_ENTRY_SIZE {
            return Err(DsarEntryError::InvalidRecordLength {
                source_label: source_label.map(str::to_owned),
                index,
                actual: bytes.len(),
                expected: TABLE_ENTRY_SIZE,
            });
        }

        let mut reader = match source_label {
            Some(label) => BinaryReader::with_source(bytes, label),
            None => BinaryReader::new(bytes),
        };
        let decompressed_offset = reader.read_u64()?;
        let compressed_offset = reader.read_u64()?;
        let decompressed_size = reader.read_u32()?;
        let compressed_size = reader.read_u32()?;
        let tag = reader.read_u8()?;
        let trailing_bytes: [u8; 7] = reader.read_bytes(7)?.try_into().expect("seven-byte read");

        validate_range(
            source_label,
            index,
            "decompressed",
            decompressed_offset,
            decompressed_size,
            decompressed_size_limit,
        )?;
        if compressed_offset < payload_start {
            return Err(DsarEntryError::BeforePayload {
                source_label: source_label.map(str::to_owned),
                index,
                offset: compressed_offset,
                payload_start,
            });
        }
        validate_range(
            source_label,
            index,
            "compressed",
            compressed_offset,
            compressed_size,
            archive_size,
        )?;

        Ok(Self {
            index,
            decompressed_offset,
            compressed_offset,
            decompressed_size,
            compressed_size,
            tag,
            trailing_bytes,
        })
    }
}

fn validate_range(
    source_label: Option<&str>,
    index: usize,
    field: &'static str,
    offset: u64,
    size: u32,
    limit: u64,
) -> Result<(), DsarEntryError> {
    let Some(end) = offset.checked_add(u64::from(size)) else {
        return Err(DsarEntryError::InvalidRange {
            source_label: source_label.map(str::to_owned),
            index,
            field,
            offset,
            size,
            limit,
            overflow: true,
        });
    };

    if end > limit {
        return Err(DsarEntryError::InvalidRange {
            source_label: source_label.map(str::to_owned),
            index,
            field,
            offset,
            size,
            limit,
            overflow: false,
        });
    }

    Ok(())
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

    fn entry_bytes(
        decompressed_offset: u64,
        compressed_offset: u64,
        decompressed_size: u32,
        compressed_size: u32,
        tag: u8,
    ) -> [u8; TABLE_ENTRY_SIZE] {
        let mut bytes = [0x55; TABLE_ENTRY_SIZE];
        bytes[0..8].copy_from_slice(&decompressed_offset.to_le_bytes());
        bytes[8..16].copy_from_slice(&compressed_offset.to_le_bytes());
        bytes[16..20].copy_from_slice(&decompressed_size.to_le_bytes());
        bytes[20..24].copy_from_slice(&compressed_size.to_le_bytes());
        bytes[24] = tag;
        bytes
    }

    #[test]
    fn parses_valid_entry_and_preserves_uninterpreted_values() {
        let bytes = entry_bytes(96, 128, 32, 24, 3);

        let entry = DsarTableEntry::parse(&bytes, 7, 256, 128, 128, Some("sample.psarc")).unwrap();

        assert_eq!(entry.index, 7);
        assert_eq!(entry.decompressed_offset, 96);
        assert_eq!(entry.compressed_offset, 128);
        assert_eq!(entry.decompressed_size, 32);
        assert_eq!(entry.compressed_size, 24);
        assert_eq!(entry.tag, 3);
        assert_eq!(entry.trailing_bytes, [0x55; 7]);
    }

    #[test]
    fn rejects_truncated_or_oversized_records() {
        let bytes = entry_bytes(0, 32, 8, 8, 3);

        assert!(matches!(
            DsarTableEntry::parse(&bytes[..31], 0, 64, 32, 8, None),
            Err(DsarEntryError::InvalidRecordLength {
                actual: 31,
                expected: TABLE_ENTRY_SIZE,
                ..
            })
        ));
        assert!(matches!(
            DsarTableEntry::parse(&[bytes.as_slice(), &[0]].concat(), 0, 64, 32, 8, None),
            Err(DsarEntryError::InvalidRecordLength {
                actual: 33,
                expected: TABLE_ENTRY_SIZE,
                ..
            })
        ));
    }

    #[test]
    fn parses_multiple_entries_with_their_table_indices() {
        let first = entry_bytes(0, 64, 8, 16, 3);
        let second = entry_bytes(8, 80, 4, 12, 7);

        let first_entry = DsarTableEntry::parse(&first, 0, 128, 64, 12, None).unwrap();
        let second_entry = DsarTableEntry::parse(&second, 1, 128, 64, 12, None).unwrap();

        assert_eq!(first_entry.index, 0);
        assert_eq!(first_entry.decompressed_offset, 0);
        assert_eq!(second_entry.index, 1);
        assert_eq!(second_entry.decompressed_offset, 8);
        assert_eq!(second_entry.tag, 7);
    }

    #[test]
    fn accepts_empty_entry_ranges_at_their_limits() {
        let bytes = entry_bytes(8, 64, 0, 0, 3);

        let entry = DsarTableEntry::parse(&bytes, 0, 64, 64, 8, None).unwrap();

        assert_eq!(entry.decompressed_size, 0);
        assert_eq!(entry.compressed_size, 0);
    }

    #[test]
    fn rejects_compressed_range_outside_archive() {
        let bytes = entry_bytes(0, 63, 8, 2, 3);

        let error = DsarTableEntry::parse(&bytes, 4, 64, 32, 8, None).unwrap_err();

        assert!(error.to_string().contains("compressed range"));
        assert!(error.to_string().contains("entry 4"));
    }

    #[test]
    fn rejects_decompressed_range_outside_declared_size() {
        let bytes = entry_bytes(4, 32, 5, 8, 3);

        let error = DsarTableEntry::parse(&bytes, 0, 64, 32, 8, None).unwrap_err();

        assert!(error.to_string().contains("decompressed range"));
    }

    #[test]
    fn rejects_offsets_before_compressed_payload() {
        let bytes = entry_bytes(0, 31, 8, 8, 3);

        let error = DsarTableEntry::parse(&bytes, 0, 64, 32, 8, None).unwrap_err();

        assert!(error.to_string().contains("precedes payload start"));
    }

    #[test]
    fn rejects_range_addition_overflow() {
        let bytes = entry_bytes(u64::MAX, 32, 1, 1, 3);

        let error = DsarTableEntry::parse(&bytes, 0, 64, 32, u64::MAX, None).unwrap_err();

        assert!(error.to_string().contains("overflows"));
    }

    #[test]
    fn summarizes_raw_tag_trailer_and_zero_size_correlations() {
        let mut regular_bytes = entry_bytes(0, 64, 8, 8, 3);
        let mut zero_size_bytes = entry_bytes(8, 80, 4, 0, 254);
        regular_bytes[25] = 0x54;
        zero_size_bytes[25] = 0x54;
        let regular = DsarTableEntry::parse(&regular_bytes, 0, 128, 64, 12, None).unwrap();
        let zero_size = DsarTableEntry::parse(&zero_size_bytes, 1, 128, 64, 12, None).unwrap();
        let mut stats = DsarMetadataStats::default();

        stats.observe(&regular);
        stats.observe(&zero_size);

        assert_eq!(stats.entries_validated, 2);
        assert_eq!(stats.untracked_raw_sequence_count, 0);
        assert_eq!(stats.raw_byte_24_counts.get(&3), Some(&1));
        assert_eq!(stats.raw_byte_24_counts.get(&254), Some(&1));
        assert_eq!(
            stats
                .raw_bytes_24_to_31_counts
                .get(&[3, 0x54, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]),
            Some(&1)
        );
        assert_eq!(
            stats
                .raw_bytes_24_to_31_counts
                .get(&[254, 0x54, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]),
            Some(&1)
        );
        assert_eq!(stats.zero_compressed_size_by_byte_24.get(&254), Some(&1));
        assert!(!stats.zero_compressed_size_by_byte_24.contains_key(&3));
    }

    #[test]
    fn caps_distinct_raw_sequence_storage() {
        let mut stats = DsarMetadataStats::default();

        for value in 0..=u8::MAX {
            let mut bytes = entry_bytes(0, 64, 8, 8, 3);
            bytes[25] = value;
            let entry =
                DsarTableEntry::parse(&bytes, usize::from(value), 128, 64, 8, None).unwrap();
            stats.observe(&entry);
        }
        let mut extra_bytes = entry_bytes(0, 64, 8, 8, 3);
        extra_bytes[26] = 1;
        let extra = DsarTableEntry::parse(&extra_bytes, 256, 128, 64, 8, None).unwrap();
        stats.observe(&extra);

        assert_eq!(
            stats.raw_bytes_24_to_31_counts.len(),
            MAX_RAW_SEQUENCE_PATTERNS
        );
        assert_eq!(stats.untracked_raw_sequence_count, 1);
    }
}
