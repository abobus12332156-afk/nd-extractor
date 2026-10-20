use crate::error::{BinaryError, BinaryErrorKind};

pub struct BinaryReader<'a> {
    data: &'a [u8],
    position: usize,
    source: Option<String>,
}

impl<'a> BinaryReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            source: None,
        }
    }

    pub fn with_source(data: &'a [u8], source: impl Into<String>) -> Self {
        Self {
            data,
            position: 0,
            source: Some(source.into()),
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn seek(&mut self, offset: usize) -> Result<(), BinaryError> {
        if offset > self.data.len() {
            return Err(BinaryError::new(
                self.source.as_deref(),
                BinaryErrorKind::InvalidSeek {
                    offset,
                    available: self.data.len(),
                },
            ));
        }

        self.position = offset;
        Ok(())
    }

    pub fn read_bytes(&mut self, length: usize) -> Result<&'a [u8], BinaryError> {
        let range = self.checked_range(self.position, length, "read")?;
        self.position = range.end;
        Ok(&self.data[range])
    }

    pub fn skip(&mut self, length: usize) -> Result<(), BinaryError> {
        let range = self.checked_range(self.position, length, "skip")?;
        self.position = range.end;
        Ok(())
    }

    pub fn read_u8(&mut self) -> Result<u8, BinaryError> {
        Ok(self.read_bytes(1)?[0])
    }

    pub fn read_u16(&mut self) -> Result<u16, BinaryError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub fn read_u32(&mut self) -> Result<u32, BinaryError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub fn read_u64(&mut self) -> Result<u64, BinaryError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub fn read_i32(&mut self) -> Result<i32, BinaryError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub fn read_f32(&mut self) -> Result<f32, BinaryError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub fn read_f64(&mut self) -> Result<f64, BinaryError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn checked_range(
        &self,
        offset: usize,
        requested: usize,
        operation: &'static str,
    ) -> Result<std::ops::Range<usize>, BinaryError> {
        let available = self.data.len().saturating_sub(offset);
        let Some(end) = offset.checked_add(requested) else {
            return Err(BinaryError::new(
                self.source.as_deref(),
                BinaryErrorKind::RangeOverflow {
                    operation,
                    offset,
                    requested,
                    available,
                },
            ));
        };

        if offset > self.data.len() || end > self.data.len() {
            return Err(BinaryError::new(
                self.source.as_deref(),
                BinaryErrorKind::UnexpectedEof {
                    operation,
                    offset,
                    requested,
                    available,
                },
            ));
        }

        Ok(offset..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_numeric_values_as_little_endian() {
        let data = [
            0xAB, // u8
            0x34, 0x12, // u16
            0x78, 0x56, 0x34, 0x12, // u32
            0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12, // u64
            0xFE, 0xFF, 0xFF, 0xFF, // i32
            0x00, 0x00, 0x80, 0x3F, // f32
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // f64
        ];
        let mut reader = BinaryReader::new(&data);

        assert_eq!(reader.read_u8().unwrap(), 0xAB);
        assert_eq!(reader.read_u16().unwrap(), 0x1234);
        assert_eq!(reader.read_u32().unwrap(), 0x1234_5678);
        assert_eq!(reader.read_u64().unwrap(), 0x1234_5678_90AB_CDEF);
        assert_eq!(reader.read_i32().unwrap(), -2);
        assert_eq!(reader.read_f32().unwrap(), 1.0);
        assert_eq!(reader.read_f64().unwrap(), 1.0);
        assert_eq!(reader.position(), data.len());
    }

    #[test]
    fn reads_ranges_and_skips_bytes() {
        let mut reader = BinaryReader::new(&[1, 2, 3, 4]);

        reader.skip(1).unwrap();
        assert_eq!(reader.read_bytes(2).unwrap(), &[2, 3]);
        assert_eq!(reader.position(), 3);
    }

    #[test]
    fn seeking_to_eof_and_zero_length_reads_are_allowed() {
        let mut reader = BinaryReader::new(&[1, 2]);

        reader.seek(2).unwrap();
        assert_eq!(reader.position(), 2);
        assert_eq!(reader.read_bytes(0).unwrap(), &[]);
        assert_eq!(reader.position(), 2);
    }

    #[test]
    fn truncated_read_reports_source_offset_and_available_length() {
        let mut reader = BinaryReader::with_source(&[1, 2, 3], "mesh.pak");
        reader.seek(2).unwrap();

        let error = reader.read_u32().unwrap_err();

        assert_eq!(error.source.as_deref(), Some("mesh.pak"));
        assert_eq!(
            error.kind,
            BinaryErrorKind::UnexpectedEof {
                operation: "read",
                offset: 2,
                requested: 4,
                available: 1,
            }
        );
        assert!(error.to_string().contains("mesh.pak"));
        assert!(error.to_string().contains("requested 4 bytes"));
        assert_eq!(reader.position(), 2);
    }

    #[test]
    fn invalid_seek_does_not_change_position() {
        let mut reader = BinaryReader::new(&[1, 2]);

        let error = reader.seek(3).unwrap_err();

        assert_eq!(
            error.kind,
            BinaryErrorKind::InvalidSeek {
                offset: 3,
                available: 2,
            }
        );
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn range_validation_detects_overflow_before_access() {
        let reader = BinaryReader::new(&[1, 2, 3]);

        let error = reader.checked_range(usize::MAX, 2, "read").unwrap_err();

        assert_eq!(
            error.kind,
            BinaryErrorKind::RangeOverflow {
                operation: "read",
                offset: usize::MAX,
                requested: 2,
                available: 0,
            }
        );
    }

    #[test]
    fn truncated_range_reports_available_bytes() {
        let mut reader = BinaryReader::new(&[1, 2, 3]);
        reader.seek(1).unwrap();

        let error = reader.skip(3).unwrap_err();

        assert_eq!(
            error.kind,
            BinaryErrorKind::UnexpectedEof {
                operation: "skip",
                offset: 1,
                requested: 3,
                available: 2,
            }
        );
        assert_eq!(reader.position(), 1);
    }
}
