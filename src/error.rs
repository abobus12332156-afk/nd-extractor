use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryErrorKind {
    UnexpectedEof {
        operation: &'static str,
        offset: usize,
        requested: usize,
        available: usize,
    },
    InvalidSeek {
        offset: usize,
        available: usize,
    },
    RangeOverflow {
        operation: &'static str,
        offset: usize,
        requested: usize,
        available: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryError {
    pub source: Option<String>,
    pub kind: BinaryErrorKind,
}

impl BinaryError {
    pub(crate) fn new(source: Option<&str>, kind: BinaryErrorKind) -> Self {
        Self {
            source: source.map(str::to_owned),
            kind,
        }
    }
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = &self.source {
            write!(f, "{source}: ")?;
        }

        match &self.kind {
            BinaryErrorKind::UnexpectedEof {
                operation,
                offset,
                requested,
                available,
            } => write!(
                f,
                "cannot {operation} at offset {offset:#x}: requested {requested} bytes, \
                 but only {available} available"
            ),
            BinaryErrorKind::InvalidSeek { offset, available } => write!(
                f,
                "cannot seek to offset {offset:#x}: input length is {available} bytes"
            ),
            BinaryErrorKind::RangeOverflow {
                operation,
                offset,
                requested,
                available,
            } => write!(
                f,
                "cannot {operation} at offset {offset:#x}: range length {requested} \
                 overflows address arithmetic ({available} bytes available)"
            ),
        }
    }
}

impl Error for BinaryError {}
