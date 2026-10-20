use std::fmt;

type BcDecoder = fn(&[u8], usize, usize, &mut [u32]) -> Result<(), &'static str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcFormat {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc7,
}

impl BcFormat {
    fn bytes_per_block(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc4 => 8,
            Self::Bc3 | Self::Bc5 | Self::Bc7 => 16,
        }
    }

    fn decoder(self) -> BcDecoder {
        match self {
            Self::Bc1 => texture2ddecoder::decode_bc1,
            Self::Bc3 => texture2ddecoder::decode_bc3,
            Self::Bc4 => texture2ddecoder::decode_bc4,
            Self::Bc5 => texture2ddecoder::decode_bc5,
            Self::Bc7 => texture2ddecoder::decode_bc7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedLevel {
    pub width: usize,
    pub height: usize,
    pub rgba8: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureDecodeError {
    InvalidDimensions { width: usize, height: usize },
    SizeOverflow,
    InvalidInputSize { expected: usize, actual: usize },
    Decoder(&'static str),
}

impl fmt::Display for TextureDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(
                    f,
                    "texture dimensions must be non-zero, got {width}x{height}"
                )
            }
            Self::SizeOverflow => f.write_str("texture dimensions overflow addressable size"),
            Self::InvalidInputSize { expected, actual } => write!(
                f,
                "linear BC input has {actual} bytes; expected exactly {expected}"
            ),
            Self::Decoder(error) => write!(f, "BC decoder rejected input: {error}"),
        }
    }
}

impl std::error::Error for TextureDecodeError {}

/// Decode one independent 4x4 BC block without making assumptions about its enclosing layout.
pub fn decode_bc_block(
    format: BcFormat,
    block_data: &[u8],
) -> Result<DecodedLevel, TextureDecodeError> {
    let expected_input = format.bytes_per_block();
    if block_data.len() != expected_input {
        return Err(TextureDecodeError::InvalidInputSize {
            expected: expected_input,
            actual: block_data.len(),
        });
    }
    let mut decoded_bgra_words = vec![0_u32; 16];
    (format.decoder())(block_data, 4, 4, &mut decoded_bgra_words)
        .map_err(TextureDecodeError::Decoder)?;
    Ok(DecodedLevel {
        width: 4,
        height: 4,
        rgba8: bgra_words_to_rgba8(decoded_bgra_words),
    })
}

pub fn write_dds_file(
    width: u32,
    height: u32,
    mip_count: u32,
    dxgi_format: u32,
    mips_data: &[&[u8]],
    output: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(output)?;

    // DDS Magic
    file.write_all(b"DDS ")?;

    // DDS_HEADER (124 bytes)
    let header_size: u32 = 124;
    let flags: u32 = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000; // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT
    let pitch_or_linear_size: u32 = 0;
    let depth: u32 = 0;

    file.write_all(&header_size.to_le_bytes())?;
    file.write_all(&flags.to_le_bytes())?;
    file.write_all(&height.to_le_bytes())?;
    file.write_all(&width.to_le_bytes())?;
    file.write_all(&pitch_or_linear_size.to_le_bytes())?;
    file.write_all(&depth.to_le_bytes())?;
    file.write_all(&mip_count.to_le_bytes())?;

    // Reserved1 [11 x u32]
    for _ in 0..11 {
        file.write_all(&0u32.to_le_bytes())?;
    }

    // PIXELFORMAT (32 bytes)
    let pf_size: u32 = 32;
    let pf_flags: u32 = 0x4; // DDPF_FOURCC
    let pf_fourcc: u32 = 0x30315844; // "DX10"
    file.write_all(&pf_size.to_le_bytes())?;
    file.write_all(&pf_flags.to_le_bytes())?;
    file.write_all(&pf_fourcc.to_le_bytes())?;
    for _ in 0..5 {
        file.write_all(&0u32.to_le_bytes())?;
    }

    // CAPS (4 u32s): dwCaps, dwCaps2, dwCaps3, dwCaps4
    let dds_caps: u32 = 0x1000 | 0x400000 | if mip_count > 1 { 0x8 } else { 0 }; // DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    file.write_all(&dds_caps.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?; // Reserved2

    // DDS_HEADER_DXT10 (20 bytes)
    file.write_all(&dxgi_format.to_le_bytes())?;
    let resource_dimension: u32 = 3; // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    file.write_all(&resource_dimension.to_le_bytes())?;
    let misc_flag: u32 = 0;
    file.write_all(&misc_flag.to_le_bytes())?;
    let array_size: u32 = 1;
    file.write_all(&array_size.to_le_bytes())?;
    let misc_flags2: u32 = 0;
    file.write_all(&misc_flags2.to_le_bytes())?;

    // Write mip levels
    for mip in mips_data {
        file.write_all(mip)?;
    }

    Ok(())
}
/// not a GPU-specific tiled or swizzled payload.
pub fn decode_bc_level(
    format: BcFormat,
    width: usize,
    height: usize,
    linear_data: &[u8],
) -> Result<DecodedLevel, TextureDecodeError> {
    if width == 0 || height == 0 {
        return Err(TextureDecodeError::InvalidDimensions { width, height });
    }

    let blocks_wide = width.div_ceil(4);
    let blocks_high = height.div_ceil(4);
    let expected_input = blocks_wide
        .checked_mul(blocks_high)
        .and_then(|blocks| blocks.checked_mul(format.bytes_per_block()))
        .ok_or(TextureDecodeError::SizeOverflow)?;
    if linear_data.len() != expected_input {
        return Err(TextureDecodeError::InvalidInputSize {
            expected: expected_input,
            actual: linear_data.len(),
        });
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or(TextureDecodeError::SizeOverflow)?;
    let rgba_size = pixel_count
        .checked_mul(4)
        .ok_or(TextureDecodeError::SizeOverflow)?;
    let mut decoded_bgra_words = vec![0_u32; pixel_count];
    (format.decoder())(linear_data, width, height, &mut decoded_bgra_words)
        .map_err(TextureDecodeError::Decoder)?;

    let rgba8 = bgra_words_to_rgba8(decoded_bgra_words);
    debug_assert_eq!(rgba8.len(), rgba_size);
    Ok(DecodedLevel {
        width,
        height,
        rgba8,
    })
}

fn bgra_words_to_rgba8(words: Vec<u32>) -> Vec<u8> {
    let mut rgba8 = Vec::with_capacity(words.len() * 4);
    for bgra in words {
        let [blue, green, red, alpha] = bgra.to_le_bytes();
        rgba8.extend_from_slice(&[red, green, blue, alpha]);
    }
    rgba8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_bc1_red_block() -> [u8; 8] {
        [0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0]
    }

    #[test]
    fn decodes_known_bc1_block_to_rgba_order() {
        let image = decode_bc_level(BcFormat::Bc1, 4, 4, &constant_bc1_red_block()).unwrap();
        assert_eq!(image.width, 4);
        assert_eq!(image.height, 4);
        assert_eq!(image.rgba8.len(), 4 * 4 * 4);
        assert!(
            image
                .rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| *pixel == [255, 0, 0, 255])
        );
    }

    #[test]
    fn decodes_bc3_bc4_and_bc5_constant_blocks() {
        let red = constant_bc1_red_block();
        let mut bc3 = [0_u8; 16];
        bc3[0] = 255;
        bc3[1] = 255;
        bc3[8..].copy_from_slice(&red);
        let image = decode_bc_level(BcFormat::Bc3, 4, 4, &bc3).unwrap();
        assert!(
            image
                .rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| *pixel == [255, 0, 0, 255])
        );

        let bc4 = [127, 127, 0, 0, 0, 0, 0, 0];
        let image = decode_bc_level(BcFormat::Bc4, 4, 4, &bc4).unwrap();
        assert!(
            image
                .rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| *pixel == [127, 0, 0, 255])
        );

        let bc5 = [255, 255, 0, 0, 0, 0, 0, 0, 127, 127, 0, 0, 0, 0, 0, 0];
        let image = decode_bc_level(BcFormat::Bc5, 4, 4, &bc5).unwrap();
        assert!(
            image
                .rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| *pixel == [255, 127, 0, 255])
        );
    }

    #[test]
    fn bc7_mode_6_known_answer_matches_independent_decoder() {
        // Hand-packed BC7 mode 6 endpoints and selectors; checked against bcdec v0.985,
        // whose decoder is available under MIT or Unlicense: https://github.com/iOrange/bcdec.
        let block = [
            0xc0, 0x3f, 0x00, 0x00, 0x00, 0xfc, 0xff, 0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let image = decode_bc_block(BcFormat::Bc7, &block).unwrap();
        let mut expected = vec![[254, 0, 0, 254]; 16];
        expected[1] = [0, 0, 254, 254];
        assert_eq!(image.width, 4);
        assert_eq!(image.height, 4);
        assert_eq!(
            image.rgba8,
            expected.into_iter().flatten().collect::<Vec<_>>()
        );
        assert!(matches!(
            decode_bc_block(BcFormat::Bc7, &block[..15]),
            Err(TextureDecodeError::InvalidInputSize {
                expected: 16,
                actual: 15
            })
        ));
    }

    #[test]
    fn decodes_partial_edge_blocks_and_rejects_bad_sizes() {
        let blocks = [
            0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0, 0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0,
        ];
        let image = decode_bc_level(BcFormat::Bc1, 5, 3, &blocks).unwrap();
        assert_eq!(image.rgba8.len(), 5 * 3 * 4);
        assert!(decode_bc_level(BcFormat::Bc1, 5, 3, &blocks[..8]).is_err());
        assert!(decode_bc_level(BcFormat::Bc1, 5, 3, &[&blocks[..], &[0][..]].concat()).is_err());
        assert!(matches!(
            decode_bc_level(BcFormat::Bc1, 0, 4, &[]),
            Err(TextureDecodeError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            decode_bc_level(BcFormat::Bc7, usize::MAX, usize::MAX, &[]),
            Err(TextureDecodeError::SizeOverflow)
        ));
    }
}
