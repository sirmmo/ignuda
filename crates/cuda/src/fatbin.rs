//! CUDA fat binary container parsing.
//!
//! Layout (reverse-engineered, little endian):
//!   container header: u32 magic 0xBA55ED50, u16 version, u16 header_size, u64 payload_size
//!   entries:          u16 kind (1 = PTX, 2 = ELF/cubin), u16 version, u32 header_size,
//!                     u64 padded_payload_size, u32 compressed_size, ..., u32 sm_arch @28,
//!                     u64 flags @40 (0x2000 = LZ4, 0x8000 = ZSTD), u64 uncompressed_size @56
//! Containers may be concatenated (and empty ones occur), so we scan them in sequence.

pub const FATBIN_MAGIC: u32 = 0xBA55_ED50;
pub const WRAPPER_MAGIC: u32 = 0x4662_43B1;

pub const KIND_PTX: u16 = 1;
pub const KIND_ELF: u16 = 2;

#[derive(Debug)]
pub struct Entry {
    pub kind: u16,
    pub arch: u32,
    pub data: Vec<u8>,
}

fn rd<const N: usize>(b: &[u8], off: usize) -> Option<[u8; N]> {
    b.get(off..off + N).map(|s| s.try_into().unwrap())
}
fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    rd::<2>(b, o).map(u16::from_le_bytes)
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    rd::<4>(b, o).map(u32::from_le_bytes)
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    rd::<8>(b, o).map(u64::from_le_bytes)
}

/// Total byte size of the fatbin container(s) beginning at `p`, so callers
/// holding only a raw pointer know how much memory to read.
///
/// # Safety
/// `p` must point at a valid fatbin container.
pub unsafe fn container_len(p: *const u8) -> usize {
    let mut off = 0usize;
    loop {
        let h = std::slice::from_raw_parts(p.add(off), 16);
        if u32_at(h, 0) != Some(FATBIN_MAGIC) {
            return off;
        }
        let hs = u16_at(h, 6).unwrap() as usize;
        let size = u64_at(h, 8).unwrap() as usize;
        off += hs + size;
        // Concatenated containers only follow empty headers.
        if size != 0 {
            let next = std::slice::from_raw_parts(p.add(off), 4);
            if u32_at(next, 0) != Some(FATBIN_MAGIC) {
                return off;
            }
        }
    }
}

pub fn parse(b: &[u8]) -> Result<Vec<Entry>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 16 <= b.len() && u32_at(b, pos) == Some(FATBIN_MAGIC) {
        let hs = u16_at(b, pos + 6).unwrap() as usize;
        let size = u64_at(b, pos + 8).unwrap() as usize;
        let mut p = pos + hs;
        let end = (p + size).min(b.len());
        while p + 64 <= end {
            let kind = u16_at(b, p).unwrap();
            let ehs = u32_at(b, p + 4).unwrap() as usize;
            let padded = u64_at(b, p + 8).unwrap() as usize;
            let csize = u32_at(b, p + 16).unwrap() as usize;
            let arch = u32_at(b, p + 28).unwrap();
            let flags = u64_at(b, p + 40).unwrap();
            let usize_ = u64_at(b, p + 56).unwrap() as usize;
            let body = b.get(p + ehs..p + ehs + padded).ok_or("truncated fatbin entry")?;
            let data = if flags & 0x2000 != 0 {
                lz4_block_decompress(&body[..csize.min(body.len())], usize_)?
            } else if flags & 0x8000 != 0 {
                return Err("ZSTD-compressed fatbin entries are not supported yet".into());
            } else {
                body.to_vec()
            };
            out.push(Entry { kind, arch, data });
            p += ehs + padded;
        }
        pos += hs + size;
    }
    Ok(out)
}

/// Choose the best PTX entry: highest architecture not above `max_arch`
/// (falling back to the lowest available).
pub fn best_ptx(entries: &[Entry], max_arch: u32) -> Option<&Entry> {
    let ptx = entries.iter().filter(|e| e.kind == KIND_PTX);
    ptx.clone()
        .filter(|e| e.arch <= max_arch)
        .max_by_key(|e| e.arch)
        .or_else(|| ptx.min_by_key(|e| e.arch))
}

pub fn lz4_block_decompress(src: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::with_capacity(expected);
    let mut i = 0;
    let len_ext = |i: &mut usize, mut n: usize| -> Result<usize, String> {
        if n == 15 {
            loop {
                let b = *src.get(*i).ok_or("lz4: truncated length")?;
                *i += 1;
                n += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        Ok(n)
    };
    while i < src.len() {
        let token = src[i];
        i += 1;
        let lit = len_ext(&mut i, (token >> 4) as usize)?;
        out.extend_from_slice(src.get(i..i + lit).ok_or("lz4: truncated literals")?);
        i += lit;
        if i >= src.len() || out.len() >= expected {
            break;
        }
        let off = u16::from_le_bytes([src[i], *src.get(i + 1).ok_or("lz4: truncated offset")?]) as usize;
        i += 2;
        if off == 0 || off > out.len() {
            return Err("lz4: bad match offset".into());
        }
        let mlen = len_ext(&mut i, (token & 15) as usize)? + 4;
        let start = out.len() - off;
        for k in 0..mlen {
            let c = out[start + k];
            out.push(c);
        }
    }
    out.truncate(expected);
    Ok(out)
}
