// SPDX-License-Identifier: CC0-1.0

//! `ChaCha20` block function using aarch64 Neon, processing four blocks in parallel.
//!
//! Uses a transposed state layout: each 128-bit NEON register holds the same
//! state word from four consecutive blocks. In this layout the diagonal round
//! does not need lane shuffles — it is expressed by choosing different vector
//! combinations for the four quarter rounds. Only the byte-level transpose
//! before storing the keystream needs shuffles.

use core::arch::aarch64;

use super::{Key, Nonce, WORD_1, WORD_2, WORD_3, WORD_4};

/// Byte-shuffle table for rotate-left-by-8 within each 32-bit lane.
///
/// Each u32 lane's bytes `[b0, b1, b2, b3]` are permuted to `[b3, b0, b1, b2]`,
/// which is equivalent to a left-rotation by 8 bits.
static ROT8_TABLE: [u8; 16] = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14];

/// XOR four consecutive `ChaCha20` blocks of keystream into `chunk`, starting
/// at `start_block`.
#[inline]
#[allow(clippy::too_many_lines)]
pub(super) fn apply_4_blocks(
    chunk: &mut [u8; 4 * 64],
    key: &Key,
    nonce: &Nonce,
    start_block: u32,
) {
    // SAFETY: NEON is guaranteed on aarch64 (enforced by the module cfg gate);
    // `chunk` is a valid mutable 256-byte array; the routine only reads and
    // writes within it.
    unsafe {
        // Initial state, one broadcast register per state word. Key and nonce
        // indices are hardcoded to match the scalar `State::new` construction
        // in `chacha20.rs`.
        let init0 = aarch64::vdupq_n_u32(WORD_1);
        let init1 = aarch64::vdupq_n_u32(WORD_2);
        let init2 = aarch64::vdupq_n_u32(WORD_3);
        let init3 = aarch64::vdupq_n_u32(WORD_4);
        let init4 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[0], key.0[1], key.0[2], key.0[3]]));
        let init5 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[4], key.0[5], key.0[6], key.0[7]]));
        let init6 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[8], key.0[9], key.0[10], key.0[11]]));
        let init7 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[12], key.0[13], key.0[14], key.0[15]]));
        let init8 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[16], key.0[17], key.0[18], key.0[19]]));
        let init9 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[20], key.0[21], key.0[22], key.0[23]]));
        let init10 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[24], key.0[25], key.0[26], key.0[27]]));
        let init11 = aarch64::vdupq_n_u32(u32::from_le_bytes([key.0[28], key.0[29], key.0[30], key.0[31]]));
        // Block counter is [start, start+1, start+2, start+3].
        let counter_init: [u32; 4] = [
            start_block,
            start_block.wrapping_add(1),
            start_block.wrapping_add(2),
            start_block.wrapping_add(3),
        ];
        let init12 = aarch64::vld1q_u32(counter_init.as_ptr());
        let init13 = aarch64::vdupq_n_u32(u32::from_le_bytes([nonce.0[0], nonce.0[1], nonce.0[2], nonce.0[3]]));
        let init14 = aarch64::vdupq_n_u32(u32::from_le_bytes([nonce.0[4], nonce.0[5], nonce.0[6], nonce.0[7]]));
        let init15 = aarch64::vdupq_n_u32(u32::from_le_bytes([nonce.0[8], nonce.0[9], nonce.0[10], nonce.0[11]]));

        // Working state.
        let (mut w0, mut w1, mut w2, mut w3) = (init0, init1, init2, init3);
        let (mut w4, mut w5, mut w6, mut w7) = (init4, init5, init6, init7);
        let (mut w8, mut w9, mut w10, mut w11) = (init8, init9, init10, init11);
        let (mut w12, mut w13, mut w14, mut w15) = (init12, init13, init14, init15);

        // 20 rounds = 10 column + 10 diagonal.
        for _ in 0..10 {
            // Column round: quarter-rounds on the four state columns.
            quarter_round(&mut w0, &mut w4, &mut w8, &mut w12);
            quarter_round(&mut w1, &mut w5, &mut w9, &mut w13);
            quarter_round(&mut w2, &mut w6, &mut w10, &mut w14);
            quarter_round(&mut w3, &mut w7, &mut w11, &mut w15);
            // Diagonal round: quarter-rounds on the four state diagonals. In the
            // transposed layout, this needs no lane shuffles.
            quarter_round(&mut w0, &mut w5, &mut w10, &mut w15);
            quarter_round(&mut w1, &mut w6, &mut w11, &mut w12);
            quarter_round(&mut w2, &mut w7, &mut w8, &mut w13);
            quarter_round(&mut w3, &mut w4, &mut w9, &mut w14);
        }

        // Add the initial state back in.
        w0 = aarch64::vaddq_u32(w0, init0);
        w1 = aarch64::vaddq_u32(w1, init1);
        w2 = aarch64::vaddq_u32(w2, init2);
        w3 = aarch64::vaddq_u32(w3, init3);
        w4 = aarch64::vaddq_u32(w4, init4);
        w5 = aarch64::vaddq_u32(w5, init5);
        w6 = aarch64::vaddq_u32(w6, init6);
        w7 = aarch64::vaddq_u32(w7, init7);
        w8 = aarch64::vaddq_u32(w8, init8);
        w9 = aarch64::vaddq_u32(w9, init9);
        w10 = aarch64::vaddq_u32(w10, init10);
        w11 = aarch64::vaddq_u32(w11, init11);
        w12 = aarch64::vaddq_u32(w12, init12);
        w13 = aarch64::vaddq_u32(w13, init13);
        w14 = aarch64::vaddq_u32(w14, init14);
        w15 = aarch64::vaddq_u32(w15, init15);

        // Transpose per 4-word group into per-block vectors and XOR into the buffer.
        xor_into(chunk, 0, w0, w1, w2, w3); // bytes  0..16 of each block
        xor_into(chunk, 16, w4, w5, w6, w7); // bytes 16..32
        xor_into(chunk, 32, w8, w9, w10, w11); // bytes 32..48
        xor_into(chunk, 48, w12, w13, w14, w15); // bytes 48..64
    }
}

/// The `ChaCha20` quarter round applied to four independent blocks in parallel.
///
/// Each argument is a `uint32x4_t` holding the same state word across four
/// blocks; the mutation applies the round to all four blocks simultaneously.
#[inline(always)]
unsafe fn quarter_round(
    a: &mut aarch64::uint32x4_t,
    b: &mut aarch64::uint32x4_t,
    c: &mut aarch64::uint32x4_t,
    d: &mut aarch64::uint32x4_t,
) {
    *a = aarch64::vaddq_u32(*a, *b);
    *d = rotl16(aarch64::veorq_u32(*d, *a));

    *c = aarch64::vaddq_u32(*c, *d);
    *b = rotl12(aarch64::veorq_u32(*b, *c));

    *a = aarch64::vaddq_u32(*a, *b);
    *d = rotl8(aarch64::veorq_u32(*d, *a));

    *c = aarch64::vaddq_u32(*c, *d);
    *b = rotl7(aarch64::veorq_u32(*b, *c));
}

/// Rotate left by 16 within each 32-bit lane, via a 16-bit reversal.
#[inline(always)]
unsafe fn rotl16(x: aarch64::uint32x4_t) -> aarch64::uint32x4_t {
    aarch64::vreinterpretq_u32_u16(aarch64::vrev32q_u16(aarch64::vreinterpretq_u16_u32(x)))
}

/// Rotate left by 12 within each 32-bit lane, via shift-left plus
/// shift-right-and-insert.
///
/// ```text
/// Split each 32-bit lane by bit position:
///     H = bits 20..31 of x  (top 12)
///     L = bits  0..19 of x  (low 20)
///
///     x            = |   H   |          L          |
///     rotl(x, 12)  = |          L          |   H   |
///
/// Step 1: vshlq_n_u32::<12>(x)    = |          L          | 0000 0000 000 |
/// Step 2: vsriq_n_u32::<20>(_, x):
///           (x >> 20)             = | 0000 0000 0000 0000 000 |   H   |
///           vsri preserves the top 20 bits of step 1 (= L)
///           and writes the low 12 bits of (x >> 20) (= H)
///           into step 1's low 12 bits.
///           result                = |          L          |   H   |  = rotl(x, 12)
/// ```
#[inline(always)]
unsafe fn rotl12(x: aarch64::uint32x4_t) -> aarch64::uint32x4_t {
    aarch64::vsriq_n_u32::<20>(aarch64::vshlq_n_u32::<12>(x), x)
}

/// Rotate left by 8 within each 32-bit lane, via a byte-permute table.
#[inline(always)]
unsafe fn rotl8(x: aarch64::uint32x4_t) -> aarch64::uint32x4_t {
    let table = aarch64::vld1q_u8(ROT8_TABLE.as_ptr());
    aarch64::vreinterpretq_u32_u8(aarch64::vqtbl1q_u8(aarch64::vreinterpretq_u8_u32(x), table))
}

/// Rotate left by 7 within each 32-bit lane, via shift-left plus
/// shift-right-and-insert.
///
/// Same construction as [`rotl12`], with `(7, 25)` in place of `(12, 20)`:
///
/// ```text
///     H = bits 25..31 of x  (top 7)
///     L = bits  0..24 of x  (low 25)
///
///     x            = | H |             L              |
///     rotl(x, 7)   = |             L              | H |
/// ```
#[inline(always)]
unsafe fn rotl7(x: aarch64::uint32x4_t) -> aarch64::uint32x4_t {
    aarch64::vsriq_n_u32::<25>(aarch64::vshlq_n_u32::<7>(x), x)
}

/// Transpose four "same-word-across-blocks" vectors into four "block words"
/// vectors and XOR each into the corresponding block's 16-byte slot inside
/// `chunk`, at word-group `offset` (one of `{0, 16, 32, 48}`).
///
/// Given `v_i[j]` = word `i` of block `j`, this writes for each block `j`
/// the four consecutive words `[v_0[j], v_1[j], v_2[j], v_3[j]]` into
/// `chunk[j * 64 + offset .. j * 64 + offset + 16]`, XOR'd against the
/// existing contents.
#[inline(always)]
fn xor_into(
    chunk: &mut [u8; 4 * 64],
    offset: usize,
    v0: aarch64::uint32x4_t,
    v1: aarch64::uint32x4_t,
    v2: aarch64::uint32x4_t,
    v3: aarch64::uint32x4_t,
) {
    // SAFETY: NEON is guaranteed on aarch64 (enforced by the module cfg
    // gate). `chunk` is a fixed-size `[u8; 256]` array; callers pass
    // `offset` in `{0, 16, 32, 48}` and `j` iterates `0..4`, so
    // `j * 64 + offset` is at most 240 and the 16-byte access ends at
    // byte 255 — within the array.
    unsafe {
        // Interleave pairs to build partial per-block layouts.
        let a = aarch64::vzip1q_u32(v0, v1); // [b0.w_i, b0.w_{i+1}, b1.w_i, b1.w_{i+1}]
        let b = aarch64::vzip2q_u32(v0, v1); // [b2.w_i, b2.w_{i+1}, b3.w_i, b3.w_{i+1}]
        let c = aarch64::vzip1q_u32(v2, v3); // [b0.w_{i+2}, b0.w_{i+3}, b1.w_{i+2}, b1.w_{i+3}]
        let d = aarch64::vzip2q_u32(v2, v3); // [b2.w_{i+2}, b2.w_{i+3}, b3.w_{i+2}, b3.w_{i+3}]

        // Combine halves so each output vector holds one block's four contiguous words.
        let block0 = aarch64::vcombine_u32(aarch64::vget_low_u32(a), aarch64::vget_low_u32(c));
        let block1 = aarch64::vcombine_u32(aarch64::vget_high_u32(a), aarch64::vget_high_u32(c));
        let block2 = aarch64::vcombine_u32(aarch64::vget_low_u32(b), aarch64::vget_low_u32(d));
        let block3 = aarch64::vcombine_u32(aarch64::vget_high_u32(b), aarch64::vget_high_u32(d));

        // Use the u8 load/store variants so the pointer stays u8 (NEON accepts
        // unaligned memory here, but avoiding the u8->u32 pointer cast keeps
        // clippy quiet without an allow attribute).
        let base = chunk.as_mut_ptr();
        let blocks = [block0, block1, block2, block3];
        for (j, block) in blocks.iter().enumerate() {
            let ptr = base.add(j * 64 + offset);
            let existing = aarch64::vld1q_u8(ptr);
            let block_bytes = aarch64::vreinterpretq_u8_u32(*block);
            let xored = aarch64::veorq_u8(existing, block_bytes);
            aarch64::vst1q_u8(ptr, xored);
        }
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use hex::hex;

    use super::super::{ChaCha20, Key, Nonce};
    use super::apply_4_blocks;

    /// Differential test: NEON's four-block output matches four consecutive
    /// scalar single-block outputs from `ChaCha20::get_keystream` at a
    /// variety of starting block indices.
    ///
    /// This is the specific behavior that a transposed multi-block port is
    /// most likely to get wrong (transpose direction, counter increment,
    /// diagonal round routing).
    #[test]
    fn matches_scalar_across_blocks() {
        let key = Key::new(hex!("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"));
        let nonce = Nonce::new(hex!("000000090000004a00000000"));
        let cipher = ChaCha20::new_from_block(key, nonce, 0);

        for start in [0u32, 1, 42, 1_000, u32::MAX - 3] {
            let mut neon_ks = [0u8; 4 * 64];
            apply_4_blocks(&mut neon_ks, &key, &nonce, start);

            let mut scalar_ks = [0u8; 4 * 64];
            for i in 0u32..4 {
                let ks = cipher.get_keystream(start + i);
                let base = i as usize * 64;
                scalar_ks[base..base + 64].copy_from_slice(&ks);
            }
            assert_eq!(neon_ks, scalar_ks, "mismatch at start_block={}", start);
        }
    }

    /// The XOR path must combine with existing buffer contents (not overwrite).
    #[test]
    fn xor_is_involutive() {
        let key = Key::new(hex!("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"));
        let nonce = Nonce::new(hex!("000000090000004a00000000"));

        let mut buf = [0u8; 4 * 64];
        for (i, byte) in buf.iter_mut().enumerate() {
            // Deterministic non-zero pattern, truncated to u8.
            *byte = (i * 7 + 3) as u8;
        }
        let original = buf;

        apply_4_blocks(&mut buf, &key, &nonce, 5);
        assert_ne!(buf, original);
        apply_4_blocks(&mut buf, &key, &nonce, 5);
        assert_eq!(buf, original);
    }
}
