// SPDX-License-Identifier: CC0-1.0

//! `ChaCha20` block function using `x86`/`x86_64` SSE2, processing four blocks in parallel.
//!
//! This module utilizes SIMD over the `__m128i` type provided by the SSE2 intrinsics.
//! The steps are identical to the RFC for processing a single block, however a final
//! matrix transpose is required to properly apply the state to the ciphertext.
//!
//! SSE2 is a baseline feature on `x86_64` (guaranteed by every stock Rust `x86_64-*`
//! target) and is available on `i686` targets configured for Pentium 4 or later, so
//! this backend can be gated at compile time without any runtime dispatch.

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use super::{Key, Nonce, WORD_1, WORD_2, WORD_3, WORD_4};

// The `ChaCha20` quarter round applied to four independent blocks in parallel.
//
// For a single state, the quarter round is described here:
// https://datatracker.ietf.org/doc/html/rfc7539#section-2.1
//
// Each argument is a `__m128i` holding the same state variable across four blocks.
//
// SAFETY: SSE2 intrinsics are gated by feature `sse2`.
#[inline(always)]
unsafe fn quarter_round(
    a: &mut arch::__m128i,
    b: &mut arch::__m128i,
    c: &mut arch::__m128i,
    d: &mut arch::__m128i,
) {
    // `_mm_add_epi32` is a lane-wise 32-bit add.
    // `_mm_xor_si128` is a bitwise exclusive OR over the whole register.
    *a = arch::_mm_add_epi32(*a, *b);
    *d = rotl16(arch::_mm_xor_si128(*d, *a));

    *c = arch::_mm_add_epi32(*c, *d);
    *b = rotl12(arch::_mm_xor_si128(*b, *c));

    *a = arch::_mm_add_epi32(*a, *b);
    *d = rotl8(arch::_mm_xor_si128(*d, *a));

    *c = arch::_mm_add_epi32(*c, *d);
    *b = rotl7(arch::_mm_xor_si128(*b, *c));
}

// Rotate left by 16 within each 32-bit lane.
//
// Each 32-bit lane holds two 16-bit halves; rotating left by 16 is equivalent
// to swapping them. `_mm_shufflelo_epi16` shuffles the four 16-bit words in
// the low 64 bits according to an immediate control, and `_mm_shufflehi_epi16`
// does the same for the high 64 bits. The immediate `0b10_11_00_01` selects
// input words `[1, 0, 3, 2]`, which swaps adjacent 16-bit words and therefore
// rotates each 32-bit lane left by 16.
//
// SAFETY: SSE2 intrinsics are gated by feature `sse2`.
#[inline(always)]
unsafe fn rotl16(x: arch::__m128i) -> arch::__m128i {
    let x = arch::_mm_shufflelo_epi16::<0b10_11_00_01>(x);
    arch::_mm_shufflehi_epi16::<0b10_11_00_01>(x)
}

// Rotate left by 12 within each 32-bit lane, via shift-left OR shift-right.
//
// SSE2 has no 32-bit lane rotate and no equivalent of NEON's "shift right and
// insert", so every rotate that cannot be expressed as a byte or word shuffle
// falls back to `(x << N) | (x >> (32 - N))`.
//
// SAFETY: SSE2 intrinsics are gated by feature `sse2`.
#[inline(always)]
unsafe fn rotl12(x: arch::__m128i) -> arch::__m128i {
    arch::_mm_or_si128(arch::_mm_slli_epi32::<12>(x), arch::_mm_srli_epi32::<20>(x))
}

// Rotate left by 8 within each 32-bit lane, via shift-left OR shift-right.
//
// NEON has a single-op byte-permute (`vqtbl1q_u8`) that expresses this rotate
// cheaply, but SSE2 lacks `pshufb` (that lives in SSSE3), so we fall back to
// the generic shift-and-or form.
//
// SAFETY: SSE2 intrinsics are gated by feature `sse2`.
#[inline(always)]
unsafe fn rotl8(x: arch::__m128i) -> arch::__m128i {
    arch::_mm_or_si128(arch::_mm_slli_epi32::<8>(x), arch::_mm_srli_epi32::<24>(x))
}

// Rotate left by 7 within each 32-bit lane, via shift-left OR shift-right.
//
// SAFETY: SSE2 intrinsics are gated by feature `sse2`.
#[inline(always)]
unsafe fn rotl7(x: arch::__m128i) -> arch::__m128i {
    arch::_mm_or_si128(arch::_mm_slli_epi32::<7>(x), arch::_mm_srli_epi32::<25>(x))
}

// XOR four consecutive `ChaCha20` blocks of keystream into `chunk`, starting
// at `start_block`.
#[inline]
#[allow(clippy::too_many_lines)]
pub(super) fn apply_4_blocks(chunk: &mut [u8; 4 * 64], key: &Key, nonce: &Nonce, start_block: u32) {
    // SAFETY: SSE2 intrinsics are gated by feature `sse2`.
    unsafe {
        // Initialize length 4 vectors of 32-bit values.
        //
        // `_mm_set1_epi32` broadcasts a 32-bit value to every lane of a `__m128i`.
        // Each `init*` is one word of the ChaCha state replicated across four
        // parallel blocks: https://datatracker.ietf.org/doc/html/rfc7539#section-2.3
        let init0 = arch::_mm_set1_epi32(WORD_1 as i32);
        let init1 = arch::_mm_set1_epi32(WORD_2 as i32);
        let init2 = arch::_mm_set1_epi32(WORD_3 as i32);
        let init3 = arch::_mm_set1_epi32(WORD_4 as i32);
        let init4 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[0], key.0[1], key.0[2], key.0[3]]) as i32
            );
        let init5 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[4], key.0[5], key.0[6], key.0[7]]) as i32
            );
        let init6 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[8], key.0[9], key.0[10], key.0[11]]) as i32
            );
        let init7 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[12], key.0[13], key.0[14], key.0[15]]) as i32
            );
        let init8 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[16], key.0[17], key.0[18], key.0[19]]) as i32
            );
        let init9 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[20], key.0[21], key.0[22], key.0[23]]) as i32
            );
        let init10 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[24], key.0[25], key.0[26], key.0[27]]) as i32
            );
        let init11 =
            arch::_mm_set1_epi32(
                u32::from_le_bytes([key.0[28], key.0[29], key.0[30], key.0[31]]) as i32
            );
        // The only state value that varies between blocks is the block counter.
        // We set the four lanes to [start, start+1, start+2, start+3].
        //
        // `_mm_setr_epi32` sets lanes in argument order (lane 0 first), unlike
        // `_mm_set_epi32` which reverses them.
        let init12 = arch::_mm_setr_epi32(
            start_block as i32,
            start_block.wrapping_add(1) as i32,
            start_block.wrapping_add(2) as i32,
            start_block.wrapping_add(3) as i32,
        );
        let init13 = arch::_mm_set1_epi32(u32::from_le_bytes([
            nonce.0[0], nonce.0[1], nonce.0[2], nonce.0[3],
        ]) as i32);
        let init14 = arch::_mm_set1_epi32(u32::from_le_bytes([
            nonce.0[4], nonce.0[5], nonce.0[6], nonce.0[7],
        ]) as i32);
        let init15 = arch::_mm_set1_epi32(u32::from_le_bytes([
            nonce.0[8],
            nonce.0[9],
            nonce.0[10],
            nonce.0[11],
        ]) as i32);

        // The working state that will be mutated by the quarter rounds.
        let (mut w0, mut w1, mut w2, mut w3) = (init0, init1, init2, init3);
        let (mut w4, mut w5, mut w6, mut w7) = (init4, init5, init6, init7);
        let (mut w8, mut w9, mut w10, mut w11) = (init8, init9, init10, init11);
        let (mut w12, mut w13, mut w14, mut w15) = (init12, init13, init14, init15);

        // Apply the column and diagonal rounds.
        // https://datatracker.ietf.org/doc/html/rfc7539#section-2.3
        for _ in 0..10 {
            // Column round.
            quarter_round(&mut w0, &mut w4, &mut w8, &mut w12);
            quarter_round(&mut w1, &mut w5, &mut w9, &mut w13);
            quarter_round(&mut w2, &mut w6, &mut w10, &mut w14);
            quarter_round(&mut w3, &mut w7, &mut w11, &mut w15);
            // Diagonal round.
            quarter_round(&mut w0, &mut w5, &mut w10, &mut w15);
            quarter_round(&mut w1, &mut w6, &mut w11, &mut w12);
            quarter_round(&mut w2, &mut w7, &mut w8, &mut w13);
            quarter_round(&mut w3, &mut w4, &mut w9, &mut w14);
        }

        // "At the end of 20 rounds [...] we add the original input words to the
        // output words." https://datatracker.ietf.org/doc/html/rfc7539#section-2.3
        w0 = arch::_mm_add_epi32(w0, init0);
        w1 = arch::_mm_add_epi32(w1, init1);
        w2 = arch::_mm_add_epi32(w2, init2);
        w3 = arch::_mm_add_epi32(w3, init3);
        w4 = arch::_mm_add_epi32(w4, init4);
        w5 = arch::_mm_add_epi32(w5, init5);
        w6 = arch::_mm_add_epi32(w6, init6);
        w7 = arch::_mm_add_epi32(w7, init7);
        w8 = arch::_mm_add_epi32(w8, init8);
        w9 = arch::_mm_add_epi32(w9, init9);
        w10 = arch::_mm_add_epi32(w10, init10);
        w11 = arch::_mm_add_epi32(w11, init11);
        w12 = arch::_mm_add_epi32(w12, init12);
        w13 = arch::_mm_add_epi32(w13, init13);
        w14 = arch::_mm_add_epi32(w14, init14);
        w15 = arch::_mm_add_epi32(w15, init15);

        xor_into(chunk, 0, w0, w1, w2, w3); // bytes  0..16 of each block
        xor_into(chunk, 16, w4, w5, w6, w7); // bytes 16..32
        xor_into(chunk, 32, w8, w9, w10, w11); // bytes 32..48
        xor_into(chunk, 48, w12, w13, w14, w15); // bytes 48..64
    }
}

// This function handles the final step of applying the keystream to the
// ciphertext as outlined in the RFC: https://datatracker.ietf.org/doc/html/rfc7539#section-2.4.1
//
// The memory layout is not yet in the correct format when using the vector
// representations, so there is an additional transpose step.
//
// Suppose a ChaCha state is made up of words [w0, ..., w15]. We are processing
// four blocks, [b0, b1, b2, b3].
//
// In the current state, each `__m128i` contains the word at an index `i`:
// [b0.w_i, b1.w_i, b2.w_i, b3.w_i]. We want [b0.w0, b0.w1, b0.w2, b0.w3] so we
// can XOR into the ciphertext.
//
// SAFETY: SSE2 intrinsics are gated by feature `sse2`.
#[inline(always)]
// `_mm_loadu_si128` / `_mm_storeu_si128` are the unaligned variants, so the
// cast from `*mut u8` to `*mut __m128i` does not actually require 16-byte
// alignment.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn xor_into(
    chunk: &mut [u8; 4 * 64],
    offset: usize,
    w0: arch::__m128i,
    w1: arch::__m128i,
    w2: arch::__m128i,
    w3: arch::__m128i,
) {
    // `_mm_unpacklo_epi32` interleaves the two low 32-bit lanes of its inputs:
    // [a, b, c, d] and [e, f, g, h] become [a, e, b, f].
    //
    // Here, it transforms
    // [b0.w0, b1.w0, b2.w0, b3.w0] and
    // [b0.w1, b1.w1, b2.w1, b3.w1]
    // into [b0.w0, b0.w1, b1.w0, b1.w1].
    let a = arch::_mm_unpacklo_epi32(w0, w1);
    // `_mm_unpackhi_epi32` is similar but takes the two high 32-bit lanes,
    // leaving [b2.w0, b2.w1, b3.w0, b3.w1].
    let b = arch::_mm_unpackhi_epi32(w0, w1);
    let c = arch::_mm_unpacklo_epi32(w2, w3);
    let d = arch::_mm_unpackhi_epi32(w2, w3);
    // `_mm_unpacklo_epi64` concatenates the low 64 bits of each input.
    // Combining the low 64 of `a` ([b0.w0, b0.w1]) with the low 64 of `c`
    // ([b0.w2, b0.w3]) yields exactly [b0.w0, b0.w1, b0.w2, b0.w3].
    let words0 = arch::_mm_unpacklo_epi64(a, c);
    let words1 = arch::_mm_unpackhi_epi64(a, c);
    let words2 = arch::_mm_unpacklo_epi64(b, d);
    let words3 = arch::_mm_unpackhi_epi64(b, d);

    let base = chunk.as_mut_ptr();
    let words = [words0, words1, words2, words3];
    for (j, keystream) in words.iter().enumerate() {
        // `chunk` is a `&mut [u8; 256]` with 1-byte alignment, so we must use
        // the unaligned load/store variants.
        let ptr = base.add(j * 64 + offset).cast::<arch::__m128i>();
        let plaintext = arch::_mm_loadu_si128(ptr);
        let xored = arch::_mm_xor_si128(plaintext, *keystream);
        arch::_mm_storeu_si128(ptr, xored);
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use hex::hex;

    use super::super::{ChaCha20, Key, Nonce};
    use super::apply_4_blocks;

    #[test]
    fn matches_single_block_processing() {
        let key =
            Key::new(hex!("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"));
        let nonce = Nonce::new(hex!("000000090000004a00000000"));
        let cipher = ChaCha20::new_from_block(key, nonce, 0);

        // Compares the SSE2 4-block processing with the single block processing.
        for start in [0u32, 1, 42, 1_000, u32::MAX - 3] {
            let mut sse2_ks = [0u8; 4 * 64];
            apply_4_blocks(&mut sse2_ks, &key, &nonce, start);

            let mut scalar_ks = [0u8; 4 * 64];
            for i in 0u32..4 {
                let ks = cipher.get_keystream(start + i);
                let base = i as usize * 64;
                scalar_ks[base..base + 64].copy_from_slice(&ks);
            }
            assert_eq!(sse2_ks, scalar_ks, "mismatch at start_block={}", start);
        }
    }
}
