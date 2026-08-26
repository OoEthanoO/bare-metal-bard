#pragma once
// The lane-major shared-memory layout, shared by every kernel that issues raw
// `mma.sync`. Kernel 10 derived it; kernel 11 reuses it unchanged, which is the
// point -- the two kernels differ only in HOW the bytes get into shared memory,
// so the layout has to be literally the same code and not a second copy that
// drifts.
//
// THE REGISTER MAPPING. The m16n8k8 TF32 instruction requires each of the 32
// lanes to hold four SPECIFIC elements of A:
//
//     g = laneid >> 2 (the "group"),  t = laneid & 3
//     a0 = (g, t)      a1 = (g+8, t)      a2 = (g, t+4)      a3 = (g+8, t+4)
//
// That order is worth stating precisely, because guessing it wrong is silent:
// the kernel compiles, runs at full speed, and returns garbage. The version
// with a1 and a2 swapped -- which is what the analogous f16 shape uses -- was
// found by `scripts/probe.bat` on a one-hot probe, not by reading the output.
// A tensor-core fragment layout is the kind of fact to MEASURE rather than
// recall.
//
// Given a row-major tile in shared memory those four elements sit at four
// unrelated addresses, so the load is four LDS.32. No choice of `ldm` fixes it,
// because the mapping is a property of the instruction, not of the pointer.
// So shared memory does not hold a matrix here. Each 16x8 tile of A is stored
// LANE-MAJOR: lane L's four elements at L*4 .. L*4+3, contiguous. The fragment
// load becomes one LDS.128 at `base + laneid*16`.
#include "../kernels.h"
#if BMB_TF32

namespace {
constexpr int WARPSIZE = 32;

// The native TF32 tensor-core shape on sm_80+. WMMA's m16n16k8 is two of
// these; using them directly costs nothing and makes N granular at 8.
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;

// One "unit" of shared memory is one operand tile for one mma: 16x8 for A,
// 8x16 for B (two mma's worth of B, so its load is 128-bit too). Either way
// 128 floats = 32 lanes x 4 = 512 bytes.
constexpr int UNIT = 128;

// Round-to-nearest-even into TF32. The tensor core reads only the top 19 bits
// of the register, so this is where the 13 mantissa bits are actually given up.
__device__ __forceinline__ unsigned to_tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

// D = A*B + D, one 16x8x8 TF32 matrix multiply issued by the whole warp.
// `.row.col` is the only layout the TF32 shape supports; the register mapping
// documented above is what that phrase actually means.
__device__ __forceinline__ void mma_m16n8k8(float (&d)[4],
                                            const unsigned (&a)[4],
                                            const unsigned *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// --- the two layout maps, which are the actual content of this file ---
//
// Both invert the register mapping: given a logical element, say where in
// shared memory it has to sit so the lane that needs it finds it at
// laneid*4 + (its register index).
//
// THE SWIZZLE. Without it the STORE side conflicts 8-way on A and 16-way on B,
// because a warp of staging threads varies only in bits the slot index scales
// by 4. The fix is `slot ^= f(unit)`, where f depends only on which unit the
// element belongs to. The store side sees f vary across the warp and spreads;
// the load side reads one whole unit per warp, so f is uniform there and the
// permutation is invisible. `tools/smem_banks.py` derives and checks both.
template <int BM>
__device__ __forceinline__ int a_swz(int unit) { return (unit / (BM / MMA_M)) & 7; }
template <int BN>
__device__ __forceinline__ int b_swz(int unit) { return unit % (BN / 16); }

// A element (m, k) within a BM x BK block tile.
template <int BM>
__device__ __forceinline__ int a_off(int m, int k) {
    const int unit = (k / MMA_K) * (BM / MMA_M) + m / MMA_M;
    const int slot = ((m % 8) * 4 + (k % 4)) ^ a_swz<BM>(unit);  // owning lane
    const int elem = ((k % MMA_K) / 4) * 2 + ((m % MMA_M) / 8);  // a0..a3
    return unit * UNIT + slot * 4 + elem;
}

// B element (k, n) within a BK x BN block tile. Registers b0,b1 of the n8 tile
// containing n come first, then b0,b1 of its neighbour, so a lane's four
// floats span two mma's worth of B and load as one float4.
template <int BN>
__device__ __forceinline__ int b_off(int k, int n) {
    const int unit = (k / MMA_K) * (BN / 16) + n / 16;
    const int slot = ((n % 8) * 4 + (k % 4)) ^ b_swz<BN>(unit);
    const int elem = ((k % MMA_K) / 4) + ((n % 16) / 8) * 2;
    return unit * UNIT + slot * 4 + elem;
}
}  // namespace

#endif  // BMB_TF32
