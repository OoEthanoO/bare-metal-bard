// Kernel 10: raw `mma.sync` PTX, and a shared-memory layout built backwards
// from the hardware's register mapping.
//
// Kernel 9 reached 117% of fp32 cuBLAS with WMMA and then stopped, 21% short of
// cuBLAS's own TF32 path, and five separate attempts to close that from the
// outside all made it slower (they are listed in k09_tensorcore.cu, kept so
// they are not retried). What was left was the abstraction itself, so this
// kernel removes it.
//
// WHAT WMMA HIDES. `wmma::load_matrix_sync` takes a pointer and a leading
// dimension and fills a fragment. Underneath, the m16n8k8 TF32 instruction
// requires each of the 32 lanes to hold four SPECIFIC elements of A:
//
//     g = laneid >> 2 (the "group"),  t = laneid & 3
//     a0 = (g, t)      a1 = (g+8, t)      a2 = (g, t+4)      a3 = (g+8, t+4)
//
// That order is worth stating precisely, because guessing it wrong is silent:
// the kernel compiles, runs at full speed, and returns garbage. The version
// with a1 and a2 swapped -- which is what the analogous f16 shape uses, and
// what I wrote from memory first -- was found by `scripts/probe.bat` on a
// one-hot probe, not by reading the output. A tensor-core fragment layout is
// the kind of fact to MEASURE rather than recall.
//
// Given a row-major tile in shared memory those four elements sit at four
// unrelated addresses, so the load is four LDS.32. No choice of `ldm` fixes it,
// because the mapping is a property of the instruction, not of the pointer.
//
// WHAT THIS KERNEL DOES INSTEAD. Shared memory does not have to hold a matrix.
// It only has to hold whatever makes the next read cheap. So each 16x8 tile of
// A is stored LANE-MAJOR: lane L's four elements at L*4 .. L*4+3, contiguous.
// The fragment load becomes one LDS.128 at `base + laneid*16`, which is also
// the ideal shared-memory access pattern -- 32 lanes covering 512 contiguous
// bytes, no bank conflict possible.
//
//     per warp, per k-step of 8:   WMMA   this
//       shared load instructions     24      6
//       bytes read from shared     3072   3072
//       mma issued                   16     16
//
// Four times fewer instructions to move exactly the same bytes. That framing is
// the whole experiment and it is a fair test: if the gap was instruction issue,
// this closes it; if the gap was the bytes, this changes nothing, and the bytes
// are then all that is left to blame.
//
// THE PRICE. Staging pays for it. A staging thread reads four floats along k,
// and those are four CONSECUTIVE LANES at the same register index, so one
// STS.128 becomes four STS.32. That is unavoidable rather than sloppy: the lane
// index is a function of k%4, so the four values a coalesced global read
// delivers can never belong to one lane. Loads outnumber stores three to one
// here, which is why the trade is worth making at all.
//
// AND THE STORES THEN CONFLICT, which the first working version did not
// anticipate: 8-way on A and 16-way on B, because a warp of staging threads
// varies only in bits the slot index scales by 4. The fix is an XOR swizzle --
// slot ^= f(unit), where f depends only on which unit the element belongs to.
// The store side sees f vary across the warp and spreads; the load side reads
// one whole unit per warp, so f is uniform there and the permutation is
// invisible. 8-way and 16-way become 2-way, and the fragment load stays
// perfectly conflict-free. `tools/smem_banks.py` derives and checks both.
//
// Before the swizzle this kernel ran 8% SLOWER than the WMMA one it was meant
// to beat, on a shared-memory layout whose entire justification was cheaper
// shared-memory access. Fewer instructions moving the same bytes is worth
// nothing if the bytes then arrive four at a time.
//
// ONE MORE THING THAT DID NOT WORK, measured here and kept so it is not
// retried: rounding to TF32 during staging and loading `uint4` in the inner
// loop removes 96 `cvt` per thread per K-chunk and replaces them with 32.
// It costs 4.4% (8536 -> 8158 GF/s at N=4096). Kernel 9 measured the same
// change at -4.6% on a completely different loop structure, so this is not an
// artefact of either: in the compute phase the conversions hide behind
// independent work, and in the staging path, between a global load and a
// barrier, there is nothing to hide behind.
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
// documented at the top is what that phrase actually means.
__device__ __forceinline__ void mma_m16n8k8(float (&d)[4],
                                            const unsigned (&a)[4],
                                            const unsigned *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// --- the two layout maps, which are the actual content of this kernel ---
//
// Both invert the register mapping: given a logical element, say where in
// shared memory it has to sit so the lane that needs it finds it at
// laneid*4 + (its register index).

// Which lane slot inside a unit gets swapped with which. See the comment on
// the swizzle above the kernel; f depends only on the unit, so it is uniform
// across a warp doing a fragment load and cannot disturb it.
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

template <int BM, int BN, int BK, int WM, int WN, int NUM_THREADS, int MINB>
__global__ __launch_bounds__(NUM_THREADS, MINB) void mma_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
    // No padding anywhere. Kernels 6-9 all skew their leading dimension to
    // break bank conflicts; here the layout IS the fragment, so a fragment
    // load is 32 lanes over 512 contiguous bytes and cannot conflict. The skew
    // has nothing left to fix. It also makes this the smallest shared
    // footprint of any kernel past 5: 32 KB against kernel 9's 37 KB.
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WARPSIZE;
    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WMITER = WM / MMA_M;  // 16-row tiles this warp owns
    constexpr int WNITER = WN / 16;     // 16-column (= 2 mma) units it owns

    // Staging maps, identical to kernel 9's so the fragment mechanism is the
    // only variable between the two measurements.
    const int aRow = tid / (BK / 4), aCol = tid % (BK / 4);
    constexpr int aStride = NUM_THREADS / (BK / 4);
    const int bRow = tid / (BN / 4), bCol = tid % (BN / 4);
    constexpr int bStride = NUM_THREADS / (BN / 4);

    float acc[WMITER][WNITER][2][4] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += aStride) {
            const float4 v = reinterpret_cast<const float4 *>(
                &A[(size_t)(cRow + aRow + off) * K + k0 + aCol * 4])[0];
            // One base and one XOR per element, not four calls to the map:
            // the four staged values share a unit and a register index, and
            // differ only in bits the base leaves clear, so `slot ^ j` is
            // exactly `a_off(m, k+j)`. Calling the map four times instead
            // costs 84 bytes of spill. tools/smem_banks.py checks the algebra.
            const int m = aRow + off, k = aCol * 4;
            const int unit = (k / MMA_K) * (BM / MMA_M) + m / MMA_M;
            const int base = unit * UNIT +
                             (((k % MMA_K) / 4) * 2 + ((m % MMA_M) / 8));
            const int slot = ((m % 8) * 4) ^ a_swz<BM>(unit);
            As[base + ((slot ^ 0) * 4)] = v.x;
            As[base + ((slot ^ 1) * 4)] = v.y;
            As[base + ((slot ^ 2) * 4)] = v.z;
            As[base + ((slot ^ 3) * 4)] = v.w;
        }
#pragma unroll
        for (int off = 0; off < BK; off += bStride) {
            const float4 v = reinterpret_cast<const float4 *>(
                &B[(size_t)(k0 + bRow + off) * N + cCol + bCol * 4])[0];
            // Same idea, but n scales the slot by 4 rather than by 1, so the
            // stepped bits are different. This asymmetry is the whole reason
            // the addresses go through a checked map at all.
            const int k = bRow + off, n = bCol * 4;
            const int unit = (k / MMA_K) * (BN / 16) + n / 16;
            const int base = unit * UNIT +
                             (((k % MMA_K) / 4) + ((n % 16) / 8) * 2);
            const int slot = ((n % 8) * 4 + (k % 4)) ^ b_swz<BN>(unit);
            Bs[base + ((slot ^ 0) * 4)] = v.x;
            Bs[base + ((slot ^ 4) * 4)] = v.y;
            Bs[base + ((slot ^ 8) * 4)] = v.z;
            Bs[base + ((slot ^ 12) * 4)] = v.w;
        }
        __syncthreads();

#pragma unroll
        for (int kt = 0; kt < BK / MMA_K; ++kt) {
            unsigned a[WMITER][4], b[WNITER][4];
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                const int unit = kt * (BM / MMA_M) + warpRow * WMITER + i;
                const float4 v = reinterpret_cast<const float4 *>(
                    &As[unit * UNIT + (lane ^ a_swz<BM>(unit)) * 4])[0];
                a[i][0] = to_tf32(v.x);
                a[i][1] = to_tf32(v.y);
                a[i][2] = to_tf32(v.z);
                a[i][3] = to_tf32(v.w);
            }
#pragma unroll
            for (int j = 0; j < WNITER; ++j) {
                const int unit = kt * (BN / 16) + warpCol * WNITER + j;
                const float4 v = reinterpret_cast<const float4 *>(
                    &Bs[unit * UNIT + (lane ^ b_swz<BN>(unit)) * 4])[0];
                b[j][0] = to_tf32(v.x);
                b[j][1] = to_tf32(v.y);
                b[j][2] = to_tf32(v.z);
                b[j][3] = to_tf32(v.w);
            }
#pragma unroll
            for (int i = 0; i < WMITER; ++i)
#pragma unroll
                for (int j = 0; j < WNITER; ++j) {
                    mma_m16n8k8(acc[i][j][0], a[i], &b[j][0]);
                    mma_m16n8k8(acc[i][j][1], a[i], &b[j][2]);
                }
        }
        __syncthreads();
    }

    // Epilogue. The accumulator mapping is the one place the hand-written
    // version is cheaper than WMMA for free: a lane's c0/c1 are ADJACENT
    // columns, so every store is a float2 rather than two scalars.
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < WMITER; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m0 = cRow + warpRow * WM + i * MMA_M;
                const int n0 = cCol + warpCol * WN + j * 16 + h * MMA_N + t * 2;
                float *p0 = &C[(size_t)(m0 + g) * N + n0];
                float *p1 = &C[(size_t)(m0 + g + 8) * N + n0];
                const float *r = acc[i][j][h];
                float2 o0 = reinterpret_cast<float2 *>(p0)[0];
                float2 o1 = reinterpret_cast<float2 *>(p1)[0];
                o0.x = alpha * r[0] + beta * o0.x;
                o0.y = alpha * r[1] + beta * o0.y;
                o1.x = alpha * r[2] + beta * o1.x;
                o1.y = alpha * r[3] + beta * o1.y;
                reinterpret_cast<float2 *>(p0)[0] = o0;
                reinterpret_cast<float2 *>(p1)[0] = o1;
            }
        }
    }
}

void sgemm_mma(int M, int N, int K, float alpha, const float *A, const float *B,
               float beta, float *C) {
    // WHAT THE SWEEP FOUND, and it is not what this kernel was written to
    // test. N=4096, clock pinned to 1200 MHz, median of three (full table via
    // scripts\sweep_k10.bat). `B/mma` is bytes read from shared memory per mma
    // issued -- the reuse each warp gets out of what it loads:
    //
    //     cfg  threads   WM  WN   BK   B/mma  regs  spill    GF/s
    //       0     256    32  64   32     192   128     84    8540   <- k9's shape
    //       1     128    64  64   32     128   255      4    9286   <- chosen
    //       7     128    32 128   32     160   255      4    9246
    //       2     128    64  64   32     128   168    484    8698
    //       3     256    64  64   32     128   254      0    7838   BN=256
    //       6     256    64  64   32     128   255      0    7832   BM=256
    //       4     128    64  64   16     128   128    776    4433
    //       5     128    64  64   64       -     -      -       -   smem overflow
    //
    // Configs 1 and 7 are a TIE, not a ranking: 1 wins at N=4096 by 0.4% and 7
    // wins at N=2048 by 0.6%, both inside the run-to-run spread. Config 1 is
    // the default because it is the one the model predicts, not because the
    // measurement separated them.
    //
    // The honest reading of this table is that the win came from `B/mma` and
    // not from raw `mma.sync`. Moving from config 0 to config 1 -- the same
    // instruction, the same layout, a warp tile twice as tall -- is worth 8.7%,
    // while replacing WMMA with hand-written PTX at config 0's shape was worth
    // 2.7%. The abstraction was never the problem. Shared-memory REUSE was, and
    // the reason kernel 9 could not fix it is that WMMA at this warp tile needs
    // 255 registers and spills, where the hand-written version spills 4 bytes.
    // So raw PTX mattered -- just indirectly, by making the tile affordable.
    //
    //     bytes/mma = 4096 * (WM + WN) / (WM * WN)
    //
    // which is minimised by making BOTH warp-tile dimensions large, and is
    // therefore a register-budget problem in disguise. Configs 3 and 6 test the
    // other lever, block-tile arithmetic intensity against DRAM, and lose 16%:
    // as with kernel 9, this kernel is not DRAM bound however much it looks it.
#ifndef K10_CFG
#define K10_CFG 1
#endif
#if K10_CFG == 0     // the shape kernel 9 was tuned to, for comparison
    constexpr int NUM_THREADS = 256, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 32, WN = 64, MINB = 2;    // 192 bytes/mma
#elif K10_CFG == 1
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 64, WN = 64, MINB = 2;    // 128 bytes/mma
#elif K10_CFG == 2
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 64, WN = 64, MINB = 3;
#elif K10_CFG == 3
    constexpr int NUM_THREADS = 256, BM = 128, BN = 256, BK = 32;
    constexpr int WM = 64, WN = 64, MINB = 1;
#elif K10_CFG == 4
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, MINB = 4;
#elif K10_CFG == 5
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 64;
    constexpr int WM = 64, WN = 64, MINB = 1;
#elif K10_CFG == 6
    constexpr int NUM_THREADS = 256, BM = 256, BN = 128, BK = 32;
    constexpr int WM = 64, WN = 64, MINB = 1;
#elif K10_CFG == 7   // 160 bytes/mma: wide in N only
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 32, WN = 128, MINB = 2;
#endif

    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / WARPSIZE, "warp grid");

    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
        return;
    }

    dim3 grid(N / BN, M / BM);
    mma_kernel<BM, BN, BK, WM, WN, NUM_THREADS, MINB>
        <<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

#else
// Pre-Ampere: TF32 tensor cores do not exist, so this stage of the ladder
// is not built. The entry stays so the registry and the benchmark harness
// do not have to change shape per architecture; it runs the best fp32
// kernel instead, and reports as such.
void sgemm_mma(int M, int N, int K, float alpha, const float *A,
             const float *B, float beta, float *C) {
    sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
}
#endif
