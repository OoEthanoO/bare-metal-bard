// Kernel 7: warp-level tiling.
//
// Profiling kernel 6 (ncu, N=1024) gave the diagnosis:
//     DRAM throughput          15.1%   <- global memory is a non-issue now
//     L1/TEX cache throughput  81.4%   <- saturated
//     Compute (SM) throughput  54.2%
// Shared memory and L1 share the LSU/MIO datapath, so an 81% L1 figure with
// 15% DRAM means the kernel is bound on SMEM->register traffic. The FMA pipes
// are starved waiting for operands to arrive from shared memory.
//
// Kernel 6 spends, per k step: TM+TN = 16 SMEM loads to feed TM*TN = 64 FMAs,
// a ratio of 0.25 loads per FMA. To beat it we need each loaded value to feed
// more math, which means a bigger per-thread tile -- but registers are already
// the occupancy limiter, so we cannot simply raise TM and TN.
//
// Warp tiling gets the reuse without the register blowup by inserting a level
// of blocking between the block and the thread. The block tile (BM x BN) is
// split among warps (WM x WN each), and a warp's 32 threads cover their tile
// in WMITER x WNITER strided sub-steps rather than one contiguous patch.
//
//     per k step: WMITER*TM + WNITER*TN = 8 + 16 = 24 loads
//                 WMITER*WNITER*TM*TN   = 128 FMAs
//                 -> 0.1875 loads per FMA, a 25% cut from kernel 6.
//
// Why the warp is the right granularity: the 32 threads of a warp issue in
// lockstep, so their SMEM requests coalesce into the same bank transactions.
// Arranging a warp to cover a contiguous 64x64 region means its loads are
// broadcast- and conflict-friendly by construction, which a block-level
// arrangement only achieves by accident.
//
// BK also doubles to 16, halving the number of __syncthreads() per unit of K
// and giving the scheduler longer runs of independent FMAs to hide latency.
#include "../kernels.h"

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

namespace {
constexpr int WARPSIZE = 32;
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void warptile_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
    __shared__ float As[BK * BM];  // transposed, As[k][m]
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // --- warp placement within the block tile ---
    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    // How many strided sub-tiles each warp walks, and their size. WMITER is
    // derived, not chosen: it is whatever makes the arithmetic close.
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;   // 64
    constexpr int WSUBN = WN / WNITER;   // 16

    // --- thread placement within one warp sub-tile ---
    // (WSUBM/TM) x (WSUBN/TN) = 8 x 4 = 32 threads. Lane index varies fastest
    // along the N direction so that consecutive lanes read consecutive floats
    // of Bs.
    const int laneIdx = tid % WARPSIZE;
    const int threadColInWarp = laneIdx % (WSUBN / TN);  // 0..3
    const int threadRowInWarp = laneIdx / (WSUBN / TN);  // 0..7

    // --- staging index maps (chosen for coalescing, independent of the above) ---
    const int innerRowA = tid / (BK / 4);
    const int innerColA = tid % (BK / 4);
    constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
    const int innerRowB = tid / (BN / 4);
    const int innerColB = tid % (BN / 4);
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);

    float acc[WMITER * TM][WNITER * TN] = {{0.0f}};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += rowStrideA) {
            const float4 a = CVEC4(
                A[(size_t)(cRow + innerRowA + off) * K + k0 + innerColA * 4]);
            // Transpose into SMEM: four scalar stores now, in exchange for
            // contiguous float4 register reads on every inner iteration.
            As[(innerColA * 4 + 0) * BM + innerRowA + off] = a.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + off] = a.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + off] = a.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + off] = a.w;
        }
#pragma unroll
        for (int off = 0; off < BK; off += rowStrideB) {
            VEC4(Bs[(innerRowB + off) * BN + innerColB * 4]) = CVEC4(
                B[(size_t)(k0 + innerRowB + off) * N + cCol + innerColB * 4]);
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Fetch this thread's operand slices once...
#pragma unroll
            for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
                for (int i = 0; i < TM; i += 4) {
                    VEC4(regM[wSubRow * TM + i]) =
                        CVEC4(As[k * BM + warpRow * WM + wSubRow * WSUBM +
                                 threadRowInWarp * TM + i]);
                }
            }
#pragma unroll
            for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    VEC4(regN[wSubCol * TN + j]) =
                        CVEC4(Bs[k * BN + warpCol * WN + wSubCol * WSUBN +
                                 threadColInWarp * TN + j]);
                }
            }
            // ...then spend them on WMITER*WNITER*TM*TN FMAs.
#pragma unroll
            for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
#pragma unroll
                for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            acc[wSubRow * TM + i][wSubCol * TN + j] +=
                                regM[wSubRow * TM + i] * regN[wSubCol * TN + j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
        for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const size_t r = (size_t)(cRow + warpRow * WM + wSubRow * WSUBM +
                                          threadRowInWarp * TM + i);
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const int c = cCol + warpCol * WN + wSubCol * WSUBN +
                                  threadColInWarp * TN + j;
                    float4 old = CVEC4(C[r * N + c]);
                    old.x = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 0] + beta * old.x;
                    old.y = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 1] + beta * old.y;
                    old.z = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 2] + beta * old.z;
                    old.w = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 3] + beta * old.w;
                    VEC4(C[r * N + c]) = old;
                }
            }
        }
    }
}

void sgemm_warptile(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C) {
    constexpr int NUM_THREADS = 128;
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, WNITER = 4;
    constexpr int TM = 8, TN = 4;

    // Consistency checks the compiler can prove, so a bad retune fails to
    // build instead of silently computing the wrong thing.
    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / WARPSIZE,
                  "warp grid must exactly cover the block tile");
    static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0,
                  "warp tile must divide evenly into thread tiles");
    static_assert((NUM_THREADS * 4) % BK == 0, "A staging must tile evenly");
    static_assert((NUM_THREADS * 4) % BN == 0, "B staging must tile evenly");

    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
        return;
    }

    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    warptile_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
