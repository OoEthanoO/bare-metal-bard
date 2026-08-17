// Kernel 9: tensor cores, via WMMA with TF32.
//
// Kernels 1-8 all compute on the SM's fp32 FMA pipes, and kernel 8 gets within
// a few percent of what cuBLAS achieves on those same pipes. But the card has a
// second, much wider set of matrix units sitting completely idle. Measured on
// this GPU (`./bench/sgemm --tf32`):
//
//     cuBLAS SGEMM, true fp32      7071 GF/s
//     cuBLAS SGEMM, TF32 enabled  10560 GF/s
//
// That ~1.5x is not better scheduling, it is different silicon.
//
// WHAT TF32 ACTUALLY IS, because the name is misleading: it is a 19-bit format
// with fp32's 8-bit exponent but only 10 mantissa bits -- the same range as
// fp32, the precision of fp16. Inputs are rounded to TF32 for the multiply;
// accumulation stays in full fp32. So this kernel is not a faster way to get
// the previous answer, it is a deliberately lower-precision answer that happens
// to be good enough for training neural networks. That is why it carries its
// own, looser tolerance in the registry rather than sharing the fp32 bar.
//
// WHY WMMA AND NOT RAW mma.sync: the warp-level `mma.sync` PTX instruction is
// faster still, but it requires hand-placing each operand in a specific lane
// and register, and the fragment layouts differ per shape. WMMA expresses the
// same hardware through a documented fragment abstraction. For a first contact
// with tensor cores the arithmetic is what matters, and WMMA gets there without
// a page of lane-mapping tables.
//
// The tile structure is the same idea as kernel 7: a block tile split among
// warps, each warp walking its sub-tile. What changes is the innermost step --
// instead of each thread issuing scalar FMAs over its own register tile, the
// whole warp cooperatively issues one 16x16x8 matrix multiply.
#include "../kernels.h"
#include <mma.h>

using namespace nvcuda;

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

namespace {
constexpr int WARPSIZE = 32;

// The WMMA shape available for TF32 on sm_80+.
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
}

template <int BM, int BN, int BK, int WM, int WN, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void tensorcore_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
    // Leading dimensions are padded by 8 floats. Without the skew every row of
    // a tile starts at the same shared-memory bank, and WMMA's fragment loads
    // -- which read a column of lanes across consecutive rows -- would all
    // collide. 8 keeps ldm a multiple of 8, which TF32 WMMA requires.
    constexpr int LDA = BK + 8;
    constexpr int LDB = BN + 8;
    __shared__ float As[BM * LDA];
    __shared__ float Bs[BK * LDB];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    // How many 16x16 output tiles this warp owns.
    constexpr int WMITER = WM / WMMA_M;
    constexpr int WNITER = WN / WMMA_N;

    // Staging maps, same reasoning as earlier kernels: consecutive threads
    // read consecutive floats so the global loads stay coalesced.
    const int aRow = tid / (BK / 4);
    const int aCol = tid % (BK / 4);
    constexpr int aStride = NUM_THREADS / (BK / 4);
    const int bRow = tid / (BN / 4);
    const int bCol = tid % (BN / 4);
    constexpr int bStride = NUM_THREADS / (BN / 4);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                   wmma::row_major> aFrag[WMITER];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                   wmma::row_major> bFrag[WNITER];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WMITER][WNITER];

#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Note there is no transpose here. Kernels 6-8 stored As transposed so
        // that each thread's register reads were contiguous; WMMA loads whole
        // fragments and handles the internal layout itself, so the natural
        // row-major staging is what it wants.
#pragma unroll
        for (int off = 0; off < BM; off += aStride) {
            VEC4(As[(aRow + off) * LDA + aCol * 4]) =
                CVEC4(A[(size_t)(cRow + aRow + off) * K + k0 + aCol * 4]);
        }
#pragma unroll
        for (int off = 0; off < BK; off += bStride) {
            VEC4(Bs[(bRow + off) * LDB + bCol * 4]) =
                CVEC4(B[(size_t)(k0 + bRow + off) * N + cCol + bCol * 4]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                wmma::load_matrix_sync(
                    aFrag[i], &As[(warpRow * WM + i * WMMA_M) * LDA + kk], LDA);
                // TF32 fragments are loaded as fp32 and rounded in place; this
                // is the step that actually gives up the mantissa bits.
#pragma unroll
                for (int t = 0; t < aFrag[i].num_elements; ++t)
                    aFrag[i].x[t] = wmma::__float_to_tf32(aFrag[i].x[t]);
            }
#pragma unroll
            for (int j = 0; j < WNITER; ++j) {
                wmma::load_matrix_sync(
                    bFrag[j], &Bs[kk * LDB + warpCol * WN + j * WMMA_N], LDB);
#pragma unroll
                for (int t = 0; t < bFrag[j].num_elements; ++t)
                    bFrag[j].x[t] = wmma::__float_to_tf32(bFrag[j].x[t]);
            }
            // One instruction per (i,j) drives a whole 16x16x8 multiply-
            // accumulate across the warp. Accumulation is in fp32.
#pragma unroll
            for (int i = 0; i < WMITER; ++i)
#pragma unroll
                for (int j = 0; j < WNITER; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: read C as an accumulator fragment so alpha/beta can be applied
    // elementwise. Both fragments use the same lane mapping, so x[t] in one
    // corresponds to x[t] in the other without knowing what that mapping is.
#pragma unroll
    for (int i = 0; i < WMITER; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
            float *cptr = C + (size_t)(cRow + warpRow * WM + i * WMMA_M) * N +
                          cCol + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
            wmma::load_matrix_sync(cFrag, cptr, N, wmma::mem_row_major);
#pragma unroll
            for (int t = 0; t < cFrag.num_elements; ++t)
                cFrag.x[t] = alpha * acc[i][j].x[t] + beta * cFrag.x[t];
            wmma::store_matrix_sync(cptr, cFrag, N, wmma::mem_row_major);
        }
    }
}

void sgemm_tensorcore(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    // Tuned empirically; the differences are large and were not obvious up
    // front. Measured at N=4096, fp32 cuBLAS = ~7075 GF/s:
    //
    //   threads  WM  WN   registers   GF/s
    //     128    64  64      255      7783   register-starved: 255 is the
    //                                        hardware ceiling per thread
    //     256    64  32      130      6418
    //     256    32  64      128      8122   <- chosen
    //     512    32  32      114      6702
    //
    // The 128-thread version asks each thread to hold 16 accumulator fragments
    // (4x4 x 8 floats = 128 registers before anything else) and hits the 255
    // limit, which caps blocks per SM. Spreading the same block tile over 8
    // warps halves the per-thread accumulator count.
    //
    // WM=32/WN=64 beating WM=64/WN=32 by 27% at identical register counts and
    // identical mma_sync counts is the less obvious half. The two differ only
    // in how warps are arranged over the block tile, which changes how their
    // fragment loads land in shared memory: with WN=64 there are 2 warp
    // columns and 4 warp rows, so the B fragments -- the larger of the two
    // operands here -- are shared across more warps and re-read from fewer
    // distinct SMEM regions.
    constexpr int NUM_THREADS = 256;  // 8 warps
    constexpr int BM = 128, BN = 128, BK = 32;
    constexpr int WM = 32, WN = 64;   // 4 warp rows x 2 warp cols

    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / WARPSIZE, "warp grid");
    static_assert(BM % WMMA_M == 0 && BN % WMMA_N == 0 && BK % WMMA_K == 0,
                  "block tile must be a whole number of WMMA tiles");

    // WMMA needs the global leading dimension to be a multiple of 8 for TF32;
    // N % BN == 0 with BN = 128 already guarantees it.
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
        return;
    }

    dim3 grid(N / BN, M / BM);
    tensorcore_kernel<BM, BN, BK, WM, WN, NUM_THREADS>
        <<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}
