// Transpose-aware GEMM built on the kernel-7 warp-tiling structure.
//
// The key observation that makes this cheap: the fast kernel already stores
// its A tile TRANSPOSED in shared memory (As[k][m]), because that is what
// makes the per-thread register reads contiguous. So the four transpose cases
// are not four algorithms -- they are four index maps feeding the same tile.
//
//   staging A, transA=false: A is M x K, k contiguous.
//       load a float4 along k, SCATTER it across 4 rows of As.
//   staging A, transA=true:  A is K x M, m contiguous.
//       load a float4 along m, COPY it straight into As. (cheaper!)
//
//   staging B, transB=false: B is K x N, n contiguous.
//       load a float4 along n, COPY straight into Bs.
//   staging B, transB=true:  B is N x K, k contiguous.
//       load a float4 along k, SCATTER across 4 rows of Bs.
//
// In every case the global read runs along the contiguous axis of whatever
// layout the operand actually has, so all four variants stay coalesced. The
// transpose costs only the choice between a vector store and four scalar
// stores into shared memory.
#include "gemm.h"
#include "gelu.cuh"
// Tensor cores here are TF32, which is Ampere and newer. BMB_TF32 is set by
// the build when the target arch is sm_80+; without it the whole tensor-core
// half of this file compiles out and every GEMM takes the fp32 path. That is
// not a hypothetical: the brief this project came from suggests a free Colab
// T4, which is sm_75, and until now the repo would not build there at all.
#if BMB_TF32
#include <mma.h>
using namespace nvcuda;
#endif
#include <type_traits>
#include "kernels.h"
#include <cstdio>

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

namespace {
constexpr int WARPSIZE = 32;

// Which epilogue features a given instantiation compiles in. This is a
// TEMPLATE parameter rather than a runtime check on the pointers, and that is
// not premature tidiness: with runtime `if (ep.gelu_out)`, the tanh expansion
// sits in every GEMM whether it is used or not, and the fp32 kernel measured
// 6% slower for carrying code it never executed. `if constexpr` makes the
// unused branches -- and their instruction footprint -- actually disappear.
namespace epi {
constexpr int BIAS = 1, ADD = 2, GELU = 4;
}

// ---------------------------------------------------------------- fast path
template <bool TA, bool TB, int BM, int BN, int BK, int WM, int WN, int WNITER,
          int TM, int TN, int NUM_THREADS, int EPI>
__global__ __launch_bounds__(NUM_THREADS) void gemm_fast(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C, GemmEpilogue ep) {
    // Double buffered, exactly as kernel 8 is. This GEMM was written against
    // the kernel-7 structure and kernel 8 arrived afterwards, so the model
    // never got the prefetch -- and measured at the shapes the model actually
    // runs (see --mnk in bench), kernel 8 beats kernel 7 on every one of them.
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    const int laneIdx = tid % WARPSIZE;
    const int threadColInWarp = laneIdx % (WSUBN / TN);
    const int threadRowInWarp = laneIdx / (WSUBN / TN);

    float acc[WMITER * TM][WNITER * TN] = {{0.0f}};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Staging index maps. Each is chosen so consecutive threads read
    // consecutive addresses in whichever layout the operand has.
    const int aRow = TA ? tid / (BM / 4) : tid / (BK / 4);
    const int aCol = TA ? tid % (BM / 4) : tid % (BK / 4);
    constexpr int aStride = TA ? NUM_THREADS / (BM / 4) : (NUM_THREADS * 4) / BK;
    const int bRow = TB ? tid / (BK / 4) : tid / (BN / 4);
    const int bCol = TB ? tid % (BK / 4) : tid % (BN / 4);
    constexpr int bStride = TB ? (NUM_THREADS * 4) / BK : NUM_THREADS / (BN / 4);

    // How many float4s each thread stages per tile, per operand. The count
    // differs between the transpose cases because that decides which axis of
    // the operand is walked.
    constexpr int NA = TA ? BK / aStride : BM / aStride;
    constexpr int NB = TB ? BN / bStride : BK / bStride;
    float4 aPre[NA], bPre[NB];

    // ---- prologue: chunk 0 goes straight into buffer 0 ----
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int off = i * aStride;
        if constexpr (TA) {
            VEC4(As[0][(aRow + off) * BM + aCol * 4]) =
                CVEC4(A[(size_t)(aRow + off) * M + cRow + aCol * 4]);
        } else {
            const float4 a = CVEC4(A[(size_t)(cRow + aRow + off) * K + aCol * 4]);
            As[0][(aCol * 4 + 0) * BM + aRow + off] = a.x;
            As[0][(aCol * 4 + 1) * BM + aRow + off] = a.y;
            As[0][(aCol * 4 + 2) * BM + aRow + off] = a.z;
            As[0][(aCol * 4 + 3) * BM + aRow + off] = a.w;
        }
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int off = i * bStride;
        if constexpr (TB) {
            const float4 b = CVEC4(B[(size_t)(cCol + bRow + off) * K + bCol * 4]);
            Bs[0][(bCol * 4 + 0) * BN + bRow + off] = b.x;
            Bs[0][(bCol * 4 + 1) * BN + bRow + off] = b.y;
            Bs[0][(bCol * 4 + 2) * BN + bRow + off] = b.z;
            Bs[0][(bCol * 4 + 3) * BN + bRow + off] = b.w;
        } else {
            VEC4(Bs[0][(bRow + off) * BN + bCol * 4]) =
                CVEC4(B[(size_t)(bRow + off) * N + cCol + bCol * 4]);
        }
    }
    __syncthreads();

    int cur = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int knext = k0 + BK;
        const bool more = knext < K;

        // Issue the next chunk of global loads BEFORE the arithmetic below, so
        // the round trip overlaps the FMAs instead of stalling in front of them.
        if (more) {
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int off = i * aStride;
                if constexpr (TA)
                    aPre[i] = CVEC4(A[(size_t)(knext + aRow + off) * M + cRow + aCol * 4]);
                else
                    aPre[i] = CVEC4(A[(size_t)(cRow + aRow + off) * K + knext + aCol * 4]);
            }
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                const int off = i * bStride;
                if constexpr (TB)
                    bPre[i] = CVEC4(B[(size_t)(cCol + bRow + off) * K + knext + bCol * 4]);
                else
                    bPre[i] = CVEC4(B[(size_t)(knext + bRow + off) * N + cCol + bCol * 4]);
            }
        }

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
#pragma unroll
                for (int i = 0; i < TM; i += 4)
                    VEC4(regM[wSubRow * TM + i]) =
                        CVEC4(As[cur][k * BM + warpRow * WM + wSubRow * WSUBM +
                                      threadRowInWarp * TM + i]);
#pragma unroll
            for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
#pragma unroll
                for (int j = 0; j < TN; j += 4)
                    VEC4(regN[wSubCol * TN + j]) =
                        CVEC4(Bs[cur][k * BN + warpCol * WN + wSubCol * WSUBN +
                                      threadColInWarp * TN + j]);
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

        // Land the prefetched tile in the OTHER buffer. No leading barrier is
        // needed: this iteration read buffer cur and writes cur^1, which no
        // thread touches until the next iteration. The trailing barrier orders
        // these writes against that iteration.
        if (more) {
            const int nxt = cur ^ 1;
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int off = i * aStride;
                if constexpr (TA) {
                    VEC4(As[nxt][(aRow + off) * BM + aCol * 4]) = aPre[i];
                } else {
                    As[nxt][(aCol * 4 + 0) * BM + aRow + off] = aPre[i].x;
                    As[nxt][(aCol * 4 + 1) * BM + aRow + off] = aPre[i].y;
                    As[nxt][(aCol * 4 + 2) * BM + aRow + off] = aPre[i].z;
                    As[nxt][(aCol * 4 + 3) * BM + aRow + off] = aPre[i].w;
                }
            }
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                const int off = i * bStride;
                if constexpr (TB) {
                    Bs[nxt][(bCol * 4 + 0) * BN + bRow + off] = bPre[i].x;
                    Bs[nxt][(bCol * 4 + 1) * BN + bRow + off] = bPre[i].y;
                    Bs[nxt][(bCol * 4 + 2) * BN + bRow + off] = bPre[i].z;
                    Bs[nxt][(bCol * 4 + 3) * BN + bRow + off] = bPre[i].w;
                } else {
                    VEC4(Bs[nxt][(bRow + off) * BN + bCol * 4]) = bPre[i];
                }
            }
            __syncthreads();
            cur = nxt;
        }
    }

#pragma unroll
    for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
#pragma unroll
        for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const size_t r = (size_t)(cRow + warpRow * WM + wSubRow * WSUBM +
                                          threadRowInWarp * TM + i);
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const int c = cCol + warpCol * WN + wSubCol * WSUBN +
                                  threadColInWarp * TN + j;
                    // beta == 0 is not just "multiply by zero": it means the
                    // old C is not needed, so the LOAD goes away too. Every
                    // activation GEMM in the model passes beta = 0, and the
                    // read it was doing is a full extra pass over the output.
                    float4 old = (beta != 0.0f) ? CVEC4(C[r * N + c])
                                                : make_float4(0, 0, 0, 0);
                    float4 bv = make_float4(0, 0, 0, 0);
                    if constexpr (EPI & epi::BIAS) bv = CVEC4(ep.bias[c]);
                    if constexpr (EPI & epi::ADD) {
                        const float4 av = CVEC4(ep.add[r * N + c]);
                        bv.x += av.x; bv.y += av.y; bv.z += av.z; bv.w += av.w;
                    }
                    old.x = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 0] + beta * old.x + bv.x;
                    old.y = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 1] + beta * old.y + bv.y;
                    old.z = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 2] + beta * old.z + bv.z;
                    old.w = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 3] + beta * old.w + bv.w;
                    VEC4(C[r * N + c]) = old;
                    if constexpr (EPI & epi::GELU) {
                        float4 gv;
                        gv.x = gelu_scalar(old.x); gv.y = gelu_scalar(old.y);
                        gv.z = gelu_scalar(old.z); gv.w = gelu_scalar(old.w);
                        VEC4(ep.gelu_out[r * N + c]) = gv;
                    }
                }
            }
}

#if BMB_TF32
// ------------------------------------------------------- tensor-core path
//
// The same transpose-aware idea as gemm_fast, but the inner product runs on the
// tensor cores in TF32 instead of the fp32 FMA pipes.
//
// The trick that keeps this from being four kernels: WMMA fragments could be
// loaded col_major to express a transpose, but then each of the four cases
// would need its own fragment types and its own leading dimensions. Instead the
// transpose is absorbed entirely into STAGING -- shared memory always holds
// As[m][k] and Bs[k][n], whatever layout the operand had in global memory --
// so the fragment loads and the mma below are literally identical in all four
// cases. That is the same move gemm_fast makes, for the same reason.
//
// Measured at the shapes this model actually runs, tensor cores are worth
// 1.22-1.58x over the fp32 kernel -- a wider margin than the 1.30x they manage
// on square matrices, because skinny matmuls starve the fp32 pipes harder than
// they starve the tensor cores.
// The TF32 WMMA shape available on sm_80+.
constexpr int TC_M = 16, TC_N = 16, TC_K = 8;

template <bool TA, bool TB, int BM, int BN, int BK, int WM, int WN,
          int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS, 2) void gemm_tc(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
    // Shared memory holds each operand in whichever orientation makes STAGING a
    // straight vector copy, and the fragment layout is chosen to match. The
    // first version of this kernel always stored As[m][k] and Bs[k][n], which
    // forced the transposed cases to scatter four scalars per float4 read.
    // Measured, that cost more than the tensor cores gained: TF32 beat fp32 by
    // 1.10-1.47x on NN and NT and LOST by 0.64-0.87x on TN and TT, and the
    // backward pass computes every weight gradient as TN.
    //
    // So when A arrives K x M it is stored K x M, and the fragment is loaded
    // col_major instead. Same data, same mma, no scatter.
    using ALayout = std::conditional_t<TA, wmma::col_major, wmma::row_major>;
    using BLayout = std::conditional_t<TB, wmma::col_major, wmma::row_major>;
    // Padded by 8 so a fragment load, which walks a column of lanes across
    // consecutive rows, does not put every lane in the same bank. 8 keeps the
    // leading dimension a multiple of 8, which TF32 WMMA requires.
    constexpr int LDA = (TA ? BM : BK) + 8;
    constexpr int LDB = (TB ? BK : BN) + 8;
    __shared__ float As[(TA ? BK : BM) * LDA];
    __shared__ float Bs[(TB ? BN : BK) * LDB];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WMITER = WM / TC_M;
    constexpr int WNITER = WN / TC_N;

    // Two index maps per operand: one for each transpose case, each reading
    // along whichever axis is contiguous in the layout the operand really has.
    const int aRow = TA ? tid / (BM / 4) : tid / (BK / 4);
    const int aCol = TA ? tid % (BM / 4) : tid % (BK / 4);
    constexpr int aStride = TA ? NUM_THREADS / (BM / 4) : NUM_THREADS / (BK / 4);
    constexpr int aIters = TA ? BK / aStride : BM / aStride;

    const int bRow = TB ? tid / (BK / 4) : tid / (BN / 4);
    const int bCol = TB ? tid % (BK / 4) : tid % (BN / 4);
    constexpr int bStride = TB ? NUM_THREADS / (BK / 4) : NUM_THREADS / (BN / 4);
    constexpr int bIters = TB ? BN / bStride : BK / bStride;

    wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, wmma::precision::tf32,
                   ALayout> aFrag[WMITER];
    wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, wmma::precision::tf32,
                   BLayout> bFrag[WNITER];
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> acc[WMITER][WNITER];

#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int it = 0; it < aIters; ++it) {
            const int off = it * aStride;
            if constexpr (TA) {
                // A is K x M: read along m and store K x M. Vector copy.
                VEC4(As[(aRow + off) * LDA + aCol * 4]) =
                    CVEC4(A[(size_t)(k0 + aRow + off) * M + cRow + aCol * 4]);
            } else {
                // A is M x K: read along k and store M x K. Vector copy.
                VEC4(As[(aRow + off) * LDA + aCol * 4]) =
                    CVEC4(A[(size_t)(cRow + aRow + off) * K + k0 + aCol * 4]);
            }
        }
#pragma unroll
        for (int it = 0; it < bIters; ++it) {
            const int off = it * bStride;
            if constexpr (TB) {
                // B is N x K: read along k and store N x K. Vector copy.
                VEC4(Bs[(bRow + off) * LDB + bCol * 4]) =
                    CVEC4(B[(size_t)(cCol + bRow + off) * K + k0 + bCol * 4]);
            } else {
                VEC4(Bs[(bRow + off) * LDB + bCol * 4]) =
                    CVEC4(B[(size_t)(k0 + bRow + off) * N + cCol + bCol * 4]);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += TC_K) {
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                // row_major: element (m,k) at As[m*LDA + k]; the tile starts at
                // (m0, kk). col_major: element (m,k) at As[k*LDA + m], so the
                // same tile starts at As[kk*LDA + m0]. One index, two readings.
                const int m0 = warpRow * WM + i * TC_M;
                wmma::load_matrix_sync(
                    aFrag[i], TA ? &As[kk * LDA + m0] : &As[m0 * LDA + kk], LDA);
                // Rounding to TF32 happens here rather than at staging time.
                // It is three times more of it and measurably faster: in the
                // compute phase there is enough independent work to hide it,
                // in the staging path there is not. See k09 for the numbers.
#pragma unroll
                for (int t = 0; t < aFrag[i].num_elements; ++t)
                    aFrag[i].x[t] = wmma::__float_to_tf32(aFrag[i].x[t]);
            }
#pragma unroll
            for (int j = 0; j < WNITER; ++j) {
                const int n0 = warpCol * WN + j * TC_N;
                wmma::load_matrix_sync(
                    bFrag[j], TB ? &Bs[n0 * LDB + kk] : &Bs[kk * LDB + n0], LDB);
#pragma unroll
                for (int t = 0; t < bFrag[j].num_elements; ++t)
                    bFrag[j].x[t] = wmma::__float_to_tf32(bFrag[j].x[t]);
            }
#pragma unroll
            for (int i = 0; i < WMITER; ++i)
#pragma unroll
                for (int j = 0; j < WNITER; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WMITER; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
            float *cptr = C + (size_t)(cRow + warpRow * WM + i * TC_M) * N +
                          cCol + warpCol * WN + j * TC_N;
            wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> cFrag;
            wmma::load_matrix_sync(cFrag, cptr, N, wmma::mem_row_major);
#pragma unroll
            for (int t = 0; t < cFrag.num_elements; ++t)
                cFrag.x[t] = alpha * acc[i][j].x[t] + beta * cFrag.x[t];
            wmma::store_matrix_sync(cptr, cFrag, N, wmma::mem_row_major);
        }
    }
}

#endif  // BMB_TF32

// --------------------------------------------------------------- slow path
// Guarded, scalar, handles any M/N/K and any transpose combination. Used for
// ragged shapes (e.g. an unpadded vocabulary). Correctness first; the fast
// path is what the hot shapes are aligned to hit.
template <bool TA, bool TB, int BS, int EPI>
__global__ void gemm_generic(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C,
                             GemmEpilogue ep) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS + 1];  // +1 breaks the bank conflict on Bs[k][.]

    const int col = threadIdx.x, row = threadIdx.y;
    const int cRow = blockIdx.y * BS, cCol = blockIdx.x * BS;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BS) {
        const int am = cRow + row, ak = k0 + col;
        As[row][col] = (am < M && ak < K)
                           ? (TA ? A[(size_t)ak * M + am] : A[(size_t)am * K + ak])
                           : 0.0f;
        const int bk = k0 + row, bn = cCol + col;
        Bs[row][col] = (bk < K && bn < N)
                           ? (TB ? B[(size_t)bn * K + bk] : B[(size_t)bk * N + bn])
                           : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BS; ++k) acc += As[row][k] * Bs[k][col];
        __syncthreads();
    }

    if (cRow + row < M && cCol + col < N) {
        const size_t idx = (size_t)(cRow + row) * N + cCol + col;
        const float old = (beta != 0.0f) ? C[idx] : 0.0f;
        float y = alpha * acc + beta * old;
        if constexpr (EPI & epi::BIAS) y += ep.bias[cCol + col];
        if constexpr (EPI & epi::ADD) y += ep.add[idx];
        C[idx] = y;
        if constexpr (EPI & epi::GELU) ep.gelu_out[idx] = gelu_scalar(y);
    }
}

#if BMB_TF32
// ------------------------------------------------- tensor-core path, raw PTX
// Kernel 10's structure, made transpose-aware. See src/kernels/k10_mma.cu for
// why the shared layout looks like this; the short version is that shared
// memory holds each operand tile in the order the tensor core's registers
// want it, so a fragment load is one 128-bit access instead of four 32-bit
// ones, and a 64x64 warp tile becomes affordable in registers.
//
// The transpose story is the SAME as the fp32 path's and it costs even less
// here. The lane-major layout is a function of the logical element (m,k), not
// of how the operand is stored, so all four cases share one map. What changes
// is only which axis the global float4 runs along, and therefore which bits of
// the destination slot the four staged values step through:
//
//     A, transA=false: read along k -> k%4 varies -> slot steps by 1
//     A, transA=true:  read along m -> m%8 varies -> slot steps by 4
//     B, transB=false: read along n -> n%8 varies -> slot steps by 4
//     B, transB=true:  read along k -> k%4 varies -> slot steps by 1
//
// So the transposed cases are not the expensive ones here. Both orientations
// pay four scalar shared stores; the only difference is the stride between
// them, and neither is a vector store. That is the price of the layout, and
// it is charged evenly.
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;
constexpr int UNIT = 128;  // floats per operand tile: 32 lanes x 4

__device__ __forceinline__ unsigned to_tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_m16n8k8(float (&d)[4],
                                            const unsigned (&a)[4],
                                            const unsigned *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The XOR swizzle that keeps the staging stores from collapsing onto a handful
// of banks. It permutes which lane slot inside a tile an element occupies, so
// it must be uniform across a warp doing a FRAGMENT LOAD (which reads one whole
// tile) and must VARY across a warp doing a STAGING STORE (which is what
// spreads the banks). Both hold only if it keys on the tile coordinate the
// staging warp actually walks -- and that is the axis the global read runs
// along, which is exactly what the transpose flag decides:
//
//     A, transA=false  stages along k  ->  key on kt
//     A, transA=true   stages along m  ->  key on mi
//     B, transB=false  stages along n  ->  key on nj
//     B, transB=true   stages along k  ->  key on kt
//
// Key it on the wrong one and it is warp-uniform during staging, so it does
// nothing at all: the transposed cases were 16-way conflicted and 20% slower
// than the WMMA path they replaced, while the untransposed ones were fine.
// tools/smem_banks.py simulates all four.
template <int BM, bool TA>
__device__ __forceinline__ int a_swz(int unit) {
    return (TA ? unit % (BM / MMA_M) : unit / (BM / MMA_M)) & 7;
}
template <int BN, bool TB>
__device__ __forceinline__ int b_swz(int unit) {
    return (TB ? unit / (BN / 16) : unit % (BN / 16)) & 7;
}

template <bool TA, bool TB, int BM, int BN, int BK, int WM, int WN,
          int NUM_THREADS, int MINB, int EPI>
__global__ __launch_bounds__(NUM_THREADS, MINB) void gemm_mma(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C, GemmEpilogue ep, int kBegin, int kEnd) {
    // No padding: the layout IS the fragment, so a fragment load is 32 lanes
    // over 512 contiguous bytes and cannot conflict.
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WARPSIZE;
    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    // gridDim.z is the K-split count. Each z owns a slice of K and its own
    // plane of the output workspace; with no split it is 1 and this is a no-op.
    if (gridDim.z > 1) {
        const int chunk = kEnd;  // the caller passes the chunk size here
        kBegin = blockIdx.z * chunk;
        kEnd = kBegin + chunk;
        if (kEnd > K) kEnd = K;
        C += (size_t)blockIdx.z * M * N;
    }

    constexpr int WMITER = WM / MMA_M;  // 16-row tiles this warp owns
    constexpr int WNITER = WN / 16;     // 16-column (= 2 mma) units it owns

    // Each operand reads along the contiguous axis of the layout it really
    // has, so every global load stays coalesced in all four cases.
    const int aRow = TA ? tid / (BM / 4) : tid / (BK / 4);
    const int aCol = TA ? tid % (BM / 4) : tid % (BK / 4);
    constexpr int aStride = TA ? NUM_THREADS / (BM / 4) : NUM_THREADS / (BK / 4);
    constexpr int aIters = TA ? BK / aStride : BM / aStride;

    const int bRow = TB ? tid / (BK / 4) : tid / (BN / 4);
    const int bCol = TB ? tid % (BK / 4) : tid % (BN / 4);
    constexpr int bStride = TB ? NUM_THREADS / (BK / 4) : NUM_THREADS / (BN / 4);
    constexpr int bIters = TB ? BN / bStride : BK / bStride;

    float acc[WMITER][WNITER][2][4] = {};

    for (int k0 = kBegin; k0 < kEnd; k0 += BK) {
#pragma unroll
        for (int it = 0; it < aIters; ++it) {
            const int off = it * aStride;
            // (m, k) of the FIRST of the four staged elements, and the axis
            // the other three step along.
            const int m = TA ? aCol * 4 : aRow + off;
            const int k = TA ? aRow + off : aCol * 4;
            const float4 v =
                TA ? CVEC4(A[(size_t)(k0 + aRow + off) * M + cRow + aCol * 4])
                   : CVEC4(A[(size_t)(cRow + aRow + off) * K + k0 + aCol * 4]);
            const int unit = (k / MMA_K) * (BM / MMA_M) + m / MMA_M;
            const int base =
                unit * UNIT + (((k % MMA_K) / 4) * 2 + ((m % MMA_M) / 8));
            const int slot = ((m % 8) * 4 + (k % 4)) ^ a_swz<BM, TA>(unit);
            constexpr int S = TA ? 4 : 1;  // how the slot steps per element
            As[base + ((slot ^ (0 * S)) * 4)] = v.x;
            As[base + ((slot ^ (1 * S)) * 4)] = v.y;
            As[base + ((slot ^ (2 * S)) * 4)] = v.z;
            As[base + ((slot ^ (3 * S)) * 4)] = v.w;
        }
#pragma unroll
        for (int it = 0; it < bIters; ++it) {
            const int off = it * bStride;
            const int k = TB ? bCol * 4 : bRow + off;
            const int n = TB ? bRow + off : bCol * 4;
            const float4 v =
                TB ? CVEC4(B[(size_t)(cCol + bRow + off) * K + k0 + bCol * 4])
                   : CVEC4(B[(size_t)(k0 + bRow + off) * N + cCol + bCol * 4]);
            const int unit = (k / MMA_K) * (BN / 16) + n / 16;
            const int base =
                unit * UNIT + (((k % MMA_K) / 4) + ((n % 16) / 8) * 2);
            const int slot = ((n % 8) * 4 + (k % 4)) ^ b_swz<BN, TB>(unit);
            constexpr int S = TB ? 1 : 4;
            Bs[base + ((slot ^ (0 * S)) * 4)] = v.x;
            Bs[base + ((slot ^ (1 * S)) * 4)] = v.y;
            Bs[base + ((slot ^ (2 * S)) * 4)] = v.z;
            Bs[base + ((slot ^ (3 * S)) * 4)] = v.w;
        }
        __syncthreads();

#pragma unroll
        for (int kt = 0; kt < BK / MMA_K; ++kt) {
            unsigned a[WMITER][4], b[WNITER][4];
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                const int unit = kt * (BM / MMA_M) + warpRow * WMITER + i;
                const float4 v = CVEC4(
                    As[unit * UNIT + (lane ^ a_swz<BM, TA>(unit)) * 4]);
                a[i][0] = to_tf32(v.x);
                a[i][1] = to_tf32(v.y);
                a[i][2] = to_tf32(v.z);
                a[i][3] = to_tf32(v.w);
            }
#pragma unroll
            for (int j = 0; j < WNITER; ++j) {
                const int unit = kt * (BN / 16) + warpCol * WNITER + j;
                const float4 v = CVEC4(
                    Bs[unit * UNIT + (lane ^ b_swz<BN, TB>(unit)) * 4]);
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

    // A lane's c0/c1 are adjacent columns, so the epilogue stores float2.
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
                // beta == 0 means the old C is not needed, so do not read it.
                // Every activation GEMM in the model passes beta = 0, and this
                // load was a whole extra pass over the output tensor.
                float2 o0 = {0.0f, 0.0f}, o1 = {0.0f, 0.0f};
                if (beta != 0.0f) {
                    o0 = reinterpret_cast<float2 *>(p0)[0];
                    o1 = reinterpret_cast<float2 *>(p1)[0];
                }
                // A column bias costs two floats per lane out of L1 here, and
                // saves a read AND a write of the entire output tensor as a
                // separate kernel. This is the one thing WMMA cannot express:
                // applying a per-COLUMN value needs to know which accumulator
                // register holds which column, which is exactly what the
                // fragment abstraction hides.
                float2 bv = {0.0f, 0.0f};
                if constexpr (EPI & epi::BIAS)
                    bv = reinterpret_cast<const float2 *>(&ep.bias[n0])[0];
                // The residual is a whole tensor, so unlike the bias it costs
                // a real read -- but it replaces a kernel that read TWO
                // tensors and wrote a third.
                float2 a0v = {0.0f, 0.0f}, a1v = {0.0f, 0.0f};
                if constexpr (EPI & epi::ADD) {
                    const size_t i0 = (size_t)(m0 + g) * N + n0;
                    const size_t i1 = (size_t)(m0 + g + 8) * N + n0;
                    a0v = reinterpret_cast<const float2 *>(&ep.add[i0])[0];
                    a1v = reinterpret_cast<const float2 *>(&ep.add[i1])[0];
                }
                o0.x = alpha * r[0] + beta * o0.x + bv.x + a0v.x;
                o0.y = alpha * r[1] + beta * o0.y + bv.y + a0v.y;
                o1.x = alpha * r[2] + beta * o1.x + bv.x + a1v.x;
                o1.y = alpha * r[3] + beta * o1.y + bv.y + a1v.y;
                reinterpret_cast<float2 *>(p0)[0] = o0;
                reinterpret_cast<float2 *>(p1)[0] = o1;
                if constexpr (EPI & epi::GELU) {
                    const size_t i0 = (size_t)(m0 + g) * N + n0;
                    const size_t i1 = (size_t)(m0 + g + 8) * N + n0;
                    float2 g0 = {gelu_scalar(o0.x), gelu_scalar(o0.y)};
                    float2 g1 = {gelu_scalar(o1.x), gelu_scalar(o1.y)};
                    reinterpret_cast<float2 *>(&ep.gelu_out[i0])[0] = g0;
                    reinterpret_cast<float2 *>(&ep.gelu_out[i1])[0] = g1;
                }
            }
        }
    }
}

#endif  // BMB_TF32

// Tile parameters of the fast path, matching kernel 8.
constexpr int FBM = 128, FBN = 128, FBK = 16;
constexpr int FWM = 64, FWN = 64, FWNITER = 4;
constexpr int FTM = 8, FTN = 4, FTHREADS = 128;

// Tile parameters of the tensor-core path: kernel 10's instruction and layout,
// but NOT kernel 10's tile. The ladder kernel uses a 64x64 warp tile in blocks
// of 128 threads, which is worth 8.7% on a square N=4096 because it halves
// shared-memory traffic per mma. In situ it LOSES, 1.034x against 1.044x, and
// the reason is grid size: the model's GEMMs are 4096 x 384 x 384 and friends,
// so a 128x128 block tile gives 96 blocks against 36 SMs. The machine is not
// full, and a 128-thread block then brings half as many warps per SM to hide
// latency with. The extra reuse is real and there is nothing to spend it on.
//
// So the best tile on a square benchmark is not the best tile in the model,
// and both numbers are in the repo rather than just the flattering one.
// TWO tensor-core tiles, chosen per shape at launch.
//
// A 128x128 block tile is the right default: it has the arithmetic intensity
// (32 FLOP/byte) and it is what the square benchmark is tuned on. But the
// weight-gradient matmuls in the backward are 384x384x4096 and friends -- tall
// K, tiny output -- and a 128x128 tile cuts that into a grid of NINE blocks
// against 36 SMs. Three quarters of the machine is idle no matter how good the
// kernel is.
//
// So there is a second tile, half as tall, which doubles the block count and
// fits more of them per SM. Measured at the model's own shapes, clock pinned:
//
//   shape                    blocks   128x128   64x128
//   4096x384x384  (fwd)          96      7149     6898     wide tile wins
//   4096x384x1536 (fwd)          96      7851     7722     wide tile wins
//   384x384x4096  (dW)            9      1846     2313     +25%
//
// That is not a tuning accident, it is the roofline meeting the grid: below
// about one wave the block count is what limits you, and above it the reuse
// is. cuBLAS ships dozens of kernels for exactly this reason.
constexpr int TBM = 128, TBN = 128, TBK = 32;
constexpr int TWM = 32, TWN = 64, TTHREADS = 256, TMINB = 2;

constexpr int SBM = 64, SBN = 128, SBK = 32;
constexpr int SWM = 32, SWN = 64, STHREADS = 128, SMINB = 4;

// WHAT `cp.async` DID HERE, measured and then reverted, recorded so it is not
// retried on the same reasoning.
//
// Kernel 11 replaces this kernel's register staging with a `cp.async` pipeline
// and is +8.1% at square N=4096. Doing the same to this kernel -- same layout,
// same swizzle keys, same four transpose cases, BK cut from 32 to 16 to pay for
// the stages -- is SLOWER AT EVERY SHAPE THE MODEL RUNS:
//
//   shape (NN)                    register-staged   cp.async, 3 stages
//   4096x1536x384   mlp up               10469            9818    -6.2%
//   4096x1152x384   qkv proj              9937            9246    -7.0%
//   4096x384x384    attn proj             7276            6823    -6.2%
//   4096x384x1536   mlp down              7898            7480    -5.3%
//   1536x384x4096   dW fcproj             8827            8481    -3.9%
//   384x384x4096    dW attnproj           7825            7545    -3.6%
//   384x1536x4096   dW fc                 8804            8531    -3.1%
//   384x1152x4096   dW qkv                8473            8292    -2.1%
//
// End to end that is 46.1 -> 47.4 ms/step. The fp32 path is untouched and reads
// the same to within noise in both builds, which is what makes this a clean
// measurement rather than a drifting machine.
//
// AND THE FOOTPRINT IS NOT THE REASON, which was the obvious suspect: three
// stages at BK=16 is 48 KB against the 32 KB here, and on a 100 KiB SM that
// still fits two blocks. Two stages at BK=16 is 32 KB -- byte for byte what
// this kernel uses today -- and it loses by the same ~6% (mlp up 9802, attn
// proj 6878). So the cost is BK itself, not the pipeline it was paying for.
//
// The difference from kernel 11 that survives is reuse per barrier. Kernel 11
// runs a 64x64 warp tile: 128 bytes of shared traffic per mma, and BK=16 still
// leaves it two k-steps of independent arithmetic to hide a barrier behind.
// This kernel runs 32x64 -- 192 bytes per mma, chosen when the epilogue and the
// four transpose cases mattered more than the tile did -- so halving BK doubles
// the barrier count against arithmetic that was already thinner. A pipeline
// cannot pay for itself out of a k-chunk that does not have enough work in it.
//
// Which makes the next thing to try the warp tile, not the staging: 64x64 here
// costs 128 threads instead of 256 and would have to be measured against the
// epilogue and split-K paths that the current shape was picked for.

// Switch below this many 128x128 blocks. 72 is two blocks per SM across 36
// SMs -- one full wave -- so the rule reads: if the default tile cannot fill
// the machine once, prefer the tile that makes more blocks.
// WHICH OF THE TWO TILES, AND WHY THE OLD RULE WAS WRONG TWICE OVER.
//
// This used to be `TILE_SWITCH_BLOCKS = 72` -- two blocks per SM across the 36
// SMs of the 4070 this was written on -- reading "if the wide tile cannot fill
// the machine once, prefer the tile that makes more blocks". Re-measuring it on
// a 46-SM card was worth 7.3% of a training step. The constant was stale, but
// the rule behind it was also wrong, and the second part took a control to see.
//
// Forcing the NARROW tile on every shape, this card, TF32 GFLOP/s:
//
//   shape                       N     wide    narrow
//   attn proj    4096x384x384    384   7172    8556   +19%
//   qkv proj    4096x1152x384   1152   9732   11100   +14%
//   mlp up      4096x1536x384   1536  10467   11584   +11%
//   mlp down    4096x384x1536    384   7868    9613   +22%
//
// The narrow tile wins at 384 blocks (mlp up, four full waves) as decisively as
// at 96 (attn proj, one). So it is NOT wave quantisation -- which was the first
// explanation, and does not survive the arithmetic either: 96 blocks into 92
// slots and 192 into 184 are the same 52% efficiency.
//
// THE ROOFLINE IS THE MOTIVATION AND NOT A PROOF, which is worth stating
// because the tempting version of this comment is wrong. A BM x BN tile reads
// (BM+BN)*BK*4 bytes per 2*BM*BN*BK flops, so its intensity is
// BM*BN/(2*(BM+BN)): 32.0 FLOP/byte wide, 21.3 narrow. The ridge point of the
// 4070 is far above both and this card's is much lower, which is a real and
// robust difference between the two machines and plausibly why they prefer
// different tiles.
//
// It does NOT survive as a threshold test, and pretending otherwise was a
// mistake caught only by building it. "Narrow clears the ridge" is true at
// 21.3 against 21.0 -- a 1.4% margin, and only at the PINNED 1.2 GHz clock.
// At this card's base clock the ridge is 25.4 and at boost 54.1, so the same
// card answers the same question three different ways depending on which clock
// you evaluate at. A rule that flips on 1.4% is numerology, not physics; and
// the intensity model ignores L2 reuse, which at these shapes is large.
//
// So what is here is an EMPIRICAL rule with the roofline as its motivation:
// take the narrow tile unless this machine's ridge point (at base clock, which
// is a fixed device property rather than whatever the card is clocked at right
// now) sits above the WIDE tile's intensity, i.e. unless the machine is
// bandwidth-starved enough that arithmetic intensity is what limits it.
//
//   4070      ~11.3 TFLOP/s / 256 GB/s = 44.3  >  32.0  ->  wide    (measured)
//   5070 Ti    17.1 TFLOP/s / 672 GB/s = 25.4  <  32.0  ->  narrow  (measured)
//
// Both sides clear by ~20-40%, which is the margin the knife-edge version
// lacked. It is calibrated on two cards and should be re-measured on a third
// rather than trusted; `--bench` in tools/test_gemm.cu is what does that.
constexpr double NARROW_TILE_AI = (double)SBM * SBN / (2.0 * (SBM + SBN));
constexpr double WIDE_TILE_AI = (double)TBM * TBN / (2.0 * (TBM + TBN));

static bool prefer_narrow_tile() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int dev = 0, clock_khz = 0, mem_khz = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cached = 0;  // cannot tell -- keep the tile the 4070 was tuned on
        return false;
    }
    // CUDA 13 removed cudaDeviceProp::clockRate and ::memoryClockRate; the
    // attribute API works on both, which device_query.cu already relies on.
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev);
    if (clock_khz <= 0 || mem_khz <= 0 || p.memoryBusWidth <= 0) {
        cached = 0;
        return false;
    }
    const double peak = 2.0 * 128 * p.multiProcessorCount * clock_khz * 1e3;
    const double bw = 2.0 * (p.memoryBusWidth / 8.0) * mem_khz * 1e3;
    cached = (peak / bw < WIDE_TILE_AI) ? 1 : 0;
    return cached != 0;
}

// Narrowing BN to 64 was tried, to cut the wave quantisation at N=384 where a
// 128-wide tile leaves only three block columns and the second wave runs a
// third full. It is worse everywhere -- 6144 -> 5644 GF/s at 4096x384x384 --
// because a 128x64 tile has 21 FLOP/byte against 128x128's 32, and this far
// below the 43 ridge point the lost reuse costs more than the tail does.

// Off by default: TF32 keeps fp32 range but only 10 mantissa bits, so turning
// it on changes what the model computes. It is the precision the hardware was
// built to train in, but that is a decision for the caller to make explicitly.
static bool g_tf32 = false;

// Only reached by the GEMM_USE_WMMA comparison build. gemm_tc cannot fuse a
// column bias -- see the note in gemm_mma's epilogue -- so that build pays for
// it the old way, with a separate pass, which is what it is there to represent.
__global__ void epilogue_only_k(float *C, GemmEpilogue ep, int N, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float y = C[i];
    if (ep.bias) y += ep.bias[i % (unsigned)N];
    if (ep.add) y += ep.add[i];
    C[i] = y;
    if (ep.gelu_out) ep.gelu_out[i] = gelu_scalar(y);
}

// Sum the per-split partials and apply alpha/beta/epilogue exactly once.
// Deterministic: a fixed number of partials in a fixed order, rather than
// atomics, for the same reason the bias backward avoids them.
template <int EPI>
__global__ void splitk_reduce_k(float *C, const float *partial, int N,
                                size_t total, int splits, float alpha,
                                float beta, GemmEpilogue ep) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = 0.0f;
    for (int sp = 0; sp < splits; ++sp) acc += partial[sp * total + i];
    float y = alpha * acc;
    if (beta != 0.0f) y += beta * C[i];
    if constexpr (EPI & epi::BIAS) y += ep.bias[i % (unsigned)N];
    if constexpr (EPI & epi::ADD) y += ep.add[i];
    C[i] = y;
    if constexpr (EPI & epi::GELU) ep.gelu_out[i] = gelu_scalar(y);
}

// How many K-splits this shape wants. Zero or one means "do not split".
//
// SPLIT-K IS ABOUT THE OUTPUT, NOT THE GRID. I talked myself out of this once
// by costing it for 4096x384x384, where the output is 6.3 MB and every extra
// partial is another 6.3 MB of traffic -- correctly, it loses there. The
// weight gradients are the opposite shape: 384x384 output, K=4096. The output
// is 590 KB, the partials are free, and the grid is NINE blocks on a 36-SM
// card. Same technique, opposite verdict, and the deciding quantity is the
// ratio of K to M*N rather than anything about waves.
inline int splitk_for(int blocks, int K, int BK) {
    constexpr int TARGET_BLOCKS = 144;  // 4 blocks/SM x 36 SMs
    if (blocks >= TARGET_BLOCKS) return 1;
    int s = TARGET_BLOCKS / blocks;
    const int max_by_k = K / (BK * 4);  // keep >=4 K-chunks per split
    if (s > max_by_k) s = max_by_k;
    return s < 2 ? 1 : s;
}

template <bool TA, bool TB, int EPI>
void dispatch_epi(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C, cudaStream_t stream,
                  GemmEpilogue ep) {
    // Tensor cores when asked for and the shape lines up. BK is 32 here, not
    // the fp32 path's 16, so the alignment test is stricter and a shape that
    // misses it simply falls through to fp32 rather than to the slow path.
    //
    // Below sm_80 there is no TF32 path at all and g_tf32 can never be true,
    // so this whole branch compiles out and the fp32 kernels carry everything.
#if BMB_TF32
    if (g_tf32 && M % TBM == 0 && N % TBN == 0 && K % TBK == 0) {
        dim3 grid(N / TBN, M / TBM);
#ifdef GEMM_USE_WMMA
        gemm_tc<TA, TB, TBM, TBN, TBK, 32, 64, 256>
            <<<grid, 256, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
        if (ep.bias || ep.add || ep.gelu_out) {
            const size_t total = (size_t)M * N;
            epilogue_only_k<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
                C, ep, N, total);
        }
#else
        const bool narrow =
            prefer_narrow_tile() && M % SBM == 0;
        const int bm = narrow ? SBM : TBM, bn = narrow ? SBN : TBN;
        const int bk = narrow ? SBK : TBK;
        const int blocks = (M / bm) * (N / bn);
        const int splits = splitk_for(blocks, K, bk);
        dim3 g2(N / bn, M / bm, splits);

        if (splits > 1) {
            // Partials live in a workspace, one plane per split. thread_local
            // for the same reason reduce_mean's scalar is: one host thread per
            // GPU, one workspace per device.
            static thread_local float *ws = nullptr;
            static thread_local size_t cap = 0;
            const size_t total = (size_t)M * N;
            const size_t need = total * splits;
            if (need > cap) {
                if (ws) cudaFree(ws);
                cudaMalloc(&ws, need * sizeof(float));
                cap = need;
            }
            const int chunk = ((K / splits + bk - 1) / bk) * bk;
            // The split kernels compute raw partials: alpha, beta and the
            // epilogue are applied once, by the reduction.
            if (narrow)
                gemm_mma<TA, TB, SBM, SBN, SBK, SWM, SWN, STHREADS, SMINB, 0>
                    <<<g2, STHREADS, 0, stream>>>(M, N, K, 1.0f, A, B, 0.0f, ws,
                                                  GemmEpilogue(), 0, chunk);
            else
                gemm_mma<TA, TB, TBM, TBN, TBK, TWM, TWN, TTHREADS, TMINB, 0>
                    <<<g2, TTHREADS, 0, stream>>>(M, N, K, 1.0f, A, B, 0.0f, ws,
                                                  GemmEpilogue(), 0, chunk);
            splitk_reduce_k<EPI><<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
                C, ws, N, total, splits, alpha, beta, ep);
        } else if (narrow) {
            dim3 sgrid(N / SBN, M / SBM);
            gemm_mma<TA, TB, SBM, SBN, SBK, SWM, SWN, STHREADS, SMINB, EPI>
                <<<sgrid, STHREADS, 0, stream>>>(M, N, K, alpha, A, B, beta, C,
                                                 ep, 0, K);
        } else {
            gemm_mma<TA, TB, TBM, TBN, TBK, TWM, TWN, TTHREADS, TMINB, EPI>
                <<<grid, TTHREADS, 0, stream>>>(M, N, K, alpha, A, B, beta, C,
                                                ep, 0, K);
        }
#endif
    } else
#endif  // BMB_TF32
        if (M % FBM == 0 && N % FBN == 0 && K % FBK == 0) {
        dim3 grid(N / FBN, M / FBM);
        gemm_fast<TA, TB, FBM, FBN, FBK, FWM, FWN, FWNITER, FTM, FTN, FTHREADS, EPI>
            <<<grid, FTHREADS, 0, stream>>>(M, N, K, alpha, A, B, beta, C, ep);
    } else {
        constexpr int BS = 16;
        dim3 block(BS, BS);
        dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
        gemm_generic<TA, TB, BS, EPI><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C, ep);
    }
}

// All eight combinations are instantiated. Seven is not enough: an EPI mask
// that claims a feature the caller did not supply would dereference a null
// pointer, so the mask must always describe the pointers exactly.
template <bool TA, bool TB>
void dispatch(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C, cudaStream_t stream, GemmEpilogue ep) {
    const int e = (ep.bias ? epi::BIAS : 0) | (ep.add ? epi::ADD : 0) |
                  (ep.gelu_out ? epi::GELU : 0);
#define CASE(m)                                                                    case (m):                                                                          dispatch_epi<TA, TB, (m)>(M, N, K, alpha, A, B, beta, C, stream, ep);          break;
    switch (e) {
        CASE(0)
        CASE(epi::BIAS)
        CASE(epi::ADD)
        CASE(epi::GELU)
        CASE(epi::BIAS | epi::ADD)
        CASE(epi::BIAS | epi::GELU)
        CASE(epi::ADD | epi::GELU)
        CASE(epi::BIAS | epi::ADD | epi::GELU)
    }
#undef CASE
}
}  // namespace

void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C,
          cudaStream_t stream, GemmEpilogue ep) {
    if (!transA && !transB)      dispatch<false, false>(M, N, K, alpha, A, B, beta, C, stream, ep);
    else if (!transA && transB)  dispatch<false, true >(M, N, K, alpha, A, B, beta, C, stream, ep);
    else if (transA && !transB)  dispatch<true,  false>(M, N, K, alpha, A, B, beta, C, stream, ep);
    else                         dispatch<true,  true >(M, N, K, alpha, A, B, beta, C, stream, ep);
}

void gemm_set_tf32(bool on) {
#if BMB_TF32
    g_tf32 = on;
#else
    // Pre-Ampere: asking for TF32 is not an error, it just has no effect. The
    // caller gets fp32, which is what the hardware can do.
    (void)on;
#endif
}

bool gemm_tf32_available() {
#if BMB_TF32
    return true;
#else
    return false;
#endif
}
bool gemm_tf32() { return g_tf32; }
