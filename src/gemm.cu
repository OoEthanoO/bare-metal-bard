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
#include <mma.h>
#include <type_traits>

using namespace nvcuda;
#include "kernels.h"
#include <cstdio>

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

namespace {
constexpr int WARPSIZE = 32;

// ---------------------------------------------------------------- fast path
template <bool TA, bool TB, int BM, int BN, int BK, int WM, int WN, int WNITER,
          int TM, int TN, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void gemm_fast(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
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
                    float4 old = CVEC4(C[r * N + c]);
                    old.x = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 0] + beta * old.x;
                    old.y = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 1] + beta * old.y;
                    old.z = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 2] + beta * old.z;
                    old.w = alpha * acc[wSubRow * TM + i][wSubCol * TN + j + 3] + beta * old.w;
                    VEC4(C[r * N + c]) = old;
                }
            }
}

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

// --------------------------------------------------------------- slow path
// Guarded, scalar, handles any M/N/K and any transpose combination. Used for
// ragged shapes (e.g. an unpadded vocabulary). Correctness first; the fast
// path is what the hot shapes are aligned to hit.
template <bool TA, bool TB, int BS>
__global__ void gemm_generic(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
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
        C[idx] = alpha * acc + beta * C[idx];
    }
}

// Tile parameters of the fast path, matching kernel 8.
constexpr int FBM = 128, FBN = 128, FBK = 16;
constexpr int FWM = 64, FWN = 64, FWNITER = 4;
constexpr int FTM = 8, FTN = 4, FTHREADS = 128;

// Tile parameters of the tensor-core path, matching kernel 9.
constexpr int TBM = 128, TBN = 128, TBK = 32;
constexpr int TWM = 32, TWN = 64, TTHREADS = 256;
// Narrowing BN to 64 was tried, to cut the wave quantisation at N=384 where a
// 128-wide tile leaves only three block columns and the second wave runs a
// third full. It is worse everywhere -- 6144 -> 5644 GF/s at 4096x384x384 --
// because a 128x64 tile has 21 FLOP/byte against 128x128's 32, and this far
// below the 43 ridge point the lost reuse costs more than the tail does.

// Off by default: TF32 keeps fp32 range but only 10 mantissa bits, so turning
// it on changes what the model computes. It is the precision the hardware was
// built to train in, but that is a decision for the caller to make explicitly.
static bool g_tf32 = false;

template <bool TA, bool TB>
void dispatch(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C, cudaStream_t stream) {
    // Tensor cores when asked for and the shape lines up. BK is 32 here, not
    // the fp32 path's 16, so the alignment test is stricter and a shape that
    // misses it simply falls through to fp32 rather than to the slow path.
    if (g_tf32 && M % TBM == 0 && N % TBN == 0 && K % TBK == 0) {
        dim3 grid(N / TBN, M / TBM);
        gemm_tc<TA, TB, TBM, TBN, TBK, TWM, TWN, TTHREADS>
            <<<grid, TTHREADS, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    } else if (M % FBM == 0 && N % FBN == 0 && K % FBK == 0) {
        dim3 grid(N / FBN, M / FBM);
        gemm_fast<TA, TB, FBM, FBN, FBK, FWM, FWN, FWNITER, FTM, FTN, FTHREADS>
            <<<grid, FTHREADS, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        constexpr int BS = 16;
        dim3 block(BS, BS);
        dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
        gemm_generic<TA, TB, BS><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }
}
}  // namespace

void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C,
          cudaStream_t stream) {
    if (!transA && !transB)      dispatch<false, false>(M, N, K, alpha, A, B, beta, C, stream);
    else if (!transA && transB)  dispatch<false, true >(M, N, K, alpha, A, B, beta, C, stream);
    else if (transA && !transB)  dispatch<true,  false>(M, N, K, alpha, A, B, beta, C, stream);
    else                         dispatch<true,  true >(M, N, K, alpha, A, B, beta, C, stream);
}

void gemm_set_tf32(bool on) { g_tf32 = on; }
bool gemm_tf32() { return g_tf32; }
