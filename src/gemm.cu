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

template <bool TA, bool TB>
void dispatch(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C, cudaStream_t stream) {
    if (M % FBM == 0 && N % FBN == 0 && K % FBK == 0) {
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
