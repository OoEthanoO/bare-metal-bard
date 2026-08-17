// Kernel 8: double-buffered shared memory.
//
// Kernel 7 ended latency bound: ncu showed memory at 45% and compute at 55%,
// so neither resource was the ceiling -- the SMs were simply waiting. The
// reason is visible in kernel 7's loop structure:
//
//     load global -> shared        <- ~500 cycle latency
//     __syncthreads()              <- every warp blocks until it lands
//     compute on the tile
//     __syncthreads()              <- and blocks again before overwriting
//
// The load is immediately followed by a barrier, so nothing overlaps it. Every
// K-chunk pays the full round trip to DRAM with the FMA pipes idle.
//
// Double buffering fixes this by keeping TWO tiles in shared memory. While the
// warps compute on buffer `cur`, the global loads for the *next* chunk are
// already in flight, landing in registers; they get written into buffer
// `1-cur` after the compute is issued. The memory latency is then hidden
// behind arithmetic that was going to happen anyway.
//
// It also halves the barriers. Within an iteration the warps read one buffer
// and write the other, so those cannot conflict, and a single __syncthreads()
// at the bottom is enough to order this iteration's writes against the next
// iteration's reads. One barrier per K-chunk instead of two.
//
// Costs, both real:
//   * shared memory doubles, 16 KiB -> 32 KiB per block
//   * 32 extra registers per thread to hold the in-flight tile
// Both eat into occupancy, which is why this is worth measuring rather than
// assuming. See the note in sgemm_doublebuffer() for what actually happened.
#include "../kernels.h"

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

namespace {
constexpr int WARPSIZE = 32;
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void dbuf_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
    // Two buffers each. As stays transposed (As[k][m]) so the per-thread
    // register reads remain contiguous float4s, as in kernels 6 and 7.
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

    const int aRow = tid / (BK / 4);
    const int aCol = tid % (BK / 4);
    constexpr int aStride = (NUM_THREADS * 4) / BK;
    const int bRow = tid / (BN / 4);
    const int bCol = tid % (BN / 4);
    constexpr int bStride = NUM_THREADS / (BN / 4);

    // How many float4s each thread stages per tile.
    constexpr int NA = BM / aStride;
    constexpr int NB = BK / bStride;

    float acc[WMITER * TM][WNITER * TN] = {{0.0f}};
    float regM[WMITER * TM];
    float regN[WNITER * TN];
    float4 aPre[NA];  // the in-flight tile
    float4 bPre[NB];

    // ---- prologue: stage chunk 0 straight into buffer 0 ----
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int off = i * aStride;
        const float4 a = CVEC4(A[(size_t)(cRow + aRow + off) * K + aCol * 4]);
        As[0][(aCol * 4 + 0) * BM + aRow + off] = a.x;
        As[0][(aCol * 4 + 1) * BM + aRow + off] = a.y;
        As[0][(aCol * 4 + 2) * BM + aRow + off] = a.z;
        As[0][(aCol * 4 + 3) * BM + aRow + off] = a.w;
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int off = i * bStride;
        VEC4(Bs[0][(bRow + off) * BN + bCol * 4]) =
            CVEC4(B[(size_t)(bRow + off) * N + cCol + bCol * 4]);
    }
    __syncthreads();

    int cur = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int knext = k0 + BK;
        const bool more = knext < K;

        // ---- issue the next chunk's global loads NOW ----
        // These are plain loads, but they are issued before the compute below
        // and consumed only after it, so the ~500-cycle DRAM latency is spent
        // while the FMA pipes are busy rather than while they stall.
        if (more) {
#pragma unroll
            for (int i = 0; i < NA; ++i)
                aPre[i] = CVEC4(
                    A[(size_t)(cRow + aRow + i * aStride) * K + knext + aCol * 4]);
#pragma unroll
            for (int i = 0; i < NB; ++i)
                bPre[i] = CVEC4(
                    B[(size_t)(knext + bRow + i * bStride) * N + cCol + bCol * 4]);
        }

        // ---- compute on the resident buffer ----
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

        // ---- land the prefetched tile in the OTHER buffer ----
        // Safe without a leading barrier: this iteration reads buffer `cur`
        // and writes `1-cur`, which no thread touches until the next
        // iteration. The trailing barrier orders these writes against that
        // iteration's reads, and also guarantees every thread has finished
        // reading buffer `cur` before the iteration after next overwrites it.
        if (more) {
            const int nxt = cur ^ 1;
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int off = i * aStride;
                As[nxt][(aCol * 4 + 0) * BM + aRow + off] = aPre[i].x;
                As[nxt][(aCol * 4 + 1) * BM + aRow + off] = aPre[i].y;
                As[nxt][(aCol * 4 + 2) * BM + aRow + off] = aPre[i].z;
                As[nxt][(aCol * 4 + 3) * BM + aRow + off] = aPre[i].w;
            }
#pragma unroll
            for (int i = 0; i < NB; ++i)
                VEC4(Bs[nxt][(bRow + i * bStride) * BN + bCol * 4]) = bPre[i];
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

void sgemm_doublebuffer(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
    constexpr int NUM_THREADS = 128;
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, WNITER = 4;
    constexpr int TM = 8, TN = 4;

    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / WARPSIZE, "warp grid");
    static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0, "warp tile");
    // 2 * (BM + BN) * BK * 4 bytes must fit the 48 KiB default SMEM budget.
    static_assert(2 * (BM + BN) * BK * sizeof(float) <= 48 * 1024, "smem budget");

    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
        return;
    }

    dim3 grid(N / BN, M / BM);
    dbuf_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}
