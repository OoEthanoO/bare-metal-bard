// Kernel 5: 2D register tiling.
//
// Kernel 4 amortized one B element over TM A elements. The symmetric move is
// to have each thread own a TM x TN patch of C: load TM elements of A and TN
// of B into registers, then issue TM*TN FMAs from them.
//
//     SMEM loads per FMA:  (TM + TN) / (TM * TN)
//     TM=TN=8  ->  16 loads per 64 FMAs = 0.25, versus 1.125 in kernel 4.
//
// This is the outer-product formulation: each k step is a rank-1 update of the
// register tile, accumulated over K.
//
// Blocking is widened to BM=BN=128 to raise global intensity as well:
//     per chunk   (128*8 + 8*128) floats = 8 KiB,  2*128*128*8 = 262 KFLOP
//     -> 32 FLOP/byte, versus 16 in kernel 4 and a ridge point of ~58.
//
// Threads = (BM*BN)/(TM*TN) = 256, but each tile is 1024 elements, so every
// thread now stages 4 elements of A and 4 of B in a strided loop.
//
// Register budget: 64 accumulators + 8 + 8 staging + indices ~= 80+ registers.
// At 256 threads that is ~20K of the SM's 65K register file, allowing 3
// concurrent blocks (768 of 1536 threads, 50% occupancy). Low occupancy is
// correct here -- a compute-bound kernel wants deep per-thread ILP, not many
// threads. Occupancy is a means to latency hiding, not a goal.
#include "../kernels.h"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void tile2d_kernel(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

    // This thread owns the TM x TN patch at (threadRow*TM, threadCol*TN).
    const int threadCol = tid % (BN / TN);  // 0..15
    const int threadRow = tid / (BN / TN);  // 0..15

    // Strided staging. strideA is how many rows of the A tile the block covers
    // in one pass; the loop repeats until all BM rows are staged.
    const int innerColA = tid % BK;          // 0..7
    const int innerRowA = tid / BK;          // 0..31
    constexpr int strideA = NUM_THREADS / BK;   // 32
    const int innerColB = tid % BN;          // 0..127
    const int innerRowB = tid / BN;          // 0..1
    constexpr int strideB = NUM_THREADS / BN;   // 2

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            const int r = cRow + innerRowA + off;
            As[(innerRowA + off) * BK + innerColA] =
                (r < M && k0 + innerColA < K)
                    ? A[(size_t)r * K + k0 + innerColA] : 0.0f;
        }
#pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            const int r = k0 + innerRowB + off;
            Bs[(innerRowB + off) * BN + innerColB] =
                (r < K && cCol + innerColB < N)
                    ? B[(size_t)r * N + cCol + innerColB] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Pull one column slice of A and one row slice of B into
            // registers...
#pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[k * BN + threadCol * TN + j];
            // ...then spend them on TM*TN FMAs with no further memory traffic.
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = cRow + threadRow * TM + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = cCol + threadCol * TN + j;
            if (c < N) {
                size_t idx = (size_t)r * N + c;
                C[idx] = alpha * acc[i][j] + beta * C[idx];
            }
        }
    }
}

void sgemm_tile2d(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));  // 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    tile2d_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
