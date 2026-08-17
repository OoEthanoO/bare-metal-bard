// Kernel 4: 1D register tiling.
//
// Kernel 3 raised intensity to 8 FLOP/byte against a ridge point of ~58, so it
// is still bandwidth bound -- but the bottleneck has moved. It is no longer
// global memory: it is the SMEM-to-register traffic. Every FMA in kernel 3
// needs two shared-memory loads, and shared memory tops out far below what the
// FMA pipes can consume.
//
// The fix is to give each thread more than one output. A thread that owns TM
// results in a column loads ONE element of B, holds it in a register, and
// reuses it against TM elements of A. The ratio of SMEM loads to FMAs drops
// from 2:1 to roughly (TM+1)/TM : 1 -- with TM=8 that is 9 loads per 8 FMAs
// instead of 16.
//
// Blocking: BM x BN tile of C per block, marched over K in BK-wide chunks.
//     per chunk   load  (BM*BK + BK*BN) floats,  do 2*BM*BN*BK flops
//     BM=BN=64, BK=8 -> 4 KiB loaded, 64 KFLOP -> 16 FLOP/byte, 2x kernel 3.
//
// Threads per block = (BM*BN)/TM = 512, which is exactly BM*BK = BK*BN = 512,
// so each thread stages exactly one element of each tile. That is not a
// coincidence, it is how the parameters were chosen.
#include "../kernels.h"

template <int BM, int BN, int BK, int TM>
__global__ void tile1d_kernel(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // Output mapping: this thread owns column `threadCol` and the TM
    // consecutive rows starting at threadRow*TM.
    const int threadCol = tid % BN;
    const int threadRow = tid / BN;

    // Staging mapping. Deliberately different from the output mapping: it is
    // chosen for coalescing, not for who consumes the data.
    const int innerColA = tid % BK;   // 0..7
    const int innerRowA = tid / BK;   // 0..63
    const int innerColB = tid % BN;   // 0..63
    const int innerRowB = tid / BN;   // 0..7

    float acc[TM] = {0.0f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // A tile: 8 consecutive threads cover 8 consecutive floats of one row,
        // so a warp issues 4 x 32B transactions. Rows of A are K apart, so
        // this is the best a row-major A allows without a transpose.
        As[innerRowA * BK + innerColA] =
            (cRow + innerRowA < M && k0 + innerColA < K)
                ? A[(size_t)(cRow + innerRowA) * K + k0 + innerColA] : 0.0f;
        // B tile: 64 consecutive threads cover 64 consecutive floats, so a
        // warp issues one fully contiguous 128B transaction. Ideal.
        Bs[innerRowB * BN + innerColB] =
            (k0 + innerRowB < K && cCol + innerColB < N)
                ? B[(size_t)(k0 + innerRowB) * N + cCol + innerColB] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Hoist the B element into a register once, then amortize it over
            // TM multiplies. This single line is the whole point of kernel 4.
            const float bcast = Bs[k * BN + threadCol];
#pragma unroll
            for (int t = 0; t < TM; ++t) {
                acc[t] += As[(threadRow * TM + t) * BK + k] * bcast;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < TM; ++t) {
        const int r = cRow + threadRow * TM + t;
        if (r < M && cCol + threadCol < N) {
            size_t idx = (size_t)r * N + cCol + threadCol;
            C[idx] = alpha * acc[t] + beta * C[idx];
        }
    }
}

void sgemm_tile1d(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);  // 512
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    tile1d_kernel<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
