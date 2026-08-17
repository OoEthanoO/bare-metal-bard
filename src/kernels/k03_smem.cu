// Kernel 3: shared-memory tiling (cache blocking).
//
// Kernel 2 is coalesced but still reads every operand from global memory once
// per use. A 32x32 block of C needs the same 32 rows of A for all 32 columns,
// so each element of A is fetched 32 times over the block, and each element of
// B likewise.
//
// Here the block cooperatively stages a BS x BS tile of A and of B into shared
// memory, then every thread reads from SMEM instead. Each global element is
// loaded once per block rather than BS times.
//
// Arithmetic intensity, per K-chunk:
//     loaded  2 * BS * BS floats = 8 KiB   (BS = 32)
//     flops   2 * BS * BS * BS   = 64 KFLOP
//     ratio   8 FLOP/byte
// Up 32x from the naive 0.25, but the ridge point is ~58 FLOP/byte, so this is
// still memory bound -- the reason kernels 4 and 5 exist. The ceiling here is
// roughly 256 GB/s * 8 FLOP/byte = 2 TFLOP/s.
//
// Occupancy note: 2 * 32 * 32 * 4 B = 8 KiB of SMEM per block. With 100 KiB per
// SM that allows 12 blocks, but 1024 threads/block against a 1536 thread/SM
// limit caps us at 1 block per SM. Threads, not shared memory, are the binding
// constraint at this stage.
#include "../kernels.h"

template <int BS>
__global__ void smem_kernel(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    const int col = threadIdx.x;  // 0..BS-1, contiguous within a warp
    const int row = threadIdx.y;
    const int cRow = blockIdx.y * BS;  // top-left of this block's C tile
    const int cCol = blockIdx.x * BS;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BS) {
        // Staging loads. Both are coalesced: threadIdx.x varies fastest and
        // maps to the contiguous dimension of A and of B alike. Out-of-range
        // entries are zeroed so ragged edges contribute nothing, which lets
        // the inner loop run unguarded.
        As[row][col] = (cRow + row < M && k0 + col < K)
                           ? A[(size_t)(cRow + row) * K + k0 + col] : 0.0f;
        Bs[row][col] = (k0 + row < K && cCol + col < N)
                           ? B[(size_t)(k0 + row) * N + cCol + col] : 0.0f;
        __syncthreads();

        // SMEM access pattern in the inner loop:
        //   As[row][k]  -- every lane in the warp shares `row`, so this is one
        //                  broadcast, not 32 reads.
        //   Bs[k][col]  -- lanes hold consecutive `col`, hitting 32 distinct
        //                  banks. Conflict-free.
#pragma unroll
        for (int k = 0; k < BS; ++k) {
            acc += As[row][k] * Bs[k][col];
        }
        // Second barrier: without it a fast thread could overwrite As/Bs for
        // the next k0 while a slow one is still reading this chunk.
        __syncthreads();
    }

    if (cRow + row < M && cCol + col < N) {
        size_t idx = (size_t)(cRow + row) * N + cCol + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}

void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C) {
    constexpr int BS = 32;
    dim3 block(BS, BS);
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    smem_kernel<BS><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
