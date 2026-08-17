// Kernel 2: identical arithmetic to kernel 1, one index swapped.
//
// threadIdx.x now maps to the COLUMN. Within a warp, lanes 0..31 hold
// consecutive `col` and identical `row`, so:
//
//   B[k*N + col] -> 32 consecutive floats = 128 bytes = 4 sectors, one
//                   coalesced access serving the whole warp.
//   A[row*K + k] -> every lane wants the SAME address, which the hardware
//                   serves as a broadcast, also one transaction.
//   C[row*N+col] -> consecutive, coalesced.
//
// Kernel 1 needed up to 32 separate transactions for what this does in one.
// Nothing about the flops changed; only the address pattern did.
//
// Note the block is declared (32, 32) in both kernels -- the thread-to-data
// mapping is what moved, not the block shape.
#include "../kernels.h"

__global__ void coalesced_kernel(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

void sgemm_coalesced(int M, int N, int K, float alpha, const float *A,
                     const float *B, float beta, float *C) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    coalesced_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
