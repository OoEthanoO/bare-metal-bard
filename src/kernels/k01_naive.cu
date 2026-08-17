// Kernel 1: the naive baseline, written the way it comes out the first time.
//
// One thread per output element. Each thread walks the full K dimension,
// reading one element of A and one of B per step. There is no reuse of any
// kind: every thread re-reads the same row of A that its warp-neighbours read,
// and the same column of B that every other block-row reads.
//
// Arithmetic intensity is 2 flops per 8 bytes loaded = 0.25 FLOP/byte, against
// a ridge point of ~58 FLOP/byte. This kernel is memory bound by a factor of
// ~230x, and that gap is the entire subject of this project.
//
// It is also uncoalesced: threadIdx.x maps to the ROW, which is the natural
// thing to write and exactly the wrong thing to do. Lanes 0..31 of a warp read
// A[0*K+k], A[1*K+k], ... -- addresses K floats apart, so each lane needs its
// own 32-byte memory transaction. Kernel 2 swaps this and measures the cost.
#include "../kernels.h"

__global__ void naive_kernel(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C) {
    dim3 block(32, 32);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
