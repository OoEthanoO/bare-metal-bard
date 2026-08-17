#pragma once
#include <cuda_runtime.h>

// Transpose-aware GEMM, row-major throughout:
//
//     C[M,N] = alpha * op(A) @ op(B) + beta * C
//
//     op(A) = A   with A stored M x K (lda = K)   when transA == false
//           = A^T with A stored K x M (lda = M)   when transA == true
//     op(B) = B   with B stored K x N (ldb = N)   when transB == false
//           = B^T with B stored N x K (ldb = K)   when transB == true
//
// Backprop through Y = X @ W needs dX = dY @ W^T and dW = X^T @ dY, i.e. the
// NT and TN cases. Materializing the transposes would cost an extra pass over
// the data; instead the transpose is folded into the shared-memory staging,
// which already had to reindex the operands anyway.
void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C,
          cudaStream_t stream = 0);

// Batched GEMM: `batch` independent problems, each strided by strideA/B/C.
//
// Attention needs B*NH = 96 independent (T x hs) @ (hs x T) products. Looping
// the single GEMM would cost 96 kernel launches (~5 us each) for work that
// takes tens of microseconds, so the batch index lives in gridDim.z instead.
void batched_gemm(bool transA, bool transB, int batch, int M, int N, int K,
                  float alpha, const float *A, long long strideA,
                  const float *B, long long strideB, float beta, float *C,
                  long long strideC, cudaStream_t stream = 0);
