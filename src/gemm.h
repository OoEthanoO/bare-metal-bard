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

// Work folded into the GEMM's epilogue, where the result is already sitting in
// a register with a store about to happen anyway.
//
// Each of these, as its own kernel, costs a full read AND write of the output
// tensor to perform one arithmetic operation on each element. The step profile
// put the bias at 8.2% and the residual and GELU at another 6% between them --
// second, third and fourth behind the matmuls, for three operations that do no
// arithmetic worth measuring. Fused, they cost the epilogue a couple of extra
// loads per lane and no global traffic at all.
//
// The order applied is:  y = alpha*op(A)op(B) + beta*C + bias[col] + add[i]
// then C[i] = y, and if gelu_out is set, gelu_out[i] = gelu(y).
struct GemmEpilogue {
    const float *bias = nullptr;      // one value per COLUMN
    const float *add = nullptr;       // full tensor, same shape as C (residual)
    float *gelu_out = nullptr;        // second output, gelu(y), same shape as C
};

void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C,
          cudaStream_t stream = 0, GemmEpilogue ep = GemmEpilogue());

// Batched GEMM: `batch` independent problems, each strided by strideA/B/C.
//
// Attention needs B*NH = 96 independent (T x hs) @ (hs x T) products. Looping
// the single GEMM would cost 96 kernel launches (~5 us each) for work that
// takes tens of microseconds, so the batch index lives in gridDim.z instead.
void batched_gemm(bool transA, bool transB, int batch, int M, int N, int K,
                  float alpha, const float *A, long long strideA,
                  const float *B, long long strideB, float beta, float *C,
                  long long strideC, cudaStream_t stream = 0);

// Route the matmuls above through the TF32 tensor cores where the shape allows
// it. Off by default: this changes the numerics, not just the speed. TF32 keeps
// fp32's 8-bit exponent and 10 of its 23 mantissa bits, which is the trade
// every framework makes by default on Ampere and later.
void gemm_set_tf32(bool on);
bool gemm_tf32();
