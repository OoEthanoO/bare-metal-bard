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

    // The BACKWARD of the fusion above, and the reason it is worth a slot of
    // its own: the MLP's dX gemm produces d(gelu output), and the very next
    // kernel used to read that whole tensor back, multiply it by gelu'(pre),
    // and write it out again. At 4096x1536 that is three passes over 25 MB per
    // layer to do one multiply per element.
    //
    // With this set, C[i] = dgelu(dgelu_pre[i], y) instead of y -- the
    // derivative is applied where the value is already in a register and the
    // separate pass disappears. Only the pre-activation has to be read.
    const float *dgelu_pre = nullptr;  // pre-activation, same shape as C

    // The bias gradient, computed from data the GEMM already stages.
    //
    // Every weight gradient in the backward is dW = dY^T @ X -- a TN gemm with
    // dY as the A operand -- and the bias gradient of the same layer is
    // dbias[m] = sum_k dY[k][m]: a sum along exactly the K axis the gemm
    // already iterates, over exactly the values its A staging already loads.
    // As its own kernel that sum costs a full extra pass over dY (3.9% of a
    // training step for the four of them); here it costs four FADDs per staged
    // float4 in one block column, and no extra global reads at all.
    //
    // With this set, dbias_out[m] += sum over k of op(A)[m][k], accumulated in
    // fp32 from the raw values (before any TF32 rounding), deterministically
    // (fixed-order reductions, no atomics). alpha and beta do not apply to it.
    //
    // Only the tensor-core path implements it, and only for transA=true (the
    // orientation backprop uses). Callers must check gemm_dbias_supported()
    // and fall back to a separate reduction; a call that sets this where it is
    // not supported aborts loudly rather than skipping the sum silently.
    float *dbias_out = nullptr;  // length M, accumulated +=
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

// Force the K-split count instead of deriving it from the shape and the SM
// count. Zero restores the derived rule. This exists so the rule can be
// MEASURED against a sweep rather than argued about: the derived value is one
// point on a curve, and the only way to know it is the right point is to have
// seen the curve. tools/test_gemm.cu --splitk prints it.
void gemm_set_splitk(int splits);
int gemm_splitk();

// False on pre-Ampere hardware, where the TF32 kernels are not compiled in
// at all. gemm_set_tf32(true) is then a no-op rather than an error.
bool gemm_tf32_available();

// Whether a gemm with these arguments would honor GemmEpilogue::dbias_out.
// True exactly when the tensor-core path will be taken (TF32 on, aligned
// shape) with transA set. The caller uses this to decide between fusing the
// bias gradient and running the standalone reduction -- the decision has to
// live with the caller because the fallback kernel does.
bool gemm_dbias_supported(bool transA, bool transB, int M, int N, int K);
