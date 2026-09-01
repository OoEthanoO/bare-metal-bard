#pragma once
#include <cuda_runtime.h>

// Every activation tensor is row-major (B, T, C), so the (b,t) pair is a flat
// row index and any matmul can treat it as a single N = B*T dimension. Weights
// are stored (out_features, in_features), the PyTorch convention, and consumed
// with transB = true -- the GEMM handles that without a transposed copy.

// ---- embeddings ----
void encoder_forward(float *out, const int *tokens, const float *wte,
                     const float *wpe, int B, int T, int C);
void encoder_backward(float *dwte, float *dwpe, const float *dout,
                      const int *tokens, int B, int T, int C);

// ---- layernorm ----
// mean and rstd are saved from the forward pass; recomputing them in backward
// would cost a second pass over inp for no memory saving worth having.
void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int N, int C);
void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        const float *dout, const float *inp,
                        const float *weight, const float *mean,
                        const float *rstd, int N, int C);

// ---- activations and residuals ----
void gelu_forward(float *out, const float *inp, int n);
void gelu_backward(float *dinp, const float *inp, const float *dout, int n);
void residual_forward(float *out, const float *a, const float *b, int n);
void add_inplace(float *dst, const float *src, int n);

// ---- bias handling ----
// The GEMM has no bias term, so bias is a separate add; its gradient is a
// column sum over the B*T dimension.
void bias_backward(float *dbias, const float *dout, int N, int C);

// Force the row-split count instead of deriving it from C and the SM count.
// Zero restores the derived rule. Same reason gemm_set_splitk exists: a rule
// that picks one point on a curve should be checked against the curve.
// tools/bench_nn.cu --splits prints it.
void bias_backward_set_splits(int splits);

// Same, for the layernorm backward's block count. Its rows are independent, so
// the block count trades parallelism against how many per-column partials the
// second pass has to sum.
void layernorm_backward_set_blocks(int blocks);

// ---- loss ----
// Vp is the padded vocabulary (a multiple of 128 so the head matmul hits the
// fast GEMM path); V is the real vocabulary. Columns in [V, Vp) are masked out
// of the softmax so the padding cannot absorb probability mass.
void softmax_crossentropy_forward(float *probs, float *losses,
                                  const float *logits, const int *targets,
                                  int N, int V, int Vp);
void crossentropy_softmax_backward(float *dlogits, const float *probs,
                                   const int *targets, int N, int V, int Vp,
                                   float dloss_scale);

// ---- optimizer ----
// grad_scale multiplies every gradient on the way in, which is how gradient
// clipping is applied without a separate pass over the gradient buffer.
void adamw_update(float *params, float *grads, float *m, float *v, int n,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, int step, float grad_scale);

// L2 norm over the whole gradient vector, for global-norm clipping.
float grad_global_norm(const float *grads, int n);

// ---- utility ----
void zero_buffer(float *p, size_t n);
float reduce_mean(const float *d_values, int n);
