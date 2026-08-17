// Causal multi-head self-attention, built from the batched GEMM plus two
// bandwidth-bound kernels (permute, softmax).
//
// The causal mask is never materialized as a matrix of -inf. Instead the
// softmax simply does not read past t2 > t1 and writes exact zeros there. That
// saves a (B,NH,T,T) buffer and, more usefully, halves the softmax's work --
// row t1 reduces over t1+1 entries rather than T.
#include "attention.h"
#include "gemm.h"
#include "reduce.cuh"
#include <cfloat>
#include <cmath>

namespace {
using red::block_reduce;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// qkv (B, T, 3C) -> q,k,v each (B, NH, T, hs), stored back to back in qkvr.
__global__ void permute_qkv_k(float *qkvr, const float *qkv, int B, int T,
                              int NH, int hs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * NH * T * hs;
    if (idx >= total) return;

    // Decode the destination index (b, h, t, i).
    const int i = idx % hs;
    const int t = (idx / hs) % T;
    const int h = (idx / (hs * T)) % NH;
    const int b = idx / (hs * T * NH);

    const int C = NH * hs;
    // Source row for this (b,t); the three projections sit at 0, C, 2C.
    const size_t src = (size_t)(b * T + t) * 3 * C + (size_t)h * hs + i;
    qkvr[(size_t)0 * total + idx] = qkv[src];
    qkvr[(size_t)1 * total + idx] = qkv[src + C];
    qkvr[(size_t)2 * total + idx] = qkv[src + 2 * C];
}

// Inverse of the above, used to scatter dq/dk/dv back into dqkv.
__global__ void unpermute_qkv_k(float *dqkv, const float *dqkvr, int B, int T,
                                int NH, int hs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * NH * T * hs;
    if (idx >= total) return;

    const int i = idx % hs;
    const int t = (idx / hs) % T;
    const int h = (idx / (hs * T)) % NH;
    const int b = idx / (hs * T * NH);

    const int C = NH * hs;
    const size_t dst = (size_t)(b * T + t) * 3 * C + (size_t)h * hs + i;
    dqkv[dst]         = dqkvr[(size_t)0 * total + idx];
    dqkv[dst + C]     = dqkvr[(size_t)1 * total + idx];
    dqkv[dst + 2 * C] = dqkvr[(size_t)2 * total + idx];
}

// (B, NH, T, hs) -> (B, T, C), merging heads back into the channel axis.
__global__ void unpermute_out_k(float *out, const float *in, int B, int T,
                                int NH, int hs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * NH * T * hs;
    if (idx >= total) return;
    const int i = idx % hs;
    const int t = (idx / hs) % T;
    const int h = (idx / (hs * T)) % NH;
    const int b = idx / (hs * T * NH);
    out[(size_t)(b * T + t) * (NH * hs) + (size_t)h * hs + i] = in[idx];
}

// (B, T, C) -> (B, NH, T, hs), the forward permute applied to dout.
__global__ void permute_out_k(float *out, const float *in, int B, int T,
                              int NH, int hs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * NH * T * hs;
    if (idx >= total) return;
    const int i = idx % hs;
    const int t = (idx / hs) % T;
    const int h = (idx / (hs * T)) % NH;
    const int b = idx / (hs * T * NH);
    out[idx] = in[(size_t)(b * T + t) * (NH * hs) + (size_t)h * hs + i];
}

// One block per (b, h, t1) row. Row t1 attends only to positions <= t1.
__global__ void softmax_causal_k(float *att, const float *preatt, int T) {
    const int row = blockIdx.x;
    const int t1 = row % T;
    const int len = t1 + 1;  // causal: everything after t1 is masked
    const float *x = preatt + (size_t)row * T;
    float *y = att + (size_t)row * T;

    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < len; i += blockDim.x) m = fmaxf(m, x[i]);
    m = block_reduce<true>(m);

    float s = 0.0f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const float e = __expf(x[i] - m);  // shift for stability
        y[i] = e;
        s += e;
    }
    s = block_reduce<false>(s);
    const float inv = 1.0f / s;

    for (int i = threadIdx.x; i < len; i += blockDim.x) y[i] *= inv;
    // Explicit zeros in the masked region: the att @ V matmul reads the full
    // row, so these must be 0 rather than stale memory.
    for (int i = len + threadIdx.x; i < T; i += blockDim.x) y[i] = 0.0f;
}

// For y = softmax(x): dx_i = y_i * (dy_i - sum_j dy_j y_j).
__global__ void softmax_causal_bwd_k(float *dpreatt, const float *datt,
                                     const float *att, int T) {
    const int row = blockIdx.x;
    const int t1 = row % T;
    const int len = t1 + 1;
    const float *y = att + (size_t)row * T;
    const float *dy = datt + (size_t)row * T;
    float *dx = dpreatt + (size_t)row * T;

    float dot = 0.0f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) dot += dy[i] * y[i];
    dot = block_reduce<false>(dot);

    for (int i = threadIdx.x; i < len; i += blockDim.x)
        dx[i] = y[i] * (dy[i] - dot);
    for (int i = len + threadIdx.x; i < T; i += blockDim.x) dx[i] = 0.0f;
}
}  // namespace

void attention_forward(float *out, float *qkvr, float *att, const float *qkv,
                       int B, int T, int C, int NH) {
    const int hs = C / NH;
    const int total = B * NH * T * hs;  // == B*T*C
    const float scale = 1.0f / sqrtf((float)hs);
    const long long sHead = (long long)T * hs, sAtt = (long long)T * T;

    permute_qkv_k<<<ceil_div(total, 256), 256>>>(qkvr, qkv, B, T, NH, hs);
    const float *q = qkvr, *k = qkvr + total, *v = qkvr + 2 * total;
    float *vaccum = qkvr + 3 * total;  // slot 4: head-major attention output

    // preatt = scale * q @ k^T, one (T x hs) @ (hs x T) per (b, h). k is
    // stored (T, hs) and we want k^T, which is exactly the transB case:
    // op(B) = B^T with B stored N x K = T x hs. No transposed copy needed.
    // preatt is written into `att`, which softmax then normalizes in place.
    batched_gemm(false, true, B * NH, T, T, hs, scale, q, sHead, k, sHead,
                 0.0f, att, sAtt);

    // In-place is safe: every thread reads index i and writes index i of the
    // same row, and the stride loop gives each index to exactly one thread, so
    // no thread ever reads a location another thread writes. The barriers
    // inside block_reduce separate the read-only max/sum passes from the
    // writes that follow.
    softmax_causal_k<<<B * NH * T, 128>>>(att, att, T);

    // vaccum = att @ v, one (T x T) @ (T x hs) per (b, h). The masked region
    // of att is exactly zero, so future positions contribute nothing.
    batched_gemm(false, false, B * NH, T, hs, T, 1.0f, att, sAtt, v, sHead,
                 0.0f, vaccum, sHead);

    unpermute_out_k<<<ceil_div(total, 256), 256>>>(out, vaccum, B, T, NH, hs);
}

void attention_backward(float *dqkv, float *dqkvr, float *datt, float *dpreatt,
                        const float *dout, const float *qkvr, const float *att,
                        int B, int T, int C, int NH) {
    const int hs = C / NH;
    const int total = B * NH * T * hs;
    const float scale = 1.0f / sqrtf((float)hs);
    const long long sHead = (long long)T * hs, sAtt = (long long)T * T;

    const float *q = qkvr, *k = qkvr + total, *v = qkvr + 2 * total;
    float *dq = dqkvr, *dk = dqkvr + total, *dv = dqkvr + 2 * total;
    float *dvaccum = dqkvr + 3 * total;

    permute_out_k<<<ceil_div(total, 256), 256>>>(dvaccum, dout, B, T, NH, hs);

    // out[t1,i] = sum_t2 att[t1,t2] * v[t2,i], so:
    //   datt[t1,t2] = sum_i dvaccum[t1,i] * v[t2,i]  =  dvaccum @ v^T
    //   dv[t2,i]    = sum_t1 att[t1,t2] * dvaccum[t1,i]  =  att^T @ dvaccum
    batched_gemm(false, true, B * NH, T, T, hs, 1.0f, dvaccum, sHead, v, sHead,
                 0.0f, datt, sAtt);
    batched_gemm(true, false, B * NH, T, hs, T, 1.0f, att, sAtt, dvaccum, sHead,
                 0.0f, dv, sHead);

    // datt and dpreatt may be the same buffer: the reduction reads the whole
    // row before the barrier, and the writes that follow are index-for-index.
    softmax_causal_bwd_k<<<B * NH * T, 128>>>(dpreatt, datt, att, T);

    // preatt[t1,t2] = scale * sum_i q[t1,i] * k[t2,i], so:
    //   dq[t1,i] = scale * sum_t2 dpreatt[t1,t2] * k[t2,i]  =  scale * dpreatt @ k
    //   dk[t2,i] = scale * sum_t1 dpreatt[t1,t2] * q[t1,i]  =  scale * dpreatt^T @ q
    batched_gemm(false, false, B * NH, T, hs, T, scale, dpreatt, sAtt, k, sHead,
                 0.0f, dq, sHead);
    batched_gemm(true, false, B * NH, T, hs, T, scale, dpreatt, sAtt, q, sHead,
                 0.0f, dk, sHead);

    unpermute_qkv_k<<<ceil_div(total, 256), 256>>>(dqkv, dqkvr, B, T, NH, hs);
}
