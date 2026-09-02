// Elementwise, reduction, and normalization kernels for the transformer.
//
// The GEMMs are compute bound and got all the blocking work. Everything in
// this file is the opposite: bandwidth bound, arithmetic intensity below 1,
// and the only thing that matters is touching each byte once and touching it
// with a coalesced, ideally 128-bit, access. Where a kernel has to reduce
// along a row it uses warp shuffles rather than shared memory, because the
// shuffle network does not pay the L1/MIO cost that shared memory does -- the
// exact cost that was throttling GEMM kernel 6.
#include "nn.h"
#include "gelu.cuh"
#include "reduce.cuh"
#include <cstdio>
#include <cmath>
#include <cfloat>

#define VEC4(p) (reinterpret_cast<float4 *>(&(p))[0])
#define CVEC4(p) (reinterpret_cast<const float4 *>(&(p))[0])

namespace {
using red::block_reduce;

// ------------------------------------------------------------------ encoder
__global__ void encoder_fwd_k(float *out, const int *tokens, const float *wte,
                              const float *wpe, int T, int C, int n4) {
    // One thread per float4 of the output. C is a multiple of 4 by config.
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= n4) return;
    const int idx = i4 * 4;
    const int bt = idx / C, c = idx % C;
    const int t = bt % T;
    const int tok = tokens[bt];
    float4 e = CVEC4(wte[(size_t)tok * C + c]);
    const float4 p = CVEC4(wpe[(size_t)t * C + c]);
    e.x += p.x; e.y += p.y; e.z += p.z; e.w += p.w;
    VEC4(out[idx]) = e;
}

__global__ void encoder_bwd_k(float *dwte, float *dwpe, const float *dout,
                              const int *tokens, int T, int C, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int bt = idx / C, c = idx % C;
    const int t = bt % T;
    const float g = dout[idx];
    // Many (b,t) positions share a token id, so these are genuine conflicts,
    // not an artifact of the parallelization -- atomics are the right tool.
    atomicAdd(&dwte[(size_t)tokens[bt] * C + c], g);
    atomicAdd(&dwpe[(size_t)t * C + c], g);
}

// ---------------------------------------------------------------- layernorm
// One block per row. C = 384 here, so a 128-thread block handles 3 elements
// each and both reductions (mean, then variance) stay inside the block.
__global__ void layernorm_fwd_k(float *out, float *mean, float *rstd,
                                const float *inp, const float *weight,
                                const float *bias, int C) {
    const int row = blockIdx.x;
    const float *x = inp + (size_t)row * C;
    float *y = out + (size_t)row * C;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) sum += x[i];
    const float mu = block_reduce<false>(sum) / C;

    float sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float d = x[i] - mu;
        sq += d * d;
    }
    // Biased variance (divide by C), matching LayerNorm rather than a sample
    // variance -- this must agree with the backward pass below.
    const float var = block_reduce<false>(sq) / C;
    const float rs = rsqrtf(var + 1e-5f);

    if (threadIdx.x == 0) { mean[row] = mu; rstd[row] = rs; }
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        y[i] = ((x[i] - mu) * rs) * weight[i] + bias[i];
}

// For y = w * xhat + b with xhat = (x-mu)*rstd, the input gradient is
//     dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
// The two subtracted terms are the corrections for mu and sigma themselves
// depending on x; dropping them is a classic silent-wrongness bug, so both
// means are reduced explicitly here.
__global__ void layernorm_bwd_k(float *dinp, float *part_dw, float *part_db,
                                const float *dout, const float *inp,
                                const float *weight, const float *mean,
                                const float *rstd, int N, int C) {
    // TWO ATOMICS PER ELEMENT USED TO LIVE IN THE ROW LOOP, and they were both
    // the slow part and a correctness claim this repo was not keeping.
    //
    // The old version ran one block per row and did
    //     atomicAdd(&dweight[i], dy[i] * xhat);  atomicAdd(&dbias[i], dy[i]);
    // for every element -- at N=4096, C=384 that is 3.1 MILLION atomic adds
    // contending on 768 addresses, and it measured 64 GB/s against this card's
    // 672. It also made the gradient depend on the order blocks happened to
    // finish in: two runs at the same seed produce different losses by step 20.
    // The bias reduction below already refuses atomics for exactly that reason,
    // in a comment that claimed the repo compares runs bit for bit.
    //
    // IT DOES NOT, AND REMOVING THESE DOES NOT MAKE IT SO -- worth stating
    // plainly because I believed the opposite for about ten minutes. One seed
    // agreed across two runs and I called it fixed; the next seed did not.
    // `encoder_bwd_k` still uses atomics for dwte/dwpe, and THERE the conflicts
    // are real: many (b,t) positions share a token id, so the accumulation
    // genuinely collides however the kernel is arranged. The layernorm's
    // collisions were the avoidable kind -- an artifact of one block per row --
    // and that is the whole claim being made here.
    //
    // So the accumulation moves out of the row loop. A block owns a strided set
    // of ROWS instead of one row, and within a block thread t owns columns
    // t, t+blockDim, ... for every row it visits -- a fixed 1:1 thread-to-column
    // map, so the per-column running sums live in shared memory with no atomic
    // and no race. One partial per block per column is written at the end, and
    // a second kernel sums a FIXED number of partials in a FIXED order.
    extern __shared__ float acc[];  // [C] dweight, then [C] dbias
    float *acc_dw = acc, *acc_db = acc + C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        acc_dw[i] = 0.0f;
        acc_db[i] = 0.0f;
    }
    __syncthreads();

    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const float *x = inp + (size_t)row * C;
        const float *dy = dout + (size_t)row * C;
        float *dx = dinp + (size_t)row * C;
        const float mu = mean[row], rs = rstd[row];

        float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            const float xhat = (x[i] - mu) * rs;
            const float dxhat = dy[i] * weight[i];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
        }
        const float m1 = block_reduce<false>(sum_dxhat) / C;
        const float m2 = block_reduce<false>(sum_dxhat_xhat) / C;

        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            const float xhat = (x[i] - mu) * rs;
            const float g = dy[i];
            const float dxhat = g * weight[i];
            dx[i] += rs * (dxhat - m1 - xhat * m2);
            acc_dw[i] += g * xhat;
            acc_db[i] += g;
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        part_dw[(size_t)blockIdx.x * C + i] = acc_dw[i];
        part_db[(size_t)blockIdx.x * C + i] = acc_db[i];
    }
}

// Fixed number of partials, summed in a fixed order: deterministic, and the
// same shape as bias_bwd_reduce_k below.
__global__ void layernorm_bwd_reduce_k(float *dweight, float *dbias,
                                       const float *part_dw,
                                       const float *part_db, int C,
                                       int blocks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sw = 0.0f, sb = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        sw += part_dw[(size_t)i * C + c];
        sb += part_db[(size_t)i * C + c];
    }
    dweight[c] += sw;
    dbias[c] += sb;
}

// --------------------------------------------------------------------- gelu
// GPT-2's tanh approximation, kept exactly because the model is compared
// against reference implementations that use it.


__global__ void gelu_fwd_k(float *out, const float *inp, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = CVEC4(inp[i * 4]);
    v.x = gelu_scalar(v.x); v.y = gelu_scalar(v.y);
    v.z = gelu_scalar(v.z); v.w = gelu_scalar(v.w);
    VEC4(out[i * 4]) = v;
}



__global__ void gelu_bwd_k(float *dinp, const float *inp, const float *dout, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 x = CVEC4(inp[i * 4]);
    const float4 g = CVEC4(dout[i * 4]);
    float4 d;
    d.x = dgelu_scalar(x.x, g.x); d.y = dgelu_scalar(x.y, g.y);
    d.z = dgelu_scalar(x.z, g.z); d.w = dgelu_scalar(x.w, g.w);
    VEC4(dinp[i * 4]) = d;
}

// ----------------------------------------------------------------- residual
__global__ void residual_fwd_k(float *out, const float *a, const float *b, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 x = CVEC4(a[i * 4]), y = CVEC4(b[i * 4]);
    float4 r; r.x = x.x + y.x; r.y = x.y + y.y; r.z = x.z + y.z; r.w = x.w + y.w;
    VEC4(out[i * 4]) = r;
}

__global__ void add_inplace_k(float *dst, const float *src, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 d = CVEC4(dst[i * 4]);
    const float4 s = CVEC4(src[i * 4]);
    d.x += s.x; d.y += s.y; d.z += s.z; d.w += s.w;
    VEC4(dst[i * 4]) = d;
}

// --------------------------------------------------------------------- bias
// The forward bias used to live here as its own kernel. It now rides in the
// GEMM epilogue (see gemm.h): a standalone pass reads and writes the entire
// output tensor to do one add per element, and the step profile put the four
// of them together at 8.2% of a training step.
//
// The backward now rides a GEMM too, but a different one and a different way:
// a column reduction is not something an OUTPUT epilogue can do -- it needs
// every row -- but the dW gemm's A staging touches every row of dY anyway, so
// on the tensor-core path the sum accumulates there instead (see
// GemmEpilogue::dbias_out). What is below is the fp32 path's reduction, and
// the fallback for any shape the fused path declines.

// dbias[c] = sum over all N rows of dout[n][c].
//
// The first version of this put ONE BLOCK PER COLUMN and strode down the rows,
// with a comment claiming the reads were coalesced because consecutive blocks
// own consecutive columns. They are not. Coalescing happens within a warp, and
// in that arrangement a warp's 32 threads read 32 different ROWS at the same
// column -- 32 addresses C floats apart, so 32 separate memory transactions
// fetching 32 bytes each to use 4. Neighbouring blocks do re-use those sectors
// out of L2, which is why it was merely bad rather than catastrophic, and why
// it survived: nothing about the source looks wrong.
//
// The step profile is what found it. At 5.4% of a training step for an
// operation whose floor is a single streaming read of dout, it was the second
// biggest non-matmul item left.
//
// So: threadIdx.x walks COLUMNS, and a warp reads 32 consecutive floats of one
// row. threadIdx.y walks rows, and gridDim.y splits the rows again so narrow
// tensors still fill the machine -- at C=384 the column axis alone is only 12
// blocks. The partials are written to a workspace and summed by a second
// kernel rather than atomically, because atomics would make the gradient
// depend on block scheduling order.
//
// THAT SENTENCE USED TO END "and this repo compares runs bit for bit", WHICH
// WAS NOT TRUE -- two runs at the same seed diverge by step 20, checked rather
// than assumed. Avoiding atomics here is still right; it buys determinism in
// THIS reduction, not in the step. The layernorm backward above was the other
// avoidable half, and encoder_bwd_k is the half that cannot go.
constexpr int BIAS_COLS = 32, BIAS_ROWS = 8;

__global__ void bias_bwd_k(float *partial, const float *dout, int N, int C) {
    const int c = blockIdx.x * BIAS_COLS + threadIdx.x;
    float acc = 0.0f;
    if (c < C) {
        for (int n = blockIdx.y * BIAS_ROWS + threadIdx.y; n < N;
             n += gridDim.y * BIAS_ROWS)
            acc += dout[(size_t)n * C + c];
    }
    // Reduce down the row axis. s[y][x] with x contiguous, so no bank conflict.
    __shared__ float s[BIAS_ROWS][BIAS_COLS];
    s[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
#pragma unroll
    for (int r = BIAS_ROWS / 2; r > 0; r >>= 1) {
        if (threadIdx.y < r)
            s[threadIdx.y][threadIdx.x] += s[threadIdx.y + r][threadIdx.x];
        __syncthreads();
    }
    if (threadIdx.y == 0 && c < C) partial[(size_t)blockIdx.y * C + c] = s[0][threadIdx.x];
}

// Fixed number of partials, summed in a fixed order: deterministic.
__global__ void bias_bwd_reduce_k(float *dbias, const float *partial, int C,
                                  int splits) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.0f;
    for (int i = 0; i < splits; ++i) acc += partial[(size_t)i * C + c];
    dbias[c] += acc;
}

// ------------------------------------------------------- softmax + xent
// One block per row of logits. Vp is padded up to a GEMM-friendly multiple,
// and the padded columns must be excluded: leaving them in would hand real
// probability mass to tokens that do not exist.
__global__ void softmax_xent_fwd_k(float *probs, float *losses,
                                   const float *logits, const int *targets,
                                   int V, int Vp) {
    const int row = blockIdx.x;
    const float *x = logits + (size_t)row * Vp;
    float *p = probs + (size_t)row * Vp;

    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, x[i]);
    m = block_reduce<true>(m);

    float s = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float e = __expf(x[i] - m);   // shift for numerical stability
        p[i] = e;
        s += e;
    }
    s = block_reduce<false>(s);
    const float inv = 1.0f / s;

    for (int i = threadIdx.x; i < V; i += blockDim.x) p[i] *= inv;
    for (int i = V + threadIdx.x; i < Vp; i += blockDim.x) p[i] = 0.0f;

    if (threadIdx.x == 0) losses[row] = -logf(fmaxf(p[targets[row]], 1e-30f));
}

// d(loss)/d(logit_i) = (p_i - [i == target]) * scale. The softmax and the
// cross-entropy Jacobians collapse into this one expression, which is why
// they are fused rather than written as two kernels.
__global__ void xent_softmax_bwd_k(float *dlogits, const float *probs,
                                   const int *targets, int V, int Vp,
                                   float scale) {
    const int row = blockIdx.x;
    const float *p = probs + (size_t)row * Vp;
    float *d = dlogits + (size_t)row * Vp;
    const int tgt = targets[row];
    for (int i = threadIdx.x; i < Vp; i += blockDim.x) {
        const float indicator = (i == tgt) ? 1.0f : 0.0f;
        d[i] = (i < V) ? (p[i] - indicator) * scale : 0.0f;
    }
}

// ---------------------------------------------------------------- optimizer
__global__ void adamw_k(float *params, const float *grads, float *m, float *v,
                        int n, float lr, float beta1, float beta2, float eps,
                        float wd, float bc1, float bc2, float gscale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = grads[i] * gscale;
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi; v[i] = vi;
    const float mhat = mi / bc1;
    const float vhat = vi / bc2;
    // Decoupled weight decay: applied to the parameter, not folded into the
    // gradient, which is the "W" in AdamW.
    params[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * params[i]);
}

// Partial sums of squares; the last few hundred partials are finished on the
// host, which is cheaper than a second launch for this size.
__global__ void sumsq_k(const float *g, float *partial, int n) {
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float v = g[i];
        acc += v * v;
    }
    acc = block_reduce<false>(acc);
    if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

__global__ void reduce_mean_k(const float *in, float *out, int n) {
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += in[i];
    acc = block_reduce<false>(acc);
    if (threadIdx.x == 0) *out = acc / n;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// How many blocks it takes to fill this machine, derived rather than assumed.
// gemm.cu learned this the hard way three times over: a constant written as
// "4 blocks/SM x 36 SMs" is a 4070 fact, and it is wrong SILENTLY on every
// other card -- no error, just quiet under-filling.
inline int target_blocks() {
    static int cached = 0;
    if (cached) return cached;
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ==
            cudaSuccess && sms > 0)
        cached = 4 * sms;
    else
        cached = 144;
    return cached;
}
}  // namespace

// ---------------------------------------------------------------- launchers
void encoder_forward(float *out, const int *tokens, const float *wte,
                     const float *wpe, int B, int T, int C) {
    const int n4 = B * T * C / 4;
    encoder_fwd_k<<<ceil_div(n4, 256), 256>>>(out, tokens, wte, wpe, T, C, n4);
}

void encoder_backward(float *dwte, float *dwpe, const float *dout,
                      const int *tokens, int B, int T, int C) {
    const int total = B * T * C;
    encoder_bwd_k<<<ceil_div(total, 256), 256>>>(dwte, dwpe, dout, tokens, T, C, total);
}

void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                       const float *weight, const float *bias, int N, int C) {
    layernorm_fwd_k<<<N, 128>>>(out, mean, rstd, inp, weight, bias, C);
}

static int g_ln_blocks_force = 0;
void layernorm_backward_set_blocks(int b) { g_ln_blocks_force = b; }

void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        const float *dout, const float *inp,
                        const float *weight, const float *mean,
                        const float *rstd, int N, int C) {
    // One block per row was 4096 blocks doing one row each. Now a block owns a
    // strided set of rows so the per-column sums can live in its shared memory
    // for the whole walk -- which means the block count is a tuning parameter
    // rather than the row count, and it is derived from the SM count for the
    // usual reason (see fill_blocks in gemm.cu, and the three stale 36-SM
    // constants that cost this project a session each).
    // EIGHT BLOCKS PER SM, MEASURED IN THE STEP AND NOT IN A MICRO-BENCH.
    //
    // The block count trades two things against each other: more blocks give
    // the row loop more parallelism, and each one costs the second pass another
    // partial to sum per column. Swept in situ (`--ln-blocks`), on this card:
    //
    //   blocks       92    184    368    736   1024   2048   4096
    //   ms/step    2.79   1.75   1.53   1.82   2.02   2.75   4.49
    //
    // 368 is 8 x 46 SMs. Note the far end: at one block per row -- what this
    // kernel used to launch -- the reduction has 4096 partials per column to
    // sum and the region costs 4.49 ms, nearly three times the tuned value.
    // The old kernel got away with it only because it had no partials at all,
    // having paid in atomics instead.
    int blocks = g_ln_blocks_force > 0 ? g_ln_blocks_force : 2 * target_blocks();
    if (blocks > N) blocks = N;

    static thread_local float *part = nullptr;
    static thread_local size_t cap = 0;
    const size_t need = (size_t)blocks * C * 2;
    if (need > cap) {
        if (part) cudaFree(part);
        cudaMalloc(&part, need * sizeof(float));
        cap = need;
    }
    const size_t smem = (size_t)C * 2 * sizeof(float);
    layernorm_bwd_k<<<blocks, 128, smem>>>(dinp, part, part + (size_t)blocks * C,
                                           dout, inp, weight, mean, rstd, N, C);
    layernorm_bwd_reduce_k<<<ceil_div(C, 256), 256>>>(
        dweight, dbias, part, part + (size_t)blocks * C, C, blocks);
}

void gelu_forward(float *out, const float *inp, int n) {
    gelu_fwd_k<<<ceil_div(n / 4, 256), 256>>>(out, inp, n / 4);
}
void gelu_backward(float *dinp, const float *inp, const float *dout, int n) {
    gelu_bwd_k<<<ceil_div(n / 4, 256), 256>>>(dinp, inp, dout, n / 4);
}
void residual_forward(float *out, const float *a, const float *b, int n) {
    residual_fwd_k<<<ceil_div(n / 4, 256), 256>>>(out, a, b, n / 4);
}
void add_inplace(float *dst, const float *src, int n) {
    add_inplace_k<<<ceil_div(n / 4, 256), 256>>>(dst, src, n / 4);
}

static int g_bias_splits_force = 0;
void bias_backward_set_splits(int splits) { g_bias_splits_force = splits; }

void bias_backward(float *dbias, const float *dout, int N, int C) {
    // Enough row-splits to keep every SM busy on the narrow tensors, capped so
    // the second pass stays trivial. thread_local for the same reason
    // reduce_mean is: one host thread per GPU, one workspace per device.
    const int cols = ceil_div(C, BIAS_COLS);
    // THE FOURTH STALE 36-SM CONSTANT, and the first one outside gemm.cu. This
    // read `ceil_div(144, cols)` with the comment "~4 blocks per SM at 36 SMs".
    int splits = ceil_div(target_blocks(), cols);
    if (splits < 1) splits = 1;
    if (g_bias_splits_force > 0) splits = g_bias_splits_force;
    const int max_splits = ceil_div(N, BIAS_ROWS);
    if (splits > max_splits) splits = max_splits;

    static thread_local float *partial = nullptr;
    static thread_local size_t cap = 0;
    const size_t need = (size_t)splits * C;
    if (need > cap) {
        if (partial) cudaFree(partial);
        cudaMalloc(&partial, need * sizeof(float));
        cap = need;
    }

    dim3 block(BIAS_COLS, BIAS_ROWS);
    bias_bwd_k<<<dim3(cols, splits), block>>>(partial, dout, N, C);
    bias_bwd_reduce_k<<<ceil_div(C, 256), 256>>>(dbias, partial, C, splits);
}

void softmax_crossentropy_forward(float *probs, float *losses,
                                  const float *logits, const int *targets,
                                  int N, int V, int Vp) {
    softmax_xent_fwd_k<<<N, 128>>>(probs, losses, logits, targets, V, Vp);
}
void crossentropy_softmax_backward(float *dlogits, const float *probs,
                                   const int *targets, int N, int V, int Vp,
                                   float dloss_scale) {
    xent_softmax_bwd_k<<<N, 128>>>(dlogits, probs, targets, V, Vp, dloss_scale);
}

void adamw_update(float *params, float *grads, float *m, float *v, int n,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, int step, float grad_scale) {
    const float bc1 = 1.0f - powf(beta1, (float)step);
    const float bc2 = 1.0f - powf(beta2, (float)step);
    adamw_k<<<ceil_div(n, 256), 256>>>(params, grads, m, v, n, lr, beta1, beta2,
                                       eps, weight_decay, bc1, bc2, grad_scale);
}

float grad_global_norm(const float *grads, int n) {
    constexpr int NBLK = 256;
    // THREAD_LOCAL, for the same reason reduce_mean's scalar is. This was a
    // plain static, which on one device is merely a cache and on two devices
    // is a pointer into the wrong GPU's memory for whichever rank lost the
    // race to allocate it -- and with persistent rank workers both ranks hit
    // the null check in the same microsecond. It surfaced on the 2x A40 box
    // as NaN gradients from step 1 in one run out of three.
    static thread_local float *d_partial = nullptr;
    if (!d_partial) cudaMalloc(&d_partial, NBLK * sizeof(float));
    sumsq_k<<<NBLK, 256>>>(grads, d_partial, n);
    float h[NBLK];
    cudaMemcpy(h, d_partial, NBLK * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int i = 0; i < NBLK; ++i) s += (double)h[i];
    return (float)sqrt(s);
}

void zero_buffer(float *p, size_t n) { cudaMemset(p, 0, n * sizeof(float)); }

float reduce_mean(const float *d_values, int n) {
    // One persistent scalar rather than a malloc/free pair per call; this runs
    // once per forward pass and cudaMalloc is not cheap.
    //
    // THREAD_LOCAL, not static. Data-parallel training runs one host thread per
    // GPU, and a plain `static` here would hand every rank the same pointer --
    // allocated on whichever device happened to call first. Rank 1 would then
    // reduce into rank 0's memory, and both would race to read the single
    // scalar back. Thread-local storage gives each rank its own buffer on its
    // own device, which is both correct and what makes the copy below cheap.
    //
    // The host side is PINNED. A device-to-host copy out of pageable memory
    // goes through a driver staging buffer under a process-wide lock, so two
    // ranks doing it at the same moment serialise against each other for
    // reasons that have nothing to do with either GPU.
    static thread_local float *d_out = nullptr;
    static thread_local float *h_out = nullptr;
    if (!d_out) {
        cudaMalloc(&d_out, sizeof(float));
        cudaMallocHost(&h_out, sizeof(float));
    }
    reduce_mean_k<<<1, 256>>>(d_values, d_out, n);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    return *h_out;
}
