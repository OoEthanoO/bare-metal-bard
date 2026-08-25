// GPT model implementation: allocation, init, forward, backward.
//
// Every matmul here routes through the hand-written GEMM in src/gemm.cu.
// Nothing in this file links cuBLAS.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "gpt.h"
#include "gemm.h"
#include "nn.h"
#include "attention.h"
#include "flash.h"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e__ = (x);                                                 \
        if (e__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                    cudaGetErrorString(e__), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

const char *const PARAM_NAMES[NUM_PARAM_TENSORS] = {
    "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
    "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb"};

void param_sizes(size_t *s, const GPTConfig &c) {
    const size_t L = c.n_layer, C = c.n_embd, Vp = c.padded_vocab,
                 maxT = c.max_seq_len;
    s[0]  = Vp * C;          // wte      (also the output head, tied)
    s[1]  = maxT * C;        // wpe
    s[2]  = L * C;           // ln1w
    s[3]  = L * C;           // ln1b
    s[4]  = L * 3 * C * C;   // qkvw     (3C, C)
    s[5]  = L * 3 * C;       // qkvb
    s[6]  = L * C * C;       // attprojw (C, C)
    s[7]  = L * C;           // attprojb
    s[8]  = L * C;           // ln2w
    s[9]  = L * C;           // ln2b
    s[10] = L * 4 * C * C;   // fcw      (4C, C)
    s[11] = L * 4 * C;       // fcb
    s[12] = L * C * 4 * C;   // fcprojw  (C, 4C)
    s[13] = L * C;           // fcprojb
    s[14] = C;               // lnfw
    s[15] = C;               // lnfb
}

static void point_params(Parameters &p, float *base, const size_t *s) {
    float **ptrs[NUM_PARAM_TENSORS] = {
        &p.wte, &p.wpe, &p.ln1w, &p.ln1b, &p.qkvw, &p.qkvb, &p.attprojw,
        &p.attprojb, &p.ln2w, &p.ln2b, &p.fcw, &p.fcb, &p.fcprojw, &p.fcprojb,
        &p.lnfw, &p.lnfb};
    size_t off = 0;
    for (int i = 0; i < NUM_PARAM_TENSORS; ++i) {
        *ptrs[i] = base + off;
        off += s[i];
    }
}

struct Strides {
    size_t BTC, BT3C, BT4C, BTNHTT, BT, qkvr, lse;
};

static Strides make_strides(const GPT &g) {
    const size_t B = g.B, T = g.T, C = g.config.n_embd, NH = g.config.n_head;
    Strides s;
    s.BTC = B * T * C;
    s.BT3C = B * T * 3 * C;
    s.BT4C = B * T * 4 * C;
    s.BTNHTT = B * NH * T * T;
    s.BT = B * T;
    s.qkvr = 4 * B * T * C;  // q, k, v, plus the head-major output scratch
    s.lse = B * NH * T;      // fused path keeps this instead of att
    return s;
}

void gpt_alloc(GPT &g) {
    const GPTConfig &c = g.config;
    const int B = g.B, T = g.T, L = c.n_layer;

    // --- parameters, gradients, and the two AdamW moments ---
    param_sizes(g.psize, c);
    g.num_params = 0;
    for (int i = 0; i < NUM_PARAM_TENSORS; ++i) g.num_params += g.psize[i];

    CUDA_CHECK(cudaMalloc(&g.params_mem, g.num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.grads_mem, g.num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.m_mem, g.num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.v_mem, g.num_params * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.m_mem, 0, g.num_params * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.v_mem, 0, g.num_params * sizeof(float)));
    point_params(g.params, g.params_mem, g.psize);
    point_params(g.grads, g.grads_mem, g.psize);

    // --- forward activations, one arena ---
    const Strides s = make_strides(g);
    // The fused path is where most of the activation memory goes or does not
    // go: qkvr and att are 4*B*T*C and B*NH*T*T per layer, and the fused
    // kernel needs neither -- only B*NH*T floats of log-sum-exp.
    const size_t attn_mem = g.use_flash ? s.lse : s.qkvr + s.BTNHTT;
    const size_t per_layer = s.BTC     // ln1
                           + s.BT * 2  // ln1_mean, ln1_rstd
                           + s.BT3C    // qkv
                           + attn_mem  // qkvr + att, or just lse
                           + s.BTC     // atty
                           + s.BTC     // attproj
                           + s.BTC     // residual2
                           + s.BTC     // ln2
                           + s.BT * 2  // ln2_mean, ln2_rstd
                           + s.BT4C    // fch
                           + s.BT4C    // fch_gelu
                           + s.BTC     // fcproj
                           + s.BTC;    // residual3
    const size_t global = s.BTC        // encoded
                        + s.BTC        // lnf
                        + s.BT * 2     // lnf_mean, lnf_rstd
                        + (size_t)B * T * c.padded_vocab * 2  // logits, probs
                        + s.BT;        // losses
    g.num_acts = per_layer * L + global;
    CUDA_CHECK(cudaMalloc(&g.acts_mem, g.num_acts * sizeof(float)));

    float *p = g.acts_mem;
    auto take = [&p](size_t n) { float *r = p; p += n; return r; };
    g.acts.encoded   = take(s.BTC);
    g.acts.ln1       = take(s.BTC * L);
    g.acts.ln1_mean  = take(s.BT * L);
    g.acts.ln1_rstd  = take(s.BT * L);
    g.acts.qkv       = take(s.BT3C * L);
    g.acts.qkvr      = g.use_flash ? nullptr : take(s.qkvr * L);
    g.acts.att       = g.use_flash ? nullptr : take(s.BTNHTT * L);
    g.acts.lse       = g.use_flash ? take(s.lse * L) : nullptr;
    g.acts.atty      = take(s.BTC * L);
    g.acts.attproj   = take(s.BTC * L);
    g.acts.residual2 = take(s.BTC * L);
    g.acts.ln2       = take(s.BTC * L);
    g.acts.ln2_mean  = take(s.BT * L);
    g.acts.ln2_rstd  = take(s.BT * L);
    g.acts.fch       = take(s.BT4C * L);
    g.acts.fch_gelu  = take(s.BT4C * L);
    g.acts.fcproj    = take(s.BTC * L);
    g.acts.residual3 = take(s.BTC * L);
    g.acts.lnf       = take(s.BTC);
    g.acts.lnf_mean  = take(s.BT);
    g.acts.lnf_rstd  = take(s.BT);
    g.acts.logits    = take((size_t)B * T * c.padded_vocab);
    g.acts.probs     = take((size_t)B * T * c.padded_vocab);
    g.acts.losses    = take(s.BT);

    // --- backward scratch, single layer, reused ---
    const size_t dattn_mem = g.use_flash ? s.lse : s.qkvr + s.BTNHTT;
    g.num_grad_acts = s.BTC          // dres
                    + s.BTC * 3      // dln1, dln2, datty
                    + s.BT3C         // dqkv
                    + dattn_mem      // dqkvr + datt, or just dsum
                    + s.BT4C * 2     // dfch, dfch_gelu
                    + (size_t)B * T * c.padded_vocab  // dlogits
                    + s.BTC;         // dlnf
    CUDA_CHECK(cudaMalloc(&g.grads_act_mem, g.num_grad_acts * sizeof(float)));
    p = g.grads_act_mem;
    g.gr.dres      = take(s.BTC);
    g.gr.dln1      = take(s.BTC);
    g.gr.dln2      = take(s.BTC);
    g.gr.datty     = take(s.BTC);
    g.gr.dqkv      = take(s.BT3C);
    g.gr.dqkvr     = g.use_flash ? nullptr : take(s.qkvr);
    g.gr.datt      = g.use_flash ? nullptr : take(s.BTNHTT);
    g.gr.dsum      = g.use_flash ? take(s.lse) : nullptr;
    g.gr.dfch      = take(s.BT4C);
    g.gr.dfch_gelu = take(s.BT4C);
    g.gr.dlogits   = take((size_t)B * T * c.padded_vocab);
    g.gr.dlnf      = take(s.BTC);

    CUDA_CHECK(cudaMallocHost(&g.h_tokens, (size_t)B * T * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&g.h_targets, (size_t)B * T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_tokens, (size_t)B * T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_targets, (size_t)B * T * sizeof(int)));
    // Sampling reuses the loss kernel, which indexes probs[targets[row]]. Zero
    // this so that path can never dereference uninitialized indices before the
    // first training step has written real targets.
    CUDA_CHECK(cudaMemset(g.d_targets, 0, (size_t)B * T * sizeof(int)));
}

// GPT-2 initialization: normal(0, 0.02), except the two projections that write
// into the residual stream, which are scaled by 1/sqrt(2*L) so the variance of
// the residual stream does not grow with depth.
void gpt_init(GPT &g, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> h(g.num_params, 0.0f);

    Parameters hp;
    point_params(hp, h.data(), g.psize);
    const float resid_scale = 0.02f / sqrtf(2.0f * g.config.n_layer);

    auto fill = [&](float *dst, size_t n, float std) {
        std::normal_distribution<float> d(0.0f, std);
        for (size_t i = 0; i < n; ++i) dst[i] = d(rng);
    };
    auto ones = [&](float *dst, size_t n) {
        for (size_t i = 0; i < n; ++i) dst[i] = 1.0f;
    };

    fill(hp.wte, g.psize[0], 0.02f);
    fill(hp.wpe, g.psize[1], 0.02f);
    ones(hp.ln1w, g.psize[2]);                 // ln biases stay zero
    fill(hp.qkvw, g.psize[4], 0.02f);
    fill(hp.attprojw, g.psize[6], resid_scale);
    ones(hp.ln2w, g.psize[8]);
    fill(hp.fcw, g.psize[10], 0.02f);
    fill(hp.fcprojw, g.psize[12], resid_scale);
    ones(hp.lnfw, g.psize[14]);

    CUDA_CHECK(cudaMemcpy(g.params_mem, h.data(),
                          g.num_params * sizeof(float), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------- forward
// Returns the mean cross-entropy loss. `targets` may be null for inference.
float gpt_forward(GPT &g, const int *tokens, const int *targets) {
    const GPTConfig &c = g.config;
    const int B = g.B, T = g.T, C = c.n_embd, L = c.n_layer, NH = c.n_head;
    const int Vp = c.padded_vocab, V = c.vocab_size;
    const int N = B * T;  // rows for every matmul
    const Strides s = make_strides(g);
    Activations &a = g.acts;
    Parameters &p = g.params;

    // Stage through pinned memory, then hand the copy to the stream. The
    // host-to-host memcpy is nearly free; the win is that the device copy no
    // longer blocks this thread, which is what let one rank's forward pass
    // stall another rank's on a different GPU entirely.
    memcpy(g.h_tokens, tokens, (size_t)N * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(g.d_tokens, g.h_tokens, (size_t)N * sizeof(int),
                               cudaMemcpyHostToDevice));
    encoder_forward(a.encoded, g.d_tokens, p.wte, p.wpe, B, T, C);

    float *residual = a.encoded;  // the running residual stream
    for (int l = 0; l < L; ++l) {
        float *ln1 = a.ln1 + l * s.BTC;
        float *ln1_mean = a.ln1_mean + l * s.BT, *ln1_rstd = a.ln1_rstd + l * s.BT;
        float *qkv = a.qkv + l * s.BT3C, *qkvr = a.qkvr + l * s.qkvr;
        float *att = a.att + l * s.BTNHTT, *atty = a.atty + l * s.BTC;
        float *attproj = a.attproj + l * s.BTC, *residual2 = a.residual2 + l * s.BTC;
        float *ln2 = a.ln2 + l * s.BTC;
        float *ln2_mean = a.ln2_mean + l * s.BT, *ln2_rstd = a.ln2_rstd + l * s.BT;
        float *fch = a.fch + l * s.BT4C, *fch_gelu = a.fch_gelu + l * s.BT4C;
        float *fcproj = a.fcproj + l * s.BTC, *residual3 = a.residual3 + l * s.BTC;

        const float *ln1w = p.ln1w + l * C, *ln1b = p.ln1b + l * C;
        const float *qkvw = p.qkvw + (size_t)l * 3 * C * C, *qkvb = p.qkvb + l * 3 * C;
        const float *apw = p.attprojw + (size_t)l * C * C, *apb = p.attprojb + l * C;
        const float *ln2w = p.ln2w + l * C, *ln2b = p.ln2b + l * C;
        const float *fcw = p.fcw + (size_t)l * 4 * C * C, *fcb = p.fcb + l * 4 * C;
        const float *fpw = p.fcprojw + (size_t)l * C * 4 * C, *fpb = p.fcprojb + l * C;

        layernorm_forward(ln1, ln1_mean, ln1_rstd, residual, ln1w, ln1b, N, C);
        // Weights are (out, in), so every forward matmul is the transB case.
        // The bias rides in the GEMM epilogue. As its own kernel it read and
        // wrote the whole (N, 3C) tensor to do one add per element, and the
        // profile put all four of these together at 8.2% of a training step.
        gemm(false, true, N, 3 * C, C, 1.0f, ln1, qkvw, 0.0f, qkv, 0, qkvb);
        if (g.use_flash)
            flash_attention_forward(atty, a.lse + l * s.lse, qkv, B, T, C, NH);
        else
            attention_forward(atty, qkvr, att, qkv, B, T, C, NH);
        gemm(false, true, N, C, C, 1.0f, atty, apw, 0.0f, attproj, 0, apb);
        residual_forward(residual2, residual, attproj, N * C);

        layernorm_forward(ln2, ln2_mean, ln2_rstd, residual2, ln2w, ln2b, N, C);
        gemm(false, true, N, 4 * C, C, 1.0f, ln2, fcw, 0.0f, fch, 0, fcb);
        gelu_forward(fch_gelu, fch, N * 4 * C);
        gemm(false, true, N, C, 4 * C, 1.0f, fch_gelu, fpw, 0.0f, fcproj, 0, fpb);
        residual_forward(residual3, residual2, fcproj, N * C);

        residual = residual3;
    }

    layernorm_forward(a.lnf, a.lnf_mean, a.lnf_rstd, residual, p.lnfw, p.lnfb, N, C);
    // Tied head: logits = lnf @ wte^T.
    gemm(false, true, N, Vp, C, 1.0f, a.lnf, p.wte, 0.0f, a.logits);

    if (!targets) return 0.0f;

    memcpy(g.h_targets, targets, (size_t)N * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(g.d_targets, g.h_targets, (size_t)N * sizeof(int),
                               cudaMemcpyHostToDevice));
    softmax_crossentropy_forward(a.probs, a.losses, a.logits, g.d_targets, N, V, Vp);
    return reduce_mean(a.losses, N);
}

// --------------------------------------------------------------- backward
void gpt_backward(GPT &g) {
    const GPTConfig &c = g.config;
    const int B = g.B, T = g.T, C = c.n_embd, L = c.n_layer, NH = c.n_head;
    const int Vp = c.padded_vocab, V = c.vocab_size;
    const int N = B * T;
    const Strides s = make_strides(g);
    Activations &a = g.acts;
    Parameters &p = g.params, &d = g.grads;
    Gradients &gr = g.gr;

    CUDA_CHECK(cudaMemset(g.grads_mem, 0, g.num_params * sizeof(float)));

    // Mean over B*T positions, so each position's gradient carries 1/N.
    crossentropy_softmax_backward(gr.dlogits, a.probs, g.d_targets, N, V, Vp,
                                  1.0f / N);

    // Tied head: logits = lnf @ wte^T.
    gemm(false, false, N, C, Vp, 1.0f, gr.dlogits, p.wte, 0.0f, gr.dlnf);
    gemm(true, false, Vp, C, N, 1.0f, gr.dlogits, a.lnf, 1.0f, d.wte);

    float *last_residual = a.residual3 + (size_t)(L - 1) * s.BTC;
    CUDA_CHECK(cudaMemset(gr.dres, 0, s.BTC * sizeof(float)));
    layernorm_backward(gr.dres, d.lnfw, d.lnfb, gr.dlnf, last_residual, p.lnfw,
                       a.lnf_mean, a.lnf_rstd, N, C);

    for (int l = L - 1; l >= 0; --l) {
        // The input to layer l is layer l-1's output, or the embedding.
        float *residual = (l == 0) ? a.encoded : a.residual3 + (size_t)(l - 1) * s.BTC;

        float *ln1 = a.ln1 + l * s.BTC;
        float *ln1_mean = a.ln1_mean + l * s.BT, *ln1_rstd = a.ln1_rstd + l * s.BT;
        // Backward reads the PERMUTED q/k/v (qkvr), not the raw fused qkv.
        float *qkvr = a.qkvr + l * s.qkvr;
        float *qkv = a.qkv + l * s.BT3C;
        float *att = a.att + l * s.BTNHTT, *atty = a.atty + l * s.BTC;
        float *residual2 = a.residual2 + l * s.BTC, *ln2 = a.ln2 + l * s.BTC;
        float *ln2_mean = a.ln2_mean + l * s.BT, *ln2_rstd = a.ln2_rstd + l * s.BT;
        float *fch = a.fch + l * s.BT4C, *fch_gelu = a.fch_gelu + l * s.BT4C;

        const float *ln1w = p.ln1w + l * C;
        const float *qkvw = p.qkvw + (size_t)l * 3 * C * C;
        const float *apw = p.attprojw + (size_t)l * C * C;
        const float *ln2w = p.ln2w + l * C;
        const float *fcw = p.fcw + (size_t)l * 4 * C * C;
        const float *fpw = p.fcprojw + (size_t)l * C * 4 * C;

        float *dln1w = d.ln1w + l * C, *dln1b = d.ln1b + l * C;
        float *dqkvw = d.qkvw + (size_t)l * 3 * C * C, *dqkvb = d.qkvb + l * 3 * C;
        float *dapw = d.attprojw + (size_t)l * C * C, *dapb = d.attprojb + l * C;
        float *dln2w = d.ln2w + l * C, *dln2b = d.ln2b + l * C;
        float *dfcw = d.fcw + (size_t)l * 4 * C * C, *dfcb = d.fcb + l * 4 * C;
        float *dfpw = d.fcprojw + (size_t)l * C * 4 * C, *dfpb = d.fcprojb + l * C;

        // gr.dres currently holds d(residual3). Since residual3 =
        // residual2 + fcproj, that is also d(fcproj) exactly.
        gemm(false, false, N, 4 * C, C, 1.0f, gr.dres, fpw, 0.0f, gr.dfch_gelu);
        gemm(true, false, C, 4 * C, N, 1.0f, gr.dres, fch_gelu, 1.0f, dfpw);
        bias_backward(dfpb, gr.dres, N, C);

        gelu_backward(gr.dfch, fch, gr.dfch_gelu, N * 4 * C);

        gemm(false, false, N, C, 4 * C, 1.0f, gr.dfch, fcw, 0.0f, gr.dln2);
        gemm(true, false, 4 * C, C, N, 1.0f, gr.dfch, ln2, 1.0f, dfcw);
        bias_backward(dfcb, gr.dfch, N, 4 * C);

        // Accumulates into gr.dres, turning d(residual3) into d(residual2).
        layernorm_backward(gr.dres, dln2w, dln2b, gr.dln2, residual2, ln2w,
                           ln2_mean, ln2_rstd, N, C);

        // Now gr.dres holds d(residual2) = d(attproj).
        gemm(false, false, N, C, C, 1.0f, gr.dres, apw, 0.0f, gr.datty);
        gemm(true, false, C, C, N, 1.0f, gr.dres, atty, 1.0f, dapw);
        bias_backward(dapb, gr.dres, N, C);

        // The fused backward reads the raw qkv and the attention output, both
        // already saved, and rebuilds the probabilities from lse. That is why
        // it needs no score matrix: nothing here was thrown away that cannot
        // be recomputed in registers.
        if (g.use_flash)
            flash_attention_backward(gr.dqkv, gr.dsum, gr.datty, qkv, atty,
                                     a.lse + l * s.lse, B, T, C, NH);
        else
            attention_backward(gr.dqkv, gr.dqkvr, gr.datt, gr.datt, gr.datty,
                               qkvr, att, B, T, C, NH);

        gemm(false, false, N, C, 3 * C, 1.0f, gr.dqkv, qkvw, 0.0f, gr.dln1);
        gemm(true, false, 3 * C, C, N, 1.0f, gr.dqkv, ln1, 1.0f, dqkvw);
        bias_backward(dqkvb, gr.dqkv, N, 3 * C);

        // Accumulates again, turning d(residual2) into d(layer input).
        layernorm_backward(gr.dres, dln1w, dln1b, gr.dln1, residual, ln1w,
                           ln1_mean, ln1_rstd, N, C);
    }

    encoder_backward(d.wte, d.wpe, gr.dres, g.d_tokens, B, T, C);
}

