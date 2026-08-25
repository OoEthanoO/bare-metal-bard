#pragma once
#include <cstddef>

// GPT model: allocation, initialization, forward, backward.
//
// Separated from the training driver so the gradient checker can drive the
// same code the trainer uses. A gradient check that tests a reimplementation
// of the model proves nothing about the model.

struct GPTConfig {
    int max_seq_len;   // T
    int vocab_size;    // V, the real number of distinct tokens
    int padded_vocab;  // Vp, rounded up so the head matmul hits the fast GEMM
    int n_layer;       // L
    int n_head;        // NH
    int n_embd;        // C
};

constexpr int NUM_PARAM_TENSORS = 16;

struct Parameters {
    float *wte, *wpe;
    float *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb;
    float *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb;
    float *lnfw, *lnfb;
};

// Forward values the backward pass needs to revisit.
struct Activations {
    float *encoded;
    float *ln1, *ln1_mean, *ln1_rstd;
    float *qkv, *qkvr, *att, *atty;
    float *lse;  // fused path: (B, NH, T) log-sum-exp, replaces att entirely
    float *residual2, *ln2, *ln2_mean, *ln2_rstd;
    float *fch, *fch_gelu, *residual3;
    float *lnf, *lnf_mean, *lnf_rstd;
    float *logits, *probs, *losses;
};

// Backward scratch: one layer's worth, reused across all layers.
struct Gradients {
    float *dres;
    float *dln1, *dln2, *datty;
    float *dqkv, *dqkvr, *datt;
    float *dsum;  // fused path: (B, NH, T) rowsum(dout * out)
    float *dfch, *dfch_gelu;
    float *dlogits, *dlnf;
};

struct GPT {
    GPTConfig config;
    // Fused (FlashAttention-style) attention instead of the three-kernel path.
    // Changes what gets allocated, so it must be set before gpt_alloc.
    bool use_flash;
    int B, T;

    Parameters params, grads;
    size_t psize[NUM_PARAM_TENSORS];
    size_t num_params;
    float *params_mem, *grads_mem, *m_mem, *v_mem;

    Activations acts;
    Gradients gr;
    float *acts_mem, *grads_act_mem;
    size_t num_acts, num_grad_acts;

    int *d_tokens, *d_targets;
    // Pinned staging for the two per-step uploads. A copy out of pageable
    // memory blocks the calling thread and takes a driver-wide lock; from
    // pinned memory it is a plain async DMA on the stream.
    int *h_tokens, *h_targets;
};

void gpt_alloc(GPT &g);
void gpt_init(GPT &g, unsigned seed);

// Returns mean cross-entropy loss; pass targets = nullptr to skip the loss.
float gpt_forward(GPT &g, const int *tokens, const int *targets);
void gpt_backward(GPT &g);

// Names of the parameter tensors, indexed as in param_sizes.
extern const char *const PARAM_NAMES[NUM_PARAM_TENSORS];
void param_sizes(size_t *s, const GPTConfig &c);
