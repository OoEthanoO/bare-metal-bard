// A GPT trained end to end with hand-written CUDA kernels.
//
// No PyTorch, no cuBLAS, no cuDNN. Every matmul goes through the warp-tiled
// GEMM developed in src/kernels/, and every other operation -- layernorm,
// softmax, GELU, attention, cross-entropy, AdamW -- is in src/nn.cu and
// src/attention.cu.
//
// Two structural decisions worth naming:
//
// * Weight tying. The token embedding doubles as the output head, so the
//   logits matmul is lnf @ wte^T and its gradient accumulates into the same
//   dwte the encoder writes. This is what GPT-2 does.
//
// * In-place residual gradient. The backward pass carries ONE (B,T,C) buffer
//   for the residual stream. Because residual3 = residual2 + fcproj implies
//   d(residual2) = d(residual3) + (path through ln2), and layernorm_backward
//   accumulates into its output, the same buffer can be updated in place as
//   the gradient walks back through the block. That turns what would be a
//   ~1 GB mirror of all forward activations into ~140 MB of scratch.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "gpt.h"
#include "gemm.h"
#include "nn.h"
#include "ddp.h"
#include "flash.h"
#include <chrono>
#include <thread>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e__ = (x);                                                 \
        if (e__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                    cudaGetErrorString(e__), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------- sampling
static int sample_from_probs(const float *probs, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        cdf += probs[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static std::string generate(GPT &g, const std::vector<char> &itos, int steps,
                            std::mt19937 &rng, float temperature) {
    const int T = g.T, Vp = g.config.padded_vocab, V = g.config.vocab_size;
    std::vector<int> ctx((size_t)g.B * T, 0);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::string out;

    // Start from a newline so the sample begins at a line boundary.
    int cur_len = 1;
    ctx[0] = 0;
    for (size_t i = 0; i < itos.size(); ++i)
        if (itos[i] == '\n') { ctx[0] = (int)i; break; }

    std::vector<float> row(Vp);
    for (int step = 0; step < steps; ++step) {
        gpt_forward(g, ctx.data(), nullptr);
        // The next-token distribution lives at the last filled position.
        const int pos = cur_len - 1;
        softmax_crossentropy_forward(g.acts.probs, g.acts.losses, g.acts.logits,
                                     g.d_targets, g.B * T, V, Vp);
        CUDA_CHECK(cudaMemcpy(row.data(), g.acts.probs + (size_t)pos * Vp,
                              Vp * sizeof(float), cudaMemcpyDeviceToHost));

        if (temperature != 1.0f) {
            float sum = 0.0f;
            for (int i = 0; i < V; ++i) {
                row[i] = powf(row[i], 1.0f / temperature);
                sum += row[i];
            }
            for (int i = 0; i < V; ++i) row[i] /= sum;
        }

        const int tok = sample_from_probs(row.data(), V, uni(rng));
        out.push_back(itos[tok]);

        if (cur_len < T) {
            ctx[cur_len] = tok;
            ++cur_len;
        } else {
            // Context is full: slide the window left by one.
            for (int i = 0; i < T - 1; ++i) ctx[i] = ctx[i + 1];
            ctx[T - 1] = tok;
        }
    }
    return out;
}

// ------------------------------------------------------------- data loader
struct Dataset {
    std::vector<int> train, val;
    std::vector<char> itos;
};

static Dataset load_chars(const char *path, float val_frac) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string text(n, '\0');
    if (fread(&text[0], 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);

    // Character-level vocabulary: sorted unique bytes.
    std::vector<int> present(256, 0);
    for (unsigned char ch : text) present[ch] = 1;
    std::vector<int> stoi(256, -1);
    Dataset ds;
    for (int i = 0; i < 256; ++i)
        if (present[i]) { stoi[i] = (int)ds.itos.size(); ds.itos.push_back((char)i); }

    std::vector<int> all(text.size());
    for (size_t i = 0; i < text.size(); ++i)
        all[i] = stoi[(unsigned char)text[i]];

    const size_t nval = (size_t)(all.size() * val_frac);
    const size_t ntrain = all.size() - nval;
    ds.train.assign(all.begin(), all.begin() + ntrain);
    ds.val.assign(all.begin() + ntrain, all.end());
    return ds;
}

// Samples B random windows of length T+1: inputs are [0,T), targets [1,T+1).
static void get_batch(const std::vector<int> &data, int B, int T,
                      std::mt19937 &rng, std::vector<int> &x,
                      std::vector<int> &y) {
    std::uniform_int_distribution<size_t> pick(0, data.size() - T - 2);
    for (int b = 0; b < B; ++b) {
        const size_t off = pick(rng);
        for (int t = 0; t < T; ++t) {
            x[(size_t)b * T + t] = data[off + t];
            y[(size_t)b * T + t] = data[off + t + 1];
        }
    }
}

// Validation loss over several batches.
//
// A single batch of B*T tokens is a noisy estimate -- during development the
// single-batch number swung by 0.17 nats between adjacent evals, which is
// larger than the improvement being measured over the same interval. Averaging
// over several independent batches, drawn from a dedicated RNG so evaluation
// never perturbs the training data order, makes the curve mean something.
static float evaluate(GPT &g, const std::vector<int> &data, int B, int T,
                      unsigned seed, int n_batches) {
    std::mt19937 erng(seed);
    std::vector<int> x((size_t)B * T), y((size_t)B * T);
    double total = 0.0;
    for (int i = 0; i < n_batches; ++i) {
        get_batch(data, B, T, erng, x, y);
        total += gpt_forward(g, x.data(), y.data());
    }
    return (float)(total / n_batches);
}

static void save_checkpoint(const GPT &g, const char *path,
                            const std::vector<char> &itos, bool verbose = true) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    const int magic = 0x47505431;  // "GPT1"
    fwrite(&magic, sizeof(int), 1, f);
    fwrite(&g.config, sizeof(GPTConfig), 1, f);
    const int nvocab = (int)itos.size();
    fwrite(&nvocab, sizeof(int), 1, f);
    fwrite(itos.data(), 1, itos.size(), f);
    std::vector<float> h(g.num_params);
    cudaMemcpy(h.data(), g.params_mem, g.num_params * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(h.data(), sizeof(float), g.num_params, f);
    fclose(f);
    if (verbose)
        printf("saved checkpoint -> %s (%.1f MB)\n", path,
               g.num_params * 4.0 / 1048576.0);
}

// Reads config and vocabulary from the file, so a checkpoint is
// self-describing and sampling needs no matching command-line flags.
static bool load_checkpoint(GPT &g, const char *path, std::vector<char> &itos) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    int magic = 0;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != 0x47505431) {
        fprintf(stderr, "%s is not a GPT1 checkpoint\n", path);
        fclose(f);
        return false;
    }
    if (fread(&g.config, sizeof(GPTConfig), 1, f) != 1) { fclose(f); return false; }
    int nvocab = 0;
    if (fread(&nvocab, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    itos.resize(nvocab);
    if (fread(itos.data(), 1, nvocab, f) != (size_t)nvocab) { fclose(f); return false; }

    g.T = g.config.max_seq_len;
    gpt_alloc(g);
    std::vector<float> h(g.num_params);
    if (fread(h.data(), sizeof(float), g.num_params, f) != g.num_params) {
        fprintf(stderr, "checkpoint truncated\n");
        fclose(f);
        return false;
    }
    fclose(f);
    cudaMemcpy(g.params_mem, h.data(), g.num_params * sizeof(float),
               cudaMemcpyHostToDevice);
    printf("loaded %s: %d layers, %d heads, %d embd, ctx %d, vocab %d, %.2fM params\n",
           path, g.config.n_layer, g.config.n_head, g.config.n_embd,
           g.config.max_seq_len, g.config.vocab_size, g.num_params / 1e6);
    return true;
}

// Linear warmup then cosine decay to lr_max/10. Warmup matters here because
// Adam's second-moment estimate is near-meaningless for the first few steps,
// and a full-size step taken against it can knock the model into a bad basin
// it spends thousands of steps climbing out of.
static float lr_at(int step, int warmup, int total, float lr_max) {
    const float lr_min = lr_max * 0.1f;
    if (step <= warmup) return lr_max * (float)step / (float)warmup;
    const float p = (float)(step - warmup) / (float)std::max(1, total - warmup);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159265f * p));
}

// ------------------------------------------------------------------- main
int main(int argc, char **argv) {
    const char *data_path = "data/input.txt";
    int steps = 2000, B = 16, T = 256, n_layer = 6, n_head = 6, n_embd = 384;
    float lr = 1e-3f, weight_decay = 0.1f, grad_clip = 1.0f;
    int warmup = 100;
    int eval_every = 100, sample_every = 500, sample_len = 400;
    int eval_batches = 20;
    const char *save_path = "bench/gpt.bin";
    const char *load_path = nullptr;
    float temperature = 0.8f;
    unsigned seed = 1337;
    bool use_flash = true;  // --unfused selects the three-kernel attention
    int bwd_cfg = -1;       // --bwd-cfg pins a fused-backward tile; -1 = measured rule
    bool alloc_only = false;
    int nranks = 1;  // --gpus N: data-parallel replicas
    bool tf32 = false;  // --tf32: route the matmuls through the tensor cores

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) data_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-l") && i + 1 < argc) n_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--clip") && i + 1 < argc) grad_clip = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eval") && i + 1 < argc) eval_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample") && i + 1 < argc) sample_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eval-batches") && i + 1 < argc) eval_batches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) save_path = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) load_path = argv[++i];
        else if (!strcmp(argv[i], "--len") && i + 1 < argc) sample_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--alloc-only")) alloc_only = true;
        else if (!strcmp(argv[i], "--gpus") && i + 1 < argc) nranks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf32")) tf32 = true;
        else if (!strcmp(argv[i], "--unfused")) use_flash = false;
        // Pin the fused-backward tile config, for A/B without a rebuild.
        else if (!strcmp(argv[i], "--bwd-cfg") && i + 1 < argc) bwd_cfg = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    // --load: restore a checkpoint and generate, no training, no dataset.
    if (load_path) {
        GPT lg;
        lg.B = B;
        lg.use_flash = true;
        std::vector<char> itos;
        if (!load_checkpoint(lg, load_path, itos)) return 1;
        std::mt19937 srng(seed + 1);
        printf("\n%s\n", generate(lg, itos, sample_len, srng, temperature).c_str());
        return 0;
    }

    Dataset ds = load_chars(data_path, 0.1f);
    const int V = (int)ds.itos.size();
    const int Vp = ((V + 127) / 128) * 128;  // pad so the head matmul is aligned

    // Data parallel: each rank owns a full replica of the model and a shard of
    // the batch. Ranks are placed round-robin over the physical devices, so
    // --gpus 2 on one GPU is a faithful rehearsal of the whole path (identical
    // arithmetic, identical collective) with only the transport differing.
    if (B % nranks != 0) {
        fprintf(stderr, "batch %d does not divide across %d ranks\n", B, nranks);
        return 1;
    }
    const int Bshard = B / nranks;
    gemm_set_tf32(tf32);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { fprintf(stderr, "no CUDA device\n"); return 1; }
    std::vector<int> devs(nranks);
    for (int r = 0; r < nranks; ++r) devs[r] = r % ndev;

    std::vector<GPT> reps(nranks);
    for (int r = 0; r < nranks; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        reps[r].config = {T, V, Vp, n_layer, n_head, n_embd};
        reps[r].B = Bshard;
        reps[r].T = T;
        reps[r].use_flash = use_flash;
    }
    GPT &g = reps[0];

    printf("data      %zu train / %zu val tokens, vocab %d (padded %d)\n",
           ds.train.size(), ds.val.size(), V, Vp);

    for (int r = 0; r < nranks; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        gpt_alloc(reps[r]);
        // Same seed on every rank, so the replicas start identical and the
        // all-reduce is the only thing keeping them that way.
        gpt_init(reps[r], seed);
    }
    DDP ddp;
    ddp_init(ddp, nranks, devs.data());
    CUDA_CHECK(cudaSetDevice(devs[0]));

    const double param_mb = g.num_params * 4.0 / 1048576.0;
    const double act_mb = g.num_acts * 4.0 / 1048576.0;
    const double gact_mb = g.num_grad_acts * 4.0 / 1048576.0;
    printf("model     %d layers, %d heads, %d embd, ctx %d, %s attention\n",
           n_layer, n_head, n_embd, T, use_flash ? "fused" : "unfused");
    printf("matmul    %s\n", tf32 ? "TF32 tensor cores" : "fp32");
    if (bwd_cfg >= 0) {
        flash_set_bwd_config(bwd_cfg);
        printf("bwd tile  config %d pinned (%s)\n", bwd_cfg,
               flash_bwd_config_name(bwd_cfg));
    }
    printf("params    %.2fM (%.1f MB; +%.1f MB grads, +%.1f MB adam state)\n",
           g.num_params / 1e6, param_mb, param_mb, 2 * param_mb);
    printf("memory    %.1f MB forward activations, %.1f MB backward scratch\n",
           act_mb, gact_mb);
    printf("total     %.2f GB resident (params+grads+adam+activations+scratch)\n",
           (4.0 * param_mb + act_mb + gact_mb) / 1024.0);
    if (nranks > 1) ddp_report_topology(ddp);
    // Allocation is the whole question for a context-length study, and a step
    // at long context is slow. Note that on Windows the driver will happily
    // oversubscribe VRAM into system memory, so a successful cudaMalloc is NOT
    // evidence that the model fits -- the printed total against the card's
    // memory is.
    if (alloc_only) return 0;

    float best_val = 1e30f;
    int best_step = -1;  // -1 until an evaluation actually produces one
    std::mt19937 rng(seed), sample_rng(seed + 1);
    std::vector<int> x((size_t)B * T), y((size_t)B * T);

    // Flops per step: 6*N*P for fwd+bwd through the parameters, plus the
    // attention terms which do not scale with parameter count.
    const double tokens_per_step = (double)B * T;
    const double flops_per_step =
        6.0 * tokens_per_step * g.num_params +
        12.0 * n_layer * n_head * (double)T * T * (n_embd / n_head) * B;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    double ema_ms = 0.0, ema_comm = 0.0;
    std::vector<float *> gbufs(nranks);

    for (int step = 1; step <= steps; ++step) {
        get_batch(ds.train, B, T, rng, x, y);

        // Wall-clock, not events: with several devices in flight the thing
        // worth timing is the step, and an event on one device's stream does
        // not see the others.
        for (int r = 0; r < nranks; ++r) {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        const auto t_begin = std::chrono::steady_clock::now();

        // Each rank runs its own shard of the batch. Averaging the shard means
        // reproduces the full-batch mean exactly, so N ranks of B/N compute the
        // same gradient as one rank of B.
        // One host thread per rank. CUDA's current device is per-THREAD, so
        // each thread simply selects its own GPU and the two run concurrently.
        //
        // Driving both from one thread does not work: gpt_forward blocks on a
        // token upload and on the loss reduction, so rank 1 could not start
        // until rank 0 had finished. Two GPUs then take longer than one, and
        // the profile blames nothing in particular because the stall is in the
        // host loop rather than on the device.
        std::vector<double> rank_loss(nranks, 0.0);
        if (nranks == 1) {
            CUDA_CHECK(cudaSetDevice(devs[0]));
            rank_loss[0] = gpt_forward(reps[0], x.data(), y.data());
            gpt_backward(reps[0]);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(nranks);
            for (int r = 0; r < nranks; ++r) {
                workers.emplace_back([&, r] {
                    cudaSetDevice(devs[r]);
                    const size_t off = (size_t)r * Bshard * T;
                    rank_loss[r] =
                        gpt_forward(reps[r], x.data() + off, y.data() + off);
                    gpt_backward(reps[r]);
                    cudaDeviceSynchronize();
                });
            }
            for (auto &w : workers) w.join();
        }
        double loss_sum = 0.0;
        for (int r = 0; r < nranks; ++r) loss_sum += rank_loss[r];
        const float loss = (float)(loss_sum / nranks);
        // Without this the compute/communication split is a lie: the backward
        // pass is still running on each device's default stream, and because
        // the legacy default stream synchronises with other streams, the
        // all-reduce's first transfer waits for it. That tail then gets counted
        // as communication. It read 31 ms of "comm" for a collective that
        // takes 1.5 ms in isolation.
        for (int r = 0; r < nranks; ++r) {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        const auto t_compute = std::chrono::steady_clock::now();

        // ---- the only line that needs more than one GPU to mean anything ----
        for (int r = 0; r < nranks; ++r) gbufs[r] = reps[r].grads_mem;
        ddp_allreduce(ddp, gbufs.data(), g.num_params);
        const auto t_comm = std::chrono::steady_clock::now();

        // Global-norm clipping: rescale the whole gradient vector so its L2
        // norm is at most grad_clip. Rescaling globally rather than per-tensor
        // preserves the gradient's direction, which is the point.
        //
        // The all-reduce leaves the SUM over ranks, so both the norm and the
        // update carry a 1/nranks to turn it back into the mean.
        const float lr_now = lr_at(step, warmup, steps, lr);
        float gnorm = 0.0f;
        auto opt_step = [&](int r) {
            cudaSetDevice(devs[r]);
            const float gn =
                grad_global_norm(reps[r].grads_mem, (int)reps[r].num_params) /
                nranks;
            if (r == 0) gnorm = gn;
            const float gs = ((gn > grad_clip) ? grad_clip / gn : 1.0f) / nranks;
            adamw_update(reps[r].params_mem, reps[r].grads_mem, reps[r].m_mem,
                         reps[r].v_mem, (int)reps[r].num_params, lr_now, 0.9f,
                         0.999f, 1e-8f, weight_decay, step, gs);
            cudaDeviceSynchronize();
        };
        if (nranks == 1) {
            opt_step(0);
        } else {
            std::vector<std::thread> workers;
            for (int r = 0; r < nranks; ++r) workers.emplace_back(opt_step, r);
            for (auto &w : workers) w.join();
        }
        for (int r = 0; r < nranks; ++r) {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(devs[0]));
        const auto t_end = std::chrono::steady_clock::now();

        auto msec = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const float ms = (float)msec(t_begin, t_end);
        const double comm_ms = msec(t_compute, t_comm);
        ema_ms = (step == 1) ? ms : 0.9 * ema_ms + 0.1 * ms;
        ema_comm = (step == 1) ? comm_ms : 0.9 * ema_comm + 0.1 * comm_ms;

        if (step % 10 == 0 || step == 1) {
            printf("step %5d/%d  loss %.4f  lr %.2e  |g| %.3f  %6.1f ms  %7.0f tok/s  %5.0f GFLOP/s",
                   step, steps, loss, lr_now, gnorm, ms,
                   tokens_per_step / (ms * 1e-3),
                   flops_per_step / (ms * 1e-3) / 1e9);
            if (nranks > 1)
                printf("  comm %5.1f ms (%4.1f%%)", comm_ms, 100.0 * comm_ms / ms);
            printf("\n");
            fflush(stdout);
        }

        if (eval_every > 0 && step % eval_every == 0) {
            // Fixed seed: every eval sees the SAME validation batches, so the
            // curve reflects the model changing rather than the sample.
            const float vl = evaluate(g, ds.val, B, T, 7777, eval_batches);
            const bool best = vl < best_val;
            if (best) {
                best_val = vl;
                best_step = step;
                // 10.8M params on ~1 MB of text overfits, so the final-step
                // model is not the best one. Keep the best-validation weights.
                if (save_path && *save_path) save_checkpoint(g, save_path, ds.itos, false);
            }
            printf("  [eval] step %d  val loss %.4f  (%d batches)%s\n", step, vl,
                   eval_batches, best ? "  <- best" : "");
            fflush(stdout);
        }

        if (sample_every > 0 && step % sample_every == 0) {
            const std::string txt = generate(g, ds.itos, sample_len, sample_rng, temperature);
            printf("  [sample] ------------------------------------------\n%s\n", txt.c_str());
            printf("  ---------------------------------------------------\n");
            fflush(stdout);
        }
    }

    // --eval-batches 0 suppresses evaluation entirely. That exists for the
    // profiler: `--eval 99999` still evaluated at step 0 and twice at the
    // end, and a forward-only pass over 20 validation batches swamped a
    // 2-step profile -- the first run of tools/step_profile.py reported
    // forward kernels 21x more often than their own backward kernels,
    // which is the tell.
    if (eval_batches > 0) {
        const float final_val = evaluate(g, ds.val, B, T, 7777, eval_batches);
        const float final_train = evaluate(g, ds.train, B, T, 8888, eval_batches);
        printf("\nfinal  train loss %.4f   val loss %.4f   (%d batches each)\n",
               final_train, final_val, eval_batches);
    }

    // With --eval 0 no evaluation ever runs, so there is no best to report and
    // no checkpoint was written. Printing the sentinel here claimed a best val
    // loss of 1e30 "at step 0 (checkpoint saved there)", which is three false
    // statements in one line -- and the run it appeared under was a benchmark,
    // where evaluation is switched off precisely so it cannot cost time.
    if (best_step >= 0)
        printf("best   val loss %.4f at step %d (checkpoint saved there)\n",
               best_val, best_step);

    // Generation is autoregressive -- one full forward per token -- so this is
    // not free, and at long context it dwarfs the training it is reporting on.
    // --len 0 skips it, which is what you want for a short measurement run.
    if (sample_len > 0)
        printf("\nfinal sample:\n%s\n",
               generate(g, ds.itos, sample_len, sample_rng, temperature).c_str());
    return 0;
}
