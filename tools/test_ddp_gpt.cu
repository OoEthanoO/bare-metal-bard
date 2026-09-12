// Full-batch gradients versus data-parallel training, without a vendor BLAS.
//
// First compare every gradient tensor against one rank's full-batch result.
// Then give the reference the rank-0 MEAN gradient, and check AdamW against
// the distributed SUM plus inv_ranks. This isolates optimizer normalization
// from harmless summation-order differences near zero, which Adam amplifies.
// Weights/moments persist across steps; no state is reset to hide divergence.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "../src/ddp.h"
#include "../src/gemm.h"
#include "../src/gpt.h"
#include "../src/nn.h"
#include "../src/rank_pool.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

__global__ void negate_gradient(float *g, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) g[i] = -g[i];
}

static void readback(int device, const float *src, std::vector<float> &dst) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

static bool check_loss_mean(const std::vector<float> &per_token, float reported,
                            int rank) {
    double sum = 0.0;
    for (float v : per_token) sum += v;
    const double expected = sum / per_token.size();
    if (!std::isfinite(expected) || !std::isfinite(reported) ||
        fabs(expected - reported) > 2e-6) {
        printf("FAIL rank=%d reported loss %.8f vs token mean %.8f\n",
               rank, reported, expected);
        return false;
    }
    return true;
}

// Normwise per-tensor comparison: a small tensor cannot be hidden by the much
// larger weight matrices. Explicit finiteness avoids NaNs passing max(error).
static bool compare(const GPT &g, const std::vector<float> &want,
                    const std::vector<float> &got, const char *field, int rank,
                    double atol, double rtol, double &worst_budget) {
    bool ok = true;
    size_t off = 0;
    for (int t = 0; t < NUM_PARAM_TENSORS; ++t) {
        double error = 0.0, magnitude = 0.0;
        bool finite = true;
        size_t worst_i = off;
        for (size_t i = off; i < off + g.psize[t]; ++i) {
            finite &= std::isfinite(want[i]) && std::isfinite(got[i]);
            magnitude = std::max(magnitude, fabs((double)want[i]));
            const double e = fabs((double)got[i] - want[i]);
            if (e > error) { error = e; worst_i = i; }
        }
        const double limit = atol + rtol * magnitude;
        worst_budget = std::max(worst_budget, error / limit);
        if (!finite || error > limit) {
            printf("FAIL rank=%d %s/%s index=%zu abs_err=%.3e limit=%.3e finite=%d\n",
                   rank, field, PARAM_NAMES[t], worst_i - off, error, limit, finite);
            ok = false;
        }
        off += g.psize[t];
    }
    return ok;
}

static void release_model(int device, GPT &g) {
    CUDA_CHECK(cudaSetDevice(device));
    for (float *p : {g.params_mem, g.grads_mem, g.m_mem, g.v_mem,
                     g.acts_mem, g.grads_act_mem}) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(g.d_tokens));
    CUDA_CHECK(cudaFree(g.d_targets));
    CUDA_CHECK(cudaFreeHost(g.h_tokens));
    CUDA_CHECK(cudaFreeHost(g.h_targets));
}

int main(int argc, char **argv) {
    int nranks = 2, steps = 3;
    bool tf32 = false, require_devices = false, inject = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--tf32")) tf32 = true;
        else if (!strcmp(argv[i], "--require-multi-gpu")) require_devices = true;
        else if (!strcmp(argv[i], "--inject-gradient-error")) inject = true;
        else if ((!strcmp(argv[i], "--ranks") || !strcmp(argv[i], "--steps")) && i + 1 < argc) {
            const bool rank_arg = !strcmp(argv[i], "--ranks");
            char *end = nullptr;
            const long n = strtol(argv[++i], &end, 10);
            if (*end || n < 1 || n > (rank_arg ? 8 : 20)) {
                fprintf(stderr, "invalid %s value\n", rank_arg ? "--ranks" : "--steps");
                return 2;
            }
            if (rank_arg) nranks = (int)n; else steps = (int)n;
        } else {
            fprintf(stderr, "usage: test_ddp_gpt [--tf32] [--ranks 1..8] [--steps 1..20]\n"
                            "                    [--require-multi-gpu] [--inject-gradient-error]\n");
            return 2;
        }
    }
    int ndevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    if (!ndevices || (require_devices && (nranks < 2 || ndevices < nranks))) {
        fprintf(stderr, "need %d distinct CUDA devices; found %d\n", nranks, ndevices);
        return 2;
    }
    if (tf32 && !gemm_tf32_available()) {
        fprintf(stderr, "TF32 was requested but is unavailable in this build\n");
        return 2;
    }
    gemm_set_tf32(tf32);
    constexpr int SHARD_B = 2, T = 64, C = 384, L = 2, V = 65, VP = 128;
    const int shard_tokens = SHARD_B * T, tokens = nranks * shard_tokens;
    std::vector<int> devices(nranks);
    std::vector<GPT> replicas(nranks);
    GPT reference{};
    reference.config = {T, V, VP, L, C / 64, C};
    reference.B = SHARD_B * nranks; reference.T = T; reference.use_flash = true;
    CUDA_CHECK(cudaSetDevice(0));
    gpt_alloc(reference);
    gpt_init(reference, 1337);
    for (int r = 0; r < nranks; ++r) {
        devices[r] = r % ndevices;
        GPT &g = replicas[r];
        g.config = reference.config; g.B = SHARD_B; g.T = T; g.use_flash = true;
        CUDA_CHECK(cudaSetDevice(devices[r]));
        gpt_alloc(g);
        gpt_init(g, 1337);
    }
    DDP ddp;
    ddp_init(ddp, nranks, devices.data());
    ddp_report_topology(ddp);
    printf("DDP model check: %s, global B=%d, T=%d, C=%d, L=%d, steps=%d\n",
           tf32 ? "TF32" : "FP32", reference.B, T, C, L, steps);
    printf("Checks are numerical correctness only; repeated devices are not a scaling result.\n");
    RankPool pool;
    pool.start(nranks);
    const size_t np = reference.num_params;
    std::vector<float> expected(np), actual(np), summed(np), mean(np);
    std::vector<int> x(tokens), y(tokens), device_x(shard_tokens), device_y(shard_tokens);
    std::vector<float> ref_losses(tokens), shard_losses(shard_tokens);
    std::vector<float *> gradients(nranks);
    std::vector<float> losses(nranks), norms(nranks);
    std::mt19937 rng(4242);
    bool passed = true;

    for (int step = 1; step <= steps && passed; ++step) {
        for (int b = 0; b < reference.B; ++b) {
            int previous = rng() % V;
            for (int t = 0; t < T; ++t) {
                x[b * T + t] = previous;
                y[b * T + t] = previous = rng() % V;
            }
        }
        CUDA_CHECK(cudaSetDevice(0));
        const float full_loss = gpt_forward(reference, x.data(), y.data());
        gpt_backward(reference);
        CUDA_CHECK(cudaDeviceSynchronize());
        readback(0, reference.grads_mem, expected);
        readback(0, reference.acts.losses, ref_losses);
        passed &= check_loss_mean(ref_losses, full_loss, -1);

        pool.run([&](int r) {
            CUDA_CHECK(cudaSetDevice(devices[r]));
            losses[r] = gpt_forward(replicas[r], x.data() + r * shard_tokens,
                                    y.data() + r * shard_tokens);
            gpt_backward(replicas[r]);
            CUDA_CHECK(cudaDeviceSynchronize());
        });
        double mean_loss = 0.0, worst_loss = 0.0, grad_budget = 0.0;
        for (int r = 0; r < nranks; ++r) {
            CUDA_CHECK(cudaSetDevice(devices[r]));
            CUDA_CHECK(cudaMemcpy(device_x.data(), replicas[r].d_tokens,
                                  shard_tokens * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(device_y.data(), replicas[r].d_targets,
                                  shard_tokens * sizeof(int), cudaMemcpyDeviceToHost));
            readback(devices[r], replicas[r].acts.losses, shard_losses);
            passed &= check_loss_mean(shard_losses, losses[r], r);
            for (int i = 0; i < shard_tokens; ++i) {
                const int idx = r * shard_tokens + i;
                const double e = fabs((double)ref_losses[idx] - shard_losses[i]);
                worst_loss = std::max(worst_loss, e);
                if (!std::isfinite(shard_losses[i]) || !std::isfinite(ref_losses[idx]) ||
                    e > (tf32 ? 2e-3 : 2e-5) || device_x[i] != x[idx] || device_y[i] != y[idx]) {
                    printf("FAIL rank=%d token=%d loss/input mismatch\n", r, i);
                    passed = false;
                    break;
                }
            }
            mean_loss += losses[r] / nranks;
            gradients[r] = replicas[r].grads_mem;
        }
        if (inject && step == 1) {
            CUDA_CHECK(cudaSetDevice(devices[0]));
            negate_gradient<<<256, 256>>>(gradients[0], np);
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("Injected sign reversal on rank 0: same gradient norm, wrong direction.\n");
        }
        ddp_allreduce(ddp, gradients.data(), np);
        readback(devices[0], gradients[0], summed);
        const float inv_ranks = 1.0f / nranks;
        for (size_t i = 0; i < np; ++i) mean[i] = summed[i] * inv_ranks;
        passed &= compare(reference, expected, mean, "mean gradient", 0,
                          1e-7, tf32 ? 5e-3 : 1e-4, grad_budget);
        for (int r = 1; r < nranks; ++r) {
            readback(devices[r], gradients[r], actual);
            if (memcmp(actual.data(), summed.data(), np * sizeof(float))) {
                printf("FAIL rank=%d all-reduce buffers are not byte-identical\n", r);
                passed = false;
            }
        }
        printf("step %d loss full=%.7f shards=%.7f max_token_error=%.2e gradient_budget=%.3f\n",
               step, full_loss, mean_loss, worst_loss, grad_budget);
        if (!passed) break;

        // Check optimizer scaling independently of the tiny reduction-order
        // differences just measured. The reference consumes a MEAN, while
        // each rank consumes its unchanged SUM and applies inv_ranks itself.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(reference.grads_mem, mean.data(), np * sizeof(float),
                              cudaMemcpyHostToDevice));
        double norm2 = 0.0;
        for (float v : mean) norm2 += (double)v * v;
        const float clip = (float)sqrt(norm2) * (step % 2 ? 0.25f : 2.0f);
        const float *ref_norm = adamw_clipped_update(reference.params_mem,
            reference.grads_mem, reference.m_mem, reference.v_mem, (int)np,
            1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step, clip, 1.0f);
        CUDA_CHECK(cudaDeviceSynchronize());
        const float ref_norm_value = *ref_norm;
        pool.run([&](int r) {
            CUDA_CHECK(cudaSetDevice(devices[r]));
            GPT &g = replicas[r];
            const float *p = adamw_clipped_update(g.params_mem, g.grads_mem,
                g.m_mem, g.v_mem, (int)np, 1e-4f, 0.9f, 0.999f, 1e-8f,
                0.01f, step, clip, inv_ranks);
            CUDA_CHECK(cudaDeviceSynchronize());
            norms[r] = *p;
        });
        double state_budget = 0.0;
        for (int field = 0; field < 3; ++field) {
            const float *ref = field == 0 ? reference.params_mem :
                               field == 1 ? reference.m_mem : reference.v_mem;
            readback(0, ref, expected);
            for (int r = 0; r < nranks; ++r) {
                const GPT &g = replicas[r];
                const float *src = field == 0 ? g.params_mem : field == 1 ? g.m_mem : g.v_mem;
                readback(devices[r], src, actual);
                passed &= compare(reference, expected, actual,
                    field == 0 ? "parameters" : field == 1 ? "Adam m" : "Adam v", r,
                    field == 0 ? 2e-7 : field == 1 ? 1e-9 : 1e-11,
                    field == 0 ? 2e-6 : 5e-5, state_budget);
            }
        }
        for (int r = 0; r < nranks; ++r) {
            if (!std::isfinite(norms[r]) || !std::isfinite(ref_norm_value) ||
                fabs(norms[r] - ref_norm_value) > 1e-5 * std::max(1.0f, ref_norm_value)) {
                printf("FAIL rank=%d mean gradient norm %.7f vs %.7f\n", r, norms[r], ref_norm_value);
                passed = false;
            }
        }
        printf("       optimizer %s norm=%.6f clip=%.6f state_budget=%.3f\n",
               step % 2 ? "clipped" : "unclipped", ref_norm_value, clip, state_budget);
    }
    pool.stop();
    ddp_free(ddp);
    for (int r = 0; r < nranks; ++r) release_model(devices[r], replicas[r]);
    release_model(0, reference);
    printf("DDP MODEL CHECK %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
