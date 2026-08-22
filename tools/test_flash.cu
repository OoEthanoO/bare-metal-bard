// Fused attention: correctness against the unfused path, and what it costs.
//
// The reference here is not a textbook formula, it is the attention this repo
// already trains with (src/attention.cu). That is the comparison that matters:
// the fused kernel has to reproduce what the model already does, bit-for-bit
// close enough that the gradient check and the loss curve do not move.
//
// Accuracy is reported normwise -- worst absolute error over max|ref| -- for
// the same reason the GEMM tests are. Attention outputs are convex combinations
// of value vectors, so entries near zero are cancellation, and their elementwise
// relative error measures the cancellation rather than the kernel.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "../src/attention.h"
#include "../src/flash.h"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e__ = (x);                                                 \
        if (e__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(e__), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void fill_random(std::vector<float> &v, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int)(s % 2000) - 1000) / 1000.0f;  // [-1, 1]
    }
}

struct Timing { double best_ms, median_ms; };

template <typename F>
static Timing time_it(F fn, int warmup, int iters) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> s;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(a));
        fn();
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        s.push_back(ms);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    std::sort(s.begin(), s.end());
    return {s.front(), s[s.size() / 2]};
}

int main(int argc, char **argv) {
    int B = 16, T = 256, C = 384, NH = 6, iters = 50;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-b") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) C = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) NH = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
    }
    const int hs = C / NH;
    const size_t BTC = (size_t)B * T * C;
    const size_t att_sz = (size_t)B * NH * T * T;

    printf("B=%d T=%d C=%d NH=%d (hs=%d)\n\n", B, T, C, NH, hs);

    std::vector<float> h_qkv(BTC * 3), h_ref(BTC), h_out(BTC);
    fill_random(h_qkv, 1234u);

    float *d_qkv, *d_out_ref, *d_out, *d_qkvr, *d_att, *d_lse;
    CUDA_CHECK(cudaMalloc(&d_qkv, BTC * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_ref, BTC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, BTC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkvr, BTC * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_att, att_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lse, (size_t)B * NH * T * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_qkv, h_qkv.data(), BTC * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- reference ----
    attention_forward(d_out_ref, d_qkvr, d_att, d_qkv, B, T, C, NH);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out_ref, BTC * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double maxref = 0.0;
    for (size_t i = 0; i < BTC; ++i) maxref = fmax(maxref, fabs((double)h_ref[i]));

    const Timing ref_t = time_it(
        [&] { attention_forward(d_out_ref, d_qkvr, d_att, d_qkv, B, T, C, NH); },
        5, iters);

    // Useful FLOPs: the causal half of both attention matmuls, 2 flops each.
    // The unfused path computes twice this and discards the upper triangle;
    // charging both implementations the same useful number is what makes the
    // ratio meaningful.
    const double useful =
        2.0 * 2.0 * B * NH * hs * ((double)T * (T + 1) / 2.0);
    auto gf = [&](double ms) { return useful / (ms * 1e-3) / 1e9; };

    printf("%-22s %9s %9s %10s %10s %12s\n", "impl", "best ms", "med ms",
           "GFLOP/s", "vs unfused", "norm err");
    printf("%-22s %9.3f %9.3f %10.1f %10s %12s\n", "unfused (3 kernels)",
           ref_t.best_ms, ref_t.median_ms, gf(ref_t.best_ms), "1.00x", "-");

    // ---- every fused config ----
    int best_cfg = -1;
    double best_ms = 1e30;
    for (int cfg = 0; cfg < flash_num_configs(); ++cfg) {
        CUDA_CHECK(cudaMemset(d_out, 0, BTC * sizeof(float)));
        if (!flash_attention_forward_cfg(cfg, d_out, d_lse, d_qkv, B, T, C, NH)) {
            printf("%-22s %9s (not valid for hs=%d)\n", flash_config_name(cfg),
                   "-", hs);
            continue;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, BTC * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double maxabs = 0.0;
        for (size_t i = 0; i < BTC; ++i)
            maxabs = fmax(maxabs, fabs((double)h_out[i] - (double)h_ref[i]));
        const double rel = maxabs / maxref;

        const Timing t = time_it(
            [&] {
                flash_attention_forward_cfg(cfg, d_out, d_lse, d_qkv, B, T, C, NH);
            },
            5, iters);
        if (t.best_ms < best_ms) { best_ms = t.best_ms; best_cfg = cfg; }

        char ratio[16];
        snprintf(ratio, sizeof ratio, "%.2fx", ref_t.best_ms / t.best_ms);
        printf("%-22s %9.3f %9.3f %10.1f %10s %12.2e %s\n",
               flash_config_name(cfg), t.best_ms, t.median_ms, gf(t.best_ms),
               ratio, rel, rel < 1e-5 ? "ok" : "FAIL");
    }

    // ---- what the fusion actually saves, in bytes ----
    const double att_mb = att_sz * 4.0 / 1e6;
    const double qkvr_mb = BTC * 4.0 * 4.0 / 1e6;
    printf("\nper layer, per forward pass:\n");
    printf("  score matrix (B,NH,T,T)   %8.1f MB   written once, read twice\n",
           att_mb);
    printf("  permuted qkvr (4,B,NH,T,hs) %6.1f MB   written once, read twice\n",
           qkvr_mb);
    printf("  fused saves               %8.1f MB of resident activation memory\n",
           att_mb + qkvr_mb);
    printf("  lse kept instead          %8.3f MB\n",
           (double)B * NH * T * 4.0 / 1e6);

    if (best_cfg >= 0)
        printf("\nfastest config: %s (%.3f ms)\n", flash_config_name(best_cfg),
               best_ms);
    return 0;
}
