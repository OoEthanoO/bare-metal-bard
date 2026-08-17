// Numerical gradient check for the GPT backward pass.
//
// A falling training loss does not verify a backward pass. Several real bugs
// -- a dropped correction term in layernorm, a missing scale in attention, a
// transposed dW -- still produce a descending loss, just a worse model. So the
// analytic gradients are checked against finite differences directly.
//
// METHOD: directional derivatives, not per-element differences.
//
// The naive check perturbs one scalar parameter and watches the loss. In fp32
// that barely works: individual gradients here are O(1e-4), so the loss moves
// by ~1e-6 for a reasonable epsilon, which is the same size as the forward
// pass's own rounding noise. The measurement is then mostly noise.
//
// Instead, for each parameter tensor we step along the direction of its own
// gradient, u = g/||g||:
//
//     numeric  = (L(theta + eps*u) - L(theta - eps*u)) / (2*eps)
//     analytic = <g, u> = ||g||
//
// Every element of the tensor contributes coherently, so the loss change is
// ||g|| * 2 * eps -- for qkvw that is ~1e-3 rather than ~1e-6, three orders of
// magnitude above the noise floor. A single wrong element still shows up,
// because it changes both ||g|| and the direction actually stepped.
//
// The loss is summed in double on the host rather than reduced in fp32 on the
// device, to keep the reduction from adding noise of its own.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "../src/gpt.h"

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static int g_N;  // B*T, the number of loss positions

static double loss_double(GPT &g, const int *x, const int *y) {
    gpt_forward(g, x, y);
    std::vector<float> h(g_N);
    CUDA_CHECK(cudaMemcpy(h.data(), g.acts.losses, g_N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < g_N; ++i) s += (double)h[i];
    return s / g_N;
}

int main(int argc, char **argv) {
    // Small enough to run many forward passes, large enough that every code
    // path (multi-layer residual, multi-head attention, padded vocab) is live.
    const int B = 4, T = 64, C = 128, L = 2, NH = 4, V = 65, Vp = 128;
    float eps = 1e-2f;
    if (argc > 1) eps = atof(argv[1]);

    GPT g;
    g.config = {T, V, Vp, L, NH, C};
    g.B = B; g.T = T;
    g_N = B * T;

    gpt_alloc(g);
    gpt_init(g, 1337);

    // Fixed random batch. The same tokens are used for every forward pass, so
    // the only thing changing between them is the parameter perturbation.
    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> tok(0, V - 1);
    std::vector<int> x(g_N), y(g_N);
    for (int i = 0; i < g_N; ++i) { x[i] = tok(rng); y[i] = tok(rng); }

    const double L0 = loss_double(g, x.data(), y.data());
    gpt_backward(g);

    std::vector<float> hgrad(g.num_params), hparam(g.num_params);
    CUDA_CHECK(cudaMemcpy(hgrad.data(), g.grads_mem, g.num_params * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hparam.data(), g.params_mem, g.num_params * sizeof(float),
                          cudaMemcpyDeviceToHost));

    size_t psz[NUM_PARAM_TENSORS];
    param_sizes(psz, g.config);

    printf("gradient check: B=%d T=%d C=%d L=%d NH=%d V=%d(->%d)  eps=%.0e\n",
           B, T, C, L, NH, V, Vp, eps);
    printf("loss at theta = %.6f   (ln(V) = %.6f)\n\n", L0, log((double)V));
    printf("%-10s %10s %14s %14s %10s  %s\n", "tensor", "n", "analytic",
           "numeric", "rel err", "");
    printf("---------------------------------------------------------------------------\n");

    int failures = 0, checked = 0;
    size_t off = 0;
    std::vector<float> perturbed(g.num_params);

    for (int t = 0; t < NUM_PARAM_TENSORS; ++t) {
        const size_t n = psz[t];
        const float *gt = hgrad.data() + off;

        // ||g|| for this tensor, accumulated in double.
        double norm2 = 0.0;
        for (size_t i = 0; i < n; ++i) norm2 += (double)gt[i] * (double)gt[i];
        const double gnorm = sqrt(norm2);

        if (gnorm < 1e-9) {
            // Genuinely zero gradient (e.g. an unused position embedding row)
            // carries no directional information; nothing to compare against.
            printf("%-10s %10zu %14.6e %14s %10s  skipped (zero grad)\n",
                   PARAM_NAMES[t], n, gnorm, "-", "-");
            off += n;
            continue;
        }

        // Step along u = g/||g||, so the predicted directional derivative is
        // exactly ||g||.
        perturbed = hparam;
        for (size_t i = 0; i < n; ++i)
            perturbed[off + i] = hparam[off + i] + eps * (float)(gt[i] / gnorm);
        CUDA_CHECK(cudaMemcpy(g.params_mem, perturbed.data(),
                              g.num_params * sizeof(float), cudaMemcpyHostToDevice));
        const double Lp = loss_double(g, x.data(), y.data());

        perturbed = hparam;
        for (size_t i = 0; i < n; ++i)
            perturbed[off + i] = hparam[off + i] - eps * (float)(gt[i] / gnorm);
        CUDA_CHECK(cudaMemcpy(g.params_mem, perturbed.data(),
                              g.num_params * sizeof(float), cudaMemcpyHostToDevice));
        const double Lm = loss_double(g, x.data(), y.data());

        const double numeric = (Lp - Lm) / (2.0 * eps);
        const double rel = fabs(numeric - gnorm) / std::max(gnorm, 1e-12);
        const bool ok = rel < 2e-2;
        if (!ok) ++failures;
        ++checked;

        printf("%-10s %10zu %14.6e %14.6e %10.2e  %s\n", PARAM_NAMES[t], n,
               gnorm, numeric, rel, ok ? "ok" : "FAIL");
        off += n;
    }

    // Restore and check the whole parameter vector at once: this catches an
    // error that happens to cancel within a tensor but not across tensors.
    CUDA_CHECK(cudaMemcpy(g.params_mem, hparam.data(),
                          g.num_params * sizeof(float), cudaMemcpyHostToDevice));
    {
        double norm2 = 0.0;
        for (size_t i = 0; i < g.num_params; ++i)
            norm2 += (double)hgrad[i] * (double)hgrad[i];
        const double gnorm = sqrt(norm2);

        std::vector<float> pp(g.num_params);
        for (size_t i = 0; i < g.num_params; ++i)
            pp[i] = hparam[i] + eps * (float)(hgrad[i] / gnorm);
        CUDA_CHECK(cudaMemcpy(g.params_mem, pp.data(), g.num_params * sizeof(float),
                              cudaMemcpyHostToDevice));
        const double Lp = loss_double(g, x.data(), y.data());
        for (size_t i = 0; i < g.num_params; ++i)
            pp[i] = hparam[i] - eps * (float)(hgrad[i] / gnorm);
        CUDA_CHECK(cudaMemcpy(g.params_mem, pp.data(), g.num_params * sizeof(float),
                              cudaMemcpyHostToDevice));
        const double Lm = loss_double(g, x.data(), y.data());

        const double numeric = (Lp - Lm) / (2.0 * eps);
        const double rel = fabs(numeric - gnorm) / std::max(gnorm, 1e-12);
        const bool ok = rel < 2e-2;
        if (!ok) ++failures;
        printf("---------------------------------------------------------------------------\n");
        printf("%-10s %10zu %14.6e %14.6e %10.2e  %s\n", "ALL", g.num_params,
               gnorm, numeric, rel, ok ? "ok" : "FAIL");
    }

    printf("\n%s (%d checked, %d failures)\n",
           failures ? "GRADIENT CHECK FAILED" : "gradient check passed", checked,
           failures);
    return failures ? 1 : 0;
}
