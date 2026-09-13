// Cross-entropy must read a normalized probability, even when its target is
// owned by another warp or a later iteration of that thread's strided loop.
// CPU-double oracle; no vendor BLAS. Repeated launches vary warp scheduling.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../src/nn.h"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(2); } } while (0)

static bool check(int V, int N, int repeats) {
    const int Vp = ((V + 127) / 128) * 128;
    const size_t count = (size_t)N * Vp;
    std::vector<float> logits(count, 1000.0f), probs(count), grad(count), loss(N);
    std::vector<int> targets(N);
    std::vector<double> want(V);
    double sum = 0;
    for (int c = 0; c < V; ++c) {
        // Fixed integer multiples keep the CPU and GPU inputs identical.
        want[c] = std::exp(((c * 17) % 37 - 18) * 0.25);
        sum += want[c];
    }
    for (double &p : want) p /= sum;
    for (int row = 0; row < N; ++row) {
        targets[row] = row % V;
        for (int c = 0; c < V; ++c)
            logits[(size_t)row * Vp + c] = ((c * 17) % 37 - 18) * 0.25f;
    }
    float *dx, *dp, *dg, *dl;
    int *dt;
    CUDA_CHECK(cudaMalloc(&dx, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dg, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dl, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dx, logits.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt, targets.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    bool ok = true;
    double worst_loss = 0, worst_prob = 0, worst_grad = 0;
    for (int rep = 0; rep < repeats && ok; ++rep) {
        CUDA_CHECK(cudaMemset(dp, 0xff, count * sizeof(float)));
        CUDA_CHECK(cudaMemset(dl, 0xff, N * sizeof(float)));
        softmax_crossentropy_forward(dp, dl, dx, dt, N, V, Vp);
        crossentropy_softmax_backward(dg, dp, dt, N, V, Vp, 0.125f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(loss.data(), dl, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(probs.data(), dp, count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(grad.data(), dg, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int row = 0; row < N && ok; ++row) {
            const double expected_loss = -std::log(std::max(want[targets[row]], 1e-30));
            const double le = std::abs(loss[row] - expected_loss);
            worst_loss = std::max(worst_loss, le);
            if (!std::isfinite(loss[row]) || le > 2e-5) {
                printf("FAIL V=%d rep=%d row=%d target=%d loss=%.8f expected=%.8f "
                       "final_probability=%.8f\n", V, rep, row, targets[row],
                       loss[row], expected_loss, probs[(size_t)row * Vp + targets[row]]);
                ok = false;
            }
            for (int c = 0; c < Vp && ok; ++c) {
                const size_t i = (size_t)row * Vp + c;
                const double p = c < V ? want[c] : 0;
                const double g = c < V ? (p - (c == targets[row])) * 0.125 : 0;
                const double pe = std::abs(probs[i] - p), ge = std::abs(grad[i] - g);
                worst_prob = std::max(worst_prob, pe);
                worst_grad = std::max(worst_grad, ge);
                if (!std::isfinite(probs[i]) || !std::isfinite(grad[i]) ||
                    pe > 2e-6 || ge > 2e-6 || (c >= V && (probs[i] != 0 || grad[i] != 0))) {
                    printf("FAIL V=%d rep=%d row=%d column=%d probability/gradient\n", V, rep, row, c);
                    ok = false;
                }
            }
        }
    }
    for (float *p : {dx, dp, dg, dl}) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(dt));
    printf("V=%d Vp=%d rows=%d repeats=%d max_loss_error=%.2e max_prob_error=%.2e "
           "max_grad_error=%.2e %s\n", V, Vp, N, repeats, worst_loss, worst_prob,
           worst_grad, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = check(65, 16384, 64); // production vocabulary, targets in 3 warps
    for (int V : {1, 31, 32, 33, 127, 128, 129, 257})
        ok = check(V, 4096, 8) && ok;
    printf("CROSS-ENTROPY CHECK %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
