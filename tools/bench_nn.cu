// Bandwidth bench for the non-GEMM kernels, at the shapes the model runs.
//
// WHY THIS EXISTS. The step profiler says which region costs the most; it does
// not say whether that region is doing well. For a memory-bound kernel the
// question is not "how many milliseconds" but "how close to the bandwidth this
// card actually has" -- a column reduction that moves 25 MB has a floor, and
// the only interesting number is the distance from it. Everything here is
// reported as GB/s against the measured peak, so a kernel that is already at
// the roofline can be left alone and one at a third of it cannot.
//
// Bytes are counted as the traffic the kernel CANNOT avoid: every input read
// once, every output written once. Anything above that is the kernel's own
// inefficiency and is exactly what this is meant to expose.
//
// READ THE % PEAK WITH THIS CAVEAT, WHICH COST ME AN AFTERNOON. At N=4096,
// C=384 the working set here is about 19 MB against this card's 36 MB L2, and
// the timing loop re-reads it thirty times -- so it is resident, and what gets
// measured is an L2 curve, not a DRAM one. In a training step these buffers
// compete with the whole model and mostly come from DRAM.
//
// The two regimes do not agree. Tuning the layernorm backward's block count
// here said the rewrite was 2.9x faster than the atomic version; in the step it
// was slower, until the count was re-tuned in situ. What the micro-bench DID
// get right was where the optimum sits (368 blocks in both), just not what it
// was worth. So: use this to find the shape of a curve and to spot a kernel
// sitting at a tenth of the roofline, and use `train_gpt --profile` to decide
// whether a change is worth keeping.
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../src/nn.h"

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static double time_it(void (*fn)(void *), void *arg, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) fn(arg);
    CUDA_CHECK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a);
        fn(arg);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

struct Buf {
    float *dout, *inp, *w, *mean, *rstd, *dinp, *dw, *db;
    int N, C;
};
static Buf g;

static void run_bias(void *)  { bias_backward(g.db, g.dout, g.N, g.C); }
static void run_lnbwd(void *) {
    layernorm_backward(g.dinp, g.dw, g.db, g.dout, g.inp, g.w, g.mean, g.rstd,
                       g.N, g.C);
}
static void run_gelu(void *)  { gelu_backward(g.dinp, g.inp, g.dout, g.N * g.C); }
static void run_resid(void *) { residual_forward(g.dinp, g.inp, g.dout, g.N * g.C); }

// Sweep the layernorm backward's block count at the width the model actually
// uses. The rows are independent, so this trades parallelism for the depth of
// the second pass, and the derived value is one point on that curve.
static void sweep_ln(int N, int C) {
    printf("layernorm bwd block sweep, N=%d C=%d%s", N, C, "\n");
    printf("%-10s %10s %10s%s", "blocks", "ms", "GB/s", "\n");
    const int bs[] = {46, 92, 184, 368, 736, 1024, 2048, 4096};
    const double gb = 3.0 * N * C * 4.0 / 1e9;
    for (int b : bs) {
        layernorm_backward_set_blocks(b);
        const double ms = time_it(run_lnbwd, nullptr, 30);
        printf("%-10d %10.3f %10.0f%s", b, ms, gb / (ms * 1e-3), "\n");
    }
    layernorm_backward_set_blocks(0);
    const double ms = time_it(run_lnbwd, nullptr, 30);
    printf("%-10s %10.3f %10.0f%s%s", "derived", ms, gb / (ms * 1e-3), "\n", "\n");
}

int main(int argc, char **argv) {
    int dev = 0, memclk = 0, bus = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    const double peak = 2.0 * (bus / 8.0) * memclk * 1e3 / 1e9;  // GB/s
    printf("peak bandwidth %.0f GB/s (%d-bit @ %.2f GHz)\n\n", peak, bus,
           memclk * 1e-6);

    const int N = 4096;
    const int Cs[] = {384, 1152, 1536};
    size_t maxel = (size_t)N * 1536;
    CUDA_CHECK(cudaMalloc(&g.dout, maxel * 4));
    CUDA_CHECK(cudaMalloc(&g.inp,  maxel * 4));
    CUDA_CHECK(cudaMalloc(&g.dinp, maxel * 4));
    CUDA_CHECK(cudaMalloc(&g.w,    1536 * 4));
    CUDA_CHECK(cudaMalloc(&g.dw,   1536 * 4));
    CUDA_CHECK(cudaMalloc(&g.db,   1536 * 4));
    CUDA_CHECK(cudaMalloc(&g.mean, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&g.rstd, (size_t)N * 4));
    CUDA_CHECK(cudaMemset(g.dout, 0, maxel * 4));
    CUDA_CHECK(cudaMemset(g.inp,  0, maxel * 4));
    CUDA_CHECK(cudaMemset(g.w,    0, 1536 * 4));
    // rstd multiplies; zero would make layernorm backward trivially cheap.
    std::vector<float> ones(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(g.rstd, ones.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.mean, 0, (size_t)N * 4));

    if (argc > 1 && !strcmp(argv[1], "--sweep")) {
        g.N = N; g.C = 384;
        sweep_ln(N, 384);
        return 0;
    }
    printf("%-18s %6s %6s %10s %10s %9s\n", "kernel", "N", "C", "ms", "GB/s",
           "% peak");
    printf("---------------------------------------------------------------\n");
    for (int C : Cs) {
        g.N = N; g.C = C;
        const double el = (double)N * C * 4.0 / 1e9;  // one pass, GB
        struct Case { const char *name; void (*fn)(void *); double gb; };
        const Case cases[] = {
            // reads dout, writes C floats: one pass.
            {"bias backward",   run_bias,  el},
            // reads dout + inp, writes dinp: three passes.
            {"layernorm bwd",   run_lnbwd, 3 * el},
            // reads dout + inp, writes dinp: three passes.
            {"gelu backward",   run_gelu,  3 * el},
            // reads a + b, writes out: three passes.
            {"residual fwd",    run_resid, 3 * el},
        };
        for (const Case &c : cases) {
            const double ms = time_it(c.fn, nullptr, 30);
            printf("%-18s %6d %6d %10.3f %10.0f %8.0f%%\n", c.name, N, C, ms,
                   c.gb / (ms * 1e-3), 100.0 * (c.gb / (ms * 1e-3)) / peak);
        }
        printf("\n");
    }
    return 0;
}
