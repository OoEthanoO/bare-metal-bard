// SGEMM benchmark harness.
//
// Design notes, because the measurement is as easy to get wrong as the kernel:
//
//  * This is a 55W laptop GPU. Sustained clocks sag under load, so a single
//    timed run says as much about thermals as about the kernel. We therefore
//    report BOTH the best iteration (kernel quality at peak clock) and the
//    median (what you actually get sustained).
//  * cuBLAS is re-timed in the same process, immediately after each kernel, so
//    both see a comparable thermal state. The headline number is the RATIO,
//    which cancels most of the throttling out.
//  * Correctness is checked against cuBLAS with a non-trivial alpha/beta so
//    the epilogue is exercised, not just the inner product.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels.h"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t st__ = (x);                                             \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,           \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static cublasHandle_t g_handle;

// Row-major C = alpha*A@B + beta*C using a column-major library.
//
// A column-major reader of our row-major A (M x K) sees A^T, a K x M matrix
// with leading dimension K. Same for B and C. We want C^T = B^T @ A^T, which
// in column-major terms is a plain no-transpose GEMM of the (B, A) operands
// with dimensions (N, M, K). No data is moved.
static void sgemm_cublas(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, B, N, A, K, &beta, C, N));
}

static void fill_random(std::vector<float> &v, unsigned seed) {
    // xorshift: deterministic across runs and machines, no <random> variance.
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int)(s % 2000) - 1000) / 1000.0f;  // [-1, 1]
    }
}

struct Timing {
    double best_ms;
    double median_ms;
};

// Runs `fn` warmup+iters times. C is not reset between iterations, so callers
// should time with beta = 0 to keep values bounded.
static Timing time_kernel(sgemm_fn fn, int M, int N, int K, float alpha,
                          const float *A, const float *B, float beta, float *C,
                          int warmup, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) fn(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn(M, N, K, alpha, A, B, beta, C);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back((double)ms);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    Timing t;
    t.best_ms = samples.front();
    t.median_ms = samples[samples.size() / 2];
    return t;
}

static double gflops(int M, int N, int K, double ms) {
    // 2*M*N*K flops: one multiply and one add per (m,n,k) triple.
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
}

struct Accuracy {
    double max_abs;
    double norm_rel;  // max_abs / max|ref|
    bool pass;
};

// Elementwise relative error is the wrong metric for GEMM. Each output is a
// sum of K products, so where cancellation drives an entry near zero the
// rounding noise of the other terms remains -- an entry of magnitude 1e-3 can
// carry absolute error 1e-4 while every input was computed perfectly. Chasing
// that ratio only measures cancellation.
//
// The meaningful check is normwise: the standard backward-error bound for a
// dot product is |err| <~ K * eps * sum|a_i * b_i|, which scales with the
// magnitude of the whole matrix, not with the individual entry. So we compare
// the worst absolute error against max|ref|.
//
// The separation is what makes this a real test: reordered fp32 summation
// lands around 1e-6 here, while a genuine bug (bad index, race, dropped term)
// produces errors comparable to the entries themselves, order 1e-1 or worse.
// A 1e-4 threshold sits three orders of magnitude clear of both.
static Accuracy compare(const std::vector<float> &got,
                        const std::vector<float> &ref) {
    Accuracy a{0.0, 0.0, true};
    double ref_inf = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        a.max_abs = std::max(a.max_abs, d);
        ref_inf = std::max(ref_inf, std::fabs((double)ref[i]));
        // NaN/Inf must fail regardless of what the norm says.
        if (!std::isfinite((double)got[i])) { a.pass = false; }
    }
    a.norm_rel = a.max_abs / std::max(ref_inf, 1e-30);
    if (a.norm_rel >= 1e-4) a.pass = false;
    return a;
}

static void usage(const char *prog) {
    printf("usage: %s [options]\n", prog);
    printf("  -k <ids>     comma-separated kernel ids (default: all)\n");
    printf("  -s <sizes>   comma-separated square sizes (default: 128,256,512,1024,2048,4096)\n");
    printf("  -i <n>       timed iterations (default: 20)\n");
    printf("  -w <n>       warmup iterations (default: 5)\n");
    printf("  --csv <path> append results as CSV\n");
    printf("  --no-verify  skip the correctness check\n");
    printf("  -l           list kernels and exit\n");
}

static std::vector<int> parse_ints(const char *s) {
    std::vector<int> out;
    const char *p = s;
    while (*p) {
        out.push_back(atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return out;
}

int main(int argc, char **argv) {
    std::vector<int> ids, sizes;
    int iters = 20, warmup = 5;
    bool verify = true;
    const char *csv_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-k") && i + 1 < argc)          ids = parse_ints(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)     sizes = parse_ints(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)     iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc)     warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc)  csv_path = argv[++i];
        else if (!strcmp(argv[i], "--no-verify"))            verify = false;
        else if (!strcmp(argv[i], "-l")) {
            for (int j = 0; j < NUM_KERNELS; ++j)
                printf("  %2d  %-24s %s\n", KERNELS[j].id, KERNELS[j].name, KERNELS[j].note);
            return 0;
        } else { usage(argv[0]); return 1; }
    }

    if (ids.empty())
        for (int j = 0; j < NUM_KERNELS; ++j) ids.push_back(KERNELS[j].id);
    if (sizes.empty()) sizes = {128, 256, 512, 1024, 2048, 4096};

    for (int id : ids) {
        if (!find_kernel(id)) {
            fprintf(stderr, "no kernel with id %d (use -l to list)\n", id);
            return 1;
        }
    }

    CUBLAS_CHECK(cublasCreate(&g_handle));

    int max_n = *std::max_element(sizes.begin(), sizes.end());
    size_t max_elems = (size_t)max_n * max_n;

    std::vector<float> hA(max_elems), hB(max_elems), hC0(max_elems);
    fill_random(hA, 0x12345678u);
    fill_random(hB, 0x9abcdef0u);
    fill_random(hC0, 0x0f0f0f0fu);

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), max_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), max_elems * sizeof(float), cudaMemcpyHostToDevice));

    FILE *csv = nullptr;
    if (csv_path) {
        bool exists = false;
        if (FILE *f = fopen(csv_path, "r")) { exists = true; fclose(f); }
        csv = fopen(csv_path, "a");
        if (!csv) { fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
        if (!exists) fprintf(csv, "kernel_id,kernel,size,gflops_best,gflops_median,cublas_gflops_best,pct_of_cublas\n");
    }

    printf("%-4s %-24s %6s %12s %12s %12s %8s  %s\n", "id", "kernel", "size",
           "best GF/s", "med GF/s", "cuBLAS GF/s", "%cuBLAS", "verify");
    printf("%s\n", std::string(100, '-').c_str());

    for (int id : ids) {
        const KernelEntry *ke = find_kernel(id);
        for (int n : sizes) {
            const int M = n, N = n, K = n;
            size_t elems = (size_t)M * N;

            // --- correctness, with a non-trivial epilogue ---
            Accuracy acc{0, 0, true};
            if (verify) {
                const float valpha = 0.5f, vbeta = 3.0f;
                CUDA_CHECK(cudaMemcpy(dRef, hC0.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
                sgemm_cublas(M, N, K, valpha, dA, dB, vbeta, dRef);
                CUDA_CHECK(cudaMemcpy(dC, hC0.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
                ke->fn(M, N, K, valpha, dA, dB, vbeta, dC);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());

                std::vector<float> got(elems), ref(elems);
                CUDA_CHECK(cudaMemcpy(got.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(ref.data(), dRef, elems * sizeof(float), cudaMemcpyDeviceToHost));
                acc = compare(got, ref);
            }

            // --- timing, beta = 0 so repeated iterations stay bounded ---
            CUDA_CHECK(cudaMemset(dC, 0, elems * sizeof(float)));
            Timing t  = time_kernel(ke->fn,     M, N, K, 1.0f, dA, dB, 0.0f, dC, warmup, iters);
            Timing tb = time_kernel(sgemm_cublas, M, N, K, 1.0f, dA, dB, 0.0f, dC, warmup, iters);

            double gf = gflops(M, N, K, t.best_ms);
            double gfm = gflops(M, N, K, t.median_ms);
            double gfb = gflops(M, N, K, tb.best_ms);
            double pct = 100.0 * gf / gfb;

            printf("%-4d %-24s %6d %12.1f %12.1f %12.1f %7.1f%%  %s\n", ke->id,
                   ke->name, n, gf, gfm, gfb, pct,
                   !verify ? "skipped" : (acc.pass ? "ok" : "FAIL"));
            if (verify && !acc.pass)
                printf("     max_abs=%.3e norm_rel=%.3e\n", acc.max_abs, acc.norm_rel);

            if (csv)
                fprintf(csv, "%d,%s,%d,%.2f,%.2f,%.2f,%.2f\n", ke->id, ke->name,
                        n, gf, gfm, gfb, pct);
        }
    }

    if (csv) fclose(csv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    cublasDestroy(g_handle);
    return 0;
}
