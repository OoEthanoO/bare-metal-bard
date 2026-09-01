// Correctness test for the transpose-aware GEMM, against cuBLAS, across all
// four transpose combinations and a mix of aligned, ragged, and non-square
// shapes. The backward pass depends on the NT and TN cases specifically, and a
// silent bug there would show up only as a model that trains slightly wrong --
// the worst kind of bug to chase. So it gets tested directly.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../src/gemm.h"

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %d at %d\n",(int)s,__LINE__); exit(1);} } while(0)

static cublasHandle_t handle;

// Row-major C[M,N] = alpha*op(A)@op(B) + beta*C via a column-major library.
//
// A column-major read of a row-major R x S matrix with ld = S yields its
// transpose. Working through each case:
//   transA=false: A stored MxK -> column-major view is A^T (KxM) = op(A)^T. OP_N, lda=K
//   transA=true : A stored KxM -> column-major view is (A_st)^T (MxK);
//                 op(A)^T = A_st (KxM), so transpose it back.        OP_T, lda=M
// and symmetrically for B. We then compute C^T = op(B)^T @ op(A)^T.
static void gemm_ref(bool transA, bool transB, int M, int N, int K, float alpha,
                     const float *A, const float *B, float beta, float *C) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    CUBLAS_CHECK(cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda,
                             &beta, C, N));
}

static void fill(std::vector<float> &v, unsigned s) {
    for (size_t i = 0; i < v.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int)(s % 2000) - 1000) / 1000.0f;
    }
}


// Time one gemm configuration. Both paths are timed in the same process, back
// to back, so they see the same thermal state -- the same reason the SGEMM
// harness re-times cuBLAS next to every kernel.
static double time_gemm(bool TA, bool TB, int M, int N, int K, const float *A,
                        const float *B, float *C, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) gemm(TA, TB, M, N, K, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a);
        gemm(TA, TB, M, N, K, 1.0f, A, B, 0.0f, C);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

// The same timing loop against cuBLAS, so a "% of cuBLAS" at a MODEL shape is
// measured rather than extrapolated from the square-N ladder.
//
// The math mode is set per call and restored, matching what src/bench.cu does:
// cuBLAS SGEMM is true fp32 unless opted in, so the fp32 and TF32 columns here
// are the same two denominators the rest of this repo quotes.
static double time_cublas(bool TA, bool TB, int M, int N, int K, const float *A,
                          const float *B, float *C, int iters, bool tf32) {
    CUBLAS_CHECK(cublasSetMathMode(
        handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) gemm_ref(TA, TB, M, N, K, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a);
        gemm_ref(TA, TB, M, N, K, 1.0f, A, B, 0.0f, C);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    return best;
}

int main(int argc, char **argv) {
    // --tf32 routes the same 56 cases through the tensor-core path. TF32 keeps
    // 10 mantissa bits, so the bar moves from 1e-4 to 5e-3: the point of the
    // run is that the transpose staging is right, not that the format is exact.
    bool tf32 = false, bench = false, splitk = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--tf32")) tf32 = true;
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--splitk")) splitk = true;
    }
    if (tf32) {
        gemm_set_tf32(true);
        printf("TENSOR-CORE path (TF32), tolerance 5e-3\n\n");
    }
    const double TOL = tf32 ? 5e-3 : 1e-4;
    CUBLAS_CHECK(cublasCreate(&handle));

    struct Shape { int M, N, K; const char *tag; };
    const Shape shapes[] = {
        {128, 128, 128,   "aligned tiny"},
        {256, 512, 128,   "aligned"},
        {4096, 384, 384,  "gpt: attn proj"},
        {4096, 1536, 384, "gpt: mlp up"},
        {4096, 384, 1536, "gpt: mlp down"},
        {4096, 1152, 384, "gpt: qkv proj"},
        {384, 384, 4096,  "gpt: dW attnproj"},
        {384, 1152, 4096, "gpt: dW qkv"},
        {384, 1536, 4096, "gpt: dW fc"},
        {1536, 384, 4096, "gpt: dW fcproj"},
        {4096, 128, 384,  "gpt: padded vocab head"},
        {100, 70, 33,     "ragged -> generic path"},
        {65, 65, 65,      "ragged square"},
        {130, 260, 60,    "ragged non-square"},
    };

    int failures = 0;

    // --bench answers a specific question: the tensor-core kernel is faster
    // than the fp32 one when measured standalone, and slower inside the model.
    // Timing the SAME entry point both ways, per shape and per transpose case,
    // is what tells you which of the four cases pays for it.
    if (bench) {
        printf("%-22s %6s %6s %6s  %-4s %9s %9s %9s %10s %8s\n", "shape", "M",
               "N", "K", "op", "mine f32", "mine tf32", "cuBLAS32",
               "cuBLASTF32", "% cuTF32");
        printf("------------------------------------------------------------"
               "----------------------------------\n");
        for (const Shape &s : shapes) {
            if (s.M % 128 || s.N % 128 || s.K % 32) continue;  // tc path needs this
            size_t szA = (size_t)s.M * s.K, szB = (size_t)s.K * s.N,
                   szC = (size_t)s.M * s.N;
            float *dA, *dB, *dC;
            CUDA_CHECK(cudaMalloc(&dA, szA * 4));
            CUDA_CHECK(cudaMalloc(&dB, szB * 4));
            CUDA_CHECK(cudaMalloc(&dC, szC * 4));
            CUDA_CHECK(cudaMemset(dA, 0, szA * 4));
            CUDA_CHECK(cudaMemset(dB, 0, szB * 4));
            const double flop = 2.0 * s.M * s.N * s.K;
            for (int t = 0; t < 4; ++t) {
                const bool TA = t & 1, TB = t & 2;
                const char *tag = TA ? (TB ? "TT" : "TN") : (TB ? "NT" : "NN");
                gemm_set_tf32(false);
                const double f = time_gemm(TA, TB, s.M, s.N, s.K, dA, dB, dC, 20);
                gemm_set_tf32(true);
                const double g = time_gemm(TA, TB, s.M, s.N, s.K, dA, dB, dC, 20);
                gemm_set_tf32(false);
                // Interleaved with mine rather than run as its own sweep, so
                // numerator and denominator come from the same few
                // milliseconds of machine state.
                const double cf =
                    time_cublas(TA, TB, s.M, s.N, s.K, dA, dB, dC, 20, false);
                const double ct =
                    time_cublas(TA, TB, s.M, s.N, s.K, dA, dB, dC, 20, true);
                printf("%-22s %6d %6d %6d  %-4s %9.1f %9.1f %9.1f %10.1f %7.1f%%\n",
                       s.tag, s.M, s.N, s.K, tag, flop / (f * 1e-3) / 1e9,
                       flop / (g * 1e-3) / 1e9, flop / (cf * 1e-3) / 1e9,
                       flop / (ct * 1e-3) / 1e9, 100.0 * ct / g);
            }
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }
        cublasDestroy(handle);
        return 0;
    }

    // --splitk sweeps the K-split count at every model shape, because the
    // derived rule is one point on a curve and the point of having a rule is to
    // land on the right one. Only the TN case is swept: every weight gradient
    // in the model is gemm(true, false, ...), so that is the case that has to
    // be right, and the forward shapes are here to show what the same rule
    // would do to them -- a split-K rule that helps dW and quietly costs the
    // forward pass is not an improvement.
    if (splitk) {
        gemm_set_tf32(true);
        const int MAXSP = 16;
        printf("K-split sweep, TF32, TN (what every dW runs). GF/s.%s%s", "\n", "\n");
        printf("%-20s %5s", "shape", "blk");
        for (int sp = 1; sp <= MAXSP; ++sp) printf("%6d", sp);
        printf("   %8s %8s%s", "derived", "best@", "\n");
        for (const Shape &s2 : shapes) {
            if (s2.M % 128 || s2.N % 128 || s2.K % 32) continue;
            if ((size_t)s2.M * s2.N * s2.K < 100000000ull) continue;  // skip toys
            size_t szA = (size_t)s2.M * s2.K, szB = (size_t)s2.K * s2.N,
                   szC = (size_t)s2.M * s2.N;
            float *dA, *dB, *dC;
            CUDA_CHECK(cudaMalloc(&dA, szA * 4));
            CUDA_CHECK(cudaMalloc(&dB, szB * 4));
            CUDA_CHECK(cudaMalloc(&dC, szC * 4));
            CUDA_CHECK(cudaMemset(dA, 0, szA * 4));
            CUDA_CHECK(cudaMemset(dB, 0, szB * 4));
            const double flop = 2.0 * s2.M * s2.N * s2.K;
            // The block count the narrow tile would make, which is what the
            // rule keys on.
            printf("%-20s %5d", s2.tag, (s2.M / 64) * (s2.N / 128));
            double best = 0;
            int bestsp = 0;
            for (int sp = 1; sp <= MAXSP; ++sp) {
                gemm_set_splitk(sp);
                const double t =
                    time_gemm(true, false, s2.M, s2.N, s2.K, dA, dB, dC, 20);
                const double gf = flop / (t * 1e-3) / 1e9;
                if (gf > best) { best = gf; bestsp = sp; }
                printf("%6.0f", gf);
            }
            gemm_set_splitk(0);
            const double t0 =
                time_gemm(true, false, s2.M, s2.N, s2.K, dA, dB, dC, 20);
            const double d0 = flop / (t0 * 1e-3) / 1e9;
            printf("   %8.0f %5.0f@%-2d  %+.1f%%%s", d0, best, bestsp,
                   100.0 * (best / d0 - 1.0), "\n");
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }
        cublasDestroy(handle);
        return 0;
    }

    printf("%-22s %6s %6s %6s  %-6s %10s  %s\n", "shape", "M", "N", "K", "op", "norm_rel", "");
    printf("--------------------------------------------------------------------------\n");

    for (const Shape &s : shapes) {
        size_t szA = (size_t)s.M * s.K, szB = (size_t)s.K * s.N, szC = (size_t)s.M * s.N;
        std::vector<float> hA(szA), hB(szB), hC(szC);
        fill(hA, 0x1234u); fill(hB, 0x9abcu); fill(hC, 0x5555u);

        float *dA, *dB, *dC, *dR;
        CUDA_CHECK(cudaMalloc(&dA, szA * 4));
        CUDA_CHECK(cudaMalloc(&dB, szB * 4));
        CUDA_CHECK(cudaMalloc(&dC, szC * 4));
        CUDA_CHECK(cudaMalloc(&dR, szC * 4));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * 4, cudaMemcpyHostToDevice));

        // The A and B buffers are reinterpreted with swapped dimensions for the
        // transposed cases; element count is identical so the same allocation
        // serves, which is exactly how the backward pass reuses activations.
        for (int t = 0; t < 4; ++t) {
            const bool TA = t & 1, TB = t & 2;
            const char *tag = TA ? (TB ? "TT" : "TN") : (TB ? "NT" : "NN");
            const float alpha = 0.75f, beta = 1.25f;

            CUDA_CHECK(cudaMemcpy(dC, hC.data(), szC * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dR, hC.data(), szC * 4, cudaMemcpyHostToDevice));

            gemm(TA, TB, s.M, s.N, s.K, alpha, dA, dB, beta, dC);
            gemm_ref(TA, TB, s.M, s.N, s.K, alpha, dA, dB, beta, dR);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            std::vector<float> got(szC), ref(szC);
            CUDA_CHECK(cudaMemcpy(got.data(), dC, szC * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ref.data(), dR, szC * 4, cudaMemcpyDeviceToHost));

            double maxabs = 0, refinf = 0;
            for (size_t i = 0; i < szC; ++i) {
                maxabs = std::max(maxabs, std::fabs((double)got[i] - (double)ref[i]));
                refinf = std::max(refinf, std::fabs((double)ref[i]));
            }
            double rel = maxabs / std::max(refinf, 1e-30);
            bool ok = rel < TOL;
            if (!ok) ++failures;
            printf("%-22s %6d %6d %6d  %-6s %10.2e  %s\n", s.tag, s.M, s.N, s.K,
                   tag, rel, ok ? "ok" : "FAIL");
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    }

    // ---- batched GEMM, the attention shapes ----
    //
    // These are verified indirectly by the gradient check, but only as a pass/
    // fail on the whole model. Testing them directly localizes a failure to the
    // batched kernel instead of to "somewhere in attention".
    printf("\nbatched GEMM (attention shapes), vs cuBLAS per batch\n");
    printf("--------------------------------------------------------------------------\n");
    struct BShape { int batch, M, N, K; const char *tag; };
    const BShape bshapes[] = {
        {96, 256, 256, 64, "q@k^T   (B*NH=96)"},
        {96, 256, 64, 256, "att@v   (B*NH=96)"},
        {24, 64, 64, 32,   "small heads"},
        {8, 100, 48, 33,   "ragged"},
    };

    for (const BShape &s : bshapes) {
        const size_t eA = (size_t)s.M * s.K, eB = (size_t)s.K * s.N, eC = (size_t)s.M * s.N;
        std::vector<float> hA(eA * s.batch), hB(eB * s.batch), hC(eC * s.batch);
        fill(hA, 0x2222u); fill(hB, 0x3333u); fill(hC, 0x4444u);

        float *dA, *dB, *dC, *dR;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * 4));
        CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
        CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4));
        CUDA_CHECK(cudaMalloc(&dR, hC.size() * 4));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

        for (int t = 0; t < 4; ++t) {
            const bool TA = t & 1, TB = t & 2;
            const char *tag = TA ? (TB ? "TT" : "TN") : (TB ? "NT" : "NN");
            const float alpha = 0.5f, beta = 2.0f;

            CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dR, hC.data(), hC.size() * 4, cudaMemcpyHostToDevice));

            batched_gemm(TA, TB, s.batch, s.M, s.N, s.K, alpha, dA, (long long)eA,
                         dB, (long long)eB, beta, dC, (long long)eC);
            // Reference: the same problem, one batch at a time.
            for (int b = 0; b < s.batch; ++b)
                gemm_ref(TA, TB, s.M, s.N, s.K, alpha, dA + b * eA, dB + b * eB,
                         beta, dR + b * eC);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            std::vector<float> got(hC.size()), ref(hC.size());
            CUDA_CHECK(cudaMemcpy(got.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ref.data(), dR, hC.size() * 4, cudaMemcpyDeviceToHost));

            double maxabs = 0, refinf = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                maxabs = std::max(maxabs, std::fabs((double)got[i] - (double)ref[i]));
                refinf = std::max(refinf, std::fabs((double)ref[i]));
            }
            const double rel = maxabs / std::max(refinf, 1e-30);
            const bool ok = rel < TOL;
            if (!ok) ++failures;
            printf("%-22s %6d %6d %6d  %-6s %10.2e  %s\n", s.tag, s.M, s.N, s.K,
                   tag, rel, ok ? "ok" : "FAIL");
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    }

    // ---- fused bias gradient (GemmEpilogue::dbias_out) ----
    //
    // dbias[m] = sum_k op(A)[m][k], accumulated += onto what is already there.
    // The sum is taken from the raw fp32 values BEFORE any TF32 rounding, so
    // the reference is an exact column sum and the bar is fp32 summation
    // order (1e-4), far tighter than the matmul's TF32 bar. Three properties
    // are checked per shape: the sum itself (against a double-precision CPU
    // reference, += included), that C is still the correct matmul (the fusion
    // must not disturb the output it rides on), and that two identical calls
    // give bitwise-identical dbias -- this repo forbids gradients that depend
    // on block scheduling order, so determinism is a spec, not a nicety.
    printf("\nfused bias gradient (dbias_out), TN, vs CPU column sum\n");
    printf("--------------------------------------------------------------------------\n");
    if (!gemm_dbias_supported(true, false, 384, 1536, 4096)) {
        printf("not available (needs --tf32 and the sm_80+ mma path) -- skipped\n");
    } else {
        struct DShape { int M, N, K, force_sp; const char *tag; };
        const DShape dshapes[] = {
            {384, 1536, 4096, 0, "dW fcproj-like, derived"},
            {1536, 384, 4096, 0, "dW fc-like, derived"},
            {1152, 384, 4096, 0, "dW qkv-like, derived"},
            {384, 384, 4096, 0, "dW attnproj-like"},
            {1536, 384, 384, 0, "K=384: never splits"},
            {384, 1536, 4096, 1, "forced s=1"},
            {384, 1536, 4096, 2, "forced s=2"},
            {384, 1536, 4096, 5, "forced s=5"},
            {256, 128, 128, 0, "tiny aligned"},
        };
        for (const DShape &s : dshapes) {
            const size_t szA = (size_t)s.K * s.M;  // A stored K x M (transA)
            const size_t szB = (size_t)s.K * s.N, szC = (size_t)s.M * s.N;
            std::vector<float> hA(szA), hB(szB), hPre(s.M);
            fill(hA, 0x7777u); fill(hB, 0x8888u);
            for (int m = 0; m < s.M; ++m) hPre[m] = 0.25f * (m % 7) - 0.5f;

            float *dA, *dB, *dC, *dR, *dDB;
            CUDA_CHECK(cudaMalloc(&dA, szA * 4));
            CUDA_CHECK(cudaMalloc(&dB, szB * 4));
            CUDA_CHECK(cudaMalloc(&dC, szC * 4));
            CUDA_CHECK(cudaMalloc(&dR, szC * 4));
            CUDA_CHECK(cudaMalloc(&dDB, (size_t)s.M * 4));
            CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(dC, 0, szC * 4));
            CUDA_CHECK(cudaMemset(dR, 0, szC * 4));

            gemm_set_splitk(s.force_sp);
            GemmEpilogue ep;
            ep.dbias_out = dDB;

            std::vector<float> got1(s.M), got2(s.M), gotC(szC), refC(szC);
            CUDA_CHECK(cudaMemcpy(dDB, hPre.data(), (size_t)s.M * 4,
                                  cudaMemcpyHostToDevice));
            gemm(true, false, s.M, s.N, s.K, 1.0f, dA, dB, 0.0f, dC, 0, ep);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(got1.data(), dDB, (size_t)s.M * 4,
                                  cudaMemcpyDeviceToHost));
            // Same prefill, same call: the gradient must come back bit for bit.
            CUDA_CHECK(cudaMemcpy(dDB, hPre.data(), (size_t)s.M * 4,
                                  cudaMemcpyHostToDevice));
            gemm(true, false, s.M, s.N, s.K, 1.0f, dA, dB, 0.0f, dC, 0, ep);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(got2.data(), dDB, (size_t)s.M * 4,
                                  cudaMemcpyDeviceToHost));
            gemm_set_splitk(0);

            gemm_ref(true, false, s.M, s.N, s.K, 1.0f, dA, dB, 0.0f, dR);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(gotC.data(), dC, szC * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(refC.data(), dR, szC * 4, cudaMemcpyDeviceToHost));

            // The error is judged against the sum of ABSOLUTE values, not the
            // signed sum: a column whose terms nearly cancel has a tiny result
            // sitting on a large accumulation, and fp32 rounding there is
            // condition, not a bug. Against the absolute sum the noise floor
            // is ~1e-7 and a single dropped or double-counted element is
            // ~5e-4, so 1e-6 separates them by three orders each way.
            double maxrel = 0;
            for (int m = 0; m < s.M; ++m) {
                double r = (double)hPre[m], asum = std::fabs((double)hPre[m]);
                for (int k = 0; k < s.K; ++k) {
                    const double a = (double)hA[(size_t)k * s.M + m];
                    r += a;
                    asum += std::fabs(a);
                }
                maxrel = std::max(maxrel, std::fabs((double)got1[m] - r) /
                                              std::max(asum, 1e-3));
            }
            int bitdiff = 0;
            for (int m = 0; m < s.M; ++m)
                if (memcmp(&got1[m], &got2[m], 4) != 0) ++bitdiff;
            double maxabs = 0, refinf = 0;
            for (size_t i = 0; i < szC; ++i) {
                maxabs = std::max(maxabs, std::fabs((double)gotC[i] - (double)refC[i]));
                refinf = std::max(refinf, std::fabs((double)refC[i]));
            }
            const double crel = maxabs / std::max(refinf, 1e-30);

            const bool ok = maxrel < 1e-6 && bitdiff == 0 && crel < TOL;
            if (!ok) ++failures;
            printf("%-22s %6d %6d %6d  %-6s %10.2e  %s%s%s\n", s.tag, s.M, s.N,
                   s.K, "TN", maxrel, ok ? "ok" : "FAIL",
                   bitdiff ? "  NONDETERMINISTIC" : "",
                   crel >= TOL ? "  C-CORRUPTED" : "");
            cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR); cudaFree(dDB);
        }
    }

    printf("--------------------------------------------------------------------------\n");
    printf("%s (%d failures)\n", failures ? "FAILED" : "all passed", failures);
    cublasDestroy(handle);
    return failures ? 1 : 0;
}
