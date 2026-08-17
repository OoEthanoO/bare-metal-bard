// Correctness test for the transpose-aware GEMM, against cuBLAS, across all
// four transpose combinations and a mix of aligned, ragged, and non-square
// shapes. The backward pass depends on the NT and TN cases specifically, and a
// silent bug there would show up only as a model that trains slightly wrong --
// the worst kind of bug to chase. So it gets tested directly.
#include <cstdio>
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

int main() {
    CUBLAS_CHECK(cublasCreate(&handle));

    struct Shape { int M, N, K; const char *tag; };
    const Shape shapes[] = {
        {128, 128, 128,   "aligned tiny"},
        {256, 512, 128,   "aligned"},
        {4096, 384, 384,  "gpt: attn proj"},
        {4096, 1536, 384, "gpt: mlp up"},
        {4096, 384, 1536, "gpt: mlp down"},
        {384, 384, 4096,  "gpt: dW (TN)"},
        {4096, 128, 384,  "gpt: padded vocab head"},
        {100, 70, 33,     "ragged -> generic path"},
        {65, 65, 65,      "ragged square"},
        {130, 260, 60,    "ragged non-square"},
    };

    int failures = 0;
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
            bool ok = rel < 1e-4;
            if (!ok) ++failures;
            printf("%-22s %6d %6d %6d  %-6s %10.2e  %s\n", s.tag, s.M, s.N, s.K,
                   tag, rel, ok ? "ok" : "FAIL");
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    }

    printf("--------------------------------------------------------------------------\n");
    printf("%s (%d failures)\n", failures ? "FAILED" : "all passed", failures);
    cublasDestroy(handle);
    return failures ? 1 : 0;
}
