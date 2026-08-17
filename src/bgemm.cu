// Batched GEMM for attention.
//
// The shapes here are much smaller than the MLP GEMMs -- (256 x 64) @ (64 x
// 256) per head -- so the warp-tiled kernel's 128x128 block tile does not fit;
// it would launch 4 blocks where the work wants 16. This uses a 64x64 tile
// with 4x4 register tiling, which is the same idea one size down, plus bounds
// guards so any head dimension works.
//
// The batch index is gridDim.z. All 96 (batch, head) problems are therefore
// resident in one launch, and the GPU sees 1536 blocks instead of 16 -- enough
// to fill 36 SMs, which a per-head loop never would.
#include "gemm.h"

namespace {
template <bool TA, bool TB, int BM, int BN, int BK, int TM, int TN>
__global__ void bgemm_k(int M, int N, int K, float alpha, const float *A,
                        long long strideA, const float *B, long long strideB,
                        float beta, float *C, long long strideC) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BK * BM];  // transposed for contiguous register reads
    __shared__ float Bs[BK * BN];

    // Each z-slice is an independent problem; the only thing the batch index
    // does is offset the base pointers.
    A += blockIdx.z * strideA;
    B += blockIdx.z * strideB;
    C += blockIdx.z * strideC;

    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN), threadCol = tid % (BN / TN);

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Flat strided staging with guards. Consecutive threads take
        // consecutive values of the fastest-varying index, so the global reads
        // stay coalesced in whichever layout the operand has.
        for (int i = tid; i < BM * BK; i += NUM_THREADS) {
            const int m = i / BK, k = i % BK;
            const int gm = cRow + m, gk = k0 + k;
            float val = 0.0f;
            if (gm < M && gk < K)
                val = TA ? A[(size_t)gk * M + gm] : A[(size_t)gm * K + gk];
            As[k * BM + m] = val;
        }
        for (int i = tid; i < BK * BN; i += NUM_THREADS) {
            const int k = i / BN, n = i % BN;
            const int gk = k0 + k, gn = cCol + n;
            float val = 0.0f;
            if (gk < K && gn < N)
                val = TB ? B[(size_t)gn * K + gk] : B[(size_t)gk * N + gn];
            Bs[k * BN + n] = val;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[k * BM + threadRow * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = cRow + threadRow * TM + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = cCol + threadCol * TN + j;
            if (c < N) {
                const size_t idx = (size_t)r * N + c;
                C[idx] = alpha * acc[i][j] + beta * C[idx];
            }
        }
    }
}

template <bool TA, bool TB>
void dispatch_b(int batch, int M, int N, int K, float alpha, const float *A,
                long long sA, const float *B, long long sB, float beta,
                float *C, long long sC, cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
    bgemm_k<TA, TB, BM, BN, BK, TM, TN>
        <<<grid, THREADS, 0, stream>>>(M, N, K, alpha, A, sA, B, sB, beta, C, sC);
}
}  // namespace

void batched_gemm(bool transA, bool transB, int batch, int M, int N, int K,
                  float alpha, const float *A, long long strideA,
                  const float *B, long long strideB, float beta, float *C,
                  long long strideC, cudaStream_t stream) {
    if (!transA && !transB)      dispatch_b<false, false>(batch, M, N, K, alpha, A, strideA, B, strideB, beta, C, strideC, stream);
    else if (!transA && transB)  dispatch_b<false, true >(batch, M, N, K, alpha, A, strideA, B, strideB, beta, C, strideC, stream);
    else if (transA && !transB)  dispatch_b<true,  false>(batch, M, N, K, alpha, A, strideA, B, strideB, beta, C, strideC, stream);
    else                         dispatch_b<true,  true >(batch, M, N, K, alpha, A, strideA, B, strideB, beta, C, strideC, stream);
}
