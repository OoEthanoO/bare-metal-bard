// Kernel 6: vectorized (128-bit) memory access.
//
// Two changes, and they depend on each other:
//
// 1. float4 loads. The LDG/STS instructions come in 32/64/128-bit widths. A
//    thread issuing four scalar 32-bit loads spends four instructions and four
//    memory pipeline slots to move what one LDG.E.128 moves. Kernel 5 was
//    already coalescing well, so this buys instruction count and memory-issue
//    slots rather than bytes -- but at ~5 TFLOP/s the inner loop is issue
//    bound, and that is exactly what is scarce.
//
// 2. As is stored TRANSPOSED, as As[BK][BM] instead of As[BM][BK]. In kernel 5
//    the register-tile read was As[(threadRow*TM + i)*BK + k] -- consecutive i
//    strides by BK, so the TM values a thread wants are scattered and must be
//    fetched one at a time. Transposed, the same TM values become CONTIGUOUS
//    in SMEM and are read as two float4s.
//
//    The transpose is not free: the thread loads a float4 along a row of A and
//    must scatter it into 4 different SMEM rows, which is 4 scalar stores. We
//    pay 4 scalar SMEM writes once per tile to save TM scalar SMEM reads on
//    every one of the BK inner iterations. It pays for itself immediately.
//
// This also fixes a bank conflict from kernel 5. There, Bs[k*BN + threadCol*TN]
// with TN=8 meant lane L addressed bank (8L mod 32), so lanes 0,4,8,12 all hit
// bank 0 -- a 4-way conflict. The float4 path splits a warp's 128-bit accesses
// into phases that the hardware services conflict-free.
//
// CONSTRAINT: 128-bit accesses require 16-byte alignment, so this kernel
// requires M, N, K to be multiples of the tile sizes. The launcher falls back
// to kernel 5 when they are not. A production library would ship edge kernels;
// here the fallback is honest and the transformer sizes are all aligned.
#include "../kernels.h"
#include <cassert>

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_kernel(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
    __shared__ float As[BK * BM];  // transposed: As[k][m]
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;

    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);

    // Each thread stages one float4 of A and one of B per K-chunk.
    // A: BM*BK floats = (BM*BK)/4 float4s; with 256 threads and BM=128,BK=8
    // that is 256 float4s, exactly one each.
    const int innerColA = tid % (BK / 4);   // 0..1
    const int innerRowA = tid / (BK / 4);   // 0..127
    const int innerColB = tid % (BN / 4);   // 0..31
    const int innerRowB = tid / (BN / 4);   // 0..7

    float acc[TM][TN] = {{0.0f}};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // --- stage A, transposing on the way into SMEM ---
        float4 a = CVEC4(A[(size_t)(cRow + innerRowA) * K + k0 + innerColA * 4]);
        As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;

        // --- stage B, layout unchanged, straight 128-bit copy ---
        // A warp's 32 lanes cover 32*16B = 512B contiguous. Ideal.
        VEC4(Bs[innerRowB * BN + innerColB * 4]) =
            CVEC4(B[(size_t)(k0 + innerRowB) * N + cCol + innerColB * 4]);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Both operand fetches are now contiguous float4 reads.
#pragma unroll
            for (int i = 0; i < TM; i += 4)
                VEC4(regM[i]) = CVEC4(As[k * BM + threadRow * TM + i]);
#pragma unroll
            for (int j = 0; j < TN; j += 4)
                VEC4(regN[j]) = CVEC4(Bs[k * BN + threadCol * TN + j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // Epilogue, also vectorized. Read-modify-write of C is done a float4 at a
    // time so the beta term costs one 128-bit load instead of four.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t r = (size_t)(cRow + threadRow * TM + i);
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const int c = cCol + threadCol * TN + j;
            float4 old = CVEC4(C[r * N + c]);
            old.x = alpha * acc[i][j + 0] + beta * old.x;
            old.y = alpha * acc[i][j + 1] + beta * old.y;
            old.z = alpha * acc[i][j + 2] + beta * old.z;
            old.w = alpha * acc[i][j + 3] + beta * old.w;
            VEC4(C[r * N + c]) = old;
        }
    }
}

void sgemm_vectorized(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    // Ragged edges would break both the 16-byte alignment and the unguarded
    // tile loads. Defer to the scalar kernel rather than compute garbage.
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
        return;
    }

    dim3 block((BM * BN) / (TM * TN));  // 256
    dim3 grid(N / BN, M / BM);
    vectorized_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
