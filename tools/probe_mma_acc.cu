// What layout does the m16n8k8 TF32 accumulator come back in, and what does it
// cost to feed it straight back in as an A operand?
//
// This matters because fusing attention onto tensor cores needs exactly that:
// S = Q@K^T produces an accumulator, softmax turns it into P, and P is then the
// A operand of P@V. If the two layouts agreed, the fragment could be reused in
// place and the shared-memory round-trip between the two matmuls -- the thing
// that makes the current fp32 backward only 1.06x the unfused path -- would
// simply disappear.
//
// The README's rule for this kind of fact is to measure it. A fragment layout
// guessed wrong is silent: it compiles, runs at full speed, and returns
// garbage. Kernel 10 already found that the a1/a2 order is not the one the
// analogous f16 shape uses.
//
//   scripts\probe.bat tools\probe_mma_acc.cu
#include <cstdio>
#include <cuda_runtime.h>
#include "../src/kernels.h"

#if !BMB_TF32
int main() { printf("needs sm_80+\n"); return 0; }
#else

__device__ __forceinline__ unsigned to_tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_m16n8k8(float (&d)[4], const unsigned (&a)[4],
                                            const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float AV(int r, int k) {
    return k == 0 ? 1.0f : (k == 1 ? (float)r : 0.0f);
}
__device__ __forceinline__ float BV(int k, int n) {
    return k == 0 ? (float)n : (k == 1 ? 16.0f : 0.0f);
}
__device__ __forceinline__ float DV(int r, int n) { return (float)(n + 16 * r); }

// D[r][n] = n + 16*r, built so every one of the 128 entries is distinct and
// decodes by inspection. A and B are loaded through the SAME index maps the
// GEMM kernels use, so the only unknown left in the experiment is D.
//
//   A[r][0] = 1, A[r][1] = r        B[0][n] = n, B[1][n] = 16
//   => D[r][n] = 1*n + r*16
__global__ void probe(float *out, float *perm_out) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;


    // a0=(g,t) a1=(g+8,t) a2=(g,t+4) a3=(g+8,t+4)  -- lane_major.cuh's mapping
    unsigned a[4] = {to_tf32(AV(g, t)), to_tf32(AV(g + 8, t)),
                     to_tf32(AV(g, t + 4)), to_tf32(AV(g + 8, t + 4))};
    // b0=(k0,n0) b1=(k0+4,n0)  with n0=lane/4, k0=lane%4
    unsigned b[2] = {to_tf32(BV(t, g)), to_tf32(BV(t + 4, g))};

    float d[4] = {0, 0, 0, 0};
    mma_m16n8k8(d, a, b);
    for (int i = 0; i < 4; ++i) out[lane * 4 + i] = d[i];

    // ---- second question: can the accumulator be permuted into an A operand
    // with shuffles alone? Treat d as P and try to rebuild the A fragment of
    // the SAME logical matrix, then check it against a direct load.
    //
    // Hypothesis: accumulator lane (g,t) holds columns {2t, 2t+1}; the A
    // operand wants columns {t, t+4}. Column c lives in accumulator lane
    // (g, c>>1), register (c&1) for rows 0-7 and (c&1)+2 for rows 8-15. So
    // each needed column is one __shfl_sync away inside the 4 lanes sharing g.
    const unsigned mask = 0xffffffff;
    float want[4];
    {
        const int srcA = (g << 2) | (t >> 1);        // holds column t
        const int srcB = (g << 2) | ((t + 4) >> 1);  // holds column t+4
        const int par = t & 1;  // which of the source lane's two columns we want
        // The selector has to be applied AFTER the shuffle, not inside it.
        // `__shfl_sync(mask, par ? d[1] : d[0], src)` reads the register that
        // the SOURCE lane's own `par` chose, not the one this lane asked for --
        // which lands on the neighbouring column and is off by exactly 1 in
        // this probe's encoding. So shuffle both and select locally.
        const float a0 = __shfl_sync(mask, d[0], srcA);
        const float a1 = __shfl_sync(mask, d[1], srcA);
        const float a2 = __shfl_sync(mask, d[2], srcA);
        const float a3 = __shfl_sync(mask, d[3], srcA);
        const float b0 = __shfl_sync(mask, d[0], srcB);
        const float b1 = __shfl_sync(mask, d[1], srcB);
        const float b2 = __shfl_sync(mask, d[2], srcB);
        const float b3 = __shfl_sync(mask, d[3], srcB);
        want[0] = par ? a1 : a0;  // (g,   t)
        want[1] = par ? a3 : a2;  // (g+8, t)
        want[2] = par ? b1 : b0;  // (g,   t+4)
        want[3] = par ? b3 : b2;  // (g+8, t+4)
    }
    // Ground truth: the A fragment of the matrix D[r][n] = n + 16r.
    float truth[4] = {DV(g, t), DV(g + 8, t), DV(g, t + 4), DV(g + 8, t + 4)};
    for (int i = 0; i < 4; ++i) perm_out[lane * 4 + i] = want[i] - truth[i];
}

int main() {
    float *d_out, *d_perm;
    cudaMalloc(&d_out, 128 * sizeof(float));
    cudaMalloc(&d_perm, 128 * sizeof(float));
    probe<<<1, 32>>>(d_out, d_perm);
    float h[128], hp[128];
    cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost);
    cudaMemcpy(hp, d_perm, sizeof hp, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA: %s\n", cudaGetErrorString(e)); return 1; }

    printf("m16n8k8 TF32 accumulator layout, from D[r][n] = n + 16r\n\n");
    printf("lane  reg0        reg1        reg2        reg3\n");
    for (int L = 0; L < 32; ++L) {
        printf("%4d", L);
        for (int i = 0; i < 4; ++i) {
            const int v = (int)h[L * 4 + i];
            printf("   (%2d,%d)", v / 16, v % 16);
        }
        printf("\n");
        if (L == 3) printf("     ...\n");
        if (L == 3) L = 27;  // the middle repeats the pattern; show both ends
    }

    // Restate it as a rule and check the rule against every entry.
    printf("\nchecking rule:  reg i of lane L holds "
           "(row = L/4 + 8*(i/2), col = 2*(L%%4) + i%%2)\n");
    int bad = 0;
    for (int L = 0; L < 32; ++L)
        for (int i = 0; i < 4; ++i) {
            const int v = (int)h[L * 4 + i];
            const int r = v / 16, n = v % 16;
            if (r != L / 4 + 8 * (i / 2) || n != 2 * (L % 4) + i % 2) ++bad;
        }
    printf("  %s (%d mismatches)\n", bad ? "RULE IS WRONG" : "holds for all 128", bad);

    double worst = 0;
    for (int i = 0; i < 128; ++i) worst = fmax(worst, fabs((double)hp[i]));
    printf("\naccumulator -> A operand by shuffle: max error %.1f  %s\n", worst,
           worst == 0 ? "EXACT -- no shared-memory round trip needed"
                      : "permutation hypothesis is wrong");
    return 0;
}
#endif
