// Kernel 11: `cp.async`, and the question of whether the staging registers were
// ever the thing holding kernel 10 back.
//
// WHAT KERNEL 10 DOES TODAY. Each k-chunk, every thread issues 16 LDG.128 into
// registers, then 64 STS.32 to scatter those 16 float4 into the lane-major
// layout, then a barrier, then computes. Two things about that are suspicious:
//
//   * the float4 in flight pin ~64 registers for the length of the global
//     load, and kernel 10 is register-bound at 255 with a 4-byte spill, so the
//     warp tile that produced most of its speedup is paid for out of the same
//     budget the staging is holding;
//   * nothing overlaps. The global load, the shared store and the barrier are
//     in series with the arithmetic, once per chunk.
//
// `cp.async` addresses both: it moves global->shared without the value ever
// entering a register file, and it is asynchronous, so several chunks can be in
// flight while the current one is being multiplied.
//
// THE CATCH, AND IT IS THE INTERESTING PART. `cp.async` copies a CONTIGUOUS 4,
// 8 or 16 bytes of global to a CONTIGUOUS 4, 8 or 16 bytes of shared. The
// lane-major layout is built so that one lane's four mma operands are
// contiguous in shared -- and those four operands are A[g][t], A[g+8][t],
// A[g][t+4], A[g+8][t+4], which are four unrelated addresses in a row-major A.
// So no 16-byte global chunk maps to a 16-byte shared chunk, ever. The
// hardware's fragment mapping and `cp.async`'s contiguity requirement are
// simply incompatible, and no rearrangement of the tile fixes it, because both
// ends are fixed by hardware.
//
// That leaves 4-byte `cp.async`: one instruction per element instead of one
// LDG.128 per four plus four STS.32.
//
//     per thread, per k-chunk (128x128x32, 128 threads):
//       kernel 10:  16 LDG.128 + 64 STS.32  =  80 instructions, ~64 registers
//       kernel 11:  64 cp.async.4           =  64 instructions,  0 registers
//
// 20% fewer instructions is not the point; the registers and the overlap are.
// And the thread->element map has to change to keep each individual `cp.async`
// coalesced: kernel 10 gives each thread four CONSECUTIVE k so the LDG is a
// float4, which here would make one instruction's warp-wide footprint 32 floats
// spread over 128 -- four sectors touched per 32 useful bytes. Instead each
// thread takes ONE element and consecutive lanes take consecutive k, so every
// `cp.async` is a warp reading one contiguous 128-byte row segment.
//
// WHAT THE PIPELINE COSTS. STAGES shared buffers, so 32 KB per stage at
// BK=32 -- and kernel 10's whole 32 KB footprint is what lets two blocks live
// on an SM. Two stages at BK=32 is 64 KB and one block per SM, which is four
// warps. So the pipeline has to be bought out of BK: BK=16 with two stages is
// the same 32 KB, and BK=16 with three is 48 KB. Whether depth is worth
// shallower reuse per barrier is exactly what the config sweep below measures,
// and it is not obvious in advance -- kernel 10's own sweep found BK=16
// catastrophic (4433 GF/s) when there was no pipeline to pay for it.
//
// THE RESULT, in front rather than buried: 11100 -> 11994 GF/s at N=4096,
// +8.1% at N=4096 and +5.5% at N=2048, with the same arithmetic (the error
// against the fp32 reference is 2.4e-04 for both kernels at N=4096 and 2.7e-04
// at N=2048 -- same mma sequence in the same k order, only the staging moved).
// The sweep table below carries the interesting half, which is that the
// register argument above -- the reason this kernel was written -- accounts
// for a quarter of that, and shared memory capacity decides everything else.
#include "../kernels.h"
#if BMB_TF32
#include "lane_major.cuh"

namespace {

// A generic pointer into the shared window, as the 32-bit address `cp.async`
// wants. `__cvta_generic_to_shared` is the documented way to ask; the older
// trick of casting the pointer straight to unsigned is not portable.
__device__ __forceinline__ unsigned smem_u32(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// One 4-byte global->shared copy that never touches a register.
// `.ca` caches in L1; there is no `.cg` variant at this size.
__device__ __forceinline__ void cp_async4(unsigned dst, const float *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst),
                 "l"(src));
}

// Everything issued since the last commit becomes one group. Groups complete
// in order, which is what makes `wait_group N` mean "all but the newest N are
// done" and therefore what makes a fixed-depth pipeline expressible at all.
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

}  // namespace

template <int BM, int BN, int BK, int WM, int WN, int NUM_THREADS, int STAGES,
          int MINB>
__global__ __launch_bounds__(NUM_THREADS, MINB) void cpasync_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
    // Dynamic rather than static, because a static __shared__ array is capped
    // at 48 KB and half the configurations worth measuring are over it.
    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = smem + STAGES * BM * BK;

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WARPSIZE;
    const int warpIdx = tid / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WMITER = WM / MMA_M;
    constexpr int WNITER = WN / 16;
    constexpr int A_PER = BM * BK / NUM_THREADS;  // cp.async per thread, A
    constexpr int B_PER = BK * BN / NUM_THREADS;  // ... and B

    float acc[WMITER][WNITER][2][4] = {};
    const int nchunks = K / BK;

    // One chunk of A and of B into buffer `buf`. Note what is NOT here: no
    // value, no float4, no register. Only a pair of addresses per element.
    auto fetch = [&](int buf, int k0) {
#pragma unroll
        for (int i = 0; i < A_PER; ++i) {
            const int e = tid + i * NUM_THREADS;
            const int m = e / BK, k = e % BK;
            cp_async4(smem_u32(&As[buf * (BM * BK) + a_off<BM>(m, k)]),
                      &A[(size_t)(cRow + m) * K + k0 + k]);
        }
#pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const int e = tid + i * NUM_THREADS;
            const int k = e / BN, n = e % BN;
            cp_async4(smem_u32(&Bs[buf * (BK * BN) + b_off<BN>(k, n)]),
                      &B[(size_t)(k0 + k) * N + cCol + n]);
        }
    };

    // Byte for byte kernel 10's inner loop; only the base pointers move.
    auto compute = [&](int buf) {
        const float *Ab = &As[buf * (BM * BK)];
        const float *Bb = &Bs[buf * (BK * BN)];
#pragma unroll
        for (int kt = 0; kt < BK / MMA_K; ++kt) {
            unsigned a[WMITER][4], b[WNITER][4];
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                const int unit = kt * (BM / MMA_M) + warpRow * WMITER + i;
                const float4 v = reinterpret_cast<const float4 *>(
                    &Ab[unit * UNIT + (lane ^ a_swz<BM>(unit)) * 4])[0];
                a[i][0] = to_tf32(v.x);
                a[i][1] = to_tf32(v.y);
                a[i][2] = to_tf32(v.z);
                a[i][3] = to_tf32(v.w);
            }
#pragma unroll
            for (int j = 0; j < WNITER; ++j) {
                const int unit = kt * (BN / 16) + warpCol * WNITER + j;
                const float4 v = reinterpret_cast<const float4 *>(
                    &Bb[unit * UNIT + (lane ^ b_swz<BN>(unit)) * 4])[0];
                b[j][0] = to_tf32(v.x);
                b[j][1] = to_tf32(v.y);
                b[j][2] = to_tf32(v.z);
                b[j][3] = to_tf32(v.w);
            }
#pragma unroll
            for (int i = 0; i < WMITER; ++i)
#pragma unroll
                for (int j = 0; j < WNITER; ++j) {
                    mma_m16n8k8(acc[i][j][0], a[i], &b[j][0]);
                    mma_m16n8k8(acc[i][j][1], a[i], &b[j][2]);
                }
        }
    };

    if constexpr (STAGES == 1) {
        // No pipeline: the same schedule kernel 10 runs, with cp.async doing
        // the staging. This configuration exists to separate the two claims --
        // whatever it gains over kernel 10 is the registers alone, and
        // whatever the deeper configurations gain over IT is the overlap.
        for (int c = 0; c < nchunks; ++c) {
            if (c) __syncthreads();  // nobody may overwrite what is being read
            fetch(0, c * BK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            compute(0);
        }
    } else {
        // STAGES-1 chunks in flight. The commit happens even when there is
        // nothing left to fetch, because `wait_group` counts groups and the
        // arithmetic only works if the count keeps advancing.
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < nchunks) fetch(s, s * BK);
            cp_commit();
        }
        int next = STAGES - 1;
        for (int c = 0; c < nchunks; ++c) {
            cp_wait<STAGES - 2>();  // the chunk for buffer c%STAGES has landed
            // Two jobs at once: it publishes that arrival to every thread, and
            // it guarantees the buffer this iteration is about to OVERWRITE is
            // one no thread is still reading -- that buffer was computed
            // STAGES-1 iterations ago, and every thread has passed this
            // barrier since.
            __syncthreads();
            if (next < nchunks) {
                fetch(next % STAGES, next * BK);
                ++next;
            }
            cp_commit();
            compute(c % STAGES);
        }
    }

    // Epilogue, unchanged from kernel 10: a lane's c0/c1 are adjacent columns,
    // so every store is a float2 rather than two scalars.
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < WMITER; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m0 = cRow + warpRow * WM + i * MMA_M;
                const int n0 = cCol + warpCol * WN + j * 16 + h * MMA_N + t * 2;
                float *p0 = &C[(size_t)(m0 + g) * N + n0];
                float *p1 = &C[(size_t)(m0 + g + 8) * N + n0];
                const float *r = acc[i][j][h];
                float2 o0 = reinterpret_cast<float2 *>(p0)[0];
                float2 o1 = reinterpret_cast<float2 *>(p1)[0];
                o0.x = alpha * r[0] + beta * o0.x;
                o0.y = alpha * r[1] + beta * o0.y;
                o1.x = alpha * r[2] + beta * o1.x;
                o1.y = alpha * r[3] + beta * o1.y;
                reinterpret_cast<float2 *>(p0)[0] = o0;
                reinterpret_cast<float2 *>(p1)[0] = o1;
            }
        }
    }
}

void sgemm_cpasync(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {
    // WHAT THE SWEEP FOUND. N=4096, clock pinned to 1200 MHz (nvidia-smi reads
    // 1185-1192), CUDA 13.3, sm_120, median GF/s of 20 iterations, ONE sweep --
    // enough to rank configurations 18% apart, not enough to separate 2 from 7.
    // The published k10-vs-k11 numbers are the median of three sweeps
    // (bench/results_sm120.csv, worst spread 0.9%).
    // scripts\sweep_k11.bat runs the whole table; each entry is a separate
    // compile, because STAGES and BK decide both the register allocation and
    // the shared footprint, so neither can be a runtime switch without changing
    // what is being measured.
    //
    //     cfg  BK  STAGES   smem   blocks/SM    GF/s
    //       -  32    (k10)  32 KB      2       11100   <- kernel 10, for scale
    //       0  32      1    32 KB      2       11329
    //       1  16      2    32 KB      2       11667
    //       2  16      3    48 KB      2       11911   <- chosen
    //       7  16      3    48 KB      2       11875   (WM=32, WN=128)
    //       6   8      4    32 KB      2       10736
    //       3  16      4    64 KB      1        8910
    //       4  32      2    64 KB      1        9156
    //       5  32      3    96 KB      1        9152
    //
    // Two clean readings, and one of them settles the question this kernel was
    // written to ask.
    //
    // FIRST, THE DECOMPOSITION. Config 0 is `cp.async` with no pipeline at all
    // -- the same schedule kernel 10 runs -- and it is worth 2.1%. So freeing
    // the ~64 staging registers, which the previous next-steps list called
    // "the main event", is worth 2%. Configs 1 and 2 add the overlap at the
    // same or slightly larger footprint and take it to 5.1% and 7.4%. The
    // registers were the smaller half; the asynchrony was the point, and the
    // stated reason for doing this was the wrong one.
    //
    // SECOND, THE CLIFF, which is sharper than anything else in this repo.
    // Every configuration at 32 or 48 KB beats kernel 10. Every configuration
    // at 64 KB or more loses to it by 18%, and they all land on the same
    // number (8910, 9156, 9152) regardless of BK or depth. That is not a tuning
    // curve, it is a step: `device_query` reports 100 KiB of shared memory per
    // SM, so two 48 KB blocks fit and two 64 KB blocks cannot, and a 128-thread
    // block alone on an SM is four warps with nothing to switch to. Depth
    // bought with the second block is never worth it -- the deepest pipeline
    // measured here (config 3, four stages) is the WORST entry in the table.
    //
    // The axis that matters is therefore shared memory, not stages:
    // STAGES * (BM*BK + BK*BN) * 4 must stay under half the SM's supply, and
    // every stage beyond the first has to come out of BK to do it.
#ifndef K11_CFG
#define K11_CFG 2
#endif
#if K11_CFG == 0      // no pipeline: isolates what the registers alone are worth
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 64, WN = 64, STAGES = 1, MINB = 2;   // 32 KB
#elif K11_CFG == 1    // same footprint as kernel 10, depth bought out of BK
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, STAGES = 2, MINB = 2;   // 32 KB
#elif K11_CFG == 2
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, STAGES = 3, MINB = 2;   // 48 KB
#elif K11_CFG == 3
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, STAGES = 4, MINB = 1;   // 64 KB
#elif K11_CFG == 4    // kernel 10's BK, one block per SM
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 64, WN = 64, STAGES = 2, MINB = 1;   // 64 KB
#elif K11_CFG == 5
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 32;
    constexpr int WM = 64, WN = 64, STAGES = 3, MINB = 1;   // 96 KB
#elif K11_CFG == 6    // one mma k-step per chunk, four of them in flight
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 8;
    constexpr int WM = 64, WN = 64, STAGES = 4, MINB = 2;   // 32 KB
#elif K11_CFG == 7    // kernel 10's other tied warp tile
    constexpr int NUM_THREADS = 128, BM = 128, BN = 128, BK = 16;
    constexpr int WM = 32, WN = 128, STAGES = 3, MINB = 2;  // 48 KB
#endif

    static_assert((BM / WM) * (BN / WN) == NUM_THREADS / WARPSIZE, "warp grid");
    static_assert(BM * BK % NUM_THREADS == 0, "A staging map");
    static_assert(BK * BN % NUM_THREADS == 0, "B staging map");

    constexpr size_t SMEM = (size_t)STAGES * (BM * BK + BK * BN) * sizeof(float);
    auto kernel = cpasync_kernel<BM, BN, BK, WM, WN, NUM_THREADS, STAGES, MINB>;

    // Over 48 KB the opt-in has to be requested explicitly, and whether the
    // device grants it is a property of the card. Asked once, and if the answer
    // is no this configuration cannot run here at all -- which is a real
    // outcome on a card with a smaller shared file, not an error.
    static int ok = -1;
    if (ok < 0) {
        ok = cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)SMEM) == cudaSuccess;
        if (!ok) cudaGetLastError();
    }

    if (!ok || M % BM != 0 || N % BN != 0 || K % BK != 0) {
        sgemm_mma(M, N, K, alpha, A, B, beta, C);
        return;
    }

    dim3 grid(N / BN, M / BM);
    kernel<<<grid, NUM_THREADS, SMEM>>>(M, N, K, alpha, A, B, beta, C);
}

#else
// Pre-Ampere: no TF32 tensor cores and no cp.async either, so this stage of
// the ladder is not built. The entry stays so the registry does not have to
// change shape per architecture.
void sgemm_cpasync(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {
    sgemm_tile2d(M, N, K, alpha, A, B, beta, C);
}
#endif
