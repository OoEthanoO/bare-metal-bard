// Fused causal attention: the score matrix never reaches global memory.
//
// The unfused path is three kernels and two (B, NH, T, T) buffers. Its problem
// is not that any one of those kernels is slow -- the batched GEMM is the same
// code that hits 90% of cuBLAS -- but that the intermediate has to be written
// out and read back, and the softmax over it does 5 flops per 8 bytes. On a
// card whose ridge point is 43 FLOP/byte that is 0.6% of peak, no matter how
// well the kernel is written. The only fix is to not have the intermediate.
//
// THE TRICK, in one paragraph. A softmax normally needs the whole row before it
// can produce any output, because it needs the row max and the row sum. But
// both are running statistics: after seeing part of a row you have a partial
// max m and a partial sum l, and when a later block raises the max to m', every
// quantity computed so far is corrected by the single factor exp(m - m'). So
// the output accumulator can be built block by block and rescaled in place:
//
//     m'   = max(m, rowmax(S_j))
//     l'   = l * exp(m - m') + rowsum(exp(S_j - m'))
//     O'   = O * exp(m - m') + exp(S_j - m') @ V_j
//
// with O divided by l only at the very end. Every S_j lives in registers for
// the few instructions it takes to consume it, and is then gone. This is
// FlashAttention (Dao et al. 2022), and the FLOP count is slightly *higher*
// than the unfused version -- the win is entirely in what does not get written.
//
// Two further wins fall out of the structure:
//
//   * Causality is a loop bound, not a mask. The unfused path computes all
//     T x T scores and discards the upper triangle inside the softmax. Here a
//     query block simply never visits key blocks beyond its diagonal, so the
//     attention matmuls do half the work.
//
//   * The head permute disappears. The unfused path pays a full bandwidth-bound
//     pass over 3*B*T*C to make each head's slice contiguous, because a batched
//     GEMM needs uniform strides. This kernel indexes q/k/v directly out of the
//     (B, T, 3C) projection: one block owns one head, so the head offset is a
//     constant added to the row pointer.
#include "flash.h"
#include "gemm.h"  // gemm_tf32(): attention follows the matmuls' opt-in
#include "kernels/lane_major.cuh"  // mma layout, shared with kernels 10 and 11
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>

#define VEC4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define CVEC4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Reduction across the CG threads that share a row of the S tile.
//
// Those threads are lanes of one warp differing only in the low log2(CG) bits
// of their id (tCol = tid % CG), so an xor butterfly of width CG reduces
// exactly them and leaves the result in all of them -- no broadcast needed, and
// no shared memory, which keeps these reductions off the same LSU path the
// matmul is already saturating.
template <int CG, bool IS_MAX>
__device__ __forceinline__ float row_reduce(float v) {
#pragma unroll
    for (int off = 1; off < CG; off <<= 1) {
        const float o = __shfl_xor_sync(0xffffffff, v, off);
        v = IS_MAX ? fmaxf(v, o) : v + o;
    }
    return v;
}

// Move N contiguous floats to or from shared memory, vectorized when N allows.
//
// The tile shapes want to be free to use a 2-wide column tile, which a float4
// cannot express. Rather than exclude those shapes -- and they are the ones
// that double occupancy, because a narrower tile means more threads per block
// for the same shared memory -- the width becomes a compile-time branch.
template <int N>
__device__ __forceinline__ void load_n(float *dst, const float *src) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int i = 0; i < N; i += 4) VEC4(dst[i]) = CVEC4(src[i]);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) dst[i] = src[i];
    }
}

template <int N>
__device__ __forceinline__ void store_n(float *dst, const float *src) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int i = 0; i < N; i += 4) VEC4(dst[i]) = CVEC4(src[i]);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) dst[i] = src[i];
    }
}

// ---------------------------------------------------------------- forward
//
// Grid  (ceil(T/BR), B*NH): one block owns BR query rows of one head.
// Block NT threads, arranged RG x CG over the BR x BC score tile, each thread
// holding an RT x CT patch of it in registers and an RT x OT patch of the
// output accumulator.
//
// Shared memory holds three tiles. Q and K are stored *transposed* -- Qst[i][r]
// rather than Qs[r][i] -- for the same reason kernel 6 transposed its A tile:
// the inner product runs over i, so with the transpose each thread's RT rows
// and CT columns are contiguous and load as float4. V is stored straight,
// because the second matmul reduces over keys and reads V along its head axis.
//
// K's tile and P's tile are the same allocation. Once S is in registers for a
// key block, the K tile is dead, so P can be written over it. That costs one
// extra barrier per key block and saves max(HS*BC, BC*BR) floats -- 8.5 KB at
// the default shape, 16 KB when BR = BC = HS. Shared memory is what caps
// blocks per SM here, so the saving is occupancy, which the sweep in
// test_flash then confirms or refutes for each shape rather than assuming.
template <int BR, int BC, int HS, int NT, int RT, int CT>
__global__ __launch_bounds__(NT) void flash_fwd_k(float *__restrict__ out,
                                                  float *__restrict__ lse,
                                                  const float *__restrict__ qkv,
                                                  int T, int NH, float scale) {
    constexpr int PAD = 4;          // pad rows so column walks change bank
    constexpr int RG = BR / RT;     // thread rows
    constexpr int CG = BC / CT;     // thread columns
    constexpr int OT = HS / CG;     // output columns per thread
    constexpr int V4 = HS / 4;      // float4 per head-size row
    constexpr int QW = BR + PAD;    // row width of Qst and Pst
    constexpr int KW = BC + PAD;    // row width of Kst

    static_assert(RG * CG == NT, "thread grid must cover the S tile exactly");
    static_assert(HS % CG == 0, "output tile must divide evenly");
    static_assert(32 % CG == 0, "row-mates must share a warp for the shuffle");
    static_assert(RT % 4 == 0 && CT % 4 == 0 && OT % 4 == 0, "vector loads");
    static_assert(NT % V4 == 0, "staging must tile evenly");

    extern __shared__ float smem[];
    constexpr int QSZ = HS * QW;
    constexpr int KSZ = (HS * KW > BC * QW) ? HS * KW : BC * QW;
    float *Qst = smem;         // Qst[i][r], premultiplied by 1/sqrt(hs)
    float *KPst = Qst + QSZ;   // Kst[i][c], then Pst[c][r]
    float *Vs = KPst + KSZ;    // Vs[c][j]

    const int tid = threadIdx.x;
    const int tRow = tid / CG, tCol = tid % CG;
    const int r0 = tRow * RT;   // this thread's rows within the tile
    const int c0 = tCol * CT;   // its score columns
    const int o0 = tCol * OT;   // its output columns

    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int C = NH * HS;
    const int i0 = blockIdx.x * BR;

    // Row t of this (b, h): q at +0, k at +C, v at +2C.
    const float *base = qkv + (size_t)b * T * 3 * C + h * HS;

    // --- stage Q once; it is reused by every key block ---
    {
        const int i4 = (tid % V4) * 4;
#pragma unroll
        for (int r = tid / V4; r < BR; r += NT / V4) {
            const int t = i0 + r;
            float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < T) q = CVEC4(base[(size_t)t * 3 * C + i4]);
            // Fold the 1/sqrt(hs) into Q here rather than scaling S later:
            // BR*HS multiplies once instead of BR*T per block.
            Qst[(i4 + 0) * QW + r] = q.x * scale;
            Qst[(i4 + 1) * QW + r] = q.y * scale;
            Qst[(i4 + 2) * QW + r] = q.z * scale;
            Qst[(i4 + 3) * QW + r] = q.w * scale;
        }
    }

    float acc[RT][OT];  // unnormalized output, divided by l at the end
    float mrow[RT], lrow[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        mrow[r] = -INFINITY;
        lrow[r] = 0.0f;
#pragma unroll
        for (int j = 0; j < OT; ++j) acc[r][j] = 0.0f;
    }

    // Causal bound: the last query in this block is i0+BR-1, so no key beyond
    // it is ever needed. This is where half the attention FLOPs go missing.
    const int jmax = min(T, i0 + BR);

    for (int j0 = 0; j0 < jmax; j0 += BC) {
        __syncthreads();  // previous iteration's readers of Pst/Vs are done

        // --- stage K (transposed) and V ---
        for (int lin = tid; lin < BC * V4; lin += NT) {
            const int c = lin / V4, i4 = (lin % V4) * 4;
            const int t = j0 + c;
            float4 k = make_float4(0.f, 0.f, 0.f, 0.f), v = k;
            if (t < T) {
                k = CVEC4(base[(size_t)t * 3 * C + C + i4]);
                v = CVEC4(base[(size_t)t * 3 * C + 2 * C + i4]);
            }
            KPst[(i4 + 0) * KW + c] = k.x;
            KPst[(i4 + 1) * KW + c] = k.y;
            KPst[(i4 + 2) * KW + c] = k.z;
            KPst[(i4 + 3) * KW + c] = k.w;
            VEC4(Vs[c * HS + i4]) = v;
        }
        __syncthreads();

        // --- S = (Q/sqrt(hs)) @ K^T, in registers ---
        float s[RT][CT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < CT; ++c) s[r][c] = 0.0f;

        for (int i = 0; i < HS; ++i) {
            float rq[RT], rk[CT];
#pragma unroll
            for (int r = 0; r < RT; r += 4) VEC4(rq[r]) = CVEC4(Qst[i * QW + r0 + r]);
#pragma unroll
            for (int c = 0; c < CT; c += 4) VEC4(rk[c]) = CVEC4(KPst[i * KW + c0 + c]);
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int c = 0; c < CT; ++c) s[r][c] += rq[r] * rk[c];
        }

        // --- mask, then one step of the online softmax ---
        //
        // Masked entries are -inf rather than a large negative number, so that
        // exp() gives exactly 0 and a fully-masked row is detectable (its
        // running max is still -inf) instead of quietly producing exp(0) = 1.
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            const int qg = i0 + r0 + r;
            float rmax = -INFINITY;
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                const int kg = j0 + c0 + c;
                if (kg > qg || kg >= T) s[r][c] = -INFINITY;
                rmax = fmaxf(rmax, s[r][c]);
            }
            rmax = row_reduce<CG, true>(rmax);

            const float mnew = fmaxf(mrow[r], rmax);
            const bool live = (mnew > -INFINITY);
            // exp(m - m') is 0 when m was -inf and m' is finite, which is
            // exactly right: there was nothing to carry forward.
            const float resc = live ? __expf(mrow[r] - mnew) : 1.0f;

            float rsum = 0.0f;
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                const float p = live ? __expf(s[r][c] - mnew) : 0.0f;
                s[r][c] = p;
                rsum += p;
            }
            rsum = row_reduce<CG, false>(rsum);

            mrow[r] = mnew;
            lrow[r] = lrow[r] * resc + rsum;
#pragma unroll
            for (int j = 0; j < OT; ++j) acc[r][j] *= resc;
        }

        __syncthreads();  // everyone is done reading Kst; P may overwrite it

        // P goes to shared memory transposed, Pst[c][r], so the second matmul
        // reads it the same contiguous way the first read Q.
#pragma unroll
        for (int c = 0; c < CT; ++c) {
#pragma unroll
            for (int r = 0; r < RT; r += 4) {
                float4 p;
                p.x = s[r + 0][c]; p.y = s[r + 1][c];
                p.z = s[r + 2][c]; p.w = s[r + 3][c];
                VEC4(KPst[(c0 + c) * QW + r0 + r]) = p;
            }
        }
        __syncthreads();

        // --- O += P @ V ---
        for (int c = 0; c < BC; ++c) {
            float rp[RT], rv[OT];
#pragma unroll
            for (int r = 0; r < RT; r += 4) VEC4(rp[r]) = CVEC4(KPst[c * QW + r0 + r]);
#pragma unroll
            for (int j = 0; j < OT; j += 4) VEC4(rv[j]) = CVEC4(Vs[c * HS + o0 + j]);
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int j = 0; j < OT; ++j) acc[r][j] += rp[r] * rv[j];
        }
    }

    // --- normalize and write out ---
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int qg = i0 + r0 + r;
        if (qg >= T) continue;
        const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
        float *dst = out + ((size_t)b * T + qg) * C + h * HS + o0;
#pragma unroll
        for (int j = 0; j < OT; j += 4) {
            float4 o;
            o.x = acc[r][j + 0] * inv; o.y = acc[r][j + 1] * inv;
            o.z = acc[r][j + 2] * inv; o.w = acc[r][j + 3] * inv;
            VEC4(dst[j]) = o;
        }
        // One scalar per row is the entire saved state for backward: with lse
        // in hand the backward pass reconstructs P exactly, as exp(S - lse),
        // with no second max pass and no stored score matrix.
        if (tCol == 0) lse[(size_t)bh * T + qg] = mrow[r] + __logf(lrow[r]);
    }
}

// ---- launcher ------------------------------------------------------------
//
// Shared memory exceeds the 48 KB static limit for the larger tiles, so the
// kernel takes it dynamically and opts in per launch. Ada allows 99 KB.
template <int BR, int BC, int HS, int NT, int RT, int CT>
bool launch_fwd(float *out, float *lse, const float *qkv, int B, int T, int NH,
                float scale) {
    constexpr int PAD = 4, QW = BR + PAD, KW = BC + PAD;
    constexpr int KSZ = (HS * KW > BC * QW) ? HS * KW : BC * QW;
    constexpr size_t smem = (size_t)(HS * QW + KSZ + BC * HS) * sizeof(float);

    auto kern = flash_fwd_k<BR, BC, HS, NT, RT, CT>;
    // The opt-in is PER DEVICE: each GPU holds its own copy of the kernel and
    // its own attribute table, so a process-wide "done" flag configures only
    // the device whichever rank reached it first, and the other rank's
    // launches then exceed the default 48 KB and fail -- a garbage forward on
    // rank 1 before a single byte of communication. thread_local matches the
    // one-host-thread-per-GPU rule the rest of the repo runs on (every guard
    // in this file is the same).
    static thread_local bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem) != cudaSuccess)
            return false;
        configured = true;
    }
    dim3 grid(ceil_div(T, BR), B * NH);
    kern<<<grid, NT, smem>>>(out, lse, qkv, T, NH, scale);
    return true;
}

// (id, BR, BC, NT, RT, CT, name)
// The opt-in maximum dynamic shared memory per block: 99 KiB on Ada, and the
// same on the Blackwell laptop part this now runs on. Used by both the fused
// forward and the fused backward, so it lives above the first of them.
constexpr size_t SMEM_CAP = 99 * 1024;

#if BMB_TF32
// ------------------------------------------- forward on the tensor cores
//
// The same algorithm as flash_fwd_k above, with both matmuls issued as
// mma.sync TF32 -- and, more interestingly, with P never reaching shared
// memory at all.
//
// The three operands, all laid out by kernels/lane_major.cuh, the same file
// kernels 10 and 11 use:
//
//     Q   A operand   a_off<BR>(m = query,     k = head dim)  |  S = Q @ K^T
//     K   B operand   b_off<BC>(k = head dim,  n = key)       |
//     V   B operand   b_off<HS>(k = key,       n = head dim)  |  O = P @ V
//
// Both B operands fall out of the natural layouts: `.row.col` wants its B
// operand as N x K, and K is stored (key, head dim) while V is stored
// (key, head dim) -- which is N x K for the first and K x N for the second,
// exactly as needed.
//
// THE PART THAT IS NOT FREE. P is the accumulator of the first mma and the A
// operand of the second, and for TF32 those two layouts DISAGREE: the
// accumulator holds columns {2t, 2t+1} of a row group where the A operand wants
// {t, t+4}. The f16 shapes happen to agree, which is why FlashAttention-2 gets
// this for nothing and this does not. tools/probe_mma_acc.cu measured both
// layouts rather than trusting either, and found the disagreement is confined
// to the four lanes that share a row group -- so eight shuffles convert one to
// the other exactly, and the shared-memory round trip that the fp32 kernel
// needs for P disappears.

// Accumulator tile (16 x 8) -> A fragment of the same 16 x 8 matrix.
//
// The selection MUST happen after the shuffle. Writing the obvious
// `__shfl_sync(M, par ? d[1] : d[0], src)` reads whichever register the SOURCE
// lane's own `par` chose, which silently lands on the neighbouring column.
__device__ __forceinline__ void acc_to_a(const float (&d)[4], unsigned (&a)[4],
                                         int lane) {
    constexpr unsigned M = 0xffffffffu;
    const int g = lane >> 2, t = lane & 3;
    const int sA = (g << 2) | (t >> 1);        // holds column t
    const int sB = (g << 2) | ((t + 4) >> 1);  // holds column t+4
    const int par = t & 1;
    const float a0 = __shfl_sync(M, d[0], sA), a1 = __shfl_sync(M, d[1], sA);
    const float a2 = __shfl_sync(M, d[2], sA), a3 = __shfl_sync(M, d[3], sA);
    const float b0 = __shfl_sync(M, d[0], sB), b1 = __shfl_sync(M, d[1], sB);
    const float b2 = __shfl_sync(M, d[2], sB), b3 = __shfl_sync(M, d[3], sB);
    a[0] = to_tf32(par ? a1 : a0);  // (g,   t)
    a[1] = to_tf32(par ? a3 : a2);  // (g+8, t)
    a[2] = to_tf32(par ? b1 : b0);  // (g,   t+4)
    a[3] = to_tf32(par ? b3 : b2);  // (g+8, t+4)
}

template <int BR, int BC, int HS, int NT>
__global__ __launch_bounds__(NT) void flash_fwd_mma_k(
    float *__restrict__ out, float *__restrict__ lse,
    const float *__restrict__ qkv, int T, int NH, float scale) {
    constexpr int NW = NT / WARPSIZE;  // warps, one 16-row query tile each
    constexpr int NC = BC / MMA_N;     // n-tiles of S, and k-chunks of P
    constexpr int NO = HS / MMA_N;     // n-tiles of O
    constexpr int NK = HS / MMA_K;     // k-chunks of S
    constexpr int V4 = HS / 4;

    static_assert(BR == MMA_M * NW, "one 16-row tile per warp, no remainder");
    static_assert(BC % 16 == 0 && HS % 16 == 0, "B operands pack 16 wide");
    static_assert(BC % MMA_K == 0, "P's k chunks divide the key tile");
    static_assert(NT % V4 == 0, "staging tiles evenly");

    extern __shared__ float smem[];
    float *Qs = smem;             // a_off<BR>(query, head dim)
    float *Ks = Qs + BR * HS;     // b_off<BC>(head dim, key)
    float *Vs = Ks + HS * BC;     // b_off<HS>(key, head dim)

    const int tid = threadIdx.x;
    const int lane = tid % WARPSIZE, warp = tid / WARPSIZE;
    // Every accumulator this warp holds puts rows g and g+8 in this lane, and
    // columns 2*tq and 2*tq+1 -- measured in tools/probe_mma_acc.cu.
    const int g = lane >> 2, tq = lane & 3;

    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int C = NH * HS;
    const int i0 = blockIdx.x * BR;
    const float *base = qkv + (size_t)b * T * 3 * C + h * HS;

    // --- stage Q once; every key block reuses it ---
    for (int lin = tid; lin < BR * V4; lin += NT) {
        const int r = lin / V4, i4 = (lin % V4) * 4;
        const int t = i0 + r;
        float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t < T) q = CVEC4(base[(size_t)t * 3 * C + i4]);
        // 1/sqrt(hs) folds in here: BR*HS multiplies once, not BR*T per block.
        Qs[a_off<BR>(r, i4 + 0)] = q.x * scale;
        Qs[a_off<BR>(r, i4 + 1)] = q.y * scale;
        Qs[a_off<BR>(r, i4 + 2)] = q.z * scale;
        Qs[a_off<BR>(r, i4 + 3)] = q.w * scale;
    }

    float acc_o[NO][4] = {};
    float m_[2], l_[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) { m_[u] = -INFINITY; l_[u] = 0.0f; }

    const int jmax = min(T, i0 + BR);  // causality as a loop bound
    for (int j0 = 0; j0 < jmax; j0 += BC) {
        __syncthreads();
        for (int lin = tid; lin < BC * V4; lin += NT) {
            const int c = lin / V4, i4 = (lin % V4) * 4;
            const int t = j0 + c;
            float4 k = make_float4(0.f, 0.f, 0.f, 0.f), v = k;
            if (t < T) {
                k = CVEC4(base[(size_t)t * 3 * C + C + i4]);
                v = CVEC4(base[(size_t)t * 3 * C + 2 * C + i4]);
            }
            Ks[b_off<BC>(i4 + 0, c)] = k.x;
            Ks[b_off<BC>(i4 + 1, c)] = k.y;
            Ks[b_off<BC>(i4 + 2, c)] = k.z;
            Ks[b_off<BC>(i4 + 3, c)] = k.w;
            Vs[b_off<HS>(c, i4 + 0)] = v.x;
            Vs[b_off<HS>(c, i4 + 1)] = v.y;
            Vs[b_off<HS>(c, i4 + 2)] = v.z;
            Vs[b_off<HS>(c, i4 + 3)] = v.w;
        }
        __syncthreads();

        // --- S = (Q/sqrt(hs)) @ K^T, straight into accumulators ---
        float acc_s[NC][4] = {};
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            const int au = kk * (BR / MMA_M) + warp;
            const float4 av = CVEC4(Qs[au * UNIT + (lane ^ a_swz<BR>(au)) * 4]);
            const unsigned a[4] = {to_tf32(av.x), to_tf32(av.y), to_tf32(av.z),
                                   to_tf32(av.w)};
#pragma unroll
            for (int jj = 0; jj < NC; jj += 2) {
                const int bu = kk * (BC / 16) + jj / 2;
                const float4 bv = CVEC4(Ks[bu * UNIT + (lane ^ b_swz<BC>(bu)) * 4]);
                const unsigned bb[4] = {to_tf32(bv.x), to_tf32(bv.y),
                                        to_tf32(bv.z), to_tf32(bv.w)};
                mma_m16n8k8(acc_s[jj + 0], a, &bb[0]);
                mma_m16n8k8(acc_s[jj + 1], a, &bb[2]);
            }
        }

        // --- mask, then one step of the online softmax, per row half ---
        //
        // A row lives in the four lanes sharing g, so the row reductions are a
        // width-4 butterfly -- narrower than the fp32 kernel's, because the
        // accumulator layout already put the row's columns close together.
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int qg = i0 + warp * MMA_M + g + 8 * u;
            float rmax = -INFINITY;
#pragma unroll
            for (int jj = 0; jj < NC; ++jj)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int kg = j0 + jj * MMA_N + 2 * tq + e;
                    float &v = acc_s[jj][2 * u + e];
                    if (kg > qg || kg >= T || qg >= T) v = -INFINITY;
                    rmax = fmaxf(rmax, v);
                }
            rmax = row_reduce<4, true>(rmax);

            const float mnew = fmaxf(m_[u], rmax);
            const bool live = (mnew > -INFINITY);
            const float resc = live ? __expf(m_[u] - mnew) : 1.0f;

            float rsum = 0.0f;
#pragma unroll
            for (int jj = 0; jj < NC; ++jj)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float &v = acc_s[jj][2 * u + e];
                    const float p = live ? __expf(v - mnew) : 0.0f;
                    v = p;
                    rsum += p;
                }
            rsum = row_reduce<4, false>(rsum);

            m_[u] = mnew;
            l_[u] = l_[u] * resc + rsum;
#pragma unroll
            for (int nn = 0; nn < NO; ++nn) {
                acc_o[nn][2 * u + 0] *= resc;
                acc_o[nn][2 * u + 1] *= resc;
            }
        }

        // --- O += P @ V, with P handed over in registers ---
#pragma unroll
        for (int jj = 0; jj < NC; ++jj) {
            unsigned a[4];
            acc_to_a(acc_s[jj], a, lane);
#pragma unroll
            for (int nn = 0; nn < NO; nn += 2) {
                const int bu = jj * (HS / 16) + nn / 2;
                const float4 bv = CVEC4(Vs[bu * UNIT + (lane ^ b_swz<HS>(bu)) * 4]);
                const unsigned bb[4] = {to_tf32(bv.x), to_tf32(bv.y),
                                        to_tf32(bv.z), to_tf32(bv.w)};
                mma_m16n8k8(acc_o[nn + 0], a, &bb[0]);
                mma_m16n8k8(acc_o[nn + 1], a, &bb[2]);
            }
        }
    }

    // --- normalize and write out ---
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int qg = i0 + warp * MMA_M + g + 8 * u;
        if (qg >= T) continue;
        const float inv = (l_[u] > 0.0f) ? 1.0f / l_[u] : 0.0f;
        float *dst = out + ((size_t)b * T + qg) * C + h * HS;
#pragma unroll
        for (int nn = 0; nn < NO; ++nn) {
            float2 o;
            o.x = acc_o[nn][2 * u + 0] * inv;
            o.y = acc_o[nn][2 * u + 1] * inv;
            reinterpret_cast<float2 *>(&dst[nn * MMA_N + 2 * tq])[0] = o;
        }
        if (tq == 0) lse[(size_t)bh * T + qg] = m_[u] + __logf(l_[u]);
    }
}

template <int BR, int BC, int HS, int NT>
bool launch_fwd_mma(float *out, float *lse, const float *qkv, int B, int T,
                    int NH, float scale) {
    constexpr size_t smem = (size_t)(BR * HS + HS * BC + BC * HS) * sizeof(float);
    if (smem > SMEM_CAP) return false;
    auto kern = flash_fwd_mma_k<BR, BC, HS, NT>;
    static thread_local bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem) != cudaSuccess)
            return false;
        configured = true;
    }
    dim3 grid(ceil_div(T, BR), B * NH);
    kern<<<grid, NT, smem>>>(out, lse, qkv, T, NH, scale);
    return true;
}
#endif  // BMB_TF32

// MMA=1 selects the tensor-core kernel, for which RT and CT are meaningless
// (its thread tile is the mma fragment layout, not a choice) and are passed as
// zero. Everything else is the fp32 kernel, unchanged.
//
// (id, BR, BC, NT, RT, CT, MMA, name)
#define FWD_CONFIGS(X)                                                         \
    X(0, 64, 64, 128, 4, 8, 0, "br64 bc64 t128 4x8")                           \
    X(1, 64, 32, 128, 4, 4, 0, "br64 bc32 t128 4x4")                           \
    X(2, 64, 64, 256, 4, 4, 0, "br64 bc64 t256 4x4")                           \
    X(3, 128, 32, 256, 4, 4, 0, "br128 bc32 t256 4x4")                         \
    X(4, 32, 32, 64, 4, 4, 0, "br32 bc32 t64 4x4")                             \
    X(5, 64, 32, 128, 0, 0, 1, "mma br64 bc32 t128")                           \
    X(6, 64, 64, 128, 0, 0, 1, "mma br64 bc64 t128")                           \
    X(7, 128, 32, 256, 0, 0, 1, "mma br128 bc32 t256")                         \
    X(8, 32, 32, 64, 0, 0, 1, "mma br32 bc32 t64")

// OT = HS/(BC/CT) must be a positive multiple of 4, which rules some configs
// out for small head sizes. Checked here so an unusable pair fails cleanly at
// runtime instead of failing to compile the whole file.
template <int BR, int BC, int HS, int NT, int RT, int CT>
constexpr bool cfg_ok() {
    return HS % (BC / CT) == 0 && (HS / (BC / CT)) % 4 == 0 &&
           NT % (HS / 4) == 0 && (BR / RT) * (BC / CT) == NT;
}

// The tensor-core kernel's own shape constraints: one 16-row query tile per
// warp, and B operands that pack 16 wide.
template <int BR, int BC, int HS, int NT>
constexpr bool mma_cfg_ok() {
    return BR == 16 * (NT / 32) && BC % 16 == 0 && HS % 16 == 0 &&
           BC % 8 == 0 && HS % 8 == 0 && NT % (HS / 4) == 0;
}

// Instantiate one config for one head size, or refuse it.
template <int BR, int BC, int HS, int NT, int RT, int CT, int MMA>
struct FwdInst {
    static bool run(float *out, float *lse, const float *qkv, int B, int T,
                    int NH, float scale) {
        if constexpr (MMA) {
#if BMB_TF32
            if constexpr (mma_cfg_ok<BR, BC, HS, NT>())
                return launch_fwd_mma<BR, BC, HS, NT>(out, lse, qkv, B, T, NH, scale);
            else
                return false;
#else
            return false;  // no tensor cores below sm_80
#endif
        } else if constexpr (cfg_ok<BR, BC, HS, NT, RT, CT>())
            return launch_fwd<BR, BC, HS, NT, RT, CT>(out, lse, qkv, B, T, NH, scale);
        else
            return false;
    }
};

template <int HS>
bool dispatch_cfg(int cfg, float *out, float *lse, const float *qkv, int B,
                  int T, int NH, float scale) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, MMA, name)                                   \
    case id:                                                                   \
        return FwdInst<BR, BC, HS, NT, RT, CT, MMA>::run(out, lse, qkv, B, T,   \
                                                         NH, scale);
        FWD_CONFIGS(X)
#undef X
    default: return false;
    }
}
}  // namespace

int flash_num_configs() {
    int n = 0;
#define X(id, BR, BC, NT, RT, CT, MMA, name) ++n;
    FWD_CONFIGS(X)
#undef X
    return n;
}

// -1 means "use the rule"; train_gpt --fwd-cfg N pins a config.
static int g_fwd_override = -1;

// TF32 keeps 10 mantissa bits against fp32's 23, so the tensor-core configs are
// computing a deliberately lower-precision answer rather than computing the
// same answer badly. They get their own tolerance, per the rule the SGEMM
// ladder already follows -- the alternative is loosening the bar for the fp32
// kernels, which have no excuse. Measured error is 3.8e-04; a real indexing or
// masking bug in this kernel lands at 1e-1 or worse, so the gap is wide.
double flash_config_tol(int cfg) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, MMA, name)                                   \
    case id: return MMA ? 1e-3 : 1e-5;
        FWD_CONFIGS(X)
#undef X
    default: return 1e-5;
    }
}

const char *flash_config_name(int cfg) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, MMA, name)                                   \
    case id: return name;
        FWD_CONFIGS(X)
#undef X
    default: return "?";
    }
}

// Measured, not reasoned: br64/bc32 wins at hs=64 despite br64/bc64 having the
// better arithmetic intensity on paper, because the smaller key tile fits two
// blocks per SM instead of one.
//
// The tensor-core config wins by a lot more than that -- 0.157 ms against
// 0.284 at ctx 256, 1.81x -- but it computes in TF32, so it is gated on the
// same opt-in the matmuls use rather than silently changing what the model
// computes. `--tf32` already routes every GEMM through the tensor cores; this
// makes attention follow, which it previously did not, and which is why
// attention was the last consumer in the repo still on fp32 FMAs.
int flash_default_config() {
    if (g_fwd_override >= 0) return g_fwd_override;
    return gemm_tf32() ? 5 : 1;
}

// Same escape hatch as the backward's: one binary, both arms, interleavable.
void flash_set_fwd_config(int cfg) { g_fwd_override = cfg; }

bool flash_attention_forward_cfg(int cfg, float *out, float *lse,
                                 const float *qkv, int B, int T, int C, int NH) {
    const int hs = C / NH;
    const float scale = 1.0f / sqrtf((float)hs);
    switch (hs) {
    case 32:  return dispatch_cfg<32>(cfg, out, lse, qkv, B, T, NH, scale);
    case 64:  return dispatch_cfg<64>(cfg, out, lse, qkv, B, T, NH, scale);
    case 128: return dispatch_cfg<128>(cfg, out, lse, qkv, B, T, NH, scale);
    default:  return false;
    }
}

// The tile shapes that are fastest for one head size are often not even legal
// for another -- the accumulator constraints tie the thread count to HS. So the
// default is a preference, not a requirement: try it, then fall back through
// the rest of the table until one fits.
void flash_attention_forward(float *out, float *lse, const float *qkv, int B,
                             int T, int C, int NH) {
    // Preferred first, then the best fp32 config, then anything that runs. The
    // scan below starts at config 0, which is not the fp32 tile the sweep
    // picked -- so config 1 is named explicitly rather than being fallen into.
    const int preferred[] = {flash_default_config(), 1};
    for (int c : preferred)
        if (flash_attention_forward_cfg(c, out, lse, qkv, B, T, C, NH)) return;
    for (int cfg = 0; cfg < flash_num_configs(); ++cfg)
        if (flash_attention_forward_cfg(cfg, out, lse, qkv, B, T, C, NH)) return;
    {
        // A head size the tiles cannot cover is a build-time mistake, not a
        // runtime condition to paper over.
        fprintf(stderr, "flash: no tile config for head size %d\n", C / NH);
        exit(1);
    }
}

// ---------------------------------------------------------------- backward
//
// Backward has the same shape of problem as forward plus one extra wrinkle: it
// needs P, which was deliberately never stored. FlashAttention's answer is to
// recompute it. That sounds expensive and is not, because the saved lse makes
// the recomputation exact and free of reductions:
//
//     P[i,j] = exp(S[i,j] - lse[i])
//
// -- no row max pass, no row sum pass, just an exp. Recomputing S costs one
// matmul; reading a stored score matrix back would cost 25 MB of DRAM traffic
// per layer. On a card with a 43 FLOP/byte ridge point that trade is not close.
//
// The gradients themselves:
//
//     dV[j]   = sum_i P[i,j] dO[i]
//     dP[i,j] = dO[i] . v_j
//     dS[i,j] = P[i,j] * (dP[i,j] - D[i]),   D[i] = sum_j dO[i,j] O[i,j]
//     dQ[i]   = scale * sum_j dS[i,j] k_j
//     dK[j]   = scale * sum_i dS[i,j] q_i
//
// D is the softmax Jacobian's rank-one correction term -- the same sum(dy*y)
// that softmax_causal_bwd_k computes, except that here y is the attention
// *output* rather than the probabilities. That identity is what lets the
// correction be computed without P in memory.
//
// TWO KERNELS, NOT ONE. dQ reduces over keys while dK and dV reduce over
// queries. A single kernel would have to accumulate one of them across blocks,
// i.e. through global atomics, on a tensor the size of the activations. Two
// kernels each recompute S -- one extra matmul over the causal half -- and in
// exchange every accumulator stays in registers and every write is a plain
// store. Recompute beats communication; that is this repo's whole lesson
// restated one level up.

namespace {

// D[i] = sum_j dO[i,j] * O[i,j], one warp per (b, h, t) row.
__global__ void flash_dsum_k(float *__restrict__ dsum,
                             const float *__restrict__ dout,
                             const float *__restrict__ out, int T, int NH,
                             int HS, int rows) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;

    const int t = row % T, h = (row / T) % NH, b = row / (T * NH);
    const int C = NH * HS;
    const size_t off = (size_t)(b * T + t) * C + (size_t)h * HS;

    float s = 0.0f;
    for (int i = lane; i < HS; i += 32) s += dout[off + i] * out[off + i];
    s = row_reduce<32, false>(s);
    if (lane == 0) dsum[row] = s;
}

// dK and dV: one block owns BC keys and streams every query block that can see
// them. Both accumulators live in registers for the whole kernel and are
// written exactly once.
template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int KVC>
__global__ __launch_bounds__(NT) void flash_bwd_kv_k(
    float *__restrict__ dqkv, const float *__restrict__ qkv,
    const float *__restrict__ dout, const float *__restrict__ lse,
    const float *__restrict__ dsum, int T, int NH, float scale) {
    constexpr int PAD = 4;
    constexpr int RG = BR / RT, CG = BC / CT;
    constexpr int AG = BC / AT, BG = HS / BT;
    constexpr int V4 = HS / 4;
    constexpr int KC4 = KVC / 4;  // float4s per key when staging one chunk
    constexpr int QW = BR + PAD;  // Qst/dOst row width (head-size major)
    constexpr int KW = BC + PAD;  // Kst/Vst row width (head-size major)
    constexpr int PW = BC + PAD;  // Pst/dSst row width (query major)

    static_assert(RG * CG == NT, "score tile must cover the thread block");
    static_assert(AG * BG == NT, "accumulator tile must cover it too");
    static_assert(RT % 4 == 0, "vector loads");
    static_assert(AT % 4 == 0 && BT % 4 == 0, "vector loads");
    static_assert(NT % V4 == 0, "staging must tile evenly");
    static_assert(KVC % 4 == 0 && HS % KVC == 0, "head chunk divides the head");
    static_assert(NT % KC4 == 0, "chunk staging must tile evenly");

    extern __shared__ float smem[];
    float *Qst = smem;              // [HS][QW], carries the 1/sqrt(hs)
    float *dOst = Qst + HS * QW;    // [HS][QW]
    float *Kst = dOst + HS * QW;    // [KVC][KW]
    float *Vst = Kst + KVC * KW;    // [KVC][KW]
    float *Pst = Vst + KVC * KW;    // [BR][PW]
    float *dSst = Pst + BR * PW;    // [BR][PW]
    float *lses = dSst + BR * PW;   // [BR]
    float *dss = lses + BR;         // [BR]

    const int tid = threadIdx.x;
    const int tRow = tid / CG, tCol = tid % CG;
    const int r0 = tRow * RT, c0 = tCol * CT;
    const int a0 = (tid / BG) * AT, b0 = (tid % BG) * BT;

    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int C = NH * HS;
    const int j0 = blockIdx.x * BC;

    const float *base = qkv + (size_t)b * T * 3 * C + h * HS;
    const float *dob = dout + (size_t)b * T * C + h * HS;
    const float *lserow = lse + (size_t)bh * T;
    const float *dsrow = dsum + (size_t)bh * T;

    // Stage head dimensions [hc, hc+KVC) of K and V for this block's BC keys,
    // transposed. With KVC == HS this is the whole head and runs once; with
    // KVC < HS it runs once per chunk per query block, which is the trade the
    // config table measures.
    auto stage_kv = [&](int hc) {
        const int i4 = (tid % KC4) * 4;
        for (int c = tid / KC4; c < BC; c += NT / KC4) {
            const int t = j0 + c;
            float4 k = make_float4(0.f, 0.f, 0.f, 0.f), v = k;
            if (t < T) {
                k = CVEC4(base[(size_t)t * 3 * C + C + hc + i4]);
                v = CVEC4(base[(size_t)t * 3 * C + 2 * C + hc + i4]);
            }
            Kst[(i4 + 0) * KW + c] = k.x; Kst[(i4 + 1) * KW + c] = k.y;
            Kst[(i4 + 2) * KW + c] = k.z; Kst[(i4 + 3) * KW + c] = k.w;
            Vst[(i4 + 0) * KW + c] = v.x; Vst[(i4 + 1) * KW + c] = v.y;
            Vst[(i4 + 2) * KW + c] = v.z; Vst[(i4 + 3) * KW + c] = v.w;
        }
    };

    // K and V are fixed for this block, so unchunked they are staged once and
    // every query block reuses them. This is the path every pre-chunking config
    // takes, and it is unchanged.
    if constexpr (KVC == HS) stage_kv(0);

    float dK[AT][BT], dV[AT][BT];
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int j = 0; j < BT; ++j) { dK[a][j] = 0.0f; dV[a][j] = 0.0f; }

    // Only queries at or after this key block can attend it.
    for (int i0 = (j0 / BR) * BR; i0 < T; i0 += BR) {
        __syncthreads();
        {
            const int i4 = (tid % V4) * 4;
            for (int r = tid / V4; r < BR; r += NT / V4) {
                const int t = i0 + r;
                float4 q = make_float4(0.f, 0.f, 0.f, 0.f), o = q;
                if (t < T) {
                    q = CVEC4(base[(size_t)t * 3 * C + i4]);
                    o = CVEC4(dob[(size_t)t * C + i4]);
                }
                Qst[(i4 + 0) * QW + r] = q.x * scale;
                Qst[(i4 + 1) * QW + r] = q.y * scale;
                Qst[(i4 + 2) * QW + r] = q.z * scale;
                Qst[(i4 + 3) * QW + r] = q.w * scale;
                dOst[(i4 + 0) * QW + r] = o.x; dOst[(i4 + 1) * QW + r] = o.y;
                dOst[(i4 + 2) * QW + r] = o.z; dOst[(i4 + 3) * QW + r] = o.w;
            }
            for (int r = tid; r < BR; r += NT) {
                const int t = i0 + r;
                lses[r] = (t < T) ? lserow[t] : 0.0f;
                dss[r] = (t < T) ? dsrow[t] : 0.0f;
            }
        }
        // Chunk 0 rides along with the Q/dO staging so it costs no extra
        // barrier; only chunks 1.. need a pair of their own.
        if constexpr (KVC < HS) stage_kv(0);
        __syncthreads();

        // S = Q K^T and dP = dO V^T share one loop: same shape, same tiles, and
        // interleaving them doubles the independent FMAs in flight.
        float s[RT][CT], dp[RT][CT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < CT; ++c) { s[r][c] = 0.0f; dp[r][c] = 0.0f; }

        for (int hc = 0; hc < HS; hc += KVC) {
            if constexpr (KVC < HS) {
                if (hc) {
                    __syncthreads();  // every warp is done with the last chunk
                    stage_kv(hc);
                    __syncthreads();
                }
            }
            // Deliberately NOT `#pragma unroll`. The pre-chunking loop over the
            // whole head had no pragma, and adding one here changes the codegen
            // of every unchunked config -- measured, config 0 lost 13.8% and
            // config 7 gained 10% from that alone. The chunked path is supposed
            // to be the only variable in this comparison.
            for (int ii = 0; ii < KVC; ++ii) {
                const int i = hc + ii;
                float rq[RT], rk[CT], rdo[RT], rv[CT];
#pragma unroll
                for (int r = 0; r < RT; r += 4) {
                    VEC4(rq[r]) = CVEC4(Qst[i * QW + r0 + r]);
                    VEC4(rdo[r]) = CVEC4(dOst[i * QW + r0 + r]);
                }
                load_n<CT>(rk, &Kst[ii * KW + c0]);
                load_n<CT>(rv, &Vst[ii * KW + c0]);
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int c = 0; c < CT; ++c) {
                        s[r][c] += rq[r] * rk[c];
                        dp[r][c] += rdo[r] * rv[c];
                    }
            }
        }

        // P from lse -- exact, and with no reduction. Masked entries become 0
        // rather than -inf: these are probabilities now, not scores, and 0 is
        // what makes their gradient vanish.
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            const int qg = i0 + r0 + r;
            const float li = lses[r0 + r], di = dss[r0 + r];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                const int kg = j0 + c0 + c;
                const bool live = (qg < T) && (kg < T) && (kg <= qg);
                const float p = live ? __expf(s[r][c] - li) : 0.0f;
                s[r][c] = p;
                dp[r][c] = p * (dp[r][c] - di);
            }
        }
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            store_n<CT>(&Pst[(r0 + r) * PW + c0], s[r]);
            store_n<CT>(&dSst[(r0 + r) * PW + c0], dp[r]);
        }
        __syncthreads();

        // dV[c][j] += sum_i P[i][c] dO[i][j]
        // dK[c][j] += sum_i dS[i][c] Q[i][j]
        //
        // Both reduce over queries, so one pass over i feeds both. dO and Q are
        // stored head-size major, so reading [i][j] walks a column -- the PAD on
        // QW is what keeps those BT reads in distinct banks.
        for (int i = 0; i < BR; ++i) {
            float rp[AT], rds[AT], rdo[BT], rq[BT];
#pragma unroll
            for (int a = 0; a < AT; a += 4) {
                VEC4(rp[a]) = CVEC4(Pst[i * PW + a0 + a]);
                VEC4(rds[a]) = CVEC4(dSst[i * PW + a0 + a]);
            }
#pragma unroll
            for (int j = 0; j < BT; ++j) {
                rdo[j] = dOst[(b0 + j) * QW + i];
                rq[j] = Qst[(b0 + j) * QW + i];
            }
#pragma unroll
            for (int a = 0; a < AT; ++a)
#pragma unroll
                for (int j = 0; j < BT; ++j) {
                    dV[a][j] += rp[a] * rdo[j];
                    dK[a][j] += rds[a] * rq[j];
                }
        }
    }

    // Qst already carried the scale, so dK needs no further multiply.
#pragma unroll
    for (int a = 0; a < AT; ++a) {
        const int kg = j0 + a0 + a;
        if (kg >= T) continue;
        float *dk = dqkv + ((size_t)b * T + kg) * 3 * C + C + h * HS + b0;
        float *dv = dk + C;
#pragma unroll
        for (int j = 0; j < BT; j += 4) {
            VEC4(dk[j]) = VEC4(dK[a][j]);
            VEC4(dv[j]) = VEC4(dV[a][j]);
        }
    }
}

// dQ: one block owns BR queries and streams the key blocks below the diagonal.
template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT>
__global__ __launch_bounds__(NT) void flash_bwd_q_k(
    float *__restrict__ dqkv, const float *__restrict__ qkv,
    const float *__restrict__ dout, const float *__restrict__ lse,
    const float *__restrict__ dsum, int T, int NH, float scale) {
    constexpr int PAD = 4;
    constexpr int RG = BR / RT, CG = BC / CT;
    constexpr int AG = BR / AT, BG = HS / BT;
    constexpr int V4 = HS / 4;
    constexpr int QW = BR + PAD;
    constexpr int KW = BC + PAD;

    static_assert(RG * CG == NT, "score tile must cover the thread block");
    static_assert(AG * BG == NT, "accumulator tile must cover it too");
    static_assert(RT % 4 == 0, "vector loads");
    static_assert(AT % 4 == 0 && BT % 4 == 0, "vector loads");
    static_assert(NT % V4 == 0, "staging must tile evenly");

    extern __shared__ float smem[];
    float *Qst = smem;              // [HS][QW], carries the 1/sqrt(hs)
    float *dOst = Qst + HS * QW;    // [HS][QW]
    float *Kst = dOst + HS * QW;    // [HS][KW]
    float *Vst = Kst + HS * KW;     // [HS][KW]
    float *dSst = Vst + HS * KW;    // [BC][QW], transposed: dSst[c][i]
    float *lses = dSst + BC * QW;   // [BR]
    float *dss = lses + BR;         // [BR]

    const int tid = threadIdx.x;
    const int tRow = tid / CG, tCol = tid % CG;
    const int r0 = tRow * RT, c0 = tCol * CT;
    const int a0 = (tid / BG) * AT, b0 = (tid % BG) * BT;

    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int C = NH * HS;
    const int i0 = blockIdx.x * BR;

    const float *base = qkv + (size_t)b * T * 3 * C + h * HS;
    const float *dob = dout + (size_t)b * T * C + h * HS;
    const float *lserow = lse + (size_t)bh * T;
    const float *dsrow = dsum + (size_t)bh * T;

    // Q and dO are fixed for this block; the key blocks stream past them.
    {
        const int i4 = (tid % V4) * 4;
        for (int r = tid / V4; r < BR; r += NT / V4) {
            const int t = i0 + r;
            float4 q = make_float4(0.f, 0.f, 0.f, 0.f), o = q;
            if (t < T) {
                q = CVEC4(base[(size_t)t * 3 * C + i4]);
                o = CVEC4(dob[(size_t)t * C + i4]);
            }
            Qst[(i4 + 0) * QW + r] = q.x * scale;
            Qst[(i4 + 1) * QW + r] = q.y * scale;
            Qst[(i4 + 2) * QW + r] = q.z * scale;
            Qst[(i4 + 3) * QW + r] = q.w * scale;
            dOst[(i4 + 0) * QW + r] = o.x; dOst[(i4 + 1) * QW + r] = o.y;
            dOst[(i4 + 2) * QW + r] = o.z; dOst[(i4 + 3) * QW + r] = o.w;
        }
        for (int r = tid; r < BR; r += NT) {
            const int t = i0 + r;
            lses[r] = (t < T) ? lserow[t] : 0.0f;
            dss[r] = (t < T) ? dsrow[t] : 0.0f;
        }
    }

    float dQ[AT][BT];
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int j = 0; j < BT; ++j) dQ[a][j] = 0.0f;

    const int jmax = min(T, i0 + BR);
    for (int j0 = 0; j0 < jmax; j0 += BC) {
        __syncthreads();
        {
            const int i4 = (tid % V4) * 4;
            for (int c = tid / V4; c < BC; c += NT / V4) {
                const int t = j0 + c;
                float4 k = make_float4(0.f, 0.f, 0.f, 0.f), v = k;
                if (t < T) {
                    k = CVEC4(base[(size_t)t * 3 * C + C + i4]);
                    v = CVEC4(base[(size_t)t * 3 * C + 2 * C + i4]);
                }
                Kst[(i4 + 0) * KW + c] = k.x; Kst[(i4 + 1) * KW + c] = k.y;
                Kst[(i4 + 2) * KW + c] = k.z; Kst[(i4 + 3) * KW + c] = k.w;
                Vst[(i4 + 0) * KW + c] = v.x; Vst[(i4 + 1) * KW + c] = v.y;
                Vst[(i4 + 2) * KW + c] = v.z; Vst[(i4 + 3) * KW + c] = v.w;
            }
        }
        __syncthreads();

        float s[RT][CT], dp[RT][CT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < CT; ++c) { s[r][c] = 0.0f; dp[r][c] = 0.0f; }

        for (int i = 0; i < HS; ++i) {
            float rq[RT], rk[CT], rdo[RT], rv[CT];
#pragma unroll
            for (int r = 0; r < RT; r += 4) {
                VEC4(rq[r]) = CVEC4(Qst[i * QW + r0 + r]);
                VEC4(rdo[r]) = CVEC4(dOst[i * QW + r0 + r]);
            }
            load_n<CT>(rk, &Kst[i * KW + c0]);
            load_n<CT>(rv, &Vst[i * KW + c0]);
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    s[r][c] += rq[r] * rk[c];
                    dp[r][c] += rdo[r] * rv[c];
                }
        }

#pragma unroll
        for (int r = 0; r < RT; ++r) {
            const int qg = i0 + r0 + r;
            const float li = lses[r0 + r], di = dss[r0 + r];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                const int kg = j0 + c0 + c;
                const bool live = (qg < T) && (kg < T) && (kg <= qg);
                const float p = live ? __expf(s[r][c] - li) : 0.0f;
                s[r][c] = p * (dp[r][c] - di);
            }
        }
        // dS goes out transposed, dSst[c][i], so the accumulation below reads
        // its AT queries contiguously for a fixed key.
#pragma unroll
        for (int c = 0; c < CT; ++c) {
#pragma unroll
            for (int r = 0; r < RT; r += 4) {
                float4 d;
                d.x = s[r + 0][c]; d.y = s[r + 1][c];
                d.z = s[r + 2][c]; d.w = s[r + 3][c];
                VEC4(dSst[(c0 + c) * QW + r0 + r]) = d;
            }
        }
        __syncthreads();

        // dQ[i][j] += sum_c dS[i][c] K[c][j]
        for (int c = 0; c < BC; ++c) {
            float rds[AT], rk[BT];
#pragma unroll
            for (int a = 0; a < AT; a += 4)
                VEC4(rds[a]) = CVEC4(dSst[c * QW + a0 + a]);
#pragma unroll
            for (int j = 0; j < BT; ++j) rk[j] = Kst[(b0 + j) * KW + c];
#pragma unroll
            for (int a = 0; a < AT; ++a)
#pragma unroll
                for (int j = 0; j < BT; ++j) dQ[a][j] += rds[a] * rk[j];
        }
    }

    // K was staged unscaled, so the 1/sqrt(hs) lands here.
#pragma unroll
    for (int a = 0; a < AT; ++a) {
        const int qg = i0 + a0 + a;
        if (qg >= T) continue;
        float *dq = dqkv + ((size_t)b * T + qg) * 3 * C + h * HS + b0;
#pragma unroll
        for (int j = 0; j < BT; j += 4) {
            float4 g;
            g.x = dQ[a][j + 0] * scale; g.y = dQ[a][j + 1] * scale;
            g.z = dQ[a][j + 2] * scale; g.w = dQ[a][j + 3] * scale;
            VEC4(dq[j]) = g;
        }
    }
}


#if BMB_TF32
// ------------------------------------ dK and dV on the tensor cores
//
// This is the kernel the forward's trick does NOT fit, and the way around it is
// the interesting part. Its second matmul reduces over queries:
//
//     dV[c][j] += sum_i P[i][c] dO[i][j]
//
// so the A operand is P TRANSPOSED -- M = key -- while S = Q@K^T accumulates as
// [query][key]. Transposing an accumulator is exactly the shared-memory round
// trip the port is trying to delete.
//
// So this kernel does not compute S. It computes S^T = K@Q^T directly, with
// key as M and query as N, and its accumulator is [key][query] to begin with:
// the orientation the second matmul wants. acc_to_a then applies unchanged.
//
// That costs nothing, and the reason is worth stating: the BACKWARD NEEDS NO
// ROW REDUCTIONS. The forward has to reduce along a query's keys to get the
// running max and sum, which would have been a stride-4 reduction across eight
// lanes in this orientation. Here lse and D are already known, so
//
//     P = exp(S - lse[query])        dS = P * (dP - D[query])
//
// are elementwise, and a per-query scalar is all that is needed -- which in the
// transposed layout means indexing by COLUMN rather than row. Both scalars are
// staged into shared once per query block, so the indexing costs one LDS.
//
// The operands, all through kernels/lane_major.cuh:
//
//     K, V    A operand   a_off<BC>(m = key,      k = head dim)
//     Q, dO   B operand   b_off<BR>(k = head dim, n = query)   |  S^T, dP^T
//     Q, dO   B operand   b_off<HS>(k = query,    n = head dim)|  dK, dV
//
// Q and dO appear in two B layouts each, which is four staging passes off two
// global reads. That is the whole reason this kernel wants BR = BC = 32: at 48
// KB it keeps two blocks per SM, and at BR = 64 it would be 80 KB and one.
template <int BR, int BC, int HS, int NT>
__global__ __launch_bounds__(NT) void flash_bwd_kv_mma_k(
    float *__restrict__ dqkv, const float *__restrict__ qkv,
    const float *__restrict__ dout, const float *__restrict__ lse,
    const float *__restrict__ dsum, int T, int NH, float scale) {
    constexpr int NW = NT / WARPSIZE;
    constexpr int NR = BR / MMA_N;   // n-tiles of S^T, then k-chunks of P^T
    constexpr int NO = HS / MMA_N;   // n-tiles of dK and dV
    constexpr int NK = HS / MMA_K;   // k-chunks of S^T
    constexpr int V4 = HS / 4;

    static_assert(BC == MMA_M * NW, "one 16-key tile per warp");
    static_assert(BR % 16 == 0 && HS % 16 == 0, "B operands pack 16 wide");
    static_assert(BR % MMA_K == 0, "P^T's k chunks divide the query tile");
    static_assert(NT % V4 == 0, "staging tiles evenly");

    extern __shared__ float smem[];
    float *Ks = smem;                  // a_off<BC>(key, head dim)
    float *Vs = Ks + BC * HS;          // a_off<BC>(key, head dim)
    float *Qb = Vs + BC * HS;          // b_off<BR>(head dim, query), scaled
    float *dOb = Qb + HS * BR;         // b_off<BR>(head dim, query)
    float *Qv = dOb + HS * BR;         // b_off<HS>(query, head dim), scaled
    float *dOv = Qv + BR * HS;         // b_off<HS>(query, head dim)
    float *lses = dOv + BR * HS;       // [BR]
    float *dss = lses + BR;            // [BR]

    const int tid = threadIdx.x;
    const int lane = tid % WARPSIZE, warp = tid / WARPSIZE;
    const int g = lane >> 2, tq = lane & 3;

    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int C = NH * HS;
    const int j0 = blockIdx.x * BC;

    const float *base = qkv + (size_t)b * T * 3 * C + h * HS;
    const float *dob = dout + (size_t)b * T * C + h * HS;
    const float *lserow = lse + (size_t)bh * T;
    const float *dsrow = dsum + (size_t)bh * T;

    // K and V are fixed for this block: stage once, in the A layout.
    for (int lin = tid; lin < BC * V4; lin += NT) {
        const int c = lin / V4, i4 = (lin % V4) * 4;
        const int t = j0 + c;
        float4 k = make_float4(0.f, 0.f, 0.f, 0.f), v = k;
        if (t < T) {
            k = CVEC4(base[(size_t)t * 3 * C + C + i4]);
            v = CVEC4(base[(size_t)t * 3 * C + 2 * C + i4]);
        }
        Ks[a_off<BC>(c, i4 + 0)] = k.x;
        Ks[a_off<BC>(c, i4 + 1)] = k.y;
        Ks[a_off<BC>(c, i4 + 2)] = k.z;
        Ks[a_off<BC>(c, i4 + 3)] = k.w;
        Vs[a_off<BC>(c, i4 + 0)] = v.x;
        Vs[a_off<BC>(c, i4 + 1)] = v.y;
        Vs[a_off<BC>(c, i4 + 2)] = v.z;
        Vs[a_off<BC>(c, i4 + 3)] = v.w;
    }

    float acc_dk[NO][4] = {}, acc_dv[NO][4] = {};

    // Only queries at or after this key block can attend it.
    for (int i0 = (j0 / BR) * BR; i0 < T; i0 += BR) {
        __syncthreads();
        for (int lin = tid; lin < BR * V4; lin += NT) {
            const int r = lin / V4, i4 = (lin % V4) * 4;
            const int t = i0 + r;
            float4 q = make_float4(0.f, 0.f, 0.f, 0.f), o = q;
            if (t < T) {
                q = CVEC4(base[(size_t)t * 3 * C + i4]);
                o = CVEC4(dob[(size_t)t * C + i4]);
            }
            // The 1/sqrt(hs) rides on Q in BOTH layouts, so S^T is scaled and
            // dK comes out scaled with it -- no epilogue multiply, same as the
            // fp32 kernel.
            Qb[b_off<BR>(i4 + 0, r)] = q.x * scale;
            Qb[b_off<BR>(i4 + 1, r)] = q.y * scale;
            Qb[b_off<BR>(i4 + 2, r)] = q.z * scale;
            Qb[b_off<BR>(i4 + 3, r)] = q.w * scale;
            dOb[b_off<BR>(i4 + 0, r)] = o.x;
            dOb[b_off<BR>(i4 + 1, r)] = o.y;
            dOb[b_off<BR>(i4 + 2, r)] = o.z;
            dOb[b_off<BR>(i4 + 3, r)] = o.w;
            Qv[b_off<HS>(r, i4 + 0)] = q.x * scale;
            Qv[b_off<HS>(r, i4 + 1)] = q.y * scale;
            Qv[b_off<HS>(r, i4 + 2)] = q.z * scale;
            Qv[b_off<HS>(r, i4 + 3)] = q.w * scale;
            dOv[b_off<HS>(r, i4 + 0)] = o.x;
            dOv[b_off<HS>(r, i4 + 1)] = o.y;
            dOv[b_off<HS>(r, i4 + 2)] = o.z;
            dOv[b_off<HS>(r, i4 + 3)] = o.w;
        }
        for (int r = tid; r < BR; r += NT) {
            const int t = i0 + r;
            lses[r] = (t < T) ? lserow[t] : 0.0f;
            dss[r] = (t < T) ? dsrow[t] : 0.0f;
        }
        __syncthreads();

        // S^T = K @ (Q/sqrt(hs))^T and dP^T = V @ dO^T.
        float acc_s[NR][4] = {}, acc_p[NR][4] = {};
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            const int au = kk * (BC / MMA_M) + warp;
            const float4 kv4 = CVEC4(Ks[au * UNIT + (lane ^ a_swz<BC>(au)) * 4]);
            const float4 vv4 = CVEC4(Vs[au * UNIT + (lane ^ a_swz<BC>(au)) * 4]);
            const unsigned ak[4] = {to_tf32(kv4.x), to_tf32(kv4.y),
                                    to_tf32(kv4.z), to_tf32(kv4.w)};
            const unsigned av[4] = {to_tf32(vv4.x), to_tf32(vv4.y),
                                    to_tf32(vv4.z), to_tf32(vv4.w)};
#pragma unroll
            for (int jj = 0; jj < NR; jj += 2) {
                const int bu = kk * (BR / 16) + jj / 2;
                const float4 qv4 = CVEC4(Qb[bu * UNIT + (lane ^ b_swz<BR>(bu)) * 4]);
                const float4 ov4 = CVEC4(dOb[bu * UNIT + (lane ^ b_swz<BR>(bu)) * 4]);
                const unsigned bq[4] = {to_tf32(qv4.x), to_tf32(qv4.y),
                                        to_tf32(qv4.z), to_tf32(qv4.w)};
                const unsigned bo[4] = {to_tf32(ov4.x), to_tf32(ov4.y),
                                        to_tf32(ov4.z), to_tf32(ov4.w)};
                mma_m16n8k8(acc_s[jj + 0], ak, &bq[0]);
                mma_m16n8k8(acc_s[jj + 1], ak, &bq[2]);
                mma_m16n8k8(acc_p[jj + 0], av, &bo[0]);
                mma_m16n8k8(acc_p[jj + 1], av, &bo[2]);
            }
        }

        // P^T and dS^T, elementwise. The per-query scalars are indexed by the
        // accumulator's COLUMN here, which is what the transposed orientation
        // buys and costs: one LDS each instead of a register already in hand.
        // acc_s becomes P^T, acc_p becomes dS^T.
#pragma unroll
        for (int jj = 0; jj < NR; ++jj)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rq = jj * MMA_N + 2 * tq + (i & 1);  // query in tile
                const int kg = j0 + warp * MMA_M + g + 8 * (i >> 1);
                const int qg = i0 + rq;
                const bool live = (qg < T) && (kg < T) && (kg <= qg);
                const float p = live ? __expf(acc_s[jj][i] - lses[rq]) : 0.0f;
                acc_s[jj][i] = p;
                acc_p[jj][i] = p * (acc_p[jj][i] - dss[rq]);
            }

        // dV += P^T @ dO and dK += dS^T @ Q, both with the A operand handed
        // over in registers.
#pragma unroll
        for (int jj = 0; jj < NR; ++jj) {
            unsigned ap[4], ad[4];
            acc_to_a(acc_s[jj], ap, lane);
            acc_to_a(acc_p[jj], ad, lane);
#pragma unroll
            for (int nn = 0; nn < NO; nn += 2) {
                const int bu = jj * (HS / 16) + nn / 2;
                const float4 ov4 = CVEC4(dOv[bu * UNIT + (lane ^ b_swz<HS>(bu)) * 4]);
                const float4 qv4 = CVEC4(Qv[bu * UNIT + (lane ^ b_swz<HS>(bu)) * 4]);
                const unsigned bo[4] = {to_tf32(ov4.x), to_tf32(ov4.y),
                                        to_tf32(ov4.z), to_tf32(ov4.w)};
                const unsigned bq[4] = {to_tf32(qv4.x), to_tf32(qv4.y),
                                        to_tf32(qv4.z), to_tf32(qv4.w)};
                mma_m16n8k8(acc_dv[nn + 0], ap, &bo[0]);
                mma_m16n8k8(acc_dv[nn + 1], ap, &bo[2]);
                mma_m16n8k8(acc_dk[nn + 0], ad, &bq[0]);
                mma_m16n8k8(acc_dk[nn + 1], ad, &bq[2]);
            }
        }
    }

    // Q carried the scale, so dK needs no further multiply.
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int kg = j0 + warp * MMA_M + g + 8 * u;
        if (kg >= T) continue;
        float *dk = dqkv + ((size_t)b * T + kg) * 3 * C + C + h * HS;
        float *dv = dk + C;
#pragma unroll
        for (int nn = 0; nn < NO; ++nn) {
            float2 a, c;
            a.x = acc_dk[nn][2 * u + 0]; a.y = acc_dk[nn][2 * u + 1];
            c.x = acc_dv[nn][2 * u + 0]; c.y = acc_dv[nn][2 * u + 1];
            reinterpret_cast<float2 *>(&dk[nn * MMA_N + 2 * tq])[0] = a;
            reinterpret_cast<float2 *>(&dv[nn * MMA_N + 2 * tq])[0] = c;
        }
    }
}
#endif  // BMB_TF32

#if BMB_TF32
// ------------------------------------ dQ on the tensor cores
//
// The forward's trick carries over unchanged here, because the orientations
// line up: S = Q@K^T accumulates as [query][key], and the second matmul is
//
//     dQ[i][j] += sum_c dS[i][c] K[c][j]
//
// whose A operand is dS with M = query and K = key -- the accumulator's own
// orientation. So acc_to_a applies directly and dS never reaches shared memory.
// (The dK/dV kernel is the one where this does not hold; see the note there.)
//
// K appears in TWO B layouts, because it is used as both operands:
//   Ks   b_off<BC>(k = head dim, n = key)   for  S  = Q  @ K^T
//   Kv   b_off<HS>(k = key,      n = head dim)  for dQ += dS @ K
// That is two shared buffers off one global read, which costs shared memory and
// no extra bandwidth.
template <int BR, int BC, int HS, int NT>
__global__ __launch_bounds__(NT) void flash_bwd_q_mma_k(
    float *__restrict__ dqkv, const float *__restrict__ qkv,
    const float *__restrict__ dout, const float *__restrict__ lse,
    const float *__restrict__ dsum, int T, int NH, float scale) {
    constexpr int NW = NT / WARPSIZE;
    constexpr int NC = BC / MMA_N;   // n-tiles of S, then k-chunks of dS
    constexpr int NO = HS / MMA_N;   // n-tiles of dQ
    constexpr int NK = HS / MMA_K;   // k-chunks of S
    constexpr int V4 = HS / 4;

    static_assert(BR == MMA_M * NW, "one 16-row query tile per warp");
    static_assert(BC % 16 == 0 && HS % 16 == 0, "B operands pack 16 wide");
    static_assert(NT % V4 == 0, "staging tiles evenly");

    extern __shared__ float smem[];
    float *Qs = smem;                 // a_off<BR>(query, head dim), scaled
    float *dOs = Qs + BR * HS;        // a_off<BR>(query, head dim)
    float *Ks = dOs + BR * HS;        // b_off<BC>(head dim, key)
    float *Vs = Ks + HS * BC;         // b_off<BC>(head dim, key)
    float *Kv = Vs + HS * BC;         // b_off<HS>(key, head dim)

    const int tid = threadIdx.x;
    const int lane = tid % WARPSIZE, warp = tid / WARPSIZE;
    const int g = lane >> 2, tq = lane & 3;

    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int C = NH * HS;
    const int i0 = blockIdx.x * BR;
    const float *base = qkv + (size_t)b * T * 3 * C + h * HS;
    const float *dob = dout + (size_t)b * T * C + h * HS;

    // Q and dO are fixed for this block; the key blocks stream past them.
    for (int lin = tid; lin < BR * V4; lin += NT) {
        const int r = lin / V4, i4 = (lin % V4) * 4;
        const int t = i0 + r;
        float4 q = make_float4(0.f, 0.f, 0.f, 0.f), o = q;
        if (t < T) {
            q = CVEC4(base[(size_t)t * 3 * C + i4]);
            o = CVEC4(dob[(size_t)t * C + i4]);
        }
        Qs[a_off<BR>(r, i4 + 0)] = q.x * scale;
        Qs[a_off<BR>(r, i4 + 1)] = q.y * scale;
        Qs[a_off<BR>(r, i4 + 2)] = q.z * scale;
        Qs[a_off<BR>(r, i4 + 3)] = q.w * scale;
        dOs[a_off<BR>(r, i4 + 0)] = o.x;
        dOs[a_off<BR>(r, i4 + 1)] = o.y;
        dOs[a_off<BR>(r, i4 + 2)] = o.z;
        dOs[a_off<BR>(r, i4 + 3)] = o.w;
    }

    // Row statistics are precomputed, so this kernel needs no reductions at
    // all: P is exp(S - lse) and dS is P*(dP - D), both elementwise with a
    // per-query scalar. Each lane owns rows g and g+8 of its warp's tile.
    float lse_[2], ds_[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int qg = i0 + warp * MMA_M + g + 8 * u;
        lse_[u] = (qg < T) ? lse[(size_t)bh * T + qg] : 0.0f;
        ds_[u] = (qg < T) ? dsum[(size_t)bh * T + qg] : 0.0f;
    }

    float acc_q[NO][4] = {};

    const int jmax = min(T, i0 + BR);
    for (int j0 = 0; j0 < jmax; j0 += BC) {
        __syncthreads();
        for (int lin = tid; lin < BC * V4; lin += NT) {
            const int c = lin / V4, i4 = (lin % V4) * 4;
            const int t = j0 + c;
            float4 k = make_float4(0.f, 0.f, 0.f, 0.f), v = k;
            if (t < T) {
                k = CVEC4(base[(size_t)t * 3 * C + C + i4]);
                v = CVEC4(base[(size_t)t * 3 * C + 2 * C + i4]);
            }
            Ks[b_off<BC>(i4 + 0, c)] = k.x;
            Ks[b_off<BC>(i4 + 1, c)] = k.y;
            Ks[b_off<BC>(i4 + 2, c)] = k.z;
            Ks[b_off<BC>(i4 + 3, c)] = k.w;
            Vs[b_off<BC>(i4 + 0, c)] = v.x;
            Vs[b_off<BC>(i4 + 1, c)] = v.y;
            Vs[b_off<BC>(i4 + 2, c)] = v.z;
            Vs[b_off<BC>(i4 + 3, c)] = v.w;
            Kv[b_off<HS>(c, i4 + 0)] = k.x;
            Kv[b_off<HS>(c, i4 + 1)] = k.y;
            Kv[b_off<HS>(c, i4 + 2)] = k.z;
            Kv[b_off<HS>(c, i4 + 3)] = k.w;
        }
        __syncthreads();

        // S = (Q/sqrt(hs)) @ K^T and dP = dO @ V^T share the loop: same shape,
        // same A rows, and interleaving them doubles the independent mmas.
        float acc_s[NC][4] = {}, acc_p[NC][4] = {};
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            const int au = kk * (BR / MMA_M) + warp;
            const float4 qv = CVEC4(Qs[au * UNIT + (lane ^ a_swz<BR>(au)) * 4]);
            const float4 ov = CVEC4(dOs[au * UNIT + (lane ^ a_swz<BR>(au)) * 4]);
            const unsigned aq[4] = {to_tf32(qv.x), to_tf32(qv.y), to_tf32(qv.z),
                                    to_tf32(qv.w)};
            const unsigned ao[4] = {to_tf32(ov.x), to_tf32(ov.y), to_tf32(ov.z),
                                    to_tf32(ov.w)};
#pragma unroll
            for (int jj = 0; jj < NC; jj += 2) {
                const int bu = kk * (BC / 16) + jj / 2;
                const float4 kv4 = CVEC4(Ks[bu * UNIT + (lane ^ b_swz<BC>(bu)) * 4]);
                const float4 vv4 = CVEC4(Vs[bu * UNIT + (lane ^ b_swz<BC>(bu)) * 4]);
                const unsigned bk[4] = {to_tf32(kv4.x), to_tf32(kv4.y),
                                        to_tf32(kv4.z), to_tf32(kv4.w)};
                const unsigned bv[4] = {to_tf32(vv4.x), to_tf32(vv4.y),
                                        to_tf32(vv4.z), to_tf32(vv4.w)};
                mma_m16n8k8(acc_s[jj + 0], aq, &bk[0]);
                mma_m16n8k8(acc_s[jj + 1], aq, &bk[2]);
                mma_m16n8k8(acc_p[jj + 0], ao, &bv[0]);
                mma_m16n8k8(acc_p[jj + 1], ao, &bv[2]);
            }
        }

        // P from lse -- exact, and with no reduction. Masked entries become 0
        // rather than -inf: these are probabilities now, and 0 is what makes
        // their gradient vanish. acc_s becomes dS in place.
#pragma unroll
        for (int jj = 0; jj < NC; ++jj)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int u = i >> 1;
                const int qg = i0 + warp * MMA_M + g + 8 * u;
                const int kg = j0 + jj * MMA_N + 2 * tq + (i & 1);
                const bool live = (qg < T) && (kg < T) && (kg <= qg);
                const float p = live ? __expf(acc_s[jj][i] - lse_[u]) : 0.0f;
                acc_s[jj][i] = p * (acc_p[jj][i] - ds_[u]);
            }

        // dQ += dS @ K, with dS handed over in registers.
#pragma unroll
        for (int jj = 0; jj < NC; ++jj) {
            unsigned a[4];
            acc_to_a(acc_s[jj], a, lane);
#pragma unroll
            for (int nn = 0; nn < NO; nn += 2) {
                const int bu = jj * (HS / 16) + nn / 2;
                const float4 bvv = CVEC4(Kv[bu * UNIT + (lane ^ b_swz<HS>(bu)) * 4]);
                const unsigned bb[4] = {to_tf32(bvv.x), to_tf32(bvv.y),
                                        to_tf32(bvv.z), to_tf32(bvv.w)};
                mma_m16n8k8(acc_q[nn + 0], a, &bb[0]);
                mma_m16n8k8(acc_q[nn + 1], a, &bb[2]);
            }
        }
    }

    // K was staged unscaled, so the 1/sqrt(hs) lands here.
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int qg = i0 + warp * MMA_M + g + 8 * u;
        if (qg >= T) continue;
        float *dq = dqkv + ((size_t)b * T + qg) * 3 * C + h * HS;
#pragma unroll
        for (int nn = 0; nn < NO; ++nn) {
            float2 o;
            o.x = acc_q[nn][2 * u + 0] * scale;
            o.y = acc_q[nn][2 * u + 1] * scale;
            reinterpret_cast<float2 *>(&dq[nn * MMA_N + 2 * tq])[0] = o;
        }
    }
}
#endif  // BMB_TF32

template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int QAT, int QBT, int KVC, int MMAQ, int MMAKV>
bool launch_bwd(float *dqkv, float *dsum, const float *dout, const float *qkv,
                const float *out, const float *lse, int B, int T, int NH,
                float scale) {
    constexpr int PAD = 4, QW = BR + PAD, KW = BC + PAD, PW = BC + PAD;
    // Only KVC head dimensions of K and V are resident at a time. That term is
    // what a chunked config shrinks, and shrinking it is only worth anything
    // because it is what decides how many blocks fit on an SM.
    constexpr size_t kv_smem =
        (size_t)(2 * HS * QW + 2 * KVC * KW + 2 * BR * PW + 2 * BR) * sizeof(float);
    // The tensor-core dQ kernel keeps no P tile and holds K in two B layouts
    // instead of one padded row-major tile, so its footprint is its own.
    constexpr size_t q_smem =
        MMAQ ? (size_t)(2 * BR * HS + 2 * HS * BC + BC * HS) * sizeof(float)
             : (size_t)(2 * HS * QW + 2 * HS * KW + BC * QW + 2 * BR) * sizeof(float);
    if (kv_smem > SMEM_CAP || q_smem > SMEM_CAP) return false;

    // The tensor-core dK/dV kernel holds K and V in the A layout and Q and dO
    // in TWO B layouts each, and keeps no score tile at all.
    constexpr size_t kvm_smem =
        (size_t)(2 * BC * HS + 4 * BR * HS + 2 * BR) * sizeof(float);
    if (MMAKV && kvm_smem > SMEM_CAP) return false;

    const int rows = B * NH * T;
    flash_dsum_k<<<ceil_div(rows * 32, 128), 128>>>(dsum, dout, out, T, NH, HS,
                                                    rows);
    if constexpr (MMAKV) {
#if BMB_TF32
        // One 16-key tile per warp, so this kernel sets its own block size for
        // the same reason the dQ one does.
        constexpr int NTKV = BC * 2;
        auto kv = flash_bwd_kv_mma_k<BR, BC, HS, NTKV>;
        static thread_local bool kv_conf = false;
        if (!kv_conf) {
            if (cudaFuncSetAttribute(kv, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)kvm_smem) != cudaSuccess) return false;
            kv_conf = true;
        }
        kv<<<dim3(ceil_div(T, BC), B * NH), NTKV, kvm_smem>>>(
            dqkv, qkv, dout, lse, dsum, T, NH, scale);
#else
        return false;
#endif
    } else {
        auto kv = flash_bwd_kv_k<BR, BC, HS, NT, RT, CT, AT, BT, KVC>;
        static thread_local bool kv_conf = false;
        if (!kv_conf) {
            if (cudaFuncSetAttribute(kv, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)kv_smem) != cudaSuccess) return false;
            kv_conf = true;
        }
        kv<<<dim3(ceil_div(T, BC), B * NH), NT, kv_smem>>>(dqkv, qkv, dout, lse,
                                                           dsum, T, NH, scale);
    }
    if constexpr (MMAQ) {
#if BMB_TF32
        // The two kernels no longer have to share a block size. The mma dQ
        // kernel wants exactly one 16-row query tile per warp, i.e. BR*2
        // threads -- which for the best kv shape (BR=32, 128 threads) is 64,
        // not 128. Tying them together kept the fastest kv tile from pairing
        // with the tensor-core dQ at all.
        constexpr int NTQ = BR * 2;
        auto qk = flash_bwd_q_mma_k<BR, BC, HS, NTQ>;
        static thread_local bool q_conf = false;
        if (!q_conf) {
            if (cudaFuncSetAttribute(qk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)q_smem) != cudaSuccess) return false;
            q_conf = true;
        }
        qk<<<dim3(ceil_div(T, BR), B * NH), NTQ, q_smem>>>(dqkv, qkv, dout, lse,
                                                           dsum, T, NH, scale);
#else
        return false;
#endif
    } else {
        auto qk = flash_bwd_q_k<BR, BC, HS, NT, RT, CT, QAT, QBT>;
        static thread_local bool q_conf = false;
        if (!q_conf) {
            if (cudaFuncSetAttribute(qk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)q_smem) != cudaSuccess) return false;
            q_conf = true;
        }
        qk<<<dim3(ceil_div(T, BR), B * NH), NT, q_smem>>>(dqkv, qkv, dout, lse,
                                                          dsum, T, NH, scale);
    }
    return true;
}

// The two kernels accumulate DIFFERENT tensors -- dK/dV is BC x HS, dQ is
// BR x HS -- so they need separate tile shapes. Sharing one pair would force
// BR == BC for no reason but notation, and BR != BC turns out to be the
// interesting region.
//
// WHAT THE SWEEP SETTLED, and it is not what the profiler implied. ncu says
// these kernels are bound on shared-memory-to-register traffic (L1/TEX ~88%,
// DRAM ~21%), so the indicated cure is a bigger register tile -- fewer shared
// loads per FMA. Two entries below test exactly that:
//
//   config 5   RT4 CT2   0.75 shared loads per FMA   2 blocks/SM   1.61 ms
//   config 6   RT4 CT4   0.50 shared loads per FMA   1 block /SM   2.01 ms
//
// Config 6 does a third less shared-memory work per unit of arithmetic and is
// 25% SLOWER, because the wider key tile costs the second resident block. On
// this kernel every arrangement that improves the ratio spends shared memory
// to get it, and the block count is worth more than the ratio -- the opposite
// of what the same counter reading meant for kernel 7. A profile says which
// resource is saturated. It does not say which change is affordable.
//
// -1 means "use the measured rule"; train_gpt --bwd-cfg N pins a config.
static int g_bwd_override = -1;

// CHUNKING THE HEAD DIMENSION, which is what the paragraph above says to do.
//
// The dK/dV kernel holds six shared arrays, and their sizes are
//
//     Q, dO   2 * HS * (BR+4)      K, V   2 * HS * (BC+4)
//     P, dS   2 * BR * (BC+4)
//
// so the HS terms dominate and the K/V one is the larger of them whenever
// BC > BR. K and V are also the only two operands the SECOND matmul never
// touches -- it reads P, dS, Q and dO -- so they are the only two that can be
// made non-resident without shrinking the accumulator tile. That asymmetry is
// the whole opening: staging K and V one head-chunk at a time cuts the biggest
// shared term by HS/KVC and leaves every register tile exactly as it was.
//
// What it costs is that K and V stop being staged once per block and start
// being staged once per chunk per query block, plus two barriers per chunk.
// Whether the extra blocks/SM outrun that is the measurement, not the theory.
//
// (id, BR, BC, NT, RT, CT, kv: AT, BT, dQ: QAT, QBT, KVC, MMAQ, MMAKV, name)
#define BWD_CONFIGS(X)                                                         \
    X(0, 64, 32, 128, 4, 4, 4, 4, 8, 4, 0, 0, 0, "br64 bc32 t128 kv4x4 q8x4")        \
    X(1, 64, 32, 128, 4, 4, 4, 4, 4, 8, 0, 0, 0, "br64 bc32 t128 kv4x4 q4x8")        \
    X(2, 32, 32, 64, 4, 4, 4, 8, 4, 8, 0, 0, 0, "br32 bc32 t64  kv4x8 q4x8")         \
    X(3, 32, 32, 64, 4, 4, 8, 4, 8, 4, 0, 0, 0, "br32 bc32 t64  kv8x4 q8x4")         \
    X(4, 32, 32, 64, 4, 4, 4, 4, 4, 4, 0, 0, 0, "br32 bc32 t64  kv4x4 q4x4")         \
    X(5, 32, 32, 128, 4, 2, 4, 4, 4, 4, 0, 0, 0, "br32 bc32 t128 kv4x4 q4x4")        \
    X(6, 32, 64, 128, 4, 4, 8, 4, 4, 4, 0, 0, 0, "br32 bc64 t128 kv8x4 q4x4")        \
    X(7, 64, 32, 64, 8, 4, 8, 4, 8, 8, 0, 0, 0, "br64 bc32 t64  8x4 kv8x4 q8x8")     \
    X(8, 32, 64, 128, 4, 4, 8, 4, 4, 4, 16, 0, 0, "br32 bc64 t128 kv8x4 q4x4 c16")   \
    X(9, 32, 64, 128, 4, 4, 8, 4, 4, 4, 32, 0, 0, "br32 bc64 t128 kv8x4 q4x4 c32")   \
    X(10, 32, 32, 128, 4, 2, 4, 4, 4, 4, 16, 0, 0, "br32 bc32 t128 kv4x4 q4x4 c16")  \
    X(11, 32, 32, 128, 4, 2, 4, 4, 4, 4, 32, 0, 0, "br32 bc32 t128 kv4x4 q4x4 c32")  \
    X(12, 32, 64, 128, 4, 4, 8, 4, 4, 4, 8, 0, 0, "br32 bc64 t128 kv8x4 q4x4 c8")                    \
    X(13, 64, 32, 128, 4, 4, 4, 4, 4, 8, 0, 1, 0, "br64 bc32 t128 mma-q")         \
    X(14, 32, 32, 64, 4, 4, 4, 8, 4, 8, 0, 1, 0, "br32 bc32 t64  mma-q")          \
    X(15, 32, 64, 128, 4, 4, 8, 4, 4, 4, 8, 1, 0, "br32 bc64 t128 c8 mma-q")       \
    X(16, 32, 32, 128, 4, 2, 4, 4, 4, 4, 0, 1, 0, "br32 bc32 t128 mma-q")         \
    X(17, 32, 32, 128, 4, 2, 4, 4, 4, 4, 16, 1, 0, "br32 bc32 t128 c16 mma-q")    \
    X(18, 32, 32, 128, 4, 2, 4, 4, 4, 4, 0, 1, 1, "br32 bc32 mma-q mma-kv")     \
    X(19, 32, 32, 128, 4, 2, 4, 4, 4, 4, 0, 0, 1, "br32 bc32 mma-kv only")

// KVC == 0 in the table means "no chunking"; it becomes HS here so that the
// unchunked configs keep taking the stage-once path bit for bit.
template <int HS, int KVC>
constexpr int kvc_of() { return KVC == 0 ? HS : KVC; }

// The dK/dV half of a config. Split out from the dQ half because the two
// kernels can now be chosen independently: MMAQ swaps the dQ kernel for the
// tensor-core one, whose tile is the mma fragment layout rather than a
// QAT x QBT choice, so the QAT/QBT rules stop applying to it.
template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int KVC>
constexpr bool bwd_kv_ok() {
    return (BR / RT) * (BC / CT) == NT &&        // score tile
           (BC / AT) * (HS / BT) == NT &&        // dK/dV tile
           NT % (HS / 4) == 0 && HS % BT == 0 &&
           BC % AT == 0 && AT % 4 == 0 && BT % 4 == 0 &&
           KVC % 4 == 0 && HS % KVC == 0 && NT % (KVC / 4) == 0;
}

template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int QAT, int QBT, int KVC, int MMAQ, int MMAKV>
struct BwdInst {
    static bool run(float *dqkv, float *dsum, const float *dout,
                    const float *qkv, const float *out, const float *lse, int B,
                    int T, int NH, float scale) {
        constexpr int C = kvc_of<HS, KVC>();
        // The tensor-core dQ kernel replaces the QAT/QBT tile with the mma
        // fragment layout, so it has its own shape rule: one 16-row query tile
        // per warp. The kv half of the config still has to be valid either way.
        constexpr bool q_ok =
            MMAQ ? (BC % 16 == 0 && HS % 16 == 0 && (BR * 2) % (HS / 4) == 0)
                 : ((BR / QAT) * (HS / QBT) == NT && BR % QAT == 0 &&
                    HS % QBT == 0 && QAT % 4 == 0 && QBT % 4 == 0);
        // The tensor-core dK/dV kernel replaces the AT/BT tile with the mma
        // fragment layout, so like the dQ one it has its own shape rule.
        constexpr bool kv_ok =
            MMAKV ? (BR % 16 == 0 && HS % 16 == 0 && (BC * 2) % (HS / 4) == 0 &&
                     BR % 8 == 0)
                  : bwd_kv_ok<BR, BC, HS, NT, RT, CT, AT, BT, C>();
        if constexpr (kv_ok && q_ok)
            return launch_bwd<BR, BC, HS, NT, RT, CT, AT, BT, QAT, QBT, C, MMAQ,
                              MMAKV>(
                dqkv, dsum, dout, qkv, out, lse, B, T, NH, scale);
        else
            return false;
    }
};

template <int HS>
bool dispatch_bwd(int cfg, float *dqkv, float *dsum, const float *dout,
                  const float *qkv, const float *out, const float *lse, int B,
                  int T, int NH, float scale) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, KVC, MMAQ, MMAKV, name)                 \
    case id:                                                                   \
        return BwdInst<BR, BC, HS, NT, RT, CT, AT, BT, QAT, QBT, KVC, MMAQ, MMAKV>::run(    \
            dqkv, dsum, dout, qkv, out, lse, B, T, NH, scale);
        BWD_CONFIGS(X)
#undef X
    default: return false;
    }
}
}  // namespace

int flash_num_bwd_configs() {
    int n = 0;
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, KVC, MMAQ, MMAKV, name) ++n;
    BWD_CONFIGS(X)
#undef X
    return n;
}

// Only dQ moves to TF32 in an MMAQ config -- dK and dV still come out of the
// fp32 kernel -- so the bar is raised for dQ alone rather than for all three.
// That is deliberate: if the tensor-core kernel were corrupting dK or dV, a
// blanket tolerance would hide it, and the split is what makes the measured
// 5.5e-04 / 4.2e-07 / 3.3e-07 signature readable at a glance.
void flash_bwd_config_tol(int cfg, double *dq, double *dk, double *dv) {
    int mmaq = 0, mmakv = 0;
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, KVC, MMAQ, MMAKV, name)           \
    if (cfg == id) { mmaq = MMAQ; mmakv = MMAKV; }
    BWD_CONFIGS(X)
#undef X
    *dq = mmaq ? 2e-3 : 1e-5;
    *dk = mmakv ? 2e-3 : 1e-5;
    *dv = mmakv ? 2e-3 : 1e-5;
}

const char *flash_bwd_config_name(int cfg) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, KVC, MMAQ, MMAKV, name)                 \
    case id: return name;
        BWD_CONFIGS(X)
#undef X
    default: return "?";
    }
}

// Measured. The profiler says both backward kernels are bound on
// shared-memory-to-register traffic (L1/TEX ~70%, DRAM ~21%, compute ~26%),
// which is kernel 6 disease at a different level of the hierarchy. Occupancy is
// the lesser of the two effects: config 5 doubles resident threads over config
// 3 and gains only 2.4%, because a narrower column tile also loads more per
// FMA. Fixing this properly means staging the head dimension in chunks so that
// bigger register tiles and more blocks per SM stop competing for the same
// shared memory -- the next thing to do here.
// WHAT CHUNKING SETTLED, after one false start that was my own bug.
//
// THE FALSE START, recorded because it is the more useful half. The first sweep
// said chunking was worth 8-10% at every context length. It is not. In
// restructuring the score loop I had put a `#pragma unroll` on the inner loop
// that the pre-chunking code did not have -- and it sits inside the KVC == HS
// path too, so it re-tuned the codegen of every UNCHUNKED config, which is to
// say the entire control group. Config 0 lost 13.8% to that pragma alone,
// config 7 gained 10%, and config 6 -- the one the argument rests on -- lost 8%.
// Chunking was being credited with damage the comparison had done to itself.
// A change that restructures a loop is not a controlled experiment until the
// control has been checked byte for byte.
//
// THE REAL NUMBERS. Median of three sweeps, spread under 1% on every cell,
// B*T held constant, fused backward, ms:
//
//   ctx   cfg5 bc32   cfg6 bc64   cfg8 +c16   cfg12 +c8   cfg10 bc32+c16
//   256      1.337      1.447       1.412       1.396         1.331
//   512      2.405      2.467       2.449       2.411         2.435
//   1024     4.535      4.472       4.475       4.424         4.619
//   2048     8.755      8.402       8.471       8.369         8.955
//
// Chunking is worth 0.4-3.5%, not 8-10%, and KVC=8 beats KVC=16 nearly
// everywhere -- so the chunk size picked first was wrong as well. The wide tile
// does win at long context, but mostly on its own: config 6 unchunked already
// beats config 5 by 1.4% at ctx 1024 and 4.0% at 2048, and chunking adds about
// another 1% on top of that.
//
// WHAT SURVIVES IS THE NEGATIVE HALF. `ncu` medians
// (bench/logs/flash_bwd_chunk_ncu.csv):
//
//   cfg  tile        KVC  blocks/SM  L1/TEX    SM    Occ   MIO   ms(256)
//    5   br32 bc32    --      2        78.9  34.3   16.3  0.79    1.337
//   10   br32 bc32    16      3        82.2  36.9   23.9  2.45    1.331
//    6   br32 bc64    --      1        63.8  26.0    8.3  0.25    1.447
//    8   br32 bc64    16      2        71.1  29.9   15.9  0.83    1.412
//   12   br32 bc64     8      2        74.0  30.6   15.9  1.15    1.396
//
// Config 10 is the one case where the resource was FREED rather than traded:
// the default's own tile, three resident blocks against two, nothing given up.
// Every counter says it should be faster -- highest occupancy, highest L1/TEX,
// highest SM throughput in the table -- and it ties at ctx 256 and loses
// everywhere above. Fifty percent more resident warps, for free, for nothing.
// One counter says where they went: MIO throttle 0.79 -> 2.45, a tripling, with
// the barrier stall up 0.24 -> 0.39. They arrive and they queue.
//
// The wide-tile family says the converse. Config 6 has the least saturated
// datapath of the five and the lowest throughput, because at 8% occupancy there
// are not enough warps to feed the FMA pipes however little each asks for.
// Efficiency per warp is not throughput; neither is occupancy. Both ends of
// that trade are visible in the one table.
//
// So the default is a rule rather than a constant -- the same shape as the
// GEMM's two tiles. End to end, --bwd-cfg pinning each way, three interleaved
// runs, bracketed by a ctx-256 reference that read 46.8 ms before and 46.2
// after: ctx 1024 goes 69.1 -> 68.1 ms (1.4%), ctx 2048 goes 99.4 -> 97.2
// (2.2%). The model trains at ctx 256, where the narrow tile wins and nothing
// changes.
int flash_default_bwd_config(int T) {
    if (g_bwd_override >= 0) return g_bwd_override;
    // Under --tf32 both backward kernels are tensor-core, and config 18 wins at
    // every context length measured -- 34% at ctx 256 rising to 38% at 2048 --
    // so the context-dependent rule below simply does not apply to it. That
    // rule existed because the fp32 dQ kernel was expensive enough for the kv
    // tile's shape to be worth trading against it, and the port removed the
    // trade. BR = BC = 32 is what keeps both kernels at two blocks per SM.
    if (gemm_tf32()) return 18;
    return T > 512 ? 12 : 5;
}

// A/B-ing a tile choice used to mean editing the line above and rebuilding,
// which is two builds per comparison and puts the two numbers in different
// binaries. The override makes it one binary and one run each way.
void flash_set_bwd_config(int cfg) { g_bwd_override = cfg; }

bool flash_attention_backward_cfg(int cfg, float *dqkv, float *dsum,
                                  const float *dout, const float *qkv,
                                  const float *out, const float *lse, int B,
                                  int T, int C, int NH) {
    const int hs = C / NH;
    const float scale = 1.0f / sqrtf((float)hs);
    switch (hs) {
    case 32:
        return dispatch_bwd<32>(cfg, dqkv, dsum, dout, qkv, out, lse, B, T, NH, scale);
    case 64:
        return dispatch_bwd<64>(cfg, dqkv, dsum, dout, qkv, out, lse, B, T, NH, scale);
    case 128:
        return dispatch_bwd<128>(cfg, dqkv, dsum, dout, qkv, out, lse, B, T, NH, scale);
    default:
        return false;
    }
}

void flash_attention_backward(float *dqkv, float *dsum, const float *dout,
                              const float *qkv, const float *out,
                              const float *lse, int B, int T, int C, int NH) {
    // Preferred first, then the short-context default, then anything that runs.
    // Config 12's dK/dV tile does not divide a 32-wide head, so at T > 512 with
    // hs=32 the preferred config declines -- and without config 5 named
    // explicitly here the scan below would fall to whatever happens to sit
    // earliest in the table, which is not the same as falling back to the tile
    // that was measured to be second best.
    const int preferred[] = {flash_default_bwd_config(T), 16, 12, 5};
    for (int cfg : preferred)
        if (flash_attention_backward_cfg(cfg, dqkv, dsum, dout, qkv, out, lse, B,
                                         T, C, NH))
            return;
    for (int cfg = 0; cfg < flash_num_bwd_configs(); ++cfg)
        if (flash_attention_backward_cfg(cfg, dqkv, dsum, dout, qkv, out, lse, B,
                                         T, C, NH))
            return;
    {
        fprintf(stderr, "flash: no backward tile config for head size %d\n",
                C / NH);
        exit(1);
    }
}
