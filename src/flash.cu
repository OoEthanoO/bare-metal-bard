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
// K's tile and P's tile are the same allocation. Once S is in registers the K
// tile is dead, and (when BR == HS) the two are exactly the same size. That
// costs one extra barrier per key block and buys 16 KB, which is the difference
// between one and two resident blocks per SM.
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
    static bool configured = false;
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
#define FWD_CONFIGS(X)                                                         \
    X(0, 64, 64, 128, 4, 8, "br64 bc64 t128 4x8")                              \
    X(1, 64, 32, 128, 4, 4, "br64 bc32 t128 4x4")                              \
    X(2, 64, 64, 256, 4, 4, "br64 bc64 t256 4x4")                              \
    X(3, 128, 32, 256, 4, 4, "br128 bc32 t256 4x4")                            \
    X(4, 32, 32, 64, 4, 4, "br32 bc32 t64 4x4")

// OT = HS/(BC/CT) must be a positive multiple of 4, which rules some configs
// out for small head sizes. Checked here so an unusable pair fails cleanly at
// runtime instead of failing to compile the whole file.
template <int BR, int BC, int HS, int NT, int RT, int CT>
constexpr bool cfg_ok() {
    return HS % (BC / CT) == 0 && (HS / (BC / CT)) % 4 == 0 &&
           NT % (HS / 4) == 0 && (BR / RT) * (BC / CT) == NT;
}

// Instantiate one config for one head size, or refuse it.
template <int BR, int BC, int HS, int NT, int RT, int CT>
struct FwdInst {
    static bool run(float *out, float *lse, const float *qkv, int B, int T,
                    int NH, float scale) {
        if constexpr (cfg_ok<BR, BC, HS, NT, RT, CT>())
            return launch_fwd<BR, BC, HS, NT, RT, CT>(out, lse, qkv, B, T, NH, scale);
        else
            return false;
    }
};

template <int HS>
bool dispatch_cfg(int cfg, float *out, float *lse, const float *qkv, int B,
                  int T, int NH, float scale) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, name)                                        \
    case id:                                                                   \
        return FwdInst<BR, BC, HS, NT, RT, CT>::run(out, lse, qkv, B, T, NH, scale);
        FWD_CONFIGS(X)
#undef X
    default: return false;
    }
}
}  // namespace

int flash_num_configs() {
    int n = 0;
#define X(id, BR, BC, NT, RT, CT, name) ++n;
    FWD_CONFIGS(X)
#undef X
    return n;
}

const char *flash_config_name(int cfg) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, name)                                        \
    case id: return name;
        FWD_CONFIGS(X)
#undef X
    default: return "?";
    }
}

// Measured, not reasoned: br64/bc32 wins at hs=64 despite br64/bc64 having the
// better arithmetic intensity on paper, because the smaller key tile fits two
// blocks per SM instead of one.
int flash_default_config() { return 1; }

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
    if (flash_attention_forward_cfg(flash_default_config(), out, lse, qkv, B, T,
                                    C, NH))
        return;
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
template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT>
__global__ __launch_bounds__(NT) void flash_bwd_kv_k(
    float *__restrict__ dqkv, const float *__restrict__ qkv,
    const float *__restrict__ dout, const float *__restrict__ lse,
    const float *__restrict__ dsum, int T, int NH, float scale) {
    constexpr int PAD = 4;
    constexpr int RG = BR / RT, CG = BC / CT;
    constexpr int AG = BC / AT, BG = HS / BT;
    constexpr int V4 = HS / 4;
    constexpr int QW = BR + PAD;  // Qst/dOst row width (head-size major)
    constexpr int KW = BC + PAD;  // Kst/Vst row width (head-size major)
    constexpr int PW = BC + PAD;  // Pst/dSst row width (query major)

    static_assert(RG * CG == NT, "score tile must cover the thread block");
    static_assert(AG * BG == NT, "accumulator tile must cover it too");
    static_assert(RT % 4 == 0 && CT % 4 == 0, "vector loads");
    static_assert(AT % 4 == 0 && BT % 4 == 0, "vector loads");
    static_assert(NT % V4 == 0, "staging must tile evenly");

    extern __shared__ float smem[];
    float *Qst = smem;              // [HS][QW], carries the 1/sqrt(hs)
    float *dOst = Qst + HS * QW;    // [HS][QW]
    float *Kst = dOst + HS * QW;    // [HS][KW]
    float *Vst = Kst + HS * KW;     // [HS][KW]
    float *Pst = Vst + HS * KW;     // [BR][PW]
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

    // K and V are fixed for this block: stage once, transposed.
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
        __syncthreads();

        // S = Q K^T and dP = dO V^T share one loop: same shape, same tiles, and
        // interleaving them doubles the independent FMAs in flight.
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
#pragma unroll
            for (int c = 0; c < CT; c += 4) {
                VEC4(rk[c]) = CVEC4(Kst[i * KW + c0 + c]);
                VEC4(rv[c]) = CVEC4(Vst[i * KW + c0 + c]);
            }
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    s[r][c] += rq[r] * rk[c];
                    dp[r][c] += rdo[r] * rv[c];
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
#pragma unroll
            for (int c = 0; c < CT; c += 4) {
                VEC4(Pst[(r0 + r) * PW + c0 + c]) = VEC4(s[r][c]);
                VEC4(dSst[(r0 + r) * PW + c0 + c]) = VEC4(dp[r][c]);
            }
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
    static_assert(RT % 4 == 0 && CT % 4 == 0, "vector loads");
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
#pragma unroll
            for (int c = 0; c < CT; c += 4) {
                VEC4(rk[c]) = CVEC4(Kst[i * KW + c0 + c]);
                VEC4(rv[c]) = CVEC4(Vst[i * KW + c0 + c]);
            }
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

constexpr size_t SMEM_CAP = 99 * 1024;  // Ada's opt-in maximum per block

template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int QAT, int QBT>
bool launch_bwd(float *dqkv, float *dsum, const float *dout, const float *qkv,
                const float *out, const float *lse, int B, int T, int NH,
                float scale) {
    constexpr int PAD = 4, QW = BR + PAD, KW = BC + PAD, PW = BC + PAD;
    constexpr size_t kv_smem =
        (size_t)(2 * HS * QW + 2 * HS * KW + 2 * BR * PW + 2 * BR) * sizeof(float);
    constexpr size_t q_smem =
        (size_t)(2 * HS * QW + 2 * HS * KW + BC * QW + 2 * BR) * sizeof(float);
    if (kv_smem > SMEM_CAP || q_smem > SMEM_CAP) return false;

    auto kv = flash_bwd_kv_k<BR, BC, HS, NT, RT, CT, AT, BT>;
    auto qk = flash_bwd_q_k<BR, BC, HS, NT, RT, CT, QAT, QBT>;
    static bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(kv, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)kv_smem) != cudaSuccess) return false;
        if (cudaFuncSetAttribute(qk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)q_smem) != cudaSuccess) return false;
        configured = true;
    }

    const int rows = B * NH * T;
    flash_dsum_k<<<ceil_div(rows * 32, 128), 128>>>(dsum, dout, out, T, NH, HS,
                                                    rows);
    kv<<<dim3(ceil_div(T, BC), B * NH), NT, kv_smem>>>(dqkv, qkv, dout, lse,
                                                       dsum, T, NH, scale);
    qk<<<dim3(ceil_div(T, BR), B * NH), NT, q_smem>>>(dqkv, qkv, dout, lse, dsum,
                                                      T, NH, scale);
    return true;
}

// The two kernels accumulate DIFFERENT tensors -- dK/dV is BC x HS, dQ is
// BR x HS -- so they need separate tile shapes. Sharing one pair would force
// BR == BC for no reason but notation, and BR != BC turns out to be the
// interesting region.
//
// (id, BR, BC, NT, RT, CT, kv: AT, BT, dQ: QAT, QBT, name)
#define BWD_CONFIGS(X)                                                         \
    X(0, 64, 32, 128, 4, 4, 4, 4, 8, 4, "br64 bc32 t128 kv4x4 q8x4")           \
    X(1, 64, 32, 128, 4, 4, 4, 4, 4, 8, "br64 bc32 t128 kv4x4 q4x8")           \
    X(2, 32, 32, 64, 4, 4, 4, 8, 4, 8, "br32 bc32 t64  kv4x8 q4x8")            \
    X(3, 32, 32, 64, 4, 4, 8, 4, 8, 4, "br32 bc32 t64  kv8x4 q8x4")            \
    X(4, 32, 32, 64, 4, 4, 4, 4, 4, 4, "br32 bc32 t64  kv4x4 q4x4")

template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int QAT, int QBT>
constexpr bool bwd_cfg_ok() {
    return (BR / RT) * (BC / CT) == NT &&        // score tile
           (BC / AT) * (HS / BT) == NT &&        // dK/dV tile
           (BR / QAT) * (HS / QBT) == NT &&      // dQ tile
           NT % (HS / 4) == 0 && HS % BT == 0 && HS % QBT == 0 &&
           BC % AT == 0 && BR % QAT == 0 && AT % 4 == 0 && BT % 4 == 0 &&
           QAT % 4 == 0 && QBT % 4 == 0;
}

template <int BR, int BC, int HS, int NT, int RT, int CT, int AT, int BT,
          int QAT, int QBT>
struct BwdInst {
    static bool run(float *dqkv, float *dsum, const float *dout,
                    const float *qkv, const float *out, const float *lse, int B,
                    int T, int NH, float scale) {
        if constexpr (bwd_cfg_ok<BR, BC, HS, NT, RT, CT, AT, BT, QAT, QBT>())
            return launch_bwd<BR, BC, HS, NT, RT, CT, AT, BT, QAT, QBT>(
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
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, name)                      \
    case id:                                                                   \
        return BwdInst<BR, BC, HS, NT, RT, CT, AT, BT, QAT, QBT>::run(         \
            dqkv, dsum, dout, qkv, out, lse, B, T, NH, scale);
        BWD_CONFIGS(X)
#undef X
    default: return false;
    }
}
}  // namespace

int flash_num_bwd_configs() {
    int n = 0;
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, name) ++n;
    BWD_CONFIGS(X)
#undef X
    return n;
}

const char *flash_bwd_config_name(int cfg) {
    switch (cfg) {
#define X(id, BR, BC, NT, RT, CT, AT, BT, QAT, QBT, name)                      \
    case id: return name;
        BWD_CONFIGS(X)
#undef X
    default: return "?";
    }
}

// Also measured: the 8x4 accumulator tile beats 4x8 by 25%, which is the extra
// reuse on the strided dO/Q reads paying for itself.
int flash_default_bwd_config() { return 3; }

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
    if (flash_attention_backward_cfg(flash_default_bwd_config(), dqkv, dsum,
                                     dout, qkv, out, lse, B, T, C, NH))
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
