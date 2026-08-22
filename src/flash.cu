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

int flash_default_config() { return 0; }

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

void flash_attention_forward(float *out, float *lse, const float *qkv, int B,
                             int T, int C, int NH) {
    if (!flash_attention_forward_cfg(flash_default_config(), out, lse, qkv, B, T,
                                     C, NH)) {
        // A head size the tiles cannot cover is a build-time mistake, not a
        // runtime condition to paper over.
        fprintf(stderr, "flash: no tile config for head size %d\n", C / NH);
        exit(1);
    }
}
