#pragma once
#include <cuda_runtime.h>
#include <cfloat>

// Shared block-reduction helpers.
//
// These use __shfl_xor_sync rather than shared memory. A warp shuffle moves
// data through the register file's crossbar and never touches the LSU/MIO
// path, which is precisely the resource that saturated at 81% and capped GEMM
// kernel 6. Reductions are frequent in layernorm and softmax, so keeping them
// off that path matters.
namespace red {
constexpr int WARP = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// Block-wide reduction with the result live in EVERY thread.
//
// The xor-shuffle butterfly leaves its result in all 32 lanes, not just lane
// 0, so after the second stage every warp independently holds the same total
// and no broadcast step is needed.
//
// Note the __syncthreads() around `partial`: this function is unsafe to call
// from divergent code, so callers must reach it uniformly.
//
// The LEADING __syncthreads() is not redundant. `partial` is function-static
// shared memory, so two successive calls to the same instantiation reuse the
// same array -- and layernorm calls this twice in a row (mean, then variance).
// Without the leading barrier a thread racing ahead into the second reduction
// could overwrite partial[wid] while a slower thread was still reading its
// value from the first. That is a real race producing plausible-looking but
// subtly wrong statistics, which would surface only as a model that trains
// slightly badly. Distinct IS_MAX instantiations get distinct arrays and do
// not alias, but the barrier costs nothing and removes the whole class.
template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float partial[WARP];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int nwarps = (blockDim.x + WARP - 1) / WARP;

    __syncthreads();
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) partial[wid] = v;
    __syncthreads();

    v = (lane < nwarps) ? partial[lane] : (IS_MAX ? -FLT_MAX : 0.0f);
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    return v;
}
}  // namespace red
