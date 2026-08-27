#pragma once
// `cp.async`: global->shared without the value ever entering a register file,
// and asynchronous, so several k-chunks can be in flight while the current one
// is being multiplied. Shared by kernel 11 and the model's GEMM, for the same
// reason lane_major.cuh is -- the two must not drift.
//
// THE SIZE IS FORCED, and it is worth stating because it looks like a
// pessimisation. `cp.async` copies a CONTIGUOUS 4, 8 or 16 bytes of global to a
// CONTIGUOUS 4, 8 or 16 bytes of shared. The lane-major layout exists so that
// one lane's four mma operands are contiguous in SHARED -- and those four are
// A[g][t], A[g+8][t], A[g][t+4], A[g+8][t+4], four unrelated addresses in a
// row-major A. No 16-byte global chunk maps to a 16-byte shared chunk, ever,
// and no rearrangement fixes it, because the fragment mapping and the
// contiguity requirement are both properties of the hardware. So: 4 bytes.
//
// Which changes the thread->element map. With 16-byte copies each thread reads
// four consecutive elements; here it reads ONE, and consecutive lanes take
// consecutive positions along whichever axis the operand is contiguous in, so
// each individual copy is still a warp reading one contiguous 128-byte segment.
#include "kernels.h"
#if BMB_TF32

namespace {

// A generic pointer as the 32-bit shared-window address `cp.async` wants.
// `__cvta_generic_to_shared` is the documented way to ask; the older trick of
// casting the pointer straight to unsigned is not portable.
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

#endif  // BMB_TF32
