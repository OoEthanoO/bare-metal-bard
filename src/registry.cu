#include "kernels.h"
#include <cstddef>

const KernelEntry KERNELS[] = {
    {1, "naive",     sgemm_naive,     "one thread per output, uncoalesced"},
    {2, "coalesced", sgemm_coalesced, "row/col mapping swapped"},
    {3, "smem",      sgemm_smem,      "32x32 shared-memory tiling"},
    {4, "tile1d",    sgemm_tile1d,    "1D register tiling, TM=8"},
    {5, "tile2d",    sgemm_tile2d,    "2D register tiling, 8x8 per thread"},
    {6, "vectorized",sgemm_vectorized,"float4 loads + transposed As"},
    {7, "warptile",  sgemm_warptile,  "warp-level tiling, BK=16"},
};

const int NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);

const KernelEntry *find_kernel(int id) {
    for (int i = 0; i < NUM_KERNELS; ++i) {
        if (KERNELS[i].id == id) return &KERNELS[i];
    }
    return nullptr;
}
