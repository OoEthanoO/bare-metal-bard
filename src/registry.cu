#include "kernels.h"
#include <cstddef>

const KernelEntry KERNELS[] = {
    {1, "naive",     sgemm_naive,     "one thread per output, uncoalesced", 1e-4},
    {2, "coalesced", sgemm_coalesced, "row/col mapping swapped", 1e-4},
    {3, "smem",      sgemm_smem,      "32x32 shared-memory tiling", 1e-4},
    {4, "tile1d",    sgemm_tile1d,    "1D register tiling, TM=8", 1e-4},
    {5, "tile2d",    sgemm_tile2d,    "2D register tiling, 8x8 per thread", 1e-4},
    {6, "vectorized",sgemm_vectorized,"float4 loads + transposed As", 1e-4},
    {7, "warptile",  sgemm_warptile,  "warp-level tiling, BK=16", 1e-4},
    {8, "dbuffer",   sgemm_doublebuffer, "double-buffered SMEM, 1 barrier/chunk", 1e-4},
    {9, "tensorcore",sgemm_tensorcore,   "WMMA m16n16k8 TF32 tensor cores", 4e-3},
    {10,"mma",       sgemm_mma,          "raw mma.sync, lane-major SMEM, 64x64 warp tile", 4e-3},
};

const int NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);

const KernelEntry *find_kernel(int id) {
    for (int i = 0; i < NUM_KERNELS; ++i) {
        if (KERNELS[i].id == id) return &KERNELS[i];
    }
    return nullptr;
}
