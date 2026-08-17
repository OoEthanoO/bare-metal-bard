#include "kernels.h"
#include <cstddef>

const KernelEntry KERNELS[] = {
    {1, "naive",     sgemm_naive,     "one thread per output, uncoalesced"},
    {2, "coalesced", sgemm_coalesced, "row/col mapping swapped"},
};

const int NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);

const KernelEntry *find_kernel(int id) {
    for (int i = 0; i < NUM_KERNELS; ++i) {
        if (KERNELS[i].id == id) return &KERNELS[i];
    }
    return nullptr;
}
