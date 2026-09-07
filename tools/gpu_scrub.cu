// Fill every byte of free memory on every device with a pattern, then exit.
//
// This exists to catch reads of memory that was never written. cudaMalloc
// hands back whatever the previous owner left, and a kernel that reads such
// memory before writing it behaves differently depending on which process
// ran before -- which is exactly what the two-rank anomaly did: identical
// wrong losses after some predecessors, clean after others. Run this between
// them: zeros make a stale read invisible, a NaN pattern makes it scream,
// and the difference between the two runs names the buffer.
//
//   ./bench/gpu_scrub          zeros
//   ./bench/gpu_scrub nan      0x7fc00000 in every float
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void fill_k(unsigned *p, size_t n, unsigned v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

int main(int argc, char **argv) {
    const unsigned v = (argc > 1 && !strcmp(argv[1], "nan")) ? 0x7fc00000u : 0u;
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    for (int d = 0; d < ndev; ++d) {
        cudaSetDevice(d);
        size_t free_before = 0, total = 0, filled = 0;
        cudaMemGetInfo(&free_before, &total);

        // THE BUDGET IS QUERIED, NOT DISCOVERED BY FAILING.
        //
        // This used to allocate 1 GB chunks until cudaMalloc failed, halving
        // the chunk on each failure. That terminates on Linux, where the
        // driver refuses once the card is full -- which is where this ran, on
        // rented A40s.
        //
        // It does not terminate usefully on Windows. Under WDDM the driver
        // pages device allocations out to system memory instead of failing, so
        // cudaMalloc keeps returning success long past the card's capacity:
        // the loop ran to its 4096-allocation cap, spent fifteen minutes and
        // 929 seconds of CPU, and held 0 MB of actual VRAM by the end. What it
        // was scrubbing at that point was host memory, which is not what the
        // tool is for.
        //
        // So the ceiling comes from cudaMemGetInfo. Filling what the driver
        // says is free is exactly the intent -- "every byte the next process
        // could be handed" -- and it is the same intent on both platforms.
        const size_t budget = free_before > (64u << 20) ? free_before - (64u << 20) : 0;
        size_t chunk = (size_t)1 << 30;
        void *held[4096];
        int nheld = 0;
        while (chunk >= (1u << 20) && nheld < 4096 && filled < budget) {
            if (chunk > budget - filled) {
                // Do not ask for more than the budget allows; on WDDM that
                // request would be granted and paged rather than refused.
                chunk = budget - filled;
                if (chunk < (1u << 20)) break;
            }
            void *p = nullptr;
            if (cudaMalloc(&p, chunk) != cudaSuccess) { cudaGetLastError(); chunk >>= 1; continue; }
            held[nheld++] = p;
            const size_t n = chunk / 4;
            fill_k<<<(unsigned)((n + 255) / 256), 256>>>((unsigned *)p, n, v);
            filled += chunk;
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < nheld; ++i) cudaFree(held[i]);
        printf("device %d: filled %.0f MB of %.0f MB free with 0x%08x (%d allocations)\n",
               d, filled / 1048576.0, free_before / 1048576.0, v, nheld);
    }
    return 0;
}
