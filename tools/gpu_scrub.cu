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
        size_t total = 0, filled = 0;
        // Grab memory in shrinking chunks until nothing is left, fill each.
        size_t chunk = (size_t)1 << 30;
        void *held[4096];
        int nheld = 0;
        while (chunk >= (1u << 20) && nheld < 4096) {
            void *p = nullptr;
            if (cudaMalloc(&p, chunk) != cudaSuccess) { cudaGetLastError(); chunk >>= 1; continue; }
            held[nheld++] = p;
            const size_t n = chunk / 4;
            fill_k<<<(unsigned)((n + 255) / 256), 256>>>((unsigned *)p, n, v);
            filled += chunk;
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < nheld; ++i) cudaFree(held[i]);
        cudaMemGetInfo(&total, &total);
        printf("device %d: filled %.0f MB with 0x%08x\n", d, filled / 1048576.0, v);
    }
    return 0;
}
