// Device query: establish the roofline numbers we will be measuring against.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        // Memory bandwidth: bus width is in bits, clock in kHz, DDR -> x2.
        double bw = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) * 1e3 / 1e9;

        // FP32 peak: 128 FMA lanes/SM on Ada (sm_89), 2 flops per FMA.
        double fp32 = 2.0 * 128 * p.multiProcessorCount * (p.clockRate * 1e3) / 1e9;

        printf("device %d: %s (sm_%d%d)\n", d, p.name, p.major, p.minor);
        printf("  SMs                       %d\n", p.multiProcessorCount);
        printf("  clock                     %.2f GHz\n", p.clockRate / 1e6);
        printf("  global memory             %.2f GiB\n", p.totalGlobalMem / 1073741824.0);
        printf("  memory bus                %d-bit @ %.2f GHz\n", p.memoryBusWidth, p.memoryClockRate / 1e6);
        printf("  peak bandwidth            %.1f GB/s\n", bw);
        printf("  L2 cache                  %.2f MiB\n", p.l2CacheSize / 1048576.0);
        printf("  shared mem / block        %.1f KiB (opt-in max %.1f KiB)\n",
               p.sharedMemPerBlock / 1024.0, p.sharedMemPerBlockOptin / 1024.0);
        printf("  shared mem / SM           %.1f KiB\n", p.sharedMemPerMultiprocessor / 1024.0);
        printf("  registers / block         %d\n", p.regsPerBlock);
        printf("  registers / SM            %d\n", p.regsPerMultiprocessor);
        printf("  max threads / block       %d\n", p.maxThreadsPerBlock);
        printf("  max threads / SM          %d\n", p.maxThreadsPerMultiProcessor);
        printf("  warp size                 %d\n", p.warpSize);
        printf("  peak fp32 (non-tensor)    %.1f GFLOP/s\n", fp32);
        printf("  ridge point (FLOP/byte)   %.1f\n", fp32 / bw);
    }
    return 0;
}
