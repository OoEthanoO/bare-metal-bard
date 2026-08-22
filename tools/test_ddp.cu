// Ring all-reduce: correctness, then bandwidth.
//
// The correctness half runs entirely on one GPU by giving every rank the same
// physical device. That is not a shortcut around testing -- the chunking, the
// ring order, the reduction kernel and the two phases are all exercised
// identically. Only the transport changes when the ranks are really separate,
// and the transport is the one part CUDA is responsible for rather than me.
//
// So the algorithm can be wrong here and nowhere else, which means it can be
// made right before anyone pays for a second GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../src/ddp.h"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e__ = (x);                                                 \
        if (e__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e__),  \
                    __LINE__);                                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Deterministic, and different per rank so a dropped or duplicated chunk shows.
static float value_of(int rank, size_t i) {
    return (float)((i * 2654435761u + (unsigned)rank * 40503u) % 1000) * 0.001f +
           (float)rank;
}

static bool check(int n, size_t count, const int *devs, bool verbose) {
    DDP d;
    ddp_init(d, n, devs);

    std::vector<float *> bufs(n);
    std::vector<float> host(count);
    for (int r = 0; r < n; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaMalloc(&bufs[r], count * sizeof(float)));
        for (size_t i = 0; i < count; ++i) host[i] = value_of(r, i);
        CUDA_CHECK(cudaMemcpy(bufs[r], host.data(), count * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    ddp_allreduce(d, bufs.data(), count);

    // Reference in double, so the comparison measures the collective rather
    // than fp32 summation order.
    double worst = 0.0;
    std::vector<float> got(count);
    for (int r = 0; r < n; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaMemcpy(got.data(), bufs[r], count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            double want = 0.0;
            for (int q = 0; q < n; ++q) want += (double)value_of(q, i);
            const double err = fabs((double)got[i] - want) /
                               (fabs(want) > 1.0 ? fabs(want) : 1.0);
            if (err > worst) worst = err;
        }
    }

    for (int r = 0; r < n; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaFree(bufs[r]));
    }
    ddp_free(d);

    const bool ok = worst < 1e-6;
    if (verbose)
        printf("  n=%d  count=%-9zu  worst rel err %8.2e  %s\n", n, count, worst,
               ok ? "ok" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    int forced_n = 0;
    size_t bw_count = 10800000;  // ~43 MB, the gradient buffer of the GPT here
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) forced_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) bw_count = (size_t)atoll(argv[++i]);
    }

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    printf("%d CUDA device(s) present\n\n", ngpu);

    // ---- correctness, on whatever hardware exists ----
    // Ranks are placed round-robin over the real devices, so this is a genuine
    // multi-GPU test when there are several and a faithful rehearsal when there
    // is one.
    printf("ring all-reduce correctness:\n");
    int fails = 0;
    const int ns[] = {1, 2, 3, 4, 8};
    // Sizes chosen to land on the awkward cases: not divisible by the rank
    // count, smaller than the rank count, exactly one element.
    const size_t counts[] = {1, 3, 17, 1000, 65537, 1048576};
    for (int n : ns) {
        if (forced_n && n != forced_n) continue;
        for (size_t c : counts) {
            int devs[8];
            for (int r = 0; r < n; ++r) devs[r] = r % ngpu;
            if (!check(n, c, devs, true)) ++fails;
        }
    }

    // ---- bandwidth, only meaningful with real separate devices ----
    if (ngpu > 1 || forced_n > 1) {
        const int n = forced_n ? forced_n : ngpu;
        int devs[8];
        for (int r = 0; r < n; ++r) devs[r] = r % ngpu;
        DDP d;
        ddp_init(d, n, devs);
        printf("\n");
        ddp_report_topology(d);

        std::vector<float *> bufs(n);
        for (int r = 0; r < n; ++r) {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            CUDA_CHECK(cudaMalloc(&bufs[r], bw_count * sizeof(float)));
            CUDA_CHECK(cudaMemset(bufs[r], 0, bw_count * sizeof(float)));
        }
        for (int i = 0; i < 3; ++i) ddp_allreduce(d, bufs.data(), bw_count);

        double best = 1e30;
        for (int i = 0; i < 10; ++i) {
            ddp_allreduce(d, bufs.data(), bw_count);
            const double ms = ddp_timing(d).total_ms;
            if (ms < best) best = ms;
        }
        const DDPTiming &t = ddp_timing(d);
        const double mb = bw_count * 4.0 / 1e6;
        printf("\nall-reduce of %.1f MB across %d ranks\n", mb, n);
        printf("  best            %8.3f ms\n", best);
        printf("  reduce-scatter  %8.3f ms\n", t.reduce_scatter_ms);
        printf("  all-gather      %8.3f ms\n", t.all_gather_ms);
        printf("  moved/device    %8.1f MB\n", t.bytes_per_device / 1e6);
        printf("  effective link  %8.1f GB/s\n",
               t.bytes_per_device / 1e9 / (best * 1e-3));
        printf("\nA training step here is ~86 ms. This collective is %.1f%% of that.\n",
               best / 86.0 * 100.0);

        for (int r = 0; r < n; ++r) {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            CUDA_CHECK(cudaFree(bufs[r]));
        }
        ddp_free(d);
    } else {
        printf("\nonly one GPU: correctness above is the rehearsal, bandwidth\n"
               "needs real devices and is skipped.\n");
    }

    printf("\n%s\n", fails ? "FAILURES" : "all correctness checks passed");
    return fails ? 1 : 0;
}
