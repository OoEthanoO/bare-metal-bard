// Ring all-reduce, written out rather than linked in.
#include "ddp.h"
#include <vector>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e__ = (x);                                                 \
        if (e__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                    cudaGetErrorString(e__), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace {

inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// dst += src, elementwise. The only arithmetic in a ring all-reduce; everything
// else is scheduling.
__global__ void add_into_k(float *__restrict__ dst, const float *__restrict__ src,
                           size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] += src[i];
}

// Chunk r of a `count`-element buffer split n ways. The last chunk absorbs the
// remainder, which is why every chunk is described by an explicit (off, len)
// rather than assumed equal.
struct Chunk { size_t off, len; };

inline Chunk chunk_of(size_t count, int n, int r) {
    const size_t per = ceil_div(count, (size_t)n);
    const size_t off = per * (size_t)r;
    if (off >= count) return {count, 0};
    const size_t len = (off + per <= count) ? per : count - off;
    return {off, len};
}

// One rank-to-rank transfer. Direct when the driver allows peer access,
// otherwise bounced through pinned host memory -- which is exactly the
// difference the multi-GPU section is trying to measure, so it is not hidden.
void transfer(DDP &d, int src_rank, const float *src, int dst_rank, float *dst,
              size_t len) {
    if (len == 0) return;
    // A stream belongs to the device that was current when it was created, and
    // an async copy must be issued with THAT device current. Omitting this is
    // invisible when every rank shares one GPU -- every stream belongs to
    // device 0 and device 0 is always current -- and wrong the moment the ranks
    // are really separate. It is precisely the bug a single-GPU rehearsal
    // cannot catch, and it cost one rented pod to find.
    CUDA_CHECK(cudaSetDevice(d.dev[src_rank]));
    if (d.peer[src_rank][dst_rank]) {
        CUDA_CHECK(cudaMemcpyPeerAsync(dst, d.dev[dst_rank], src, d.dev[src_rank],
                                       len * sizeof(float), d.stream[src_rank]));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d.host_stage, src, len * sizeof(float),
                                   cudaMemcpyDeviceToHost, d.stream[src_rank]));
        CUDA_CHECK(cudaStreamSynchronize(d.stream[src_rank]));
        CUDA_CHECK(cudaSetDevice(d.dev[dst_rank]));
        CUDA_CHECK(cudaMemcpyAsync(dst, d.host_stage, len * sizeof(float),
                                   cudaMemcpyHostToDevice, d.stream[dst_rank]));
        CUDA_CHECK(cudaStreamSynchronize(d.stream[dst_rank]));
    }
}

void sync_all(DDP &d) {
    for (int r = 0; r < d.n; ++r) {
        CUDA_CHECK(cudaSetDevice(d.dev[r]));
        CUDA_CHECK(cudaStreamSynchronize(d.stream[r]));
    }
}
}  // namespace

void ddp_init(DDP &d, int n, const int *devices) {
    d.n = n;
    for (int r = 0; r < n; ++r) d.dev[r] = devices[r];

    d.any_peer = false;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            d.peer[i][j] = false;
            if (i == j) continue;
            if (d.dev[i] == d.dev[j]) {
                // Same physical device: a "peer" copy is just a device-to-device
                // copy, always available. This is the single-GPU rehearsal path.
                d.peer[i][j] = true;
                d.any_peer = true;
                continue;
            }
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, d.dev[i], d.dev[j]));
            if (can) {
                CUDA_CHECK(cudaSetDevice(d.dev[i]));
                const cudaError_t e = cudaDeviceEnablePeerAccess(d.dev[j], 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();  // swallow; fall back to host staging
                    can = 0;
                }
            }
            d.peer[i][j] = can != 0;
            d.any_peer = d.any_peer || d.peer[i][j];
        }
    }

    // Peer access being ENABLED is not evidence that it WORKS. On virtualised
    // hosts -- which is most rented hardware -- the driver can advertise P2P
    // while PCIe ACS or the IOMMU quietly sends the DMA somewhere else. The
    // copy returns success, the sync returns success, and the bytes never
    // arrive. A collective that trusts the flag then computes wrong gradients
    // at full speed and says nothing, which is the worst of all outcomes.
    //
    // So: actually move bytes and check they landed. The first version moved
    // FOUR bytes, once, and on the 2x A40 box that gave three different
    // verdicts in three runs of the same binary -- both directions bad, both
    // bad again, then one direction "good", which produced a ring with one
    // peer link and one staged link, 105 ms of communication and a gradient
    // norm nobody believed. A flaky link is a broken link. So the probe now
    // moves a megabyte of pattern, three times, checks every element, and if
    // ANY pair fails the whole ring stages through the host: a ring is only as
    // trustworthy as its least trustworthy link, and it should not be mixed.
    if (getenv("DDP_NO_P2P")) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) d.peer[i][j] = false;
        printf("ddp       DDP_NO_P2P set: staging every transfer through host\n");
    }
    bool any_bad = false;
    {
        constexpr size_t PROBE = 1u << 18;  // floats: 1 MB
        std::vector<float> pattern(PROBE), back(PROBE);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!d.peer[i][j] || d.dev[i] == d.dev[j]) continue;
                float *a = nullptr, *b = nullptr;
                CUDA_CHECK(cudaSetDevice(d.dev[i]));
                CUDA_CHECK(cudaMalloc(&a, PROBE * sizeof(float)));
                CUDA_CHECK(cudaSetDevice(d.dev[j]));
                CUDA_CHECK(cudaMalloc(&b, PROBE * sizeof(float)));
                size_t bad = 0;
                for (int rep = 0; rep < 3 && bad == 0; ++rep) {
                    for (size_t k = 0; k < PROBE; ++k)
                        pattern[k] = (float)(1 + rep) * 1000.0f + 100.0f * i + j +
                                     (float)(k % 977) * 0.001f;
                    CUDA_CHECK(cudaSetDevice(d.dev[i]));
                    CUDA_CHECK(cudaMemcpy(a, pattern.data(), PROBE * sizeof(float),
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaSetDevice(d.dev[j]));
                    CUDA_CHECK(cudaMemset(b, 0, PROBE * sizeof(float)));
                    CUDA_CHECK(cudaSetDevice(d.dev[i]));
                    CUDA_CHECK(cudaMemcpyPeer(b, d.dev[j], a, d.dev[i],
                                              PROBE * sizeof(float)));
                    CUDA_CHECK(cudaSetDevice(d.dev[j]));
                    CUDA_CHECK(cudaMemcpy(back.data(), b, PROBE * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                    for (size_t k = 0; k < PROBE; ++k)
                        if (back[k] != pattern[k]) ++bad;
                }
                if (bad) {
                    printf("ddp       WARNING: peer %d->%d is enabled but moves the "
                           "wrong bytes (%zu of %zu floats wrong). Not trusted.\n",
                           d.dev[i], d.dev[j], bad, PROBE);
                    any_bad = true;
                }
                CUDA_CHECK(cudaSetDevice(d.dev[i]));
                CUDA_CHECK(cudaFree(a));
                CUDA_CHECK(cudaSetDevice(d.dev[j]));
                CUDA_CHECK(cudaFree(b));
            }
        }
    }
    if (any_bad) {
        printf("ddp       at least one peer link failed the probe: staging EVERY "
               "transfer through host memory rather than running a mixed ring\n");
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) d.peer[i][j] = false;
    }
    d.any_peer = false;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) d.any_peer = d.any_peer || d.peer[i][j];

    d.recv_cap = 0;
    d.stage_cap = 0;
    d.host_stage = nullptr;
    for (int r = 0; r < n; ++r) {
        CUDA_CHECK(cudaSetDevice(d.dev[r]));
        CUDA_CHECK(cudaStreamCreate(&d.stream[r]));
        CUDA_CHECK(cudaEventCreate(&d.start[r]));
        CUDA_CHECK(cudaEventCreate(&d.stop[r]));
        d.recv[r] = nullptr;
    }
    d.last = {0, 0, 0, 0};
}

void ddp_free(DDP &d) {
    for (int r = 0; r < d.n; ++r) {
        CUDA_CHECK(cudaSetDevice(d.dev[r]));
        if (d.recv[r]) cudaFree(d.recv[r]);
        cudaStreamDestroy(d.stream[r]);
        cudaEventDestroy(d.start[r]);
        cudaEventDestroy(d.stop[r]);
    }
    if (d.host_stage) cudaFreeHost(d.host_stage);
    d.host_stage = nullptr;
}

void ddp_allreduce(DDP &d, float **bufs, size_t count) {
    if (d.n == 1) {
        d.last = {0.0, 0.0, 0.0, 0};
        return;
    }

    const size_t per = ceil_div(count, (size_t)d.n);
    if (d.recv_cap < per) {
        for (int r = 0; r < d.n; ++r) {
            CUDA_CHECK(cudaSetDevice(d.dev[r]));
            if (d.recv[r]) CUDA_CHECK(cudaFree(d.recv[r]));
            CUDA_CHECK(cudaMalloc(&d.recv[r], per * sizeof(float)));
        }
        d.recv_cap = per;
    }
    bool need_stage = false;
    for (int i = 0; i < d.n; ++i)
        for (int j = 0; j < d.n; ++j)
            if (i != j && !d.peer[i][j]) need_stage = true;
    if (need_stage && d.stage_cap < per) {
        if (d.host_stage) CUDA_CHECK(cudaFreeHost(d.host_stage));
        CUDA_CHECK(cudaMallocHost(&d.host_stage, per * sizeof(float)));
        d.stage_cap = per;
    }

    cudaEvent_t t0, t1, t2;
    CUDA_CHECK(cudaSetDevice(d.dev[0]));
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));
    sync_all(d);
    CUDA_CHECK(cudaEventRecord(t0, d.stream[0]));

    // ---- phase 1: reduce-scatter ----
    // At step s, rank r sends chunk (r - s) and receives chunk (r - s - 1),
    // adding it into its own copy. After n-1 steps rank r holds the complete
    // sum for chunk (r + 1) mod n.
    for (int s = 0; s < d.n - 1; ++s) {
        for (int r = 0; r < d.n; ++r) {
            const int send_idx = ((r - s) % d.n + d.n) % d.n;
            const int dst = (r + 1) % d.n;
            const Chunk c = chunk_of(count, d.n, send_idx);
            transfer(d, r, bufs[r] + c.off, dst, d.recv[dst], c.len);
        }
        sync_all(d);
        for (int r = 0; r < d.n; ++r) {
            const int recv_idx = ((r - s - 1) % d.n + d.n) % d.n;
            const Chunk c = chunk_of(count, d.n, recv_idx);
            if (c.len == 0) continue;
            CUDA_CHECK(cudaSetDevice(d.dev[r]));
            const int threads = 256;
            const int blocks = (int)(ceil_div(c.len, (size_t)threads) < 1024
                                         ? ceil_div(c.len, (size_t)threads)
                                         : 1024);
            add_into_k<<<blocks, threads, 0, d.stream[r]>>>(bufs[r] + c.off,
                                                            d.recv[r], c.len);
        }
        sync_all(d);
    }
    CUDA_CHECK(cudaSetDevice(d.dev[0]));
    CUDA_CHECK(cudaEventRecord(t1, d.stream[0]));

    // ---- phase 2: all-gather ----
    // Same ring, but the arriving chunk replaces rather than accumulates.
    for (int s = 0; s < d.n - 1; ++s) {
        for (int r = 0; r < d.n; ++r) {
            const int send_idx = ((r - s + 1) % d.n + d.n) % d.n;
            const int dst = (r + 1) % d.n;
            const Chunk c = chunk_of(count, d.n, send_idx);
            transfer(d, r, bufs[r] + c.off, dst, bufs[dst] + c.off, c.len);
        }
        sync_all(d);
    }
    CUDA_CHECK(cudaSetDevice(d.dev[0]));
    CUDA_CHECK(cudaEventRecord(t2, d.stream[0]));
    sync_all(d);

    float ms1 = 0.f, ms2 = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, t0, t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms2, t1, t2));
    d.last.reduce_scatter_ms = ms1;
    d.last.all_gather_ms = ms2;
    d.last.total_ms = ms1 + ms2;
    // 2*(n-1) steps, S/n bytes each.
    d.last.bytes_per_device =
        (size_t)(2 * (d.n - 1)) * per * sizeof(float);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaEventDestroy(t2);
}

const DDPTiming &ddp_timing(const DDP &d) { return d.last; }

void ddp_report_topology(const DDP &d) {
    printf("ddp       %d rank(s) on device(s)", d.n);
    for (int r = 0; r < d.n; ++r) printf(" %d", d.dev[r]);
    bool same = true;
    for (int r = 1; r < d.n; ++r) same = same && (d.dev[r] == d.dev[0]);
    if (same && d.n > 1) printf("  [SIMULATED: one physical GPU]");
    printf("\n");

    if (d.n > 1) {
        printf("links     ");
        bool all_peer = true;
        for (int i = 0; i < d.n; ++i)
            for (int j = 0; j < d.n; ++j)
                if (i != j) all_peer = all_peer && d.peer[i][j];
        if (all_peer)
            printf("peer-to-peer enabled on every pair\n");
        else if (d.any_peer)
            printf("peer-to-peer on some pairs; others staged through host\n");
        else
            printf("NO peer access -- every transfer staged through host memory\n");
    }
}
