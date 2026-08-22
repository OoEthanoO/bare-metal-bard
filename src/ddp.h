#pragma once
#include <cstddef>
#include <cuda_runtime.h>

// Data-parallel training across GPUs, with a hand-written all-reduce.
//
// Same rule as the rest of this repo: no vendor library in the path. NCCL is to
// collectives what cuBLAS is to matmul, and skipping it is the point -- the
// interesting number is not how fast NVIDIA's ring is, it is where the ring
// stops being free.
//
// THE ALGORITHM. A naive all-reduce sends the whole gradient buffer to one rank,
// sums, and sends it back: 2*(n-1)*S bytes over one link, which serializes on
// that rank's bandwidth. The ring instead splits the buffer into n chunks and
// runs two phases, each of n-1 steps, with every device sending and receiving
// simultaneously:
//
//   reduce-scatter  after n-1 steps, device d owns the fully summed chunk d
//   all-gather      after n-1 steps, every device has every summed chunk
//
// Each step moves S/n bytes per link, so the total per device is
// 2*(n-1)/n * S -- independent of n in the limit, and every link is busy the
// whole time. That is the standard result, and it is why bandwidth-optimal
// collectives look like rings.
//
// WHAT WE ARE ACTUALLY MEASURING. For this model S = 43 MB of gradients per
// step. Whether that is free depends entirely on the link: NVLink moves it in
// well under a millisecond, PCIe gen4 x16 (~25 GB/s peak, less in practice)
// takes several, and a step is ~86 ms. The ratio of those two numbers is the
// whole multi-GPU story, and it is what ddp_timing() reports.

struct DDPTiming {
    double reduce_scatter_ms;
    double all_gather_ms;
    double total_ms;
    size_t bytes_per_device;  // moved by one device across both phases
};

struct DDP {
    int n;                 // number of ranks
    int dev[8];            // CUDA device per rank (may repeat; see below)
    bool peer[8][8];       // whether rank i can write rank j's memory directly
    bool any_peer;         // false => every transfer staged through host memory
    float *recv[8];        // per-rank landing buffer, one chunk plus slack
    size_t recv_cap;
    float *host_stage;     // used only when peer access is unavailable
    size_t stage_cap;
    cudaStream_t stream[8];
    cudaEvent_t start[8], stop[8];
    DDPTiming last;
};

// `devices` may name the SAME physical device more than once. That is not a
// mistake: it makes the entire ring -- chunking, peer copies, the reduction
// kernel, the ordering -- exercisable on one GPU, so the algorithm can be
// verified before renting anything. Only the transport differs.
void ddp_init(DDP &d, int n, const int *devices);
void ddp_free(DDP &d);

// In-place sum-all-reduce of `count` floats. bufs[r] must live on dev[r].
// Every buffer holds the same total afterwards.
void ddp_allreduce(DDP &d, float **bufs, size_t count);

// Timings from the most recent ddp_allreduce.
const DDPTiming &ddp_timing(const DDP &d);

// Human-readable summary of what the links actually are, printed once at
// startup so a run's numbers carry their topology.
void ddp_report_topology(const DDP &d);
