# A CUDA matmul, and a language model built on top of it

A hand-written CUDA SGEMM taken from 1.2% to **90.6% of cuBLAS**, then used to
train a GPT from scratch. No PyTorch, no cuBLAS, no cuDNN in the training path —
every matmul, layernorm, softmax, attention, GELU, cross-entropy and AdamW
kernel here is written from scratch.

**Hardware:** RTX 4070 Laptop (Ada, sm_89) — 36 SMs, 256 GB/s, 8 GB, 55 W.

---

## Results: the matmul

![SGEMM progression](docs/sgemm_bars.svg)

At N=4096, fp32, SM clock pinned to 1200 MHz:

| # | kernel | GFLOP/s | % of cuBLAS | what changed |
|---|--------|--------:|------------:|--------------|
| 1 | naive | 82.8 | 1.2% | one thread per output element |
| 2 | coalesced | 647.7 | 9.1% | swapped which index maps to `threadIdx.x` |
| 3 | smem | 814.6 | 11.5% | 32×32 shared-memory tile |
| 4 | tile1d | 2576.8 | 36.3% | 8 outputs per thread |
| 5 | tile2d | 5273.8 | 74.4% | 8×8 register tile (outer product) |
| 6 | vectorized | 6260.2 | 88.3% | `float4` loads + transposed A tile |
| 7 | warptile | **6420.4** | **90.6%** | block → warp → thread blocking |

**78× from first kernel to last.** Peaks at 95.1% of cuBLAS at N=1024 and 92.5%
at N=6144.

![scaling](docs/sgemm_scaling.svg)

### The one measurement that mattered

The single most useful number in the project came from the profiler, not from
reading. After kernel 6 I was at 88% and had no idea what was left. `ncu` said:

```
DRAM Throughput          15.1%     <- global memory is not the problem
L1/TEX Cache Throughput  81.4%     <- saturated
Compute (SM) Throughput  54.2%
```

Shared memory and L1 share the same LSU/MIO datapath. 81% L1 with 15% DRAM means
the kernel was bound on **shared-memory→register traffic** — the FMA units were
idle waiting for operands to arrive from SMEM, and no amount of further work on
global memory access would have helped.

Warp tiling cuts SMEM loads per FMA from 0.25 to 0.1875 by adding a blocking
level between the block and the thread. Measured after:

```
L1/TEX Cache Throughput  81.4%  ->  50.0%
```

The kernel is now **latency bound**: neither memory (45%) nor compute (55%) is
saturated, at ~25% occupancy. Double-buffering the global→shared load is the
next lever.

### Why the ridge point explains the whole sequence

The roofline ridge point for this card at 1200 MHz is **43 FLOP/byte** — every
byte read from HBM must feed 43 flops to reach peak. Arithmetic intensity per
kernel:

| kernel | FLOP/byte | bound by |
|---|---:|---|
| naive | 0.25 | DRAM, by ~170× |
| smem | 8 | DRAM |
| tile1d | 16 | DRAM |
| tile2d / vectorized | 32 | approaching compute |
| warptile | 32 (+ less SMEM traffic) | latency |

Every optimization in the list is the same move applied at a different level of
the hierarchy: *load a value once, then spend it on as much arithmetic as
possible before letting it go.*

---

## Results: the language model

A 10.8M-parameter GPT (6 layers, 6 heads, 384 embd, 256 ctx, weight-tied head)
trained on TinyShakespeare at character level, entirely on the kernels in this
repo.

```
params    10.80M (41.2 MB; +41.2 MB grads, +82.4 MB adam state)
memory    952.4 MB forward activations, 146.0 MB backward scratch
batch     16 x 256 = 4096 tokens/step
speed     88.6 ms/step, 46,261 tokens/s, ~3320 GFLOP/s end to end
loss      4.174 (= ln 65, uniform guess) -> 1.5157 val at step 2750
```

![training curve](docs/training_curve.svg)

Validation bottoms at **1.5157** around step 2750 and then climbs -- 10.8M
parameters on 1 MB of text overfits well before the LR schedule ends, so the
best-validation checkpoint is the one kept, not the last. Full curve, config and
sample: [`docs/training.md`](docs/training.md).

Sampled from that checkpoint at temperature 0.8:

```
GLOUCESTER:
He shall not be the curtain: an of your
good chorn shall the gaze on the of your tried
on prison, or brother the other reapt
of one to supper, whom our pride to king
In queen exposed fall outterprised, which in the
good trooping high tune of mortal hour
```

The end-to-end 3320 GFLOP/s sits below the standalone GEMM peak (6420) because a
training step is not all GEMM: layernorm, softmax, GELU, the attention permutes
and the optimizer are all bandwidth-bound work at arithmetic intensity below 1,
and the attention score matrices alone move 25 MB per layer per pass.

### The backward pass is gradient-checked

A falling loss does not verify a backward pass — a dropped term in layernorm's
input gradient still descends, just worse. `tools/test_grad.cu` checks analytic
gradients against finite differences using **directional derivatives**: for each
parameter tensor it steps along `u = g/‖g‖`, so the predicted change is exactly
`‖g‖` and every element contributes coherently. That puts the signal ~3 orders
of magnitude above fp32 forward-pass noise, where a naive per-element check
would be buried in it.

All 16 parameter tensors agree to 1e-5…2e-3 relative. And the check is
*sensitive*: deliberately dropping the `xhat·mean(dxhat·xhat)` term from
layernorm backward makes 14 of 16 tensors fail.

---

## Writeup

A longer writeup -- the profiling story, the measurement problem, the charts --
is a Next.js app under [`site/`](site/), deployed on Vercel. Every number on the
page is generated from `bench/results.csv` and the training log by
`tools/make_site_data.py`, so the page cannot drift from the measurements.

Run it locally:

```bash
npm --prefix site install && npm --prefix site run dev
```

Regenerate the page data after a new benchmark or training run:

```bash
python3 tools/plot_results.py
python3 tools/plot_training.py bench/logs/train_final.log docs/training_curve.svg
python3 tools/make_site_data.py
cp docs/*.svg site/public/
```

Deploying: the Next app lives in `site/`, not the repo root, so set **Root
Directory = `site`** when importing the project on Vercel. Everything else is
default.

## Build and run

Requires CUDA 12.x. On Ubuntu 26.04 the toolkit needs gcc ≤ 13 as host compiler
(the default is 15); the Makefile sets `-ccbin g++-13`.

```bash
sudo apt install nvidia-cuda-toolkit gcc-13 g++-13
```

Fetch the dataset (TinyShakespeare, ~1.1 MB; not committed):

```bash
./scripts/get_data.sh
```

Pin the clock first — see *Methodology* below for why this is not optional:

```bash
./scripts/gpu_clocks.sh lock 1200
```

Benchmark the matmuls against cuBLAS:

```bash
make && ./bench/sgemm
```

Verify the GEMM against cuBLAS (56 shape × transpose combinations,
including the batched kernel used by attention):

```bash
make test
```

Gradient-check the model:

```bash
make bench/test_grad && ./bench/test_grad
```

Train the GPT (about 7.5 minutes for 5000 steps on a 4070 Laptop):

```bash
make gpt && ./bench/train_gpt -n 5000
```

Generate text from a saved checkpoint:

```bash
./bench/train_gpt --load bench/gpt.bin --len 1000 --temp 0.8
```

Regenerate the charts:

```bash
python3 tools/plot_results.py
```

---

## Methodology

**Clocks are pinned, and this matters more than it sounds.** Left alone, this
55 W mobile part boosts to 3105 MHz and then falls back as it hits the power
cap. Measured cuBLAS SGEMM at N=2048 swung between **7.6 and 12.7 TFLOP/s**
across runs on thermal state alone — a 60% swing in the denominator of every
"% of cuBLAS" claim, which would have let me report almost any number I wanted.
Pinning to 1200 MHz (the highest clock that holds under sustained load; 1500
does not) makes runs reproducible: best-vs-median now agrees to under 1%.

**cuBLAS is re-timed immediately after each kernel** in the same process, so
both see comparable thermal state and the ratio cancels what drift remains.

**Correctness uses a normwise bound, not elementwise relative error.** Each GEMM
output is a sum of K products, so where cancellation drives an entry near zero
the rounding noise of the other terms remains — an entry of magnitude 1e-3 can
carry 1e-4 of absolute error while every input was computed correctly. Chasing
that ratio measures cancellation, not correctness. Comparing worst absolute
error against `max|ref|` separates cleanly: reordered fp32 summation lands
around 1e-6, a real bug lands at 1e-1 or worse.

---

## Layout

```
src/
  kernels/          the seven SGEMM stages, one file each, heavily commented
  gemm.cu           transpose-aware GEMM (NN/NT/TN/TT) built on kernel 7
  bgemm.cu          batched GEMM for attention
  nn.cu             layernorm, GELU, softmax+cross-entropy, AdamW, encoder
  attention.cu      causal multi-head attention, forward and backward
  gpt.cu            model: allocation, init, forward, backward
  train_gpt.cu      training loop, sampling, data
  reduce.cuh        warp-shuffle block reductions
tools/
  test_gemm.cu      GEMM vs cuBLAS, all transpose combinations
  test_grad.cu      directional-derivative gradient check
  device_query.cu   roofline numbers for this GPU
  plot_results.py   CSV -> SVG charts
scripts/
  gpu_clocks.sh     lock/unlock SM clock for reproducible benchmarks
```

### Notes on the transpose-aware GEMM

Backprop through `Y = X·Wᵀ` needs `dX = dY·W` and `dW = dYᵀ·X`. Rather than
materializing transposed copies (an extra bandwidth-bound pass over the data),
the transpose folds into the shared-memory staging. The A tile was *already*
stored transposed — that is what makes the per-thread register reads contiguous —
so each of the four cases is just a different index map, and all four stay
coalesced. The only cost difference is whether staging does one vector store or
four scalar stores.

---

## What I'd do next

1. **Double-buffer** the global→shared load, to attack the latency bound that
   kernel 7 ends on.
2. **Tensor cores** (`mma.sync`) — sm_89 has them, and fp32 SGEMM leaves them
   completely unused.
3. **Fuse attention** (FlashAttention-style) — the current implementation
   materializes the full (B, NH, T, T) score matrix, 25 MB per layer, which is
   pure bandwidth that a fused kernel would never spend.
4. **Multi-GPU**, where communication rather than compute becomes the limit.
