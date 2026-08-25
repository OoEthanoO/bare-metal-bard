# A CUDA matmul, and a language model built on top of it

A hand-written CUDA SGEMM taken from 1.2% of cuBLAS to **matching it in fp32**
(95%) and **passing it with tensor cores** (117%), then used to train a GPT from
scratch — including a **fused FlashAttention-style kernel** that never
materializes the score matrix, which doubled the context length this 8 GB card
can train. No PyTorch, no cuBLAS, no cuDNN in the training path —
every matmul, layernorm, softmax, attention, GELU, cross-entropy and AdamW
kernel here is written from scratch.

**Hardware:** RTX 4070 Laptop (Ada, sm_89) — 36 SMs, 256 GB/s, 8 GB, 55 W.

**Writeup:** <https://bare-metal-bard.vercel.app>

---

## Results: the matmul

![SGEMM progression](docs/sgemm_bars.svg)

At N=4096, fp32, SM clock pinned to 1200 MHz, CUDA 13.3. Every cell is the
**median of three independent sweeps** — see *Methodology* for why that is not
paranoia:

| # | kernel | GFLOP/s | % of cuBLAS | what changed |
|---|--------|--------:|------------:|--------------|
| 1 | naive | 82.8 | 1.2% | one thread per output element |
| 2 | coalesced | 648.0 | 9.2% | swapped which index maps to `threadIdx.x` |
| 3 | smem | 814.7 | 11.5% | 32×32 shared-memory tile |
| 4 | tile1d | 2600.3 | 36.8% | 8 outputs per thread |
| 5 | tile2d | 5255.4 | 74.3% | 8×8 register tile (outer product) |
| 6 | vectorized | 6244.1 | 88.3% | `float4` loads + transposed A tile |
| 7 | warptile | 6398.9 | 90.5% | block → warp → thread blocking |
| 8 | dbuffer | **6711.6** | **95.0%** | double-buffered SMEM, one barrier/chunk |
| 9 | tensorcore | **8298.9** | **117.4%** | WMMA m16n16k8 TF32 tensor cores |

**100× from first kernel to last.** Kernel 8 reaches 99.5% of cuBLAS at N=1024,
and both top kernels do better still at sizes that divide the 128×128 block tile
evenly, where no thread block is left partly idle: kernel 8 exceeds cuBLAS at
N=1536 (104.0%) and N=6144 (104.2%), and kernel 9 reaches ~121% at both.

These numbers were re-measured on the machine the GPU actually lives in. An
earlier set, taken on a Linux box with CUDA 12.4, is in the git history, and all
nine kernels reproduce across the two machines to within ~1%.

### The tensor-core number, stated honestly

The baseline above is cuBLAS in **true fp32**, which is what cuBLAS does by
default — TF32 has been opt-in since CUDA 11. Kernel 9 uses tensor cores, so it
is not computing the same thing. Both comparisons, at N=4096:

| | GFLOP/s | vs fp32 cuBLAS | vs TF32 cuBLAS |
|---|--------:|---------------:|---------------:|
| cuBLAS, true fp32 (default) | 7069 | 100% | — |
| cuBLAS, TF32 tensor cores | 10505 | 149% | 100% |
| `dbuffer` (mine, fp32) | 6712 | 95.0% | 63.7% |
| `tensorcore` (mine, TF32) | 8287 | 117.4% | 78.9% |

So kernel 9 beats fp32 cuBLAS by 17% and trails cuBLAS's own tensor-core path
by 21%. Quoting only the first would be the flattering half.

TF32 is not free speed: it keeps fp32's 8-bit exponent but only 10 mantissa
bits. Measured normwise error against an fp32 reference goes from **6.2e-08**
for the fp32 kernels to **2.4e-04** — about 4000× more. That is a deliberate
trade, and it is why kernel 9 carries its own tolerance in the registry rather
than loosening the bar for every kernel.

Reproduce the TF32 column with:

```bash
./bench/sgemm --tf32
```

![scaling](docs/sgemm_scaling.svg)

### The compiler optimized the thing it could see

Kernel 9 was sitting at 8064 GF/s until a question about a version number: the
writeup said CUDA 12.5, but 13.3 was also installed. Building the same source
with both, same machine, same pinned clock, N=4096:

| kernel | CUDA 12.5 | CUDA 13.3 | |
|---|--------:|--------:|---|
| naive … dbuffer | — | — | agree within 1% |
| tensorcore | **8064** | **6499** | **−19.4%** |

Eight of nine kernels are indistinguishable. The tensor-core kernel loses a
fifth of its throughput — three runs each, 8049/8051/8064 against
6511/6505/6496 — at *identical* numerical error, so it is still genuinely TF32,
just slower. `ptxas -v` gives the whole story in two lines:

```
12.5:  128 registers, 12 bytes spill stores
13.3:  142 registers,  0 spills
```

nvcc 13.3 spent 14 more registers to eliminate a 12-byte spill. That is a good
trade in isolation and a bad one here, because at 256 threads per block it
crosses an occupancy cliff. Registers are allocated per warp in multiples of 8:

- 128 regs → 4096/warp → 65536/4096 = **16 warps/SM** → 2 resident blocks
- 142 regs → rounds to 144 → 4608/warp → 65536/4608 = **14 warps/SM** → 1 block

Halving resident blocks to avoid twelve bytes of spill. The compiler optimized
what it could see — the spill — and could not see what it cost.

The fix is to say what the kernel needs, rather than hoping the register
allocator infers it:

```cuda
__global__ __launch_bounds__(NUM_THREADS, 2) void tensorcore_kernel(...)
```

The second argument is minimum blocks per SM. Both toolkits then allocate 128
registers and accept the spill, and this is *not* a 13.3 workaround — it is
faster on both:

| | before | after |
|---|--------:|--------:|
| CUDA 12.5 | 8064 (113.9%) | **8250 (116.7%)** |
| CUDA 13.3 | 6499 (91.6%) | **8290 (117.3%)** |

A spilled byte is cheap; a resident block is not. `__launch_bounds__` is how you
tell the compiler which one you are buying.

With the fix in place the two toolkits agree across the board, so the repo
builds with **whichever is newest** and prints which one it used;
`scripts\build.bat --cuda 12.5` pins the older one. Everything published here is
now measured on 13.3.

The lasting lesson is not that one compiler release regressed. It is that a 19%
loss passed every correctness test, every gradient check and every loss curve
without a murmur, and surfaced only because someone asked why a page said 12.5.
Performance regressions are invisible to correctness testing by construction —
the only thing that catches them is measuring on purpose, which is what the
benchmark harness in this repo is for.

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
saturated, at ~25% occupancy — which is what kernel 8 goes after. The cause was
structural rather than in any counter: the global load was immediately followed
by a barrier, so the ~500-cycle DRAM round trip never overlapped anything.
Double buffering keeps two tiles resident so the next chunk's loads fly while
the current one computes, and cuts barriers from two per K-chunk to one.

Warp cycles per issued instruction across the three: **5.75 → 3.08 → 2.85**.

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
| dbuffer | 32 (latency overlapped) | compute / issue |
| tensorcore | 64 (BK=32) | tensor core issue rate |

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
memory    665.0 MB forward activations, 98.1 MB backward scratch
batch     16 x 256 = 4096 tokens/step
speed     75.1 ms/step, 54,516 tokens/s, ~3917 GFLOP/s end to end
          (67.6 ms and 60,568 tok/s with --tf32; see below)
loss      4.174 (= ln 65, uniform guess) -> 1.5138 val at step 2400
```

![training curve](docs/training_curve.svg)

Validation bottoms at **1.5138** around step 2400 and then climbs -- 10.8M
parameters on 1 MB of text overfits well before the LR schedule ends, so the
best-validation checkpoint is the one kept, not the last. Full curve, config and
sample: [`docs/training.md`](docs/training.md).

Sampled from that checkpoint at temperature 0.8:

```
Is carried an old for Sirrah Paris' in Edward's blood,
When I, that remember'd up my soul,
With manner which we say twelve pass into the king,
Off his limbs and unto the wars' garden,
To steal the first alone, in this word,
Such a thought broughts dead, and thou like a doit
To unsistake the tongue of the higher.
```

This run uses the [fused attention](#fused-attention-the-score-matrix-never-exists)
described below. Running the same seed and configuration on the unfused path is
the control:

| | step | tokens/s | best val | at step | final train |
|---|---:|---:|---:|---:|---:|
| fused | **86.2 ms** | **47,531** | **1.5035** | 2400 | 0.6463 |
| unfused | 91.9 ms | 44,574 | 1.5147 | 2400 | 0.6493 |

Both bottom out at step 2400 and land 0.011 nats apart, which is what the
identical-computation claim should look like in practice rather than in theory:
fp32 addition is not associative, so a different summation order gives a
trajectory that differs in the last digits and then diverges chaotically over
5000 steps. What matters is that neither the shape of the curve nor the quality
of the model moved. The fused kernels are faster and smaller, not different.

The end-to-end 3917 GFLOP/s sits below the standalone GEMM peak because a
training step is not all GEMM. Profiling every kernel launch with `ncu` and
aggregating by category — this is the **unfused** path, and it is the
measurement that motivated the fused attention below:

| category | share of kernel time |
|---|---:|
| GEMM (matmul) | 65.8% |
| GEMM (attention, batched) | 14.5% |
| bias add / column reduce | 5.7% |
| attention softmax | 3.5% |
| attention permute | 3.1% |
| GELU | 2.9% |
| residual add | 2.3% |
| layernorm | 2.0% |
| cross-entropy, embeddings, optimizer | 0.2% |
| **all GEMM** | **80.3%** |
| **everything else** | **19.7%** |

(Relative shares. The profiled run includes evaluation forward passes and
`ncu`'s replay inflates absolute times, so only the ratios are meaningful.
Eval passes are forward-only and forward is less GEMM-heavy than backward, so
the true share for a pure training step is if anything slightly higher.)

Two things follow. First, ~20% of the time goes to kernels with arithmetic
intensity below 1 — pure bandwidth work that no amount of matmul tuning
touches; the attention score matrices alone move 25 MB per layer per pass, which
is what a fused FlashAttention-style kernel would eliminate. That became
[the next piece of work](#fused-attention-the-score-matrix-never-exists), and
the three attention rows in this table (14.5 + 3.5 + 3.1 = 21.1%) are what it
went after. Second, even the
80% does not run at the headline number: training's matmuls are far skinnier
than the square N=4096 benchmark (K=384 for the attention projections), and
small K leaves less work to amortize each tile load against.

That last point deserved measuring rather than asserting, so `./bench/sgemm`
now takes `--mnk M,N,K`. At the four shapes the model actually runs, with the
clock pinned:

| shape (M×N×K) | what it is | k7 *(in use)* | k8 | k9 (TF32) |
|---|---|---:|---:|---:|
| 4096×1152×384 | qkv projection | 6112 | 6517 | **7482** |
| 4096×384×384 | attention out | 4068 | 4353 | **6176** |
| 4096×1536×384 | MLP up | 5651 | 6026 | **7396** |
| 4096×384×1536 | MLP down | 4377 | 4658 | **6929** |

Two things fall out, and both are free performance the model is not taking.
**The model's GEMM is built on kernel 7, and kernel 8 beats it at every one of
these shapes** by 5–8% — the double buffering that was worth 6% on square
matrices is worth as much or more here. And **the tensor-core kernel is
1.22–1.58× faster than kernel 7** at these shapes, a wider margin than the 1.30×
it manages at square N=4096, because skinny matmuls punish the fp32 kernel more
than they punish the tensor cores.

Since GEMM is ~80% of a step, the second is worth something like 20% of
end-to-end training time, and TF32 is the precision the hardware was built to
train in. Both were done, and together they took **85.0 ms/step to 67.6 ms**.

Double buffering the model GEMM was the easy half: same tile parameters as
kernel 8, arithmetic untouched (loss and gradient norm identical to every
printed digit), 11.4% off the step.

The tensor cores took two attempts, and the second is the instructive one.
Standalone they are 1.22–1.58x faster at these shapes; wired into training the
first version was **2.4% slower**. Timing the same entry point per transpose
case (`./bench/test_gemm --bench`) said why, and the split was perfectly clean:

| case | first version |
|---|---|
| NN, NT (A not transposed) | 1.10–1.47x — tensor cores win |
| TN, TT (A transposed) | 0.64–0.87x — tensor cores lose |

WMMA wants A stored m-major, so `As[m][k]` forces the transposed case to
scatter four scalars per `float4` read, where the fp32 kernel stores `As[k][m]`
and gets a straight vector copy. **The backward pass computes every weight
gradient as TN**, so the forward won and the backward handed it straight back.

The fix is to store each operand in whichever orientation makes staging a
vector copy and choose the *fragment layout* to match — `col_major` when the
operand arrived transposed. Same data, same `mma`, no scatter. TN went to
0.96–1.33x, TT to 1.13–1.43x, and end-to-end TF32 went from 2.4% slower to
**9.8% faster**.

### Does TF32 cost the model anything?

Full 5000-step runs, same seed and configuration:

| | step | tokens/s | best val | at step |
|---|---:|---:|---:|---:|
| fp32 | 75.1 ms | 54,516 | 1.5138 | 2400 |
| TF32 | **67.6 ms** | **60,568** | 1.5178 | 2800 |

The validation losses differ by 0.004 nats. That is comfortably inside the
run-to-run spread this setup shows from fp32 summation order alone — across
configurations I have measured 1.5035, 1.5056, 1.5138, 1.5147, 1.5178 and
1.5194 on the same data and seed. So the honest reading is that **TF32 buys 10%
of training time and costs nothing measurable in quality**, which is exactly
why every framework defaults to it on Ampere and later.

It stays opt-in here (`--tf32`) rather than becoming the default, because it
changes what the model computes and one 5000-step run on 1 MB of Shakespeare is
not enough evidence to make that choice silently for someone else.

### How much is still on the table

Against cuBLAS's own TF32 path at the same shapes:

| shape | mine | cuBLAS TF32 | |
|---|---:|---:|---:|
| 4096×1152×384 | 7466 | 9192 | 81.2% |
| 4096×1536×384 | 7384 | 9039 | 81.7% |
| 4096×384×384 | 6209 | 8870 | 70.0% |
| 4096×384×1536 | 6949 | 9955 | 69.8% |

The split is by **N**, and the reason looks obvious: a 128-wide block tile gives
only three block columns at N=384, so with ~72 blocks resident the grid is 1.33
waves and the second wave runs a third full. The wide shapes are 4.0 and 5.3
waves and land at ~81%.

Narrowing `BN` to 64 doubles the block columns and is **worse everywhere** —
6145 → 5644 GF/s at 4096×384×384. A 128×64 tile has an arithmetic intensity of
21 FLOP/byte against 128×128's 32, and this far below the card's 43 ridge point
the reuse lost costs more than the tail wasted. So the wave-quantisation story
is true and is not the binding constraint, which is the sort of thing only a
measurement tells you.

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

---

## Fused attention: the score matrix never exists

The profile above says attention costs ~21% of a training step across three
kernels. None of them is badly written — the batched GEMM is the same code that
reaches 90% of cuBLAS. The cost is structural. A `(B, NH, T, T)` score matrix
gets written to global memory and read back, and the softmax over it does about
5 flops per 8 bytes on a card whose ridge point is **43 FLOP/byte**. That is
0.6% of peak no matter how good the kernel is. The only fix is to not have the
intermediate.

A softmax normally needs the whole row before it can emit anything, because it
needs the row max and the row sum. But both are *running* statistics. After
seeing part of a row you hold a partial max `m` and a partial sum `l`, and when
a later block raises the max to `m'`, everything computed so far is corrected by
one factor:

```
m' = max(m, rowmax(S_j))
l' = l * exp(m - m') + rowsum(exp(S_j - m'))
O' = O * exp(m - m') + exp(S_j - m') @ V_j
```

with `O` divided by `l` only at the end. Each score tile lives in registers for
the few instructions it takes to consume it and is then gone. This is
FlashAttention (Dao et al. 2022). Note that it does *more* arithmetic than the
unfused version, not less — the entire win is in what never gets written.

Two more wins fall out of the structure, and on this hardware they matter as
much as the famous one:

- **Causality becomes a loop bound, not a mask.** The unfused path computes all
  `T x T` scores and throws away the upper triangle inside the softmax. A query
  block here simply never visits key blocks past its diagonal, so both attention
  matmuls do half the work.
- **The head permute disappears.** The unfused path pays a full bandwidth-bound
  pass over `3*B*T*C` to make each head's slice contiguous, because a batched
  GEMM needs uniform strides. The fused kernel indexes q/k/v straight out of the
  `(B, T, 3C)` projection — one block owns one head, so the head offset is a
  constant on the row pointer, and the `qkvr` buffer goes away with it.

Backward never stores the score matrix either. It rebuilds the probabilities
from the saved log-sum-exp, which is exact and needs no reductions at all:

```
P[i,j] = exp(S[i,j] - lse[i])
```

Recomputing `S` costs one matmul. Reading a stored score matrix back costs 25 MB
of DRAM per layer. On this card that trade is not close. It is two kernels
rather than one, because `dQ` reduces over keys while `dK` and `dV` reduce over
queries — fusing them would push one of the three through global atomics on a
tensor the size of the activations. **Recompute beats communication**, which is
this repo's whole lesson restated one level up.

### Measured

`B=16, T=256, C=384, NH=6`, SM clock pinned to 1200 MHz, best of 50:

| | unfused | fused | |
|---|--------:|------:|---|
| attention forward | 1.130 ms | **0.333 ms** | 3.40x |
| attention backward | 1.905 ms | **1.597 ms** | 1.19x |
| forward activations | 952.4 MB | **665.0 MB** | -30% |
| backward scratch | 146.0 MB | **98.1 MB** | -33% |

One caveat on that backward ratio, since this repo has already been bitten twice
by a moving denominator. The fused backward takes 1.60 ms on **both** toolkits.
The unfused reference takes 1.83 ms on CUDA 12.5 and 1.91 ms on 13.3, so the
same kernel reads as 1.14x or 1.19x depending on which compiler built the thing
it is being compared against. The fused number is the one that did not move.

Full log, including every tile configuration and the ragged-shape checks:
[`bench/logs/flash_pinned.txt`](bench/logs/flash_pinned.txt).

Accuracy against the attention this repo already trains with: **2.09e-07**
normwise on the output, and 3.00e-07 / 4.19e-07 / 3.26e-07 on dq / dk / dv.
That is fp32 rounding, the same order as the GEMM tests. The full model still
gradient-checks through the fused path, all 16 tensors. Thirty steps each way
give train loss 2.6105 vs 2.6103 and val 2.6234 vs 2.6235 — the two paths
compute the same thing.

```bash
./bench/test_flash            # sweeps every tile config, both directions
./bench/train_gpt --unfused   # the three-kernel path, for comparison
```

### The backward is only 1.19x, and here is why

The forward result is most of the story; the backward is nearly a wash, and it
is worth saying why rather than quoting the forward alone.

The first answer was a guess: the fastest backward config ran 64 threads per
block at 46 KB of shared memory — two blocks per SM, 8% occupancy. So I added a
config with twice the threads. It gained 2.4%, and became the default. `ncu` on
that default explains why the other 97.6% did not arrive:

| | L1/TEX | DRAM | Compute | Occupancy |
|---|---:|---:|---:|---:|
| `flash_fwd` | 71.4% | 40.3% | 38.4% | 16.1% |
| `flash_bwd_kv` | **87.7%** | 23.9% | 35.8% | 16.4% |
| `flash_bwd_q` | **88.2%** | 21.0% | 34.3% | 16.3% |

L1/TEX at 88% with DRAM at 21% is a signature this project has met before —
kernel 6 showed 81% against 15%. Shared memory and L1 share the LSU/MIO
datapath, so this is shared-memory→register traffic and the FMA pipes are
starved. **The same disease as kernel 6, at a different level of the
hierarchy.**

Note what doubling the occupancy actually did: it took L1/TEX *up*, from 71% to
88%, and bought 2.4%. More warps issuing against an already-saturated datapath
is not a fix, it is more queueing.

Counting loads per FMA agrees. The forward spends 12 shared loads on 32 FMAs in
its `P@V` loop (0.375); the backward accumulation spends 16 on 32 (0.5), and its
score loop is worse still. That is also why more threads did not help: a
narrower column tile buys occupancy by loading *more* per unit of arithmetic, so
the two effects very nearly cancel.

The cure is kernel 7's — bigger register tiles — but here it collides with
shared memory, because staging the whole head dimension at once makes the tiles
proportional to `BR`. Chunking the head dimension the way a GEMM chunks `K`
would stop register tiles and blocks-per-SM competing for the same resource, and
DRAM at 21% says there is bandwidth to spare for the re-staging. That is the
next thing to do here.

Reproduce the table with `scripts\profile_flash.bat` (needs elevation — reading
GPU performance counters is admin-only on Windows).

### What the memory actually buys: twice the context

The forward speedup is nice. The memory is the part that changes what the card
can do, because the unfused footprint is *quadratic* in context length and the
fused one is *linear*. Total resident memory at `B=16` (params + grads + Adam
state + activations + backward scratch), measured with `--alloc-only`:

| ctx | fused | unfused |
|----:|------:|--------:|
| 256 | 0.91 GB | 1.23 GB |
| 512 | 1.65 GB | 2.64 GB |
| 1024 | 3.15 GB | 6.42 GB |
| 2048 | **6.13 GB** | 17.94 GB |
| 3072 | 9.12 GB | out of memory |

This card has 8 GB. The unfused path tops out at **ctx 1024**; the fused path
trains at **ctx 2048** — 1118 ms/step, 29,303 tok/s, 3678 GFLOP/s — and the
unfused path would need 17.94 GB to do the same, nearly three times the card.
**Fusing attention doubled the context this GPU can train.**

One caveat found while measuring this, worth knowing on Windows: a successful
`cudaMalloc` is not evidence that a model fits. WDDM will oversubscribe VRAM
into system memory and page, so the 15.7 GB allocation above *succeeded* and
then ran at a crawl. The number that means something is the total against the
card's memory, which is why `train_gpt` now prints it.

## Writeup

A longer writeup -- the profiling story, the measurement problem, the charts --
is a Next.js app under [`site/`](site/), live at
<https://bare-metal-bard.vercel.app> and deployed on Vercel (auto-deploys on
push to `main`). Every number on the
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

Deploying: the Next app lives in `site/`, not the repo root, so the Vercel
project has **Root Directory = `site`**. Everything else is default.

## Build and run

Requires CUDA 12.x. On Ubuntu 26.04 the toolkit needs gcc ≤ 13 as host compiler
(the default is 15); the Makefile sets `-ccbin g++-13`.

```bash
sudo apt install nvidia-cuda-toolkit gcc-13 g++-13
```

On Windows the Makefile does not apply (no `make`, and nvcc drives MSVC rather
than gcc). `scripts\build.bat` is the equivalent, and builds the same targets:

```
scripts\build.bat            REM all targets
scripts\build.bat sgemm      REM just one
```

It calls `vcvars64.bat` itself, so it works from any shell, and it builds with
the **newest complete toolkit installed** — printing which one, because nvcc's
version changes the generated SASS and the cuBLAS beside it is the denominator
of every "% of cuBLAS" number here. To pin an older one deliberately:

```
scripts\build.bat --cuda 12.5 sgemm
```

"Complete" is checked rather than assumed. CUDA 13 ships `crt` and `nvvm` as
separate installer components, so a partial install has an `nvcc` that dies on
the first `#include`; the script skips those instead of defaulting to one. The
ambient `CUDA_PATH` is deliberately ignored — it records whichever installer ran
last, not the newest toolkit present, and trusting it is exactly how this repo
spent a session benchmarking on the wrong one.

Two more CUDA 13 changes worth knowing: the runtime DLLs moved from `bin\` to
`bin\x64\`, and `cudaDeviceProp::clockRate` was removed after being deprecated
through 12.x (`device_query.cu` uses the attribute API, which works on both).
CUDA 12.5 also refuses MSVC newer than 19.40 via a version check in
`host_config.h` — the Build Tools ship 19.44, so the script passes
`-allow-unsupported-compiler`; 13.x supports 19.4x directly.

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

Check the fused attention against the unfused path — accuracy, time and memory,
sweeping every tile configuration in both directions:

```bash
make bench/test_flash && ./bench/test_flash
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

**Every published cell is the median of three independent sweeps**, not one.
Pinning the clock fixes the *within-run* variance; it does not protect against a
bad run. It caught me twice in one sitting. Once a whole sweep came back with
kernels 8 and 9 down 12–15% at N≤1024 — I wrote a paragraph explaining it as
Windows launch overhead, and three later sweeps disagreed by under 1%, so the
explanation was for a phenomenon that did not exist. Once a fused-attention
speedup read 1.84× on unpinned clocks and 1.12× pinned. Both times the error
flattered the result, which is what makes it worth guarding against rather than
apologising for.

`tools/merge_runs.py` takes several sweeps, publishes the per-cell median, and
prints the spread; anything over 3% is flagged as not-a-result-yet. Current
worst spread across the whole table is **1.0%**. The rule this encodes: a number
measured once is a hypothesis.

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
  flash.cu          fused attention: no score matrix, no permute (default)
  gpt.cu            model: allocation, init, forward, backward
  train_gpt.cu      training loop, sampling, data
  reduce.cuh        warp-shuffle block reductions
tools/
  test_gemm.cu      GEMM vs cuBLAS, all transpose combinations
  test_flash.cu     fused vs unfused attention: accuracy, time, memory
  test_grad.cu      directional-derivative gradient check
  device_query.cu   roofline numbers for this GPU
  merge_runs.py     median several sweeps, flag cells that disagree
  flash_memory.py   context-length memory sweep, fused vs unfused
  plot_results.py   CSV -> SVG charts
scripts/
  gpu_clocks.sh     lock/unlock SM clock for reproducible benchmarks
  build.bat         Windows build (nvcc + MSVC), the Makefile's equivalent
  profile_flash.bat ncu counters for the fused kernels; needs elevation
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

1. **Raw `mma.sync`** instead of WMMA, and this is now the *only* candidate
   left rather than a guess. Kernel 9 trails cuBLAS's own TF32 path by ~21%,
   and four attempts to close it from the outside all failed, measured at
   N=4096:

   | attempt | result | |
   |---|---:|---|
   | grouped block scheduling | 8290 → 7677 | −7.4% |
   | bigger tile, 256×128 | 8290 → 6930 | −16.4% |
   | `BK` 32 → 16 | 8290 → 7899 | −4.7% |
   | double buffering | 7899 → 7518 | −4.8% (both `BK`=16) |

   `ncu` reads DRAM 64.6%, compute 36.9%, L2 hit 58%, occupancy 32.8% —
   *neither* counter saturated, which is the latency signature kernel 7 had.
   But unlike kernel 7 it does not respond to the cures. The bigger tile is the
   informative failure: it lifts arithmetic intensity from 32 to 42.7
   FLOP/byte, past the 43 ridge point, and made things 16% worse. So the kernel
   is not bandwidth bound however much the DRAM counter looks like it, and
   double buffering not helping says the latency is not on the global path
   either. What is left is the abstraction: WMMA fixes the fragment layout and
   forces a shared-memory round trip that `mma.sync` + `ldmatrix` would skip.
2. **`cp.async`** for the staging copies, so kernel 8's prefetch bypasses
   registers entirely rather than costing 32 of them per thread.
3. ~~**Fuse attention** (FlashAttention-style)~~ — [done](#fused-attention-the-score-matrix-never-exists).
   Forward is 3.40x and activation memory is down 30%. The backward is only
   1.19x and the profiler says why: shared-memory→register traffic, kernel 6's
   problem at a different level. **Chunk the head dimension** the way a GEMM
   chunks K, so that bigger register tiles and more blocks per SM stop
   competing for the same shared memory. DRAM sits at 21%, so there is
   bandwidth to pay for the re-staging.
4. **Multi-GPU** — started, and the first measurements are in
   [`bench/logs/multigpu_a40.txt`](bench/logs/multigpu_a40.txt). A ring
   all-reduce written from scratch (no NCCL), data-parallel training behind
   `--gpus N`, run on 2x A40. Two findings, neither the expected one:

   **Peer-to-peer was advertised and did not work.** Every `cudaMemcpyPeerAsync`
   returned success, every sync returned success, and the bytes never arrived —
   PCIe ACS/IOMMU misconfiguration on a virtualised host. Silent: the collective
   produced wrong gradients at full speed. `ddp_init` now sends four bytes across
   each enabled pair and checks they land before trusting it.

   **Communication is 6% of a step, and two GPUs are still 1.9x slower than
   one.** The wire is fine; the host loop is not. One process driving both
   devices, with blocking calls inside the step, keeps them from overlapping.
   Giving each rank its own thread helped (159 -> 149 ms) and is not enough.
   "Communication becomes the bottleneck" is the lesson everyone quotes, and it
   is not what the measurement says — which is exactly why it was worth
   measuring. Next: one process per GPU, or a step with no blocking calls.
