# Compact GEMM block: RTX 5080 Laptop

Measured September 12, 2026 on native Windows, RTX 5080 Laptop (60 SMs,
sm_120), MUX off / NVIDIA display inactive, driver 616.64. Built with CUDA
13.3 (`nvcc` 13.3.73) and MSVC. Training uses only this repository's kernels;
cuBLAS remains a benchmark/correctness reference in the separate GEMM test.

## Result

The compact 64x64 block uses four 32x32 warps. The automatic rule selects it
for aligned shapes with `N <= 384 && K == 384`. The previous 64x128 block
remains available through `--compact-block -1`; `0` uses the rule and `1`
forces compact. This is a measured rule for the current model/device, not a
claim of optimal dispatch for arbitrary GPUs or shapes.

Four alternating 500-step runs in one binary, evaluation and sampling off.
Each median uses the 36 reported steps 150, 160, ..., 500. These are samples
of the step timer, not the median of all 351 individual steps in that range.

| Run | Mode | Median ms | Min / max ms | Busy clock range |
|---|---|---:|---:|---|
| [1](../bench/logs/compact_step_pinned_run1.txt) | old | 25.410 | 25.35 / 25.50 | 1162-1192 MHz |
| [2](../bench/logs/compact_step_pinned_run2.txt) | rule | 25.270 | 25.23 / 25.33 | 1162-1192 MHz |
| [3](../bench/logs/compact_step_pinned_run3.txt) | old | 25.415 | 25.35 / 25.47 | 1162-1192 MHz |
| [4](../bench/logs/compact_step_pinned_run4.txt) | rule | 25.280 | 25.20 / 25.38 | 1162-1192 MHz |

Mean of the two run medians: **25.4125 -> 25.275 ms**, a **0.541% time
reduction**, about **162,057 tokens/s** at 4096 tokens per step. All four
runs have 28 busy clock samples and pass the monitor's stable-below-pin
criterion. Peak reported power is 80.78-82.27 W.

Two attempts before repinning are retained as rejected evidence:
[unpinned run 1](../bench/logs/compact_step_run1.txt) and
[unpinned run 2](../bench/logs/compact_step_run2.txt). Their 2062-2407 MHz
clocks make them ineligible for this comparison. They do not enter any
reported result. The temporary 1200 MHz lock was released after measurement.

The original 5,000-step checkpoint is unchanged. These short runs measure
speed and regression behavior, not a new best validation loss.

## Why the smaller tile helps

At output 4096x384, the normal tile launches 192 blocks, or 3.2 blocks per
SM on average; compact launches 384. NN/NT use 112-126 registers per thread
and retain the same four-block residency limit. The deeper grid can fill
slots the old grid leaves empty. Shared storage falls from 25.5 to 17 KiB,
and input arithmetic intensity falls from 21.3 to 16 FLOP/byte. A wider or
deeper multiplication generally loses from the additional input traffic.

`test_gemm --tf32 --block` interleaves both arms in both orders and keeps
the minimum of the two arm timings at each shape. The first sweep covered
NN/TN only. The new sweep also covers NT, the weight layout actually used
by every forward projection. These are unfused microbenchmarks; the
training runs above check the combined effect with fused epilogues.

| Operation | Run 1 gain | Run 2 gain | Run 3 gain |
|---|---:|---:|---:|
| attention projection NN | +7.9% | +9.8% | +8.6% |
| attention projection NT | +9.8% | +11.3% | +10.8% |
| vocab head NT | +24.5% | +25.7% | +26.6% |

Full sweeps, including the losing shapes:
[run 1](../bench/logs/compact_block_nt_run1.txt),
[run 2](../bench/logs/compact_block_nt_run2.txt),
[run 3](../bench/logs/compact_block_nt_run3.txt).
Busy clocks: 1185-1192, 1177-1192, 1177-1192 MHz respectively; seven samples
per run. The [initial NN/TN sweep](../bench/logs/compact_block_ab_5080.txt)
is retained as well. No cuBLAS-relative percentage is inferred from these
tile-to-tile comparisons.

## Reproduce on Windows

From the repository root, build and pin the reference clock:

```powershell
scripts\build.bat test_gemm train_gpt test_grad test_flash
scripts\gpu_clocks.bat lock 1200
scripts\measure.bat bench\test_gemm.exe --tf32 --block
```

Run the training command four times, alternating mode `-1, 0, -1, 0`.
The recorded runs used this warm-up before each arm (the wrapper also
loads the CUDA runtime paths):

```powershell
scripts\measure.bat bench\sgemm.exe -s 4096 -k 10 -w 200 -i 50 --no-verify
scripts\measure.bat bench\train_gpt.exe -n 500 --tf32 --eval 0 --eval-batches 0 --sample 0 --len 0 --compact-block -1
```

Keep the full clock verdict. Exclude an unpinned or moving-clock run rather
than interpreting its speed as a kernel change. Restore normal GPU clocks
when finished:

```powershell
scripts\gpu_clocks.bat unlock
```

## Correctness and a test-harness repair

All GEMM shape/transpose cases, batched GEMM and fused bias-gradient cases
passed in [old](../bench/logs/compact_gemm_check_-1.txt),
[automatic](../bench/logs/compact_gemm_check_0.txt) and
[forced compact](../bench/logs/compact_gemm_check_1.txt) modes.

TF32 directional-derivative checks passed for all 16 parameter tensors and
the full parameter vector at widths
[128](../bench/logs/compact_grad_128.txt),
[256](../bench/logs/compact_grad_256.txt),
[384](../bench/logs/compact_grad_384.txt),
[512](../bench/logs/compact_grad_512.txt), and
[768](../bench/logs/compact_grad_768.txt).
The [30-step training check](../bench/logs/compact_train_check.txt) reports
losses 4.2625, 3.1102, 2.7430, 2.6151 at steps 1, 10, 20, 30. With
`BMB_COVER=1`, every one of training's 15 dispatch tags appears in the
passing test logs, including compact NN/NT and the residual epilogue.

The attention test initially [printed 23 failures and exited with code
zero](../bench/logs/compact_flash_check.txt). That was a harness defect:
under `--tf32`, the backward sweep inherited TF32 forward `out/lse`, but
its per-kernel tolerances budgeted only rounding inside the backward
kernel. FP32 backward variants consequently showed about 4e-4 error
against a 1e-5 threshold. The ragged checks also hard-coded FP32 tolerance
while selecting the TF32 dispatcher.

The repaired test prepares an FP32 forward fixture for the isolated
backward sweep, preserving each component's existing per-kernel tolerance.
It separately checks the composed default forward/backward at the full
training shape and five ragged/short contexts, with the forward rounding
budget added to each backward component's budget. It rejects non-finite
values, counts forward failures, and returns nonzero on any failure.
No attention implementation or per-kernel tolerance was changed.

The repaired suite passes in [FP32](../bench/logs/compact_flash_fp32.txt),
[TF32](../bench/logs/compact_flash_tf32.txt), and at a
[single-token main shape](../bench/logs/compact_flash_single_token.txt).
At the training shape, composed TF32 errors are 4.15e-4 forward and
4.13e-4 / 4.84e-4 / 2.91e-4 for dQ / dK / dV against the FP32 reference.
The isolated FP32 checks retain their 1e-5 bars.
