# Checking the model across ranks

`test_ddp` verifies the ring's elementwise sum. `test_ddp_gpt` verifies the
model around it: input uploads, loss reduction, full-batch gradient
equivalence, replica agreement, gradient averaging, clipping, and AdamW
state. It links the same handwritten CUDA implementation as `train_gpt`,
without cuBLAS, PyTorch, or NCCL.

The default fixture has two layers, width 384, six heads, context 64 and
two sequences per rank. It exercises the model's compact GEMM and fused
attention paths under `--tf32`. A deterministic stream generates new inputs
and next-token targets each step. No dataset, checkpoint or network access
is needed. Parameters and optimizer moments persist for all three steps.

## What a passing check establishes

1. A reference model processes the complete batch. Persistent rank workers
   process contiguous, equal-size shards of that same batch. Device tokens
   and targets are read back and checked exactly. Every token loss is
   compared with the corresponding full-batch loss; each reported scalar
   loss must also match a double-precision host average of its token losses.
2. The production ring sums the shard gradients. All ranks must receive
   byte-identical buffers, and their mean is compared against the
   full-batch reference separately for each of the 16 parameter tensors.
   Every value is examined, and non-finite values fail explicitly.
3. The reference is then given the rank-0 **mean** gradient. Each distributed
   optimizer consumes its original **sum**, applying `1/nranks` internally.
   Updated weights, Adam's first moments, second moments and reported
   gradient norms are compared on every rank. The clip threshold alternates
   between one quarter and twice the host-computed mean-gradient norm, so
   both clipping branches execute.

Step 3 deliberately isolates optimizer normalization from the reduction
comparison in step 2. Small differences in gradients near zero can become
large relative differences after Adam divides by its second moment. Using
the same mathematical gradient for the optimizer check makes a failure
locatable without giving Adam a loose tolerance. The model is not reset
between steps; the updated weights and moments feed the next step.

The worker implementation is shared through `src/rank_pool.h`. Its code
was extracted from the trainer without changing its behavior, so the test
uses the same thread lifetime and thread-local CUDA scratch ownership.

## Numerical limits

Each tensor comparison checks its maximum absolute error against
`atol + rtol * max(abs(reference tensor))`. The printed `gradient_budget`
and `state_budget` are the largest error/limit ratios; at most 1 passes.
This avoids letting a large weight matrix hide an error in a small bias
tensor. It is not an elementwise relative-error test near zero.

| Quantity | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Mean gradient, FP32 | 1e-7 | 1e-4 |
| Mean gradient, TF32 | 1e-7 | 5e-3 |
| Parameters after AdamW | 2e-7 | 2e-6 |
| Adam first moment | 1e-9 | 5e-5 |
| Adam second moment | 1e-11 | 5e-5 |

Per-token loss limits are 2e-5 in FP32 and 2e-3 in TF32. The reported loss
versus its own host mean uses 2e-6 in both modes. Gradients use a wider TF32
budget because changing batch partition can change GEMM reduction order and
where quantization occurs. The optimizer comparison retains the same limits
in both modes.

## Run on Windows

From the repository root:

```powershell
scripts\build.bat test_ddp_gpt
scripts\measure.bat bench\test_ddp_gpt.exe --ranks 2
scripts\measure.bat bench\test_ddp_gpt.exe --tf32 --ranks 3
```

The wrapper locates the installed CUDA runtime. Its clock verdict is not
used here: these are correctness checks and do not require a GPU clock lock.
Run the host-staged transport in a PowerShell process with:

```powershell
$env:DDP_NO_P2P = '1'
scripts\measure.bat bench\test_ddp_gpt.exe --tf32 --ranks 3
Remove-Item Env:DDP_NO_P2P
```

`--ranks` accepts 1 through 8; `--steps` accepts 1 through 20. Ranks are
placed round-robin over available CUDA devices. Repeated devices are
explicitly labeled as a rehearsal, and no multi-GPU speedup is reported.

On a machine with separate GPUs, require distinct devices:

```powershell
scripts\measure.bat bench\test_ddp_gpt.exe --tf32 --ranks 2 --require-multi-gpu
```

This fails with exit code 2 if there are not enough devices. The existing
cloud experiment script now runs this gate before its anomaly hunt and
timings, with default and host-staged transport. FP32-only hardware skips
the TF32 gate. The cloud script changes have not been executed on rented
hardware as part of this Windows run.

## Prove that the check rejects wrong results

```powershell
scripts\measure.bat bench\test_ddp_gpt.exe --tf32 --ranks 2 --inject-gradient-error
```

This reverses every gradient on rank 0 before all-reduce. Its norm is
unchanged, and both ranks still receive the same summed buffer, but the
gradient direction is wrong. The expected result is `DDP MODEL CHECK FAILED`
and exit code 1. It is a deliberate negative control, not a training mode.

Passing on one 5080 does not resolve the historical two-A40 anomaly or
validate physical peer transfers. This test gives that investigation a
per-tensor comparison and failure exit code instead of relying on matching
printed losses and norms.

## Verified on September 12, 2026

Native Windows, RTX 5080 Laptop, MUX off, CUDA 13.3.73, driver 616.64.
No clock pin was applied and no speedup is claimed from these checks.

The [complete test log](../bench/logs/ddp_model_5080.txt) records:

- Ten normal-transport cases: ranks 1, 2, 3, 5 and 8 in both FP32 and TF32.
- Four forced host-staging cases: FP32 with 3 ranks; TF32 with 2, 3 and 5.
- Three updates per case, alternating clipped, unclipped, clipped. All 14
  cases pass. The largest gradient error uses 15.5% of its tolerance budget;
  the largest optimizer-state error uses 1.9% of its budget.
- Both sign-reversal controls fail with exit code 1: FP32 reaches 15,759
  times its gradient error budget and TF32 reaches 325 times its budget.
- The distinct-device guard fails with exit code 2 on this single GPU.

[Memory-access checking](../bench/logs/ddp_model_memcheck_5080.txt) on the
three-rank staged path and
[uninitialized-read checking](../bench/logs/ddp_model_initcheck_5080.txt)
on the three-rank direct rehearsal each report zero errors, with two TF32
updates per run and `--error-exitcode 99` enabled.

The rebuilt production trainer also passes
[30-step single- and two-rank regressions](../bench/logs/ddp_model_trainer_smoke_5080.txt).
Both report loss 4.2625, 3.1102, 2.7430 and 2.6151 at steps 1, 10, 20 and
30; the two-rank device input checks report no mismatches. Their unlocked
timings are smoke-test output, not benchmark results.
