# A passing ring is not a passing distributed model

Status, September 13, 2026: **reproduced, not resolved**. Do not use the
two-A40 training timings as an unconditional correctness or scaling result.

## What the rented host established

The experiment used two NVIDIA A40s, driver 570.195.03 and CUDA 12.6.85.
Their connection was `SYS`, across CPU NUMA nodes, not NVLink. Both advertised
peer-copy directions returned incorrect bytes in the existing 1 MiB probe.
The implementation detected this and staged the entire ring through host
memory. See the [machine record](../bench/logs/a40_ddp_20260913/machine.txt),
[topology](../bench/logs/a40_ddp_20260913/topology.txt) and
[toolkit version](../bench/logs/a40_ddp_20260913/toolkit.txt).

The model checker was built from main `6a782e4`, with the production-shape
fixture added: six layers, width 384, six heads, context 256, eight sequences
per rank, global batch 16. No vendor BLAS is linked into this checker.
`sgemm`, used as a separate predecessor process, does link cuBLAS for its
benchmark comparison; that is not part of training.

| Check, in execution order | Result |
|---|---|
| Attention and analytical/numerical gradients | Pass |
| Compact model, FP32 and TF32 | Pass |
| Compact model, forced host staging | Pass |
| Production model, FP32 and TF32, three updates each | Pass |
| Ring correctness and transfer test | Pass |
| `sgemm -k 8,9 -s 4096`, then production TF32 model | **Fail on the first update** |

The [complete driver log](../bench/logs/a40_ddp_20260913/driver.log) preserves
that order. The script stopped at the first failed gate; later repetitions
and sanitizer gates did not execute.

The passing [production TF32 run](../bench/logs/a40_ddp_20260913/model_production_tf32.txt)
reported first-step loss 4.2631011 for the full batch and 4.2631009 for the
average of the shards. Its per-token losses were identical. The failing
[post-GEMM run](../bench/logs/a40_ddp_20260913/after_gemm_1.txt) instead reported
4.2648859 and 4.2612445, respectively. Both ranks failed at token 1, and all
16 gradient tensors exceeded their existing tolerance. The largest gradient
error used **214.871 times** its tolerance budget.

The old checker stopped scanning a rank's tokens at its first mismatch, so
the printed `max_token_error=1.54e-01` is only a **lower bound**, not the
maximum over the whole batch. The updated checker scans every token, counts
all mismatches and prints the first mismatching input, target and loss.

A follow-up sanitizer command failed to start because `compute-sanitizer`
was not on that SSH shell's path; its exit status was 127, not a sanitizer
verdict. Both the failed launch and its predecessor log are retained. There
is **no A40 sanitizer pass** from this experiment.

At the user's stop request, all raw logs were copied locally and the pod
was terminated. The service then reported zero pods, zero network volumes
and $0/hour current spend. This is zero ongoing cost, not a free experiment.

## What this does not establish

The loss mismatch was observed before the test's all-reduce, but the losses
were read after backward. Backward memory corruption therefore remains a
candidate. Peer setup and its probe also happened before any forward pass;
host-staged gradient exchange alone does not remove those effects.

Repeated failure digits do not exclude a race, nor does a predecessor's
effect prove an uninitialized read. The earlier free-VRAM scrub on the
5080 did not guarantee that the next process reused those physical bytes.
A single-device pass cannot clear a two-device failure on another GPU
architecture. Different C++ normal-distribution implementations also mean
Linux and Windows seed-1337 losses need not match each other.

## Narrower controls now available

`test_ddp_gpt` has three new switches:

- `--production-shape`: the six-layer, context-256 fixture above.
- `--forward-only`: retain the same setup and input/loss comparisons, but
  call no backward, all-reduce or optimizer. Parameters stay unchanged and
  inputs vary between steps. A pass is explicitly only a forward check.
- `--poison-scratch`: before each step, fill the actual allocated activation,
  gradient-scratch and parameter-gradient arenas with NaNs. Parameters and
  optimizer moments are not poisoned. This tests overwrite behavior without
  relying on reuse across processes; it is not an exhaustive memory checker.

On a failed model comparison, the checker reads back parameters, embeddings,
each layer's normalization outputs/statistics, QKV, attention output and
log-sum-exp, residuals, MLP pre/post-activations, final normalization and
logits. It compares the reference's matching contiguous batch shard, using
the reference's own layer strides. Each line reports differing and non-finite
element counts and the largest absolute difference. This locates a boundary
to inspect; it does not prove that boundary caused the corruption.

`DDP_NO_P2P=1` now skips peer capability queries, enablement and copy probes,
as well as staging all transfers. Previously it only changed the transfer
flags **after enabling peer access**. Use a fresh process for this control;
the switch does not revoke mappings enabled by another caller earlier in
the same CUDA context. The default peer path is unchanged.

## Reproduce without turning a failed test into a benchmark

Native Windows, on the installed CUDA toolkit:

```powershell
scripts\build.bat test_xent test_ddp_gpt test_ddp
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_ddp_local.ps1
```

On an already-provisioned two-GPU host, with an independent cost watchdog:

```bash
bash scripts/cloud_validate_ddp.sh bench/cloud-validation
# If that gate fails, collect isolation controls, not performance results:
bash scripts/cloud_diagnose_forward.sh bench/cloud-forward-diagnostics
```

The first script stops on failure. The diagnostic script deliberately records
each subject's exit status and continues its controls, returning nonzero if
any fails. Each control gets a fresh GEMM predecessor. It compares normal,
forward-only, no-peer-setup and poisoned modes, then memory checking and
uninitialized-read checking. Poisoning is **not** combined with initcheck:
that would mark the suspect data initialized and hide the read being tested.
These expanded cloud controls have not yet run on two physical GPUs.

## A separate race caught locally, and fixed

The new production-size scratch-poisoning check failed on the Windows 5080
at step 3. Two rank-1 token losses were wrong, yet **every traced parameter
and activation through logits was byte-identical**, and gradients remained
within tolerance. The [pre-fix model trace](../bench/logs/ddp_poison_before_xent_fix_5080.txt)
preserves this result.

In `softmax_xent_fwd_k`, each thread normalized its own probability entries,
then thread 0 read the target's probability without a block barrier. A target
owned by another warp could still hold its unnormalized exponential. This
explains why the final probabilities and the following backward kernel could
be right while the reported loss was wrong. Initialized memory alone does
not imply correctly synchronized memory.

The fix makes the target probability's owning thread write the loss from its
own normalized value. It adds no block barrier, preserves the existing
probability clamp, and handles vocabularies wider than the 128-thread block.
No throughput improvement is claimed.

`test_xent` uses a CPU-double oracle for loss, probability and gradient,
poisons the output buffers, and gives padding logits a deliberately enormous
value to check that padding is excluded. It exercises vocabularies 1, 31,
32, 33, 65, 127, 128, 129 and 257: boundaries within a warp, across warps,
and across iterations of a thread's loop. The production vocabulary uses
16,384 rows over 64 launches; the others use 4,096 rows over eight launches.

The [old kernel failed](../bench/logs/xent_before_fix_5080.txt) at vocabulary
sizes 65, 128, 129 and 257. At V=65, target 32, it returned loss 2.50000000
instead of 4.62291547, although the final probability was 0.00982411.
The [fixed kernel passes all nine cases](../bench/logs/xent_after_fix_5080.txt),
with maximum loss error below 7e-7, and
[memory-access checking](../bench/logs/xent_memcheck_5080.txt) reports zero errors.

This is **not yet the explanation for the A40 failure**: that failure also
changed gradients. A loss-only race does not account for those differences.
The two-device isolation experiment remains necessary.

## Post-fix Windows verification

RTX 5080 Laptop, driver 616.64, CUDA 13.3.73. No clock lock was applied and
MUX state was not re-measured; these are correctness checks, not timings.

- The [local matrix](../bench/logs/ddp_diagnostics_5080/) passes all nine
  normal cases: compact FP32/TF32, production FP32/TF32, production poisoned
  FP32/TF32, production forward-only with and without poisoning, and compact
  three-rank host staging with poisoning. Each runs three steps. The two
  forward-only cases do not perform optimizer updates.
- The sign-reversed gradient is still rejected with exit 1. Combining that
  injection with forward-only mode is rejected with exit 2. The trace for
  the sign-reversal control shows matching forward values, as expected.
- [Global and shared memory initialization checking](../bench/logs/ddp_diagnostics_initcheck_5080.txt)
  passes two full production TF32 updates with `--error-exitcode 99` and
  reports zero errors. Scratch poisoning was **off** for this check.
- Rebuilt [one-rank](../bench/logs/xent_trainer_ranks1_5080.txt) and
  [two-rank](../bench/logs/xent_trainer_ranks2_5080.txt) trainers both complete
  30 steps, reporting losses 4.2625, 3.1102, 2.7430 and 2.6151 at steps
  1, 10, 20 and 30. Evaluation, sampling and checkpoint writing were disabled.

All local ranks shared one physical GPU. These results do not validate
physical peer transport or close the two-A40 gradient anomaly.
