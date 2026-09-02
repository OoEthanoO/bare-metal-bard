#!/usr/bin/env bash
# Run the whole multi-GPU experiment on a rented box, in one command.
#
# Rented GPU-hours cost money and debugging on them costs the most, so
# everything this script runs has already been verified on one GPU with ranks
# sharing a device. What is genuinely untested until now is the wire.
#
#   git clone https://github.com/OoEthanoO/bare-metal-bard && cd bare-metal-bard
#   ./scripts/cloud_multigpu.sh 2>&1 | tee multigpu.log
#
# Then send back multigpu.log. Nothing here writes outside the repo directory.
set -uo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

# Containers usually run as root with no sudo installed, so ask for the tool
# rather than assuming the wrapper.
SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"

log "prerequisites"
# A non-interactive ssh session does not source the image's profile, so the
# toolkit that IS installed at /usr/local/cuda is not on PATH. That cost one
# rented pod: the check below said "no nvcc" on a *-devel image.
export PATH="/usr/local/cuda/bin:$PATH"
missing=""
for t in git make curl nvcc; do
  command -v "$t" >/dev/null || missing="$missing $t"
done
if [ -n "$missing" ]; then
  echo "missing:$missing"
  if command -v apt-get >/dev/null; then
    $SUDO apt-get update -qq >/dev/null 2>&1
    # nvcc comes from the toolkit, not build-essential; if the image lacks it
    # the container was a runtime image and the right fix is a -devel one.
    $SUDO apt-get install -y -qq git make curl >/dev/null 2>&1
  fi
  command -v nvcc >/dev/null || {
    echo "NO nvcc: this image has the CUDA runtime but not the toolkit."
    echo "Redeploy with a *-devel image (e.g. nvidia/cuda:12.6.2-devel-ubuntu22.04)."
    exit 1
  }
fi
echo "ok"

log "machine"
date -u
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,pcie.link.gen.max,pcie.link.width.max \
           --format=csv 2>/dev/null || nvidia-smi
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "GPUs: $NGPU"

log "interconnect topology"
# The single most important line in this log. NV# means NVLink, PIX/PXB/PHB/SYS
# are PCIe paths of decreasing quality, and SYS means it crosses the CPU sockets.
nvidia-smi topo -m 2>/dev/null || echo "(nvidia-smi topo unavailable)"

log "peer access -- read this before anything else"
echo "GeForce cards (4090/3090) have peer-to-peer DISABLED in the driver, so"
echo "every transfer is staged through host memory. Datacenter cards (A40,"
echo "L40S, A100) allow direct PCIe P2P. Which one this is changes what the"
echo "numbers below mean, so it is measured rather than assumed."

for i in $(seq 0 $((NGPU-1))); do
  for j in $(seq 0 $((NGPU-1))); do
    [ "$i" = "$j" ] && continue
    echo -n "  $i -> $j: "
    nvidia-smi topo -p2p r 2>/dev/null | sed -n "$((i+2))p" | awk -v c=$((j+2)) '{print $c}' \
      || echo "?"
  done
done

log "build"
nvcc --version | tail -2
make clean >/dev/null 2>&1
# ARCH is auto-detected from compute_cap; override with `make ARCH=sm_80` if
# the detection ever goes wrong.
# Shown, not piped to tail: nvcc takes a couple of minutes on nine kernels and
# a silent terminal looks like a hang. -j because the box has cores going spare.
echo "compiling with -j$(nproc); nvcc is slow, give it ~2 minutes"
make -j"$(nproc)" bench/test_ddp bench/train_gpt bench/sgemm bench/test_flash bench/test_grad || { echo BUILD FAILED; exit 1; }

log "correctness gate: attention, gradients, and the two-rank config that produced NaN"
# The laptop's GPU was unavailable when the cross-device fixes were made, so
# this box is where they are verified. test_flash and test_grad are the
# single-device gate; the 2-rank B=16 run is the exact configuration whose
# forward returned garbage on rank 1 (loss 9.24 at step 1, |g| 2402, then NaN)
# when the flash kernels' shared-memory opt-in was guarded per process
# instead of per device, and grad_global_norm cached its scratch on device 0.
./bench/test_flash 2>&1 | tail -3
./bench/test_grad 2>&1 | tail -2
./scripts/get_data.sh >/dev/null 2>&1 || true
# One hundred short attempts, because the residual anomaly (one run in
# twenty-one read step-1 loss 4.2701 / |g| 32.75 instead of 4.2783 / 15.023)
# needs a rate, a rank, and a phase. --ddp-trace prints each rank's own loss
# and checksums the gradient across ranks after the all-reduce. Counted
# automatically; only the deviant attempts are printed in full.
ATTEMPTS=${ATTEMPTS:-100}
bad=0
for i in $(seq 1 "$ATTEMPTS"); do
  out=$(./bench/train_gpt -n 2 --gpus 2 --ddp-trace --eval 10000 --sample 10000 --len 0 2>&1         | grep -E "^step +1/|rank losses|post-allreduce" | head -3)
  if echo "$out" | grep -q "loss 4.2783 " && echo "$out" | grep -q "(identical)"; then
    printf '.'
  else
    bad=$((bad+1)); printf '
--- attempt %d DEVIATES ---
%s
' "$i" "$out"
  fi
done
printf '
anomaly hunt: %d of %d attempts deviated from loss 4.2783 / identical checksums
' "$bad" "$ATTEMPTS"

log "single-GPU sanity: the matmul still is what it was"
# Clock pinning needs root and is not always permitted on rented boxes; if it
# fails the numbers are noisier but the multi-GPU ratio still holds.
$SUDO nvidia-smi -lgc 1200 >/dev/null 2>&1 && echo "clock pinned to 1200 MHz" \
  || echo "could not pin clock (fine; ratios still hold, absolutes are noisier)"
./bench/sgemm -k 8,9 -s 4096 2>&1 | tail -3

log "ring all-reduce: correctness, then the wire"
./bench/test_ddp

log "data-parallel training, 1 rank"
./scripts/get_data.sh >/dev/null 2>&1 || true
./bench/train_gpt -n 40 --eval 10000 --sample 10000 --len 0 2>&1 | tail -8

for n in 2 4 8; do
  [ "$n" -gt "$NGPU" ] && continue
  log "data-parallel training, $n ranks"
  ./bench/train_gpt -n 40 --gpus "$n" --eval 10000 --sample 10000 --len 0 2>&1 | tail -10
done

log "the prediction check: the configuration that measured 149 ms on 2x A40"
# bench/logs/multigpu_a40.txt, finding 2: global batch 32 took 77.5 ms on one
# rank and 149.1 on two, with comm at 6% -- the host loop, not the wire. The
# per-step thread spawning that caused it (every thread_local per-device cache
# rebuilt through cudaMallocHost each step) is now fixed with persistent rank
# workers, and the prediction ON RECORD before this script was run: two ranks
# land near one B=16 shard's step time plus comm (~41 ms + comm on A40-class
# cards), not 149. Whatever this prints, it goes in the log next to that
# prediction.
for cfg in "1 16" "1 32" "2 32" "4 32" "4 64"; do
  set -- $cfg
  [ "$1" -gt "$NGPU" ] && continue
  echo "--- $1 rank(s), global batch $2 ---"
  ./bench/train_gpt -n 30 -b "$2" --gpus "$1" --eval 10000 --sample 10000 --len 0 2>&1 \
    | grep -E "^step +30|^links"
done

log "where the two-rank step goes: host-side phase trace, three transports"
# The same 2-rank B=32 configuration three ways. (a) both ranks on ONE device
# (CUDA_VISIBLE_DEVICES=0): the rehearsal, which on the laptop shows ~0.4 ms
# of host overhead -- if this box agrees, the remaining gap is specific to
# driving two devices. (b) two devices, host staging forced. (c) two devices,
# whatever the probe allows. --ddp-trace prints per-rank wall time for issuing
# the forward (until the loss readback returns), issuing the backward, waiting
# for it, and the optimizer, so a host stall shows up as issue time.
echo "--- (a) 2 ranks on ONE device, B=32 ---"
CUDA_VISIBLE_DEVICES=0 ./bench/train_gpt -n 30 -b 32 --gpus 2 --ddp-trace --eval 10000 --sample 10000 --len 0 2>&1   | grep -E "^step +30|^links|trace|rank [01]" | tail -4
echo "--- (b) 2 ranks on two devices, host staging forced, B=32 ---"
DDP_NO_P2P=1 ./bench/train_gpt -n 30 -b 32 --gpus 2 --ddp-trace --eval 10000 --sample 10000 --len 0 2>&1   | grep -E "^step +30|^links|trace|rank [01]" | tail -4
echo "--- (c) 2 ranks on two devices, probe decides, B=32 ---"
./bench/train_gpt -n 30 -b 32 --gpus 2 --ddp-trace --eval 10000 --sample 10000 --len 0 2>&1   | grep -E "^step +30|^links|^ddp  |trace|rank [01]" | tail -6

log "weak scaling: keep the per-GPU batch fixed, grow the global batch"
# Strong scaling (fixed global batch) shrinks each GPU's work until launch
# overhead dominates and says more about the model size than the interconnect.
# Weak scaling holds per-GPU work constant, so any slowdown IS communication.
for n in 1 2 4 8; do
  [ "$n" -gt "$NGPU" ] && continue
  B=$((16 * n))
  echo "--- $n rank(s), global batch $B (16/GPU) ---"
  ./bench/train_gpt -n 30 -b "$B" --gpus "$n" --eval 10000 --sample 10000 --len 0 2>&1 \
    | grep -E "^step +30|^ddp|^links"
done

log "done"
echo "send back this log"
