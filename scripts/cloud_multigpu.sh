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
make -j"$(nproc)" bench/test_ddp bench/train_gpt bench/sgemm || { echo BUILD FAILED; exit 1; }

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
