#!/usr/bin/env bash
# Lock the GPU to a fixed SM clock so benchmark numbers are reproducible.
#
# Why this is necessary: this is a 55W mobile RTX 4070. Left alone it boosts to
# 3105 MHz, then falls back as it hits the power cap -- measured cuBLAS SGEMM
# throughput at N=2048 swung between 7.6 and 12.7 TFLOP/s across runs purely
# from thermal state. That is a 60% swing in the denominator of every "% of
# cuBLAS" claim, which makes unlocked numbers meaningless.
#
# 1200 MHz was chosen empirically: it is the highest round clock the card holds
# without deviating under a sustained dense-GEMM load (verified over a 30s run,
# 39-47W, 68-83C, clock pinned the whole time). A 1500 MHz lock does NOT hold;
# the power cap drags it to ~1320 and it wanders.
set -euo pipefail

CLOCK="${2:-1200}"

case "${1:-status}" in
  lock)
    pkexec nvidia-smi -pm 1
    pkexec nvidia-smi -lgc "${CLOCK},${CLOCK}"
    echo "SM clock locked to ${CLOCK} MHz"
    ;;
  unlock)
    pkexec nvidia-smi -rgc
    echo "SM clock lock released"
    ;;
  status)
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu \
               --format=csv
    ;;
  *)
    echo "usage: $0 {lock [MHz]|unlock|status}" >&2
    exit 1
    ;;
esac
