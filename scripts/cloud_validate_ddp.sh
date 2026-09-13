#!/usr/bin/env bash
# Bounded, fail-closed validation of two physical GPUs. Run from the repo root.
# This script does not provision resources or guarantee cloud cleanup; arrange
# an independent watchdog before launching a paid instance.
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
out=${1:-bench/cloud-validation}
mkdir -p "$out"
run() {
    local name=$1; shift
    printf '\n=== %s ===\n' "$name"
    "$@" 2>&1 | tee "$out/$name.txt"
}
run machine nvidia-smi
run topology nvidia-smi topo -m
run toolkit nvcc --version
# Bound host parallelism: each nvcc invocation compiles several large units.
run build make -j4 bench/train_gpt bench/test_ddp_gpt bench/test_ddp bench/test_flash bench/test_grad bench/test_xent bench/sgemm
run crossentropy ./bench/test_xent
run attention ./bench/test_flash
run gradients ./bench/test_grad
run model_fp32 ./bench/test_ddp_gpt --ranks 2 --require-multi-gpu
run model_tf32 ./bench/test_ddp_gpt --tf32 --ranks 2 --require-multi-gpu
run model_staged env DDP_NO_P2P=1 ./bench/test_ddp_gpt --tf32 --ranks 2 --require-multi-gpu
run model_production_fp32 ./bench/test_ddp_gpt --production-shape --ranks 2 --require-multi-gpu
run model_production_tf32 ./bench/test_ddp_gpt --production-shape --tf32 --ranks 2 --require-multi-gpu
run model_production_poisoned ./bench/test_ddp_gpt --production-shape --tf32 --poison-scratch --ranks 2 --require-multi-gpu
run model_production_forward ./bench/test_ddp_gpt --production-shape --tf32 --forward-only --ranks 2 --require-multi-gpu
run ring ./bench/test_ddp
# A predecessor workload used to change the next training run's first loss.
# Compare actual gradients, not hard-coded historical/host-library loss text.
for attempt in 1 2 3; do
    run "predecessor_gemm_$attempt" ./bench/sgemm -k 8,9 -s 4096
    run "after_gemm_$attempt" ./bench/test_ddp_gpt --production-shape --tf32 --ranks 2 --require-multi-gpu
    run "predecessor_ring_$attempt" ./bench/test_ddp
    run "after_ring_$attempt" ./bench/test_ddp_gpt --production-shape --tf32 --ranks 2 --require-multi-gpu
done
run memcheck compute-sanitizer --tool memcheck --error-exitcode 99 ./bench/test_ddp_gpt --tf32 --ranks 2 --require-multi-gpu --steps 2
run initcheck compute-sanitizer --tool initcheck --error-exitcode 99 ./bench/test_ddp_gpt --tf32 --ranks 2 --require-multi-gpu --steps 2
printf '\nALL TWO-DEVICE CORRECTNESS GATES PASSED\n'
