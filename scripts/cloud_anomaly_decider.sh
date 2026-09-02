#!/usr/bin/env bash
# The deciding experiment for the host-correlated anomaly, on ONE pod in the
# datacenter that reproduced it (README, multi-GPU item): first the isolated
# sequence and its one-variable variants, then the exact prefix of
# cloud_multigpu.sh that read 4.2701 / 32.750 three sessions running. If the
# isolated sequence reproduces here, it is the host class and not the script;
# if only the prefix does, it is something the prefix leaves behind.
set -uo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
hostname; nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
make -j"$(nproc)" bench/train_gpt bench/test_ddp bench/sgemm >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
./scripts/get_data.sh >/dev/null 2>&1 || true
pre()  { ./bench/train_gpt -n 40 --eval 10000 --sample 10000 --len 0 >/dev/null 2>&1; }
subj() { ./bench/train_gpt "$@" --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +(1|10)/|rank losses|post-allreduce|^links" | head -5; }

echo "=== part 1: isolated sequence (clean 9/9 on another datacenter's A40) ==="
for i in 1 2 3; do pre; echo "--- A$i control ---"; subj -n 40 --gpus 2; done
pre; echo "--- B CUDA_LAUNCH_BLOCKING=1 ---"; CUDA_LAUNCH_BLOCKING=1 subj -n 40 --gpus 2
pre; echo "--- C traced ---"; subj -n 40 --gpus 2 --ddp-trace

echo "=== part 2: the exact prefix of cloud_multigpu.sh that reproduced it ==="
./bench/sgemm -k 8,9 -s 4096 >/dev/null 2>&1
./bench/test_ddp 2>&1 | grep -E "passed|FAIL|effective"
echo "--- 1 rank, n40 (as in the script) ---"
./bench/train_gpt -n 40 --eval 10000 --sample 10000 --len 0 2>&1 | tail -8 | grep -E "^step +(1|40)/"
echo "--- 2 ranks, n40 (the reproducing section) ---"
./bench/train_gpt -n 40 --gpus 2 --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +(1|10)/|rank losses|^links" | head -5
echo "--- 2 ranks, n40, again, immediately ---"
./bench/train_gpt -n 40 --gpus 2 --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +(1|10)/|rank losses|^links" | head -5
echo "=== done ==="
