#!/usr/bin/env bash
# Bisect the deterministic two-rank anomaly. A40 HOSTS ONLY: it reproduced
# three times out of three there (bench/logs/multigpu_a40_run{3,4,5}*.txt) --
# a 2-rank B=16 40-step run following a 1-rank 40-step run reads step-1 loss
# 4.2701 / |g| 32.750 instead of 4.2783 / 15.023 -- and zero times out of
# eight cases on an A6000 host whose peer links work. So: the host-staged
# transport, plus timing. Each case below changes one thing. Step 1 only.
set -uo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
make -j"$(nproc)" bench/train_gpt >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
./scripts/get_data.sh >/dev/null 2>&1 || true
nvidia-smi --query-gpu=index,name --format=csv,noheader
pre() { ./bench/train_gpt -n 40 --eval 10000 --sample 10000 --len 0 >/dev/null 2>&1; }
subj() { ./bench/train_gpt "$@" --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +(1|10)/|rank losses|post-allreduce|^links" | head -5; }
for i in 1 2 3; do
  pre; echo "--- A$i control: 1-rank n40, then 2-rank n40 (A40 hosts read 4.2701 / 32.750) ---"; subj -n 40 --gpus 2
done
pre; echo "--- B control under CUDA_LAUNCH_BLOCKING=1 (no stream-ordering race can survive this) ---"
CUDA_LAUNCH_BLOCKING=1 subj -n 40 --gpus 2
pre; echo "--- C control, traced ---"; subj -n 40 --gpus 2 --ddp-trace
pre; echo "--- D control with DDP_NO_P2P=1 (already staged on A40 hosts; on others forces it) ---"
DDP_NO_P2P=1 subj -n 40 --gpus 2
pre; echo "--- E control, subject short (n2) ---"; subj -n 2 --gpus 2
echo "--- F no predecessor (fresh after 10 s idle), 2-rank n40 ---"; sleep 10; subj -n 40 --gpus 2
echo "=== done ==="
