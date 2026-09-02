#!/usr/bin/env bash
# Bisect the deterministic two-rank anomaly, on a rented box, in one command.
#
# What is known (bench/logs/multigpu_a40_run{3,4,5}*.txt): a 2-rank B=16 run
# of 40 steps that follows a 1-rank 40-step run reads step-1 loss 4.2701 and
# |g| 32.750 -- the same digits in three separate sessions -- where a clean
# run reads 4.2783 / 15.023 (rank losses 4.2873 / 4.2693). One hundred and
# fifteen short two-rank runs never showed it. `steps` is used nowhere before
# the first forward, so the step count is not the cause by itself; the
# suspects are the predecessor process (a trained model's leftovers in
# device memory, read before being written), the trace flag, and needing
# two real devices. Each case below changes one thing. Step 1 only.
set -uo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
make -j"$(nproc)" bench/train_gpt >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
./scripts/get_data.sh >/dev/null 2>&1 || true
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "GPUs: $NGPU"

run() {  # run <label> <predecessor args...> -- <subject args...>
  local label="$1"; shift
  local pre=() sub=()
  while [ "$1" != "--" ]; do pre+=("$1"); shift; done; shift
  sub=("$@")
  [ "${#pre[@]}" -gt 0 ] && ./bench/train_gpt "${pre[@]}" --eval 10000 --sample 10000 --len 0 >/dev/null 2>&1
  echo "--- $label ---"
  ./bench/train_gpt "${sub[@]}" --eval 10000 --sample 10000 --len 0 2>&1 \
    | grep -E "^step +1/|rank losses|post-allreduce" | head -3
}

run "A control: 1-rank n40, then 2-rank n40 (expect 4.2701 / 32.750)" \
    -n 40 -- -n 40 --gpus 2
run "B same, subject traced (rank losses + checksum)" \
    -n 40 -- -n 40 --gpus 2 --ddp-trace
run "C same predecessor, subject is SHORT (n2)" \
    -n 40 -- -n 2 --gpus 2
run "D predecessor is a 2-rank n2, subject n40" \
    -n 2 --gpus 2 -- -n 40 --gpus 2
run "E predecessor 1-rank n2 (untrained), subject n40" \
    -n 2 -- -n 40 --gpus 2
./bench/train_gpt -n 40 --eval 10000 --sample 10000 --len 0 >/dev/null 2>&1
echo "--- F predecessor 1-rank n40, subject 2-rank n40 with both ranks on ONE device ---"
CUDA_VISIBLE_DEVICES=0 ./bench/train_gpt -n 40 --gpus 2 --ddp-trace --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +1/|rank losses|post-allreduce" | head -3
run "G no predecessor at all (sleep 5), subject n40" \
    -- -n 40 --gpus 2 --ddp-trace
run "H predecessor 1-rank n40 B=32, subject 2-rank n40 B=32" \
    -n 40 -b 32 -- -n 40 -b 32 --gpus 2 --ddp-trace
echo "=== done ==="
