#!/usr/bin/env bash
# The two-rank anomaly, third decider. Run 10 (bench/logs/multigpu_a40_run10_decider2.txt):
# it reproduces after sgemm alone, after test_ddp alone, and after
# test_ddp + 1-rank, with the SAME wrong numbers each time -- 4.2748 / 4.2655
# on the two ranks, |g| 32.750 -- and not after sgemm + 1-rank or after
# itself. The trace does not hide it and the all-reduce is consistent, so
# the fault is in the forward, on both ranks, and the stale content it reads
# is binary in effect: fresh memory clean, anything else the same wrong.
#
# So: scrub. tools/gpu_scrub.cu fills every free byte on every device. Zeros
# between predecessor and subject should make it clean; a NaN pattern should
# turn the stale rows into NaN loss, which is a signature no filter can hide.
# Then compute-sanitizer initcheck, with its output kept whole this time.
set -uo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
hostname; nvidia-smi --query-gpu=index,name --format=csv,noheader
make -j"$(nproc)" bench/train_gpt bench/test_ddp bench/sgemm >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
nvcc -O2 -std=c++17 tools/gpu_scrub.cu -o bench/gpu_scrub || { echo SCRUB BUILD FAILED; exit 1; }
./scripts/get_data.sh >/dev/null 2>&1 || true
T2()  { ./bench/train_gpt -n 2 --gpus 2 --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +1/|rank losses" | head -2; }
SG()  { ./bench/sgemm -k 8,9 -s 4096 >/dev/null 2>&1; }

echo "=== 1. reproduce: sgemm then 2-rank (expect 4.2748 / 4.2655) ==="; SG; T2
echo "=== 2. sgemm, scrub with ZEROS, then 2-rank (clean would confirm a stale read) ==="; SG; ./bench/gpu_scrub; T2
echo "=== 3. sgemm, scrub with NaN, then 2-rank (NaN would name the read as unconditional) ==="; SG; ./bench/gpu_scrub nan; T2
echo "=== 4. fresh: 2-rank after itself, then after NaN scrub ==="; T2; ./bench/gpu_scrub nan; T2
echo "=== 5. 1-rank after NaN scrub (does the single-rank path read stale memory too?) ==="
./bench/gpu_scrub nan; ./bench/train_gpt -n 2 --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +1/"
echo "=== 6. compute-sanitizer initcheck, one two-rank step after sgemm, output kept whole ==="
SG
compute-sanitizer --tool initcheck ./bench/train_gpt -n 1 --gpus 2 --eval 10000 --sample 10000 --len 0 > sanitizer.log 2>&1
echo "exit $?; $(wc -l < sanitizer.log) lines"; grep -vE "WARNING: peer|at least one peer" sanitizer.log | head -40
echo "..."; grep -E "ERROR SUMMARY|Uninitialized" sanitizer.log | sort | uniq -c | sort -rn | head -10
echo "=== done ==="
