#!/usr/bin/env bash
# The two-rank anomaly, second decider. Run 9 (bench/logs/multigpu_a40_run9_decider.txt)
# settled the first question: the isolated sequence is clean everywhere, and
# what reproduces it -- 4.2748 / 4.2655 on the two ranks against 4.2873 /
# 4.2693, both ranks wrong, on the same host class the "clean" bisection ran
# on -- is the PREFIX: sgemm, then test_ddp, then a 1-rank run, then the
# 2-rank run. Run the 2-rank run again immediately and it is clean. So
# something in the two-rank process reads device memory before writing it,
# and what it finds there depends on which process ran before. test_ddp is
# the predecessor that touches both devices.
#
# compute-sanitizer's initcheck exists for exactly this: it names the kernel
# and the address of every read of uninitialized device memory. It is slow,
# so it runs one step. Then the predecessor is bisected.
set -uo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
hostname; nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
make -j"$(nproc)" bench/train_gpt bench/test_ddp bench/sgemm >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
./scripts/get_data.sh >/dev/null 2>&1 || true
T2()  { ./bench/train_gpt -n 2 --gpus 2 --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +1/|rank losses" | head -2; }
T1()  { ./bench/train_gpt -n 40 --eval 10000 --sample 10000 --len 0 >/dev/null 2>&1; }
SG()  { ./bench/sgemm -k 8,9 -s 4096 >/dev/null 2>&1; }
DDP() { ./bench/test_ddp >/dev/null 2>&1; }

echo "=== 1. reproduce: sgemm, test_ddp, 1-rank, then 2-rank (expect 4.2748 / 4.2655) ==="
SG; DDP; T1; T2
echo "=== 2. compute-sanitizer initcheck on one two-rank step after the same prefix ==="
SG; DDP; T1
SAN=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null | head -1)
echo "sanitizer: $SAN"
"$SAN" --tool initcheck --track-unused-memory no ./bench/train_gpt -n 1 --gpus 2 --eval 10000 --sample 10000 --len 0 2>&1 \
  | grep -vE "WARNING: peer|at least one peer" | grep -E "Uninitialized|initcheck|at 0x|by thread|in .*_k|Saved host|Host Frame:./bench|^=+ .*[Ee]rror|^step +1/|rank losses|ERROR SUMMARY" | head -60
echo "=== 3. predecessor bisection (2-rank directly after each) ==="
echo "--- after sgemm only ---";     SG; T2
echo "--- after test_ddp only ---";  DDP; T2
echo "--- after test_ddp then 1-rank ---"; DDP; T1; T2
echo "--- after sgemm then 1-rank ---";    SG; T1; T2
echo "--- after 2-rank (self) ---";  T2
echo "=== 4. and the same prefix with the two-rank run traced (does the trace hide it?) ==="
SG; DDP; T1
./bench/train_gpt -n 2 --gpus 2 --ddp-trace --eval 10000 --sample 10000 --len 0 2>&1 | grep -E "^step +1/|rank losses|post-allreduce" | head -3
echo "=== done ==="
