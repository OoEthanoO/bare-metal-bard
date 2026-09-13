#!/usr/bin/env bash
# Diagnostic companion to cloud_validate_ddp.sh, NOT a performance gate.
# Run on an already-provisioned two-GPU host with an independent cost watchdog.
# Logs failed cases and continues controls; exits nonzero if ANY case fails.
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
out=${1:-bench/cloud-forward-diagnostics}
mkdir -p "$out"
test -x ./bench/sgemm
test -x ./bench/test_ddp_gpt
failed=0
subject=(./bench/test_ddp_gpt --production-shape --tf32 --ranks 2 --require-multi-gpu --steps 1)
diagnose() {
    local name=$1; shift
    # Every control gets its own predecessor, not the potentially healing
    # process left by the previous control. A failing predecessor is fatal.
    ./bench/sgemm -k 8,9 -s 4096 > "$out/$name-predecessor.txt" 2>&1
    local code=0
    "$@" > "$out/$name.txt" 2>&1 || code=$?
    printf '%s\n' "$code" > "$out/$name.exit"
    printf '%s: exit %s (see %s/%s.txt)\n' "$name" "$code" "$out" "$name"
    if ((code != 0)); then failed=1; fi
}
# Separate forward correctness from backward/collectives, then remove peer
# setup itself. The last control poisons the actual allocations in-process.
diagnose full "${subject[@]}"
diagnose forward "${subject[@]}" --forward-only
diagnose staged_forward env DDP_NO_P2P=1 "${subject[@]}" --forward-only
diagnose staged_full env DDP_NO_P2P=1 "${subject[@]}"
diagnose poisoned_forward "${subject[@]}" --forward-only --poison-scratch
diagnose poisoned_full "${subject[@]}" --poison-scratch
# Do NOT poison under initcheck: that would mark unwritten data initialized.
diagnose initcheck compute-sanitizer --tool initcheck --error-exitcode 99 "${subject[@]}" --forward-only
diagnose memcheck compute-sanitizer --tool memcheck --error-exitcode 99 "${subject[@]}" --forward-only
printf 'Diagnostics complete; any failed case: %s. No performance claim.\n' "$failed"
exit "$failed"
