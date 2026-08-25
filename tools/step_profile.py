#!/usr/bin/env python3
"""Aggregate an ncu per-kernel dump into where a training step spends its time.

The point of this file is to stop the next optimisation being chosen by
intuition. Every time this project has guessed at a bottleneck it has been
wrong -- occupancy for the fused backward, bandwidth for the tensor-core
kernel, the interconnect for multi-GPU. The profile has been right every time.

usage: python3 tools/step_profile.py bench/logs/step_ncu_fp32.csv [more.csv ...]

ncu replays kernels to collect counters, so absolute times are inflated. Only
the SHARES are meaningful, and shares are what this prints.

Profile with `scripts/profile_step.bat`, which passes `--eval-batches 0`. That
matters: with evaluation on, a forward-only pass over 20 validation batches
runs before and after the two profiled steps, and it swamps everything. The
tell is the launch counts -- forward kernels appearing many times more often
than their own backward kernels means you are profiling evaluation.
"""
import csv
import re
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit(__doc__)

# Substring -> category. First match wins, so order matters: the fused
# attention kernels must be tested before the generic "gemm" catch-all.
CATEGORIES = [
    ("flash_fwd", "attention (fused fwd)"),
    ("flash_bwd_kv", "attention (fused bwd dK/dV)"),
    ("flash_bwd_q", "attention (fused bwd dQ)"),
    ("flash_dsum", "attention (fused bwd rowsum)"),
    ("permute", "attention permute"),
    ("softmax_causal", "attention softmax"),
    ("bgemm", "GEMM (attention, batched)"),
    ("batched", "GEMM (attention, batched)"),
    ("gemm_mma", "GEMM (tensor core, mma.sync)"),
    ("gemm_tc", "GEMM (tensor core, WMMA)"),
    ("gemm_fast", "GEMM (fp32)"),
    ("gemm_generic", "GEMM (ragged fallback)"),
    ("gemm", "GEMM (other)"),
    ("layernorm", "layernorm"),
    ("gelu", "GELU"),
    ("residual", "residual add"),
    ("bias", "bias add / column reduce"),
    ("crossentropy", "cross-entropy"),
    ("softmax", "cross-entropy"),
    ("adamw", "optimizer"),
    ("encoder", "embeddings"),
    ("norm", "grad norm"),
    ("reduce", "reductions"),
    ("add_into", "all-reduce"),
]


def categorise(name):
    low = name.lower()
    for key, cat in CATEGORIES:
        if key in low:
            return cat
    return "other: " + name.split("(")[0][:40]


def load(path):
    """ncu --csv emits banner lines before the header; find the real header."""
    rows, header = [], None
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for line in f:
            if header is None:
                if line.startswith('"ID"') or line.startswith("ID,"):
                    header = next(csv.reader([line]))
                continue
            try:
                rows.append(next(csv.reader([line])))
            except Exception:
                pass
    if header is None:
        sys.exit("no CSV header in %s -- did ncu fail? check the file" % path)

    def col(*names):
        for n in names:
            for i, h in enumerate(header):
                if h.strip().lower() == n:
                    return i
        return None

    i_name = col("kernel name", "kernel")
    i_val = col("metric value")
    i_unit = col("metric unit")
    if i_name is None or i_val is None:
        sys.exit("unexpected ncu columns: %s" % header)

    total = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        if len(r) <= max(i_name, i_val):
            continue
        try:
            v = float(r[i_val].replace(",", ""))
        except ValueError:
            continue
        unit = r[i_unit].strip().lower() if i_unit is not None and len(r) > i_unit else ""
        # Normalise to microseconds so mixed units cannot silently skew shares.
        scale = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0,
                 "msecond": 1e3, "ms": 1e3, "second": 1e6}.get(unit, 1.0)
        cat = categorise(r[i_name])
        total[cat] += v * scale
        counts[cat] += 1
    return total, counts


for path in sys.argv[1:]:
    total, counts = load(path)
    grand = sum(total.values())
    if grand == 0:
        sys.exit("no kernel durations found in %s" % path)
    print("\n%s" % path)
    print("%-34s %10s %8s %8s" % ("category", "time (us)", "share", "launches"))
    print("-" * 64)
    gemm_share = 0.0
    for cat, v in sorted(total.items(), key=lambda kv: -kv[1]):
        share = 100.0 * v / grand
        if cat.startswith("GEMM"):
            gemm_share += share
        print("%-34s %10.1f %7.1f%% %8d" % (cat, v, share, counts[cat]))
    print("-" * 64)
    print("%-34s %10s %7.1f%%" % ("all GEMM", "", gemm_share))
    print("%-34s %10s %7.1f%%" % ("everything else", "", 100.0 - gemm_share))
