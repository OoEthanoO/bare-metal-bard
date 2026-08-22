#!/usr/bin/env python3
"""Measure how attention memory scales with context length, fused vs unfused.

The point of the fused kernel is not only that it is faster. Unfused attention
allocates a (B, NH, T, T) score matrix, so its footprint is QUADRATIC in context
length while the fused path is LINEAR. That is what decides the longest context
the card can train, so it deserves to be measured rather than reasoned about.

train_gpt --alloc-only allocates everything, prints the totals and exits, which
makes this cheap: a real step at ctx 4096 takes seconds, allocation takes
milliseconds.

usage: python3 tools/flash_memory.py [--exe bench/train_gpt] [-b 16]
writes:  bench/flash_memory.csv
"""
import csv
import os
import re
import subprocess
import sys

EXE = "bench/train_gpt"
BATCH = 16
CTX = [256, 512, 1024, 2048, 3072, 4096]
OUT = "bench/flash_memory.csv"

args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "--exe" and i + 1 < len(args):
        EXE = args[i + 1]
    elif a == "-b" and i + 1 < len(args):
        BATCH = int(args[i + 1])

if not os.path.exists(EXE) and os.path.exists(EXE + ".exe"):
    EXE = EXE + ".exe"
if not os.path.exists(EXE):
    sys.exit("no %s -- build it first" % EXE)

PARAMS = re.compile(r"^params\s+([\d.]+)M", re.M)
MEM = re.compile(r"^memory\s+([\d.]+) MB forward activations, ([\d.]+) MB", re.M)
TOTAL = re.compile(r"^total\s+([\d.]+) GB", re.M)


def measure(ctx, fused):
    # CreateProcess on Windows rejects a relative path with forward slashes.
    cmd = [os.path.abspath(EXE), "-t", str(ctx), "-b", str(BATCH), "--alloc-only"]
    if not fused:
        cmd.append("--unfused")
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    # A hard allocation failure is a real data point, not an error: it is the
    # answer to "does this context fit".
    if "out of memory" in out:
        return None
    m_p, m_m, m_t = PARAMS.search(out), MEM.search(out), TOTAL.search(out)
    if not (m_p and m_m and m_t):
        sys.exit("could not parse output for ctx=%d fused=%s:\n%s" % (ctx, fused, out))
    return {
        "ctx": ctx,
        "attn": "fused" if fused else "unfused",
        "params_m": float(m_p.group(1)),
        "acts_mb": float(m_m.group(1)),
        "scratch_mb": float(m_m.group(2)),
        "total_gb": float(m_t.group(1)),
    }


rows = []
for ctx in CTX:
    for fused in (True, False):
        r = measure(ctx, fused)
        if r is None:
            rows.append({"ctx": ctx, "attn": "fused" if fused else "unfused",
                         "params_m": "", "acts_mb": "", "scratch_mb": "",
                         "total_gb": ""})
            print("ctx %5d %-8s out of memory" % (ctx, "fused" if fused else "unfused"))
        else:
            rows.append(r)
            print("ctx %5d %-8s %8.2f GB" % (ctx, r["attn"], r["total_gb"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ctx", "attn", "params_m", "acts_mb",
                                      "scratch_mb", "total_gb"])
    w.writeheader()
    w.writerows(rows)
print("wrote %s" % OUT)
