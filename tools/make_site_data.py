#!/usr/bin/env python3
"""Generate site/app/data.ts from the actual benchmark CSV and training log.

Every number on the writeup page comes from a measured run rather than being
typed in by hand, so the page cannot drift from the results.

usage: python3 tools/make_site_data.py
"""
import csv
import json
import os
import re
from collections import defaultdict

CSV = "bench/results.csv"
CSV_TF32 = "bench/results_tf32.csv"
LOG = "bench/logs/train_final.log"
OUT = "site/app/data.ts"

NOTES = {
    "naive": "one thread per output element, uncoalesced",
    "coalesced": "swapped which index maps to threadIdx.x",
    "smem": "32x32 shared-memory tile",
    "tile1d": "8 outputs per thread (TM=8)",
    "tile2d": "8x8 register tile, outer-product form",
    "vectorized": "float4 loads + transposed A tile",
    "warptile": "block -> warp -> thread blocking",
    "dbuffer": "double-buffered SMEM, one barrier per chunk",
    "tensorcore": "WMMA m16n16k8 TF32 tensor cores",
    "mma": "raw mma.sync PTX, lane-major SMEM, 64x64 warp tile",
}
INTENSITY = {
    "naive": 0.25, "coalesced": 0.25, "smem": 8, "tile1d": 16,
    "tile2d": 32, "vectorized": 32, "warptile": 32, "dbuffer": 32,
    "tensorcore": 64,
}

# ---- benchmark ----
rows, order, cublas = defaultdict(dict), [], {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        name, size = r["kernel"], int(r["size"])
        if name not in order:
            order.append(name)
        rows[name][size] = (float(r["gflops_best"]), float(r["pct_of_cublas"]))
        cublas[size] = float(r["cublas_gflops_best"])

BIG = max(cublas)
kernels = []
for i, name in enumerate(order):
    gf, pct = rows[name][BIG]
    kernels.append({
        "id": i + 1, "name": name, "gflops": gf, "pct": pct,
        "note": NOTES.get(name, ""), "intensity": INTENSITY.get(name, 0),
        "bySize": {str(s): rows[name][s][1] for s in sorted(rows[name])},
    })

# ---- TF32 comparison ----
# cuBLAS SGEMM defaults to true fp32; TF32 is opt-in. Carrying both baselines
# keeps the tensor-core kernel's numbers honest: it beats fp32 cuBLAS but
# trails cuBLAS's own tensor-core path.
tf32 = {}
if os.path.exists(CSV_TF32):
    with open(CSV_TF32) as f:
        for r in csv.DictReader(f):
            tf32.setdefault(r["kernel"], {})[r["size"]] = {
                "gflops": float(r["gflops_best"]),
                "pct": float(r["pct_of_cublas"]),
                "cublasTf32": float(r["cublas_gflops_best"]),
            }

# ---- training ----
train, evals = [], []
step_re = re.compile(r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+).*?([\d.]+) ms\s+(\d+) tok/s\s+(\d+) GFLOP/s")
eval_re = re.compile(r"\[eval\]\s+step\s+(\d+)\s+val loss\s+([\d.]+)")
total_steps = 0
text = open(LOG).read() if os.path.exists(LOG) else ""
for ln in text.splitlines():
    m = step_re.match(ln)
    if m:
        total_steps = int(m.group(2))
        train.append({"step": int(m.group(1)), "loss": float(m.group(3)),
                      "ms": float(m.group(4)), "tok": int(m.group(5)),
                      "gflops": int(m.group(6))})
    m = eval_re.search(ln)
    if m:
        evals.append({"step": int(m.group(1)), "loss": float(m.group(2))})

best = min(evals, key=lambda e: e["loss"]) if evals else {"step": 0, "loss": 0}
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0

# Prefer the sample generated from the BEST-validation checkpoint. The sample
# printed at the end of training comes from the final (overfit) weights, which
# is not the model worth showing.
sample = ""
if os.path.exists("bench/logs/sample_best.txt"):
    s = open("bench/logs/sample_best.txt").read()
    j = s.find("\n\n")
    sample = (s[j:] if j >= 0 else s).strip()
if not sample:
    i = text.rfind("final sample:")
    if i >= 0:
        sample = text[i + len("final sample:"):].strip()

m_best = re.search(r"best\s+val loss\s+([\d.]+) at step (\d+)", text)
m_final = re.search(r"final\s+train loss\s+([\d.]+)\s+val loss\s+([\d.]+)", text)

training = {
    "totalSteps": total_steps,
    "bestVal": float(m_best.group(1)) if m_best else best["loss"],
    "bestStep": int(m_best.group(2)) if m_best else best["step"],
    "finalTrain": float(m_final.group(1)) if m_final else (train[-1]["loss"] if train else 0),
    "finalVal": float(m_final.group(2)) if m_final else 0,
    "msPerStep": med([t["ms"] for t in train]),
    "tokPerSec": med([t["tok"] for t in train]),
    "gflops": med([t["gflops"] for t in train]),
    "curve": [{"step": t["step"], "loss": t["loss"]} for t in train],
    "evals": evals,
    "sample": sample[:2600],
}

# ---- fused attention ----
# Parsed from the benchmark and profiler logs rather than typed in, for the
# same reason as everything else on the page: a number that is retyped is a
# number that can drift from the run it claims to describe.
FLASH_LOG = "bench/logs/flash_pinned.txt"
FLASH_MEM = "bench/flash_memory.csv"
FLASH_NCU = "bench/logs/flash_ncu.txt"


# Anchor on the numeric columns, not on whitespace runs: config names are
# internally padded for alignment ("br32 bc32 t64  kv4x8"), so splitting on
# 2+ spaces would cut them in half.
ROW = re.compile(r"^(?P<name>.*?)\s{2,}(?P<best>[\d.]+)\s+(?P<med>[\d.]+)\s+"
                 r"(?P<gf>[\d.]+)\s+(?P<sp>[\d.]+)x(?P<tail>.*)$")
ERR = re.compile(r"\d\.\d+e[+-]\d+")


def _rows(text, header_key):
    """Rows of one table in test_flash output: from its header to the blank line."""
    out, seen = [], False
    for ln in text.splitlines():
        if not seen:
            seen = ln.startswith(header_key) and "best ms" in ln
            continue
        if not ln.strip():
            break
        m = ROW.match(ln.rstrip())
        # "name  -  (not valid for hs=64)" -- a config this head size cannot run
        if not m:
            continue
        out.append({"name": m.group("name").strip(),
                    "ms": float(m.group("best")),
                    "medMs": float(m.group("med")),
                    "gflops": float(m.group("gf")),
                    "speedup": float(m.group("sp")),
                    "errs": [float(x) for x in ERR.findall(m.group("tail"))]})
    return out


flash = {}
if os.path.exists(FLASH_LOG):
    ftext = open(FLASH_LOG).read()

    def _dir(header_key):
        ref, cfgs = None, []
        for rec in _rows(ftext, header_key):
            if rec["name"].startswith("unfused"):
                ref = rec
            else:
                cfgs.append(rec)
        best = min(cfgs, key=lambda r: r["ms"]) if cfgs else None
        return {"unfused": ref, "best": best, "configs": cfgs}

    m = re.search(r"B=(\d+) T=(\d+) C=(\d+) NH=(\d+)", ftext)
    flash["shape"] = ({"B": int(m.group(1)), "T": int(m.group(2)),
                       "C": int(m.group(3)), "NH": int(m.group(4))} if m else {})
    flash["fwd"] = _dir("impl")
    flash["bwd"] = _dir("backward")

    # Accuracy: worst error across every config, so the figure quoted is the
    # weakest one rather than the most flattering.
    def _worst(rows, i):
        vals = [r["errs"][i] for r in rows if len(r["errs"]) > i]
        return max(vals) if vals else 0

    fwd_rows = [r for r in _rows(ftext, "impl") if r["errs"]]
    bwd_rows = [r for r in _rows(ftext, "backward") if len(r["errs"]) >= 3]
    flash["err"] = {
        "out": _worst(fwd_rows, 0),
        "dq": _worst(bwd_rows, 0),
        "dk": _worst(bwd_rows, 1),
        "dv": _worst(bwd_rows, 2),
    }

    mem = re.search(r"fused saves\s+([\d.]+) MB", ftext)
    flash["savedMbPerLayer"] = float(mem.group(1)) if mem else 0

if os.path.exists(FLASH_MEM):
    by_ctx = defaultdict(dict)
    with open(FLASH_MEM) as f:
        for r in csv.DictReader(f):
            by_ctx[int(r["ctx"])][r["attn"]] = (
                float(r["total_gb"]) if r["total_gb"] else None)
    flash["memory"] = [{"ctx": c, "fused": by_ctx[c].get("fused"),
                        "unfused": by_ctx[c].get("unfused")}
                       for c in sorted(by_ctx)]

if os.path.exists(FLASH_NCU):
    # One block of counters per profiled kernel launch; the launches repeat per
    # layer, so take the first of each kernel.
    ntext = open(FLASH_NCU).read()
    prof, want = [], [("flash_fwd_k", "forward"),
                      ("flash_bwd_kv_k", "backward dK/dV"),
                      ("flash_bwd_q_k", "backward dQ")]
    for sym, label in want:
        i = ntext.find(sym + "<")
        if i < 0:
            continue
        blk = ntext[i:i + 6000]
        def g(pat):
            m = re.search(pat + r"\s+%?\s+([\d.]+)", blk)
            return float(m.group(1)) if m else 0
        prof.append({
            "name": label,
            "l1": g(r"L1/TEX Cache Throughput"),
            "dram": g(r"DRAM Throughput"),
            "compute": g(r"Compute \(SM\) Throughput"),
            "occupancy": g(r"Achieved Occupancy"),
        })
    if prof:
        flash["profile"] = prof

data = {
    "kernels": kernels,
    "tf32": tf32,
    "cublas": cublas[BIG],
    "benchSize": BIG,
    "sizes": sorted(cublas),
    "training": training,
    "flash": flash,
}

hdr = ("// GENERATED by tools/make_site_data.py -- do not edit.\n"
       "// Numbers come from bench/results.csv and bench/logs/train_final.log.\n\n")
body = "export const data = " + json.dumps(data, indent=2) + " as const;\n"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(hdr + body)
print("wrote", OUT)
print("kernels: %d, best %s %.1f GF/s (%.1f%%)"
      % (len(kernels), kernels[-1]["name"], kernels[-1]["gflops"], kernels[-1]["pct"]))
print("training: %d steps, best val %.4f @ %d, %.0f tok/s"
      % (training["totalSteps"], training["bestVal"], training["bestStep"],
         training["tokPerSec"]))
