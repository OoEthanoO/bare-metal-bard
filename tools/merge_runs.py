#!/usr/bin/env python3
"""Combine several independent benchmark sweeps into one CSV, and report spread.

WHY THIS EXISTS. bench/sgemm already reports best and median across iterations
*within* a run, which guards against a slow iteration. It does not guard against
a slow RUN. Twice in one session a single sweep produced numbers 12-15% below
what three later sweeps agreed on, and both times the outlier nearly reached the
README -- once it actually did.

So the published CSV is the median across independent runs, and the spread is
printed. A cell whose runs disagree by more than a couple of percent is not a
result yet, it is a measurement that needs repeating.

usage: python3 tools/merge_runs.py out.csv run1.csv run2.csv run3.csv [...]
"""
import csv
import statistics
import sys

if len(sys.argv) < 4:
    sys.exit(__doc__)

out_path, run_paths = sys.argv[1], sys.argv[2:]

# (kernel_id, kernel, size) -> list of rows, one per run
cells = {}
for p in run_paths:
    with open(p) as f:
        for r in csv.DictReader(f):
            cells.setdefault((int(r["kernel_id"]), r["kernel"], int(r["size"])), []).append(r)

fields = ["kernel_id", "kernel", "size", "gflops_best", "gflops_median",
          "cublas_gflops_best", "pct_of_cublas"]

worst_spread, worst_cell = 0.0, None
rows_out = []
for key in sorted(cells):
    kid, name, size = key
    runs = cells[key]
    best = [float(r["gflops_best"]) for r in runs]
    med = [float(r["gflops_median"]) for r in runs]
    cub = [float(r["cublas_gflops_best"]) for r in runs]
    pct = [float(r["pct_of_cublas"]) for r in runs]

    # Spread of the per-run bests, relative to the median. This is the number
    # that would have caught the bad runs.
    m = statistics.median(best)
    spread = (max(best) - min(best)) / m * 100.0 if m else 0.0
    if spread > worst_spread:
        worst_spread, worst_cell = spread, (name, size)

    rows_out.append({
        "kernel_id": kid, "kernel": name, "size": size,
        "gflops_best": f"{statistics.median(best):.2f}",
        "gflops_median": f"{statistics.median(med):.2f}",
        "cublas_gflops_best": f"{statistics.median(cub):.2f}",
        "pct_of_cublas": f"{statistics.median(pct):.2f}",
    })
    flag = "  <-- unstable" if spread > 3.0 else ""
    print(f"{name:<12}{size:>6}  median {statistics.median(best):9.1f} GF/s"
          f"  spread {spread:5.1f}%{flag}")

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print(f"\nwrote {out_path} ({len(rows_out)} cells, median of {len(run_paths)} runs)")
print(f"worst spread: {worst_spread:.1f}% at {worst_cell[0]} N={worst_cell[1]}")
if worst_spread > 3.0:
    print("WARNING: a cell varies more than 3% across runs; do not publish it yet")
