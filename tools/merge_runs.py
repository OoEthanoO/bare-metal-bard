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

    # Two spreads, and the FLAG WATCHES THE MEDIANS, not the bests.
    #
    # This used to flag on the spread of the per-run `best`s, on the reasoning
    # that it is the number that would have caught the bad runs. It would have
    # -- and it also fires when nothing is wrong, because `best` is an
    # extreme-value statistic and one lucky iteration moves it on its own.
    # Measured, warptile N=2048 on a quiet 5080:
    #
    #   per-run best     7964.7  7277.0  7356.6   -> 9.3%   flagged
    #   per-run median   7274.8  7228.4  7205.5   -> 1.0%   fine
    #
    # Six standalone repeats of that cell then read 7403-7500 best and
    # 7347-7390 median, so there was nothing unstable about it. The 9.3% was
    # one fast iteration inside run 1.
    #
    # The two failure modes separate cleanly on which statistic moves. A bad
    # RUN -- the 12-15% low outliers this tool was written for -- is slow in
    # every iteration, so it drags the median down with it. A lucky ITERATION
    # moves only the best. So the median spread is what distinguishes "this
    # cell is not a result yet" from "one iteration got a clean shot at the
    # machine", and the best spread is kept in the output because it is the
    # contention signature: on a machine sharing its GPU with a desktop the
    # two diverge, and that gap is worth seeing rather than hiding.
    m = statistics.median(best)
    spread_best = (max(best) - min(best)) / m * 100.0 if m else 0.0
    mm = statistics.median(med)
    spread = (max(med) - min(med)) / mm * 100.0 if mm else 0.0
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
    # A best spread much wider than the median spread is the contention
    # signature rather than an unstable cell, so it is named as itself.
    if spread_best > 3.0 and spread <= 3.0:
        flag = "  (best spread %.1f%% -- one fast iteration)" % spread_best
    print(f"{name:<12}{size:>6}  median {statistics.median(med):9.1f} GF/s"
          f"  spread {spread:5.1f}%{flag}")

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print(f"\nwrote {out_path} ({len(rows_out)} cells, median of {len(run_paths)} runs)")
print(f"worst median spread: {worst_spread:.1f}% at {worst_cell[0]} N={worst_cell[1]}")
if worst_spread > 3.0:
    print("WARNING: a cell varies more than 3% across runs; do not publish it yet")
