#!/usr/bin/env python3
"""Render the SGEMM optimization progression from bench/results.csv to SVG.

Writes plain SVG with no plotting dependency. Colors are chosen to read on both
light and dark backgrounds, since the chart is meant for a README and a blog
post and GitHub serves both themes.

usage: python3 tools/plot_results.py [results.csv] [outdir]
"""
import csv
import math
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "bench/results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs"

# Slow -> fast. Deliberately not a red/green ramp: those two are the most
# common colorblind confusion, and the ordering here is already carried by
# position, so the hue only needs to show progression.
RAMP = ["#b3543f", "#c07f3c", "#c9a63c", "#9aa845", "#6a9f5c", "#3f8f74",
        "#2f7d8a", "#2f6f9a", "#5a5aa8", "#7a4fa0"]
CUBLAS = "#8a6fb0"
FG, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#d8d8d8"


def load(path):
    rows = defaultdict(dict)
    order, cublas = [], {}
    with open(path) as f:
        for r in csv.DictReader(f):
            kid, name, size = int(r["kernel_id"]), r["kernel"], int(r["size"])
            if name not in order:
                order.append(name)
            rows[name][size] = (float(r["gflops_best"]), float(r["pct_of_cublas"]))
            cublas[size] = float(r["cublas_gflops_best"])
    return order, rows, cublas


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_chart(order, rows, cublas, size, path):
    """Horizontal bars: GFLOP/s per kernel at one matrix size."""
    W, H = 860, 60 + 38 * len(order) + 60
    L, R = 150, 90                      # margins for labels and value text
    plot_w = W - L - R
    peak = cublas[size]
    # Kernels 8 and 9 exceed cuBLAS at some sizes, so the axis cannot simply be
    # scaled to the cuBLAS line or those bars run off the plot.
    fastest = max(rows[n][size][0] for n in order)
    xmax = max(peak, fastest) * 1.10

    p = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">' % (W, H)]
    p.append('<style>text{fill:%s}.m{fill:%s}'
             '@media(prefers-color-scheme:dark){text{fill:#e8e8e8}.m{fill:#a8a8a8}'
             '.grid{stroke:#3a3a3a}}</style>' % (FG, MUTED))
    p.append('<text x="%d" y="26" font-size="16" font-weight="600">'
             'Hand-written SGEMM vs cuBLAS &#183; N=%d</text>' % (16, size))
    p.append('<text x="%d" y="45" font-size="11.5" class="m">'
             'RTX 4070 Laptop (sm_89), SM clock pinned 1200 MHz, fp32</text>' % 16)

    y0 = 66
    # x gridlines every 1000 GFLOP/s
    step = 1000
    v = 0
    while v <= xmax:
        x = L + plot_w * v / xmax
        p.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d" '
                 'stroke="%s" stroke-width="1"/>' % (x, y0 - 6, x, y0 + 38 * len(order), GRID))
        p.append('<text x="%.1f" y="%d" font-size="10" text-anchor="middle" '
                 'class="m">%d</text>' % (x, y0 + 38 * len(order) + 16, v))
        v += step

    for i, name in enumerate(order):
        gf, pct = rows[name][size]
        y = y0 + 38 * i
        w = plot_w * gf / xmax
        p.append('<text x="%d" y="%.1f" font-size="12.5" text-anchor="end">%s</text>'
                 % (L - 12, y + 17, esc(name)))
        p.append('<rect x="%d" y="%.1f" width="%.1f" height="22" rx="3" fill="%s"/>'
                 % (L, y + 2, max(w, 1.5), RAMP[min(i, len(RAMP) - 1)]))
        p.append('<text x="%.1f" y="%.1f" font-size="11.5" font-weight="600">'
                 '%.0f  <tspan class="m" font-weight="400">(%.1f%%)</tspan></text>'
                 % (L + max(w, 1.5) + 8, y + 17, gf, pct))

    # cuBLAS reference
    xc = L + plot_w * peak / xmax
    p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" '
             'stroke-width="2" stroke-dasharray="5,4"/>'
             % (xc, y0 - 6, xc, y0 + 38 * len(order), CUBLAS))
    p.append('<text x="%.1f" y="%d" font-size="11" font-weight="600" '
             'text-anchor="middle" fill="%s">cuBLAS %.0f</text>'
             % (xc, y0 - 12, CUBLAS, peak))
    p.append('<text x="%d" y="%d" font-size="10.5" class="m">GFLOP/s</text>'
             % (L + plot_w // 2 - 20, y0 + 38 * len(order) + 34))
    p.append("</svg>")
    open(path, "w").write("\n".join(p))
    return path


def line_chart(order, rows, cublas, path):
    """% of cuBLAS across matrix sizes: shows where each kernel holds up."""
    sizes = sorted(cublas)
    W, H = 860, 430
    L, R, TOP, BOT = 62, 150, 62, 52
    pw, ph = W - L - R, H - TOP - BOT

    def X(i):
        return L + pw * i / max(1, len(sizes) - 1)

    ymax = max(100.0, max(rows[n][s][1] for n in order for s in sizes if s in rows[n]))
    ymax = 20.0 * math.ceil(ymax / 20.0)  # round up to a gridline

    def Y(p):
        return TOP + ph * (1 - p / ymax)

    p = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">' % (W, H)]
    p.append('<style>text{fill:%s}.m{fill:%s}'
             '@media(prefers-color-scheme:dark){text{fill:#e8e8e8}.m{fill:#a8a8a8}'
             '.grid{stroke:#3a3a3a}}</style>' % (FG, MUTED))
    p.append('<text x="16" y="26" font-size="16" font-weight="600">'
             'Fraction of cuBLAS achieved, by matrix size</text>')
    p.append('<text x="16" y="45" font-size="11.5" class="m">'
             'higher is better &#183; 100%% = matching cuBLAS SGEMM (fp32)</text>')

    for pct in range(0, int(ymax) + 1, 20):
        y = Y(pct)
        p.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                 'stroke="%s" stroke-width="1"/>' % (L, y, L + pw, y, GRID))
        p.append('<text x="%d" y="%.1f" font-size="10" text-anchor="end" '
                 'class="m">%d%%</text>' % (L - 8, y + 3.5, pct))
    for i, s in enumerate(sizes):
        p.append('<text x="%.1f" y="%d" font-size="10.5" text-anchor="middle" '
                 'class="m">%d</text>' % (X(i), TOP + ph + 20, s))
    p.append('<text x="%d" y="%d" font-size="10.5" text-anchor="middle" '
             'class="m">matrix size N (N x N x N)</text>' % (L + pw // 2, TOP + ph + 40))

    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
             'stroke-width="1.6" stroke-dasharray="5,4"/>'
             % (L, Y(100), L + pw, Y(100), "#8a6fb0"))
    p.append('<text x="%d" y="%.1f" font-size="10" font-weight="600" fill="%s">'
             'cuBLAS</text>' % (L + 6, Y(100) - 5, "#8a6fb0"))

    for k, name in enumerate(order):
        col = RAMP[min(k, len(RAMP) - 1)]
        pts, last = [], 0
        for i, s in enumerate(sizes):
            if s in rows[name]:
                pct = rows[name][s][1]
                pts.append("%.1f,%.1f" % (X(i), Y(pct)))
                last = pct
        if not pts:
            continue
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.4" '
                 'stroke-linejoin="round"/>' % (" ".join(pts), col))
        for pt in pts:
            x, y = pt.split(",")
            p.append('<circle cx="%s" cy="%s" r="3" fill="%s"/>' % (x, y, col))
        p.append('<text x="%d" y="%.1f" font-size="11.5" fill="%s" '
                 'font-weight="600">%s</text>'
                 % (L + pw + 10, Y(last) + 4, col, esc(name)))
    p.append("</svg>")
    open(path, "w").write("\n".join(p))
    return path


def main():
    order, rows, cublas = load(CSV)
    biggest = max(cublas)
    a = bar_chart(order, rows, cublas, biggest, "%s/sgemm_bars.svg" % OUT)
    b = line_chart(order, rows, cublas, "%s/sgemm_scaling.svg" % OUT)
    print("wrote", a)
    print("wrote", b)


if __name__ == "__main__":
    main()
