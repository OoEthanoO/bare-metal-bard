#!/usr/bin/env python3
"""Render the training curve from a train_gpt log to SVG.

usage: python3 tools/plot_training.py bench/logs/train_5000.log docs/training_curve.svg
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "bench/logs/train_5000.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/training_curve.svg"

TRAIN = re.compile(r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)")
EVAL = re.compile(r"\[eval\]\s+step\s+(\d+)\s+val loss\s+([\d.]+)")

train, val, total = [], [], 1
for line in open(LOG):
    m = TRAIN.match(line)
    if m:
        train.append((int(m.group(1)), float(m.group(3))))
        total = int(m.group(2))
        continue
    m = EVAL.search(line)
    if m:
        val.append((int(m.group(1)), float(m.group(2))))

if not train:
    sys.exit("no training lines found in %s" % LOG)

W, H = 860, 400
L, R, TOP, BOT = 62, 110, 60, 52
pw, ph = W - L - R, H - TOP - BOT

xmax = max(total, train[-1][0])
ymin = min(min(v for _, v in train), min([v for _, v in val] or [9]))
ymax = max(v for _, v in train)
ymin = max(0.0, ymin - 0.15)
ymax = ymax + 0.1

X = lambda s: L + pw * s / xmax
Y = lambda v: TOP + ph * (1 - (v - ymin) / (ymax - ymin))

TRAIN_C, VAL_C, GRID, FG, MUTED = "#3f8f74", "#b3543f", "#d8d8d8", "#2b2b2b", "#6b6b6b"

p = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
     'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">' % (W, H)]
p.append('<style>text{fill:%s}.m{fill:%s}'
         '@media(prefers-color-scheme:dark){text{fill:#e8e8e8}.m{fill:#a8a8a8}'
         '.grid{stroke:#3a3a3a}}</style>' % (FG, MUTED))
p.append('<text x="16" y="26" font-size="16" font-weight="600">'
         'GPT training on hand-written CUDA kernels</text>')
p.append('<text x="16" y="45" font-size="11.5" class="m">'
         'TinyShakespeare, char-level &#183; 10.8M params &#183; '
         'cross-entropy (nats/char)</text>')

# horizontal gridlines
steps_y = 0.5
v = round(ymin * 2) / 2
while v <= ymax:
    if v >= ymin:
        y = Y(v)
        p.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                 'stroke-width="1"/>' % (L, y, L + pw, y, GRID))
        p.append('<text x="%d" y="%.1f" font-size="10" text-anchor="end" class="m">'
                 '%.1f</text>' % (L - 8, y + 3.5, v))
    v += steps_y

for s in range(0, xmax + 1, max(1, xmax // 5)):
    p.append('<text x="%.1f" y="%d" font-size="10.5" text-anchor="middle" class="m">'
             '%d</text>' % (X(s), TOP + ph + 20, s))
p.append('<text x="%d" y="%d" font-size="10.5" text-anchor="middle" class="m">'
         'step</text>' % (L + pw // 2, TOP + ph + 40))

# ln(vocab): the loss a uniform-random model would achieve
import math
base = math.log(65)
if ymin <= base <= ymax:
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1.2" '
             'stroke-dasharray="4,4"/>' % (L, Y(base), L + pw, Y(base), MUTED))
    p.append('<text x="%d" y="%.1f" font-size="10" class="m">ln(65) = %.2f  '
             '(uniform guess)</text>' % (L + 6, Y(base) - 5, base))

def poly(pts, color, width):
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
            % (" ".join("%.1f,%.1f" % (X(s), Y(v)) for s, v in pts), color, width))

p.append(poly(train, TRAIN_C, "1.6"))
if val:
    p.append(poly(val, VAL_C, "2.4"))
    for s, v in val:
        p.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (X(s), Y(v), VAL_C))

# right-edge labels
p.append('<text x="%d" y="%.1f" font-size="11.5" fill="%s" font-weight="600">'
         'train %.3f</text>' % (L + pw + 8, Y(train[-1][1]) + 4, TRAIN_C, train[-1][1]))
if val:
    p.append('<text x="%d" y="%.1f" font-size="11.5" fill="%s" font-weight="600">'
             'val %.3f</text>' % (L + pw + 8, Y(val[-1][1]) + 4, VAL_C, val[-1][1]))
p.append("</svg>")

open(OUT, "w").write("\n".join(p))
print("wrote", OUT)
print("final train %.4f  final val %.4f  (%d steps)"
      % (train[-1][1], val[-1][1] if val else float("nan"), train[-1][0]))
