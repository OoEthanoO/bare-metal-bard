#!/usr/bin/env python3
"""Generate docs/training.md from a train_gpt log.

Keeps the writeup's numbers derived from the actual run rather than
transcribed by hand.

usage: python3 tools/make_training_doc.py bench/logs/train_final.log
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "bench/logs/train_final.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/training.md"

text = open(LOG).read()
lines = text.splitlines()

step_re = re.compile(r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+).*?([\d.]+) ms\s+(\d+) tok/s\s+(\d+) GFLOP/s")
eval_re = re.compile(r"\[eval\]\s+step\s+(\d+)\s+val loss\s+([\d.]+)")

steps, evals = [], []
for ln in lines:
    m = step_re.match(ln)
    if m:
        steps.append((int(m.group(1)), float(m.group(3)), float(m.group(4)),
                      int(m.group(5)), int(m.group(6))))
    m = eval_re.search(ln)
    if m:
        evals.append((int(m.group(1)), float(m.group(2))))

header = [ln for ln in lines[:8] if ln.strip() and not ln.startswith("step")]
final = [ln for ln in lines if ln.startswith("final ") or ln.startswith("best ")]

# Last sample block in the log.
sample = ""
idx = text.rfind("final sample:")
if idx >= 0:
    sample = text[idx + len("final sample:"):].strip()
else:
    b = text.rfind("[sample]")
    if b >= 0:
        seg = text[b:].split("\n", 1)[1]
        sample = seg.split("  ------")[0].strip()

med_ms = sorted(s[2] for s in steps)[len(steps) // 2] if steps else 0
med_tok = sorted(s[3] for s in steps)[len(steps) // 2] if steps else 0
med_gf = sorted(s[4] for s in steps)[len(steps) // 2] if steps else 0
best_val = min(evals, key=lambda e: e[1]) if evals else (0, float("nan"))

doc = []
doc.append("# Training a GPT on hand-written CUDA kernels\n")
doc.append("Every kernel in the forward and backward pass is in this repo. "
           "The training binary does not link cuBLAS, cuDNN, or any other "
           "vendor math library — check the `bench/train_gpt` rule in the "
           "Makefile.\n")

doc.append("## Configuration\n")
doc.append("```")
doc.extend(header)
doc.append("```\n")

doc.append("## Throughput\n")
doc.append("| metric | median over %d logged steps |" % len(steps))
doc.append("|---|---:|")
doc.append("| step time | %.1f ms |" % med_ms)
doc.append("| tokens/s | %s |" % f"{med_tok:,}")
doc.append("| end-to-end | %s GFLOP/s |" % f"{med_gf:,}")
doc.append("")
# The bandwidth-bound tail is a different list depending on which attention the
# run used, so read it out of the log rather than asserting one of them.
fused = "fused attention" in text
tail = ("layernorm, softmax, GELU and the optimizer are all bandwidth-bound work "
        "at arithmetic intensity below 1. The attention score matrices no longer "
        "cost anything: the fused kernel keeps them in registers and never writes "
        "them to memory."
        if fused else
        "layernorm, softmax, GELU, the attention permutes and the optimizer are "
        "all bandwidth-bound work at arithmetic intensity below 1, and the "
        "attention score matrices alone move 25 MB per layer per pass.")
doc.append("The end-to-end figure counts `6*N*P` for the parameter matmuls plus "
           "the attention terms. It sits below the standalone GEMM peak "
           "(~6420 GFLOP/s) because a training step is not all GEMM: " + tail + "\n")

doc.append("## Loss curve\n")
doc.append("![training curve](training_curve.svg)\n")
if evals:
    doc.append("| step | val loss |")
    doc.append("|---:|---:|")
    stride = max(1, len(evals) // 12)
    for s, v in evals[::stride]:
        doc.append("| %d | %.4f |" % (s, v))
    doc.append("")
    doc.append("Best validation loss **%.4f** at step %d.\n" % (best_val[1], best_val[0]))

if final:
    doc.append("```")
    doc.extend(final)
    doc.append("```\n")

doc.append("A uniform guess over the 65-character vocabulary would score "
           "ln(65) = 4.174 nats/char, which is where training starts.\n")

doc.append("## Sample\n")
doc.append("Generated from the trained model at temperature 0.8:\n")
doc.append("```")
doc.append(sample)
doc.append("```\n")

doc.append("Reproduce with:\n")
doc.append("```bash")
doc.append("./bench/train_gpt --load bench/gpt.bin --len 1000 --temp 0.8")
doc.append("```")

open(OUT, "w").write("\n".join(doc) + "\n")
print("wrote", OUT)
print("steps parsed: %d, evals: %d, best val %.4f @ %d"
      % (len(steps), len(evals), best_val[1], best_val[0]))
