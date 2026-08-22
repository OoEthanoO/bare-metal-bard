# Training a GPT on hand-written CUDA kernels

Every kernel in the forward and backward pass is in this repo. The training binary does not link cuBLAS, cuDNN, or any other vendor math library — check the `bench/train_gpt` rule in the Makefile.

## Configuration

```
data      1003855 train / 111539 val tokens, vocab 65 (padded 128)
model     6 layers, 6 heads, 384 embd, ctx 256, fused attention
params    10.80M (41.2 MB; +41.2 MB grads, +82.4 MB adam state)
memory    665.0 MB forward activations, 98.1 MB backward scratch
total     0.91 GB resident (params+grads+adam+activations+scratch)
```

## Throughput

| metric | median over 501 logged steps |
|---|---:|
| step time | 85.7 ms |
| tokens/s | 47,814 |
| end-to-end | 3,435 GFLOP/s |

The end-to-end figure counts `6*N*P` for the parameter matmuls plus the attention terms. It sits below the standalone GEMM peak (~6420 GFLOP/s) because a training step is not all GEMM: layernorm, softmax, GELU and the optimizer are all bandwidth-bound work at arithmetic intensity below 1. The attention score matrices no longer cost anything: the fused kernel keeps them in registers and never writes them to memory.

## Loss curve

![training curve](training_curve.svg)

| step | val loss |
|---:|---:|
| 100 | 2.5258 |
| 500 | 1.8961 |
| 900 | 1.7079 |
| 1300 | 1.6034 |
| 1700 | 1.5570 |
| 2100 | 1.5276 |
| 2500 | 1.5089 |
| 2900 | 1.5209 |
| 3300 | 1.5526 |
| 3700 | 1.6157 |
| 4100 | 1.6962 |
| 4500 | 1.8050 |
| 4900 | 1.9079 |

Best validation loss **1.5056** at step 2400.

```
final  train loss 0.6484   val loss 1.9316   (20 batches each)
best   val loss 1.5056 at step 2400 (checkpoint saved there)
final sample:
```

A uniform guess over the 65-character vocabulary would score ln(65) = 4.174 nats/char, which is where training starts.

## Sample

Generated from the trained model at temperature 0.8:

```
What say you,--

CORIOLANUS:
That would the gods them half by day name them well.
Hear me speak their man with fond thunderous wreaths;
But thus I might be before the murderous knaves.

BRAKENBURY:
I am not yet but done that by your name,
But told you me name with at all their life.

CLARENCE:
Are they gone to make an ambush for the Tower:
If you please to the treason of God
Directing of the suppl
```

Reproduce with:

```bash
./bench/train_gpt --load bench/gpt.bin --len 1000 --temp 0.8
```
