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
| step time | 75.1 ms |
| tokens/s | 54,516 |
| end-to-end | 3,917 GFLOP/s |

The end-to-end figure counts `6*N*P` for the parameter matmuls plus the attention terms. It sits below the standalone GEMM peak (~6420 GFLOP/s) because a training step is not all GEMM: layernorm, softmax, GELU and the optimizer are all bandwidth-bound work at arithmetic intensity below 1. The attention score matrices no longer cost anything: the fused kernel keeps them in registers and never writes them to memory.

## Loss curve

![training curve](training_curve.svg)

| step | val loss |
|---:|---:|
| 100 | 2.5259 |
| 500 | 1.8929 |
| 900 | 1.7072 |
| 1300 | 1.6013 |
| 1700 | 1.5605 |
| 2100 | 1.5254 |
| 2500 | 1.5190 |
| 2900 | 1.5304 |
| 3300 | 1.5620 |
| 3700 | 1.6304 |
| 4100 | 1.7116 |
| 4500 | 1.8183 |
| 4900 | 1.9330 |

Best validation loss **1.5138** at step 2400.

```
final  train loss 0.6344   val loss 1.9758   (20 batches each)
best   val loss 1.5138 at step 2400 (checkpoint saved there)
final sample:
```

A uniform guess over the 65-character vocabulary would score ln(65) = 4.174 nats/char, which is where training starts.

## Sample

Generated from the trained model at temperature 0.8:

```
What say you?' lay down by me?

CLARENCE:
O, I will; and so heavens it too!

CLARENCE:
O, do not srew: in the king, may make is purged.

GLOUCESTER:
Must he needs die like to-morrow?

BUCKINGHAM:
Lord mayor, may strike, take to your hand.

GLOUCESTER:

HASTINGS:
To, pardon the cause; I can read him to-night.

GLOUCESTER:
I see those be patient.

BUCKINGHAM:
Then know, is King Henry, Escape?

LADY
```

Reproduce with:

```bash
./bench/train_gpt --load bench/gpt.bin --len 1000 --temp 0.8
```
