# Training a GPT on hand-written CUDA kernels

Every kernel in the forward and backward pass is in this repo. The training binary does not link cuBLAS, cuDNN, or any other vendor math library — check the `bench/train_gpt` rule in the Makefile.

## Configuration

```
data      1003855 train / 111539 val tokens, vocab 65 (padded 128)
model     6 layers, 6 heads, 384 embd, ctx 256
params    10.80M (41.2 MB; +41.2 MB grads, +82.4 MB adam state)
memory    952.4 MB forward activations, 146.0 MB backward scratch
batch     16 x 256 = 4096 tokens/step
```

## Throughput

| metric | median over 501 logged steps |
|---|---:|
| step time | 88.5 ms |
| tokens/s | 46,261 |
| end-to-end | 3,324 GFLOP/s |

The end-to-end figure counts `6*N*P` for the parameter matmuls plus the attention terms. It sits below the standalone GEMM peak (~6420 GFLOP/s) because a training step is not all GEMM: layernorm, softmax, GELU, the attention permutes and the optimizer are all bandwidth-bound work at arithmetic intensity below 1, and the attention score matrices alone move 25 MB per layer per pass.

## Loss curve

![training curve](training_curve.svg)

| step | val loss |
|---:|---:|
| 250 | 2.2121 |
| 500 | 1.8965 |
| 750 | 1.7542 |
| 1000 | 1.6774 |
| 1250 | 1.6186 |
| 1500 | 1.5736 |
| 1750 | 1.5464 |
| 2000 | 1.5243 |
| 2250 | 1.5228 |
| 2500 | 1.5193 |
| 2750 | 1.5157 |
| 3000 | 1.5500 |
| 3250 | 1.5602 |
| 3500 | 1.6114 |
| 3750 | 1.6493 |
| 4000 | 1.6978 |
| 4250 | 1.7593 |
| 4500 | 1.8364 |
| 4750 | 1.9210 |
| 5000 | 1.9931 |

Best validation loss **1.5157** at step 2750.

```
final  train loss 0.6285   val loss 1.9931   (20 batches each)
best   val loss 1.5157 at step 2750 (checkpoint saved there)
final sample:
```

A uniform guess over the 65-character vocabulary would score ln(65) = 4.174 nats/char, which is where training starts.

## Sample

Generated from the trained model at temperature 0.8:

```
Shall be fearful tenant guilty than merry
Yield-kerchange; that God, makes this day of thee,
And take the honourable to the other tender
Who professes and the dukes at a fair horse.
So, my child, earls, and my death, hath the world
That is not Rome deposed of the king's King of Night
See, every messages: and, look for life
One ribbon gross toward the side. Bring is sprang Tybalt's
And tell him to Coventry, wherein what he tells
With nothing sunshine for mhe wanton'd reigns,
But that my servant knew but the wind?
Go, indeed, sirrah, that I set aGremio;
How if I cannot tell you, since did I dream?
For I have must try soundly, I see thee Duke of York.

GLOUCESTER:
I saw shive power to her little of you.

YORK:
O uncle, Oath-bed, my liege; his majesty
Gives my sceptre from my own gage:
God my let me and fair well-apparel shake him!

KING EDWARD IV:
Why, so youthless are you on my wisdom,
And soon put your enemies, that you can
The heavens you on your accursed of the gold
At painting your a
```

Reproduce with:

```bash
./bench/train_gpt --load bench/gpt.bin --len 1000 --temp 0.8
```
