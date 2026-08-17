#pragma once

// Causal multi-head self-attention.
//
// Buffer shapes (hs = C / NH):
//   qkv    (B, T, 3C)        fused QKV projection output, [q|k|v] along 3C
//   qkvr   (4, B, NH, T, hs) slots 0-2 are q/k/v permuted head-contiguous;
//                            slot 3 is scratch for the head-major attention
//                            output before heads are merged back. Size is
//                            4*B*T*C floats, not 3.
//   att    (B, NH, T, T)     post-softmax weights, saved for backward
//   out    (B, T, C)
//
// The permute exists because the qkv projection produces heads interleaved
// along the channel axis, while the per-head matmuls need each head's (T, hs)
// slice contiguous and identically strided. Paying one bandwidth-bound pass to
// rearrange turns 96 awkward strided problems into one batched GEMM.
void attention_forward(float *out, float *qkvr, float *att, const float *qkv,
                       int B, int T, int C, int NH);

// dpreatt may alias datt; they are used in sequence, not simultaneously.
void attention_backward(float *dqkv, float *dqkvr, float *datt, float *dpreatt,
                        const float *dout, const float *qkvr, const float *att,
                        int B, int T, int C, int NH);
