#pragma once
#include <cuda_runtime.h>

// Fused causal multi-head attention -- FlashAttention-style.
//
// The unfused path in attention.cu materializes preatt and att, both
// (B, NH, T, T). At B=16, T=256, NH=6 that is 25 MB per layer written and read
// back twice per forward pass, and again in backward. Arithmetic intensity of
// the softmax over it is under 1 FLOP/byte on a card whose roofline ridge point
// is 43, so that traffic is pure loss.
//
// The fused kernel never writes the score matrix to global memory. It walks the
// key/value blocks in a loop, keeping the running softmax statistics (max and
// sum) in registers and rescaling the output accumulator whenever the max
// moves. What leaves the kernel is the (B, T, C) output and one scalar per row:
//
//     lse[b,h,t] = m + log(l)     (log-sum-exp of that row's scores)
//
// which is all the backward pass needs to reconstruct the softmax exactly.
//
// Two consequences beyond the bandwidth saving:
//   * Causality becomes a loop bound rather than a mask. The unfused path
//     computes the full T x T product and throws away half. This one skips
//     whole key blocks above the diagonal, halving the attention FLOPs.
//   * The head permute disappears. The kernel reads q, k and v straight out of
//     the fused (B, T, 3C) projection with a strided row pointer, so the
//     separate permuted qkvr buffer (4*B*T*C floats per layer) is not needed.
//
// Buffer shapes:
//   qkv  (B, T, 3C)   fused QKV projection output, [q|k|v] along 3C
//   out  (B, T, C)
//   lse  (B, NH, T)   log-sum-exp per row, saved for backward
//   dqkv (B, T, 3C)   written, not accumulated
//   dsum (B, NH, T)   backward scratch: rowsum(dout * out)

void flash_attention_forward(float *out, float *lse, const float *qkv, int B,
                             int T, int C, int NH);

void flash_attention_backward(float *dqkv, float *dsum, const float *dout,
                              const float *qkv, const float *out,
                              const float *lse, int B, int T, int C, int NH);

// ---- tuning harness ----
//
// Tile shapes are picked by measurement, not by argument, so the benchmark
// needs to reach every candidate configuration. flash_attention_forward() calls
// whichever config flash_default_config() names.
int flash_num_configs();
const char *flash_config_name(int cfg);
int flash_default_config();
// Returns false if the config cannot run for this head size.
bool flash_attention_forward_cfg(int cfg, float *out, float *lse,
                                 const float *qkv, int B, int T, int C, int NH);

// ---- backward tuning harness ----
int flash_num_bwd_configs();
const char *flash_bwd_config_name(int cfg);
int flash_default_bwd_config();
bool flash_attention_backward_cfg(int cfg, float *dqkv, float *dsum,
                                  const float *dout, const float *qkv,
                                  const float *out, const float *lse, int B,
                                  int T, int C, int NH);
