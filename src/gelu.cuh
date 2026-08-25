#pragma once

// GELU, in a header because two translation units need it now: nn.cu, which
// owns the standalone kernels, and gemm.cu, which applies it in the GEMM
// epilogue so the activation never makes a separate pass over memory.
//
// The tanh approximation rather than the exact erf form, because that is what
// GPT-2 used and this model is checked against its shapes and behaviour. The
// two differ by ~1e-3 at the extremes, which is a different model, not a less
// accurate implementation of the same one.
__device__ __forceinline__ float gelu_scalar(float x) {
    const float s = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(s * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float dgelu_scalar(float x, float g) {
    const float s = 0.7978845608028654f;
    const float inner = s * (x + 0.044715f * x * x * x);
    const float t = tanhf(inner);
    const float sech2 = 1.0f - t * t;
    const float dinner = s * (1.0f + 3.0f * 0.044715f * x * x);
    return g * (0.5f * (1.0f + t) + 0.5f * x * sech2 * dinner);
}
