#pragma once

// All kernels compute, in ROW-MAJOR layout:
//     C = alpha * A @ B + beta * C
// with A: M x K, B: K x N, C: M x N.
//
// Row-major is used throughout the project because it matches how the
// transformer code later stores activations. cuBLAS is column-major, so the
// reference wrapper transposes via the (B,A) operand swap -- see bench.cu.
typedef void (*sgemm_fn)(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C);

struct KernelEntry {
    int id;
    const char *name;
    sgemm_fn fn;
    const char *note;
    // Normwise tolerance against the fp32 cuBLAS reference. Every fp32 kernel
    // uses 1e-4; the tensor-core kernel needs more room because TF32 keeps only
    // 10 mantissa bits, so it is computing a deliberately lower-precision
    // answer rather than computing the same answer badly. Making this per
    // kernel keeps that concession explicit and local instead of loosening the
    // bar for kernels that have no excuse.
    double tol;
};

extern const KernelEntry KERNELS[];
extern const int NUM_KERNELS;

// Returns nullptr if no kernel with that id is registered.
const KernelEntry *find_kernel(int id);

// --- kernel launch wrappers, one per optimization stage ---
void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_coalesced(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_smem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_tile1d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_tile2d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_vectorized(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_warptile(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_doublebuffer(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
void sgemm_tensorcore(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
