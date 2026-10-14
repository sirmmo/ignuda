/* Exercises ignuda's cublasGemmEx family with FP16 inputs (the path ggml uses
 * on an iGPU). C = alpha*A*B + beta*C, column-major. A/B are FP16; C is FP32
 * or FP16. Also checks the strided-batched variant. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef void *cublasHandle_t;
extern int cublasCreate_v2(cublasHandle_t *);
extern int cublasDestroy_v2(cublasHandle_t);
extern int cublasGemmEx(cublasHandle_t, int, int, int, int, int, const void *, const void *, int, int,
                        const void *, int, int, const void *, void *, int, int, int, int);
extern int cublasGemmStridedBatchedEx(cublasHandle_t, int, int, int, int, int, const void *, const void *,
                                      int, int, long long, const void *, int, int, long long, const void *,
                                      void *, int, int, long long, int, int, int);
extern int cublasGemmBatchedEx(cublasHandle_t, int, int, int, int, int, const void *, const void *const *,
                               int, int, const void *const *, int, int, const void *, void *const *, int,
                               int, int, int, int);
extern int cudaMalloc(void **, size_t);
extern int cudaMemcpy(void *, const void *, size_t, int);
extern int cudaFree(void *);
extern int cudaDeviceSynchronize(void);

#define CUDA_R_32F 0
#define CUDA_R_16F 2
#define CUBLAS_COMPUTE_32F 68
#define CUBLAS_GEMM_DEFAULT (-1)

/* IEEE float <-> half (round to nearest even). */
static uint16_t f2h(float v) {
    uint32_t x;
    memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* too small -> 0 */
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    } else if (exp >= 0x1f) {
        return (uint16_t)(sign | 0x7c00u); /* overflow -> inf */
    }
    uint32_t h = (uint32_t)(exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return (uint16_t)(sign | h);
}
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            do { e++; m <<= 1; } while (!(m & 0x400));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | (0xffu << 23) | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static int check(const char *name, const float *got, const float *ref, int n) {
    int bad = 0;
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
        double e = fabs(got[i] - ref[i]);
        if (e > maxerr) maxerr = e;
        bad += e > 1e-2;
    }
    printf("%s: %s (max abs err %g)\n", name, bad ? "FAIL" : "ok", maxerr);
    return bad != 0;
}

int main(void) {
    const int M = 40, N = 24, K = 16;
    int fails = 0;
    float *A = malloc(sizeof(float) * M * K), *B = malloc(sizeof(float) * K * N);
    uint16_t *Ah = malloc(2 * M * K), *Bh = malloc(2 * K * N);
    float *C = malloc(sizeof(float) * M * N), *ref = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++) { A[i] = (float)((i % 7) - 3); Ah[i] = f2h(A[i]); }
    for (int i = 0; i < K * N; i++) { B[i] = (float)((i % 5) - 2); Bh[i] = f2h(B[i]); }
    float alpha = 2.0f, beta = 0.0f;
    for (int c = 0; c < N; c++)
        for (int r = 0; r < M; r++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += A[(size_t)k * M + r] * B[(size_t)c * K + k];
            ref[(size_t)c * M + r] = alpha * acc;
        }

    void *dA, *dB, *dC;
    cudaMalloc(&dA, 2 * M * K);
    cudaMalloc(&dB, 2 * K * N);
    cudaMalloc(&dC, sizeof(float) * M * N);
    cudaMemcpy(dA, Ah, 2 * M * K, 1);
    cudaMemcpy(dB, Bh, 2 * K * N, 1);

    cublasHandle_t h;
    if (cublasCreate_v2(&h)) { printf("cublasCreate failed\n"); return 1; }

    /* FP16 x FP16 -> FP32 */
    int st = cublasGemmEx(h, 0, 0, M, N, K, &alpha, dA, CUDA_R_16F, M, dB, CUDA_R_16F, K, &beta, dC,
                          CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st) { printf("GemmEx f16->f32 status %d\n", st); return 1; }
    cudaDeviceSynchronize();
    cudaMemcpy(C, dC, sizeof(float) * M * N, 2);
    fails += check("GemmEx f16->f32", C, ref, M * N);

    /* FP16 x FP16 -> FP16 */
    void *dCh;
    cudaMalloc(&dCh, 2 * M * N);
    st = cublasGemmEx(h, 0, 0, M, N, K, &alpha, dA, CUDA_R_16F, M, dB, CUDA_R_16F, K, &beta, dCh, CUDA_R_16F,
                      M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st) { printf("GemmEx f16->f16 status %d\n", st); return 1; }
    cudaDeviceSynchronize();
    uint16_t *Ch = malloc(2 * M * N);
    cudaMemcpy(Ch, dCh, 2 * M * N, 2);
    for (int i = 0; i < M * N; i++) C[i] = h2f(Ch[i]);
    fails += check("GemmEx f16->f16", C, ref, M * N);

    /* Strided batched: 2 independent FP16 GEMMs -> FP32. */
    const int BATCH = 2;
    uint16_t *Ab = malloc(2 * M * K * BATCH), *Bb = malloc(2 * K * N * BATCH);
    float *refb = malloc(sizeof(float) * M * N * BATCH);
    for (int bi = 0; bi < BATCH; bi++) {
        for (int i = 0; i < M * K; i++) Ab[bi * M * K + i] = f2h((float)(((i + bi) % 7) - 3));
        for (int i = 0; i < K * N; i++) Bb[bi * K * N + i] = f2h((float)(((i + bi) % 5) - 2));
        for (int c = 0; c < N; c++)
            for (int r = 0; r < M; r++) {
                float acc = 0;
                for (int k = 0; k < K; k++)
                    acc += h2f(Ab[bi * M * K + (size_t)k * M + r]) * h2f(Bb[bi * K * N + (size_t)c * K + k]);
                refb[bi * M * N + (size_t)c * M + r] = alpha * acc;
            }
    }
    void *dAb, *dBb, *dCb;
    cudaMalloc(&dAb, 2 * M * K * BATCH);
    cudaMalloc(&dBb, 2 * K * N * BATCH);
    cudaMalloc(&dCb, sizeof(float) * M * N * BATCH);
    cudaMemcpy(dAb, Ab, 2 * M * K * BATCH, 1);
    cudaMemcpy(dBb, Bb, 2 * K * N * BATCH, 1);
    st = cublasGemmStridedBatchedEx(h, 0, 0, M, N, K, &alpha, dAb, CUDA_R_16F, M, (long long)M * K, dBb,
                                    CUDA_R_16F, K, (long long)K * N, &beta, dCb, CUDA_R_32F, M,
                                    (long long)M * N, BATCH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st) { printf("GemmStridedBatchedEx status %d\n", st); return 1; }
    cudaDeviceSynchronize();
    float *Cb = malloc(sizeof(float) * M * N * BATCH);
    cudaMemcpy(Cb, dCb, sizeof(float) * M * N * BATCH, 2);
    fails += check("GemmStridedBatchedEx f16->f32", Cb, refb, M * N * BATCH);

    /* Batched (array of device pointers): reuse the two strided batches, but
     * pass per-batch pointers through device-resident pointer arrays. */
    void *hAp[BATCH], *hBp[BATCH], *hCp[BATCH];
    void *dCb2;
    cudaMalloc(&dCb2, sizeof(float) * M * N * BATCH);
    for (int bi = 0; bi < BATCH; bi++) {
        hAp[bi] = (char *)dAb + (size_t)bi * M * K * 2;
        hBp[bi] = (char *)dBb + (size_t)bi * K * N * 2;
        hCp[bi] = (char *)dCb2 + (size_t)bi * M * N * sizeof(float);
    }
    void *dAp, *dBp, *dCp;
    cudaMalloc(&dAp, sizeof(void *) * BATCH);
    cudaMalloc(&dBp, sizeof(void *) * BATCH);
    cudaMalloc(&dCp, sizeof(void *) * BATCH);
    cudaMemcpy(dAp, hAp, sizeof(void *) * BATCH, 1);
    cudaMemcpy(dBp, hBp, sizeof(void *) * BATCH, 1);
    cudaMemcpy(dCp, hCp, sizeof(void *) * BATCH, 1);
    st = cublasGemmBatchedEx(h, 0, 0, M, N, K, &alpha, (const void *const *)dAp, CUDA_R_16F, M,
                             (const void *const *)dBp, CUDA_R_16F, K, &beta, (void *const *)dCp, CUDA_R_32F,
                             M, BATCH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st) { printf("GemmBatchedEx status %d\n", st); return 1; }
    cudaDeviceSynchronize();
    float *Cb2 = malloc(sizeof(float) * M * N * BATCH);
    cudaMemcpy(Cb2, dCb2, sizeof(float) * M * N * BATCH, 2);
    fails += check("GemmBatchedEx f16->f32", Cb2, refb, M * N * BATCH);

    cublasDestroy_v2(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCh);
    cudaFree(dAb); cudaFree(dBb); cudaFree(dCb);
    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL OK");
    return fails != 0;
}
