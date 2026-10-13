/* Driver-API test: loads kernels.ptx and runs every kernel, checking results. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *s = 0; cuGetErrorName(r, &s); \
    fprintf(stderr, "%s:%d %s -> %d %s\n", __FILE__, __LINE__, #x, r, s ? s : "?"); exit(1); } } while (0)

static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    char *b = malloc(n + 1); fread(b, 1, n, f); b[n] = 0; fclose(f); return b;
}

int main(int argc, char **argv) {
    const int n = 1 << 16, N = 128;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction f;
    char name[256]; int fails = 0;
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuDeviceGetName(name, sizeof name, dev));
    printf("device: %s\n", name);
    CK(cuCtxCreate(&ctx, 0, dev));
    CK(cuModuleLoadData(&mod, slurp(argc > 1 ? argv[1] : "kernels.ptx")));

    float *a = malloc(n * 4), *b = malloc(n * 4), *c = malloc(n * 4);
    for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2 * i; }
    CUdeviceptr da, db, dc;
    CK(cuMemAlloc(&da, n * 4)); CK(cuMemAlloc(&db, n * 4)); CK(cuMemAlloc(&dc, n * 4));
    CK(cuMemcpyHtoD(da, a, n * 4)); CK(cuMemcpyHtoD(db, b, n * 4));

    /* vadd */
    int nn = n; void *args1[] = { &da, &db, &dc, &nn };
    CK(cuModuleGetFunction(&f, mod, "vadd"));
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args1, 0));
    CK(cuMemcpyDtoH(c, dc, n * 4));
    int bad = 0; for (int i = 0; i < n; i++) bad += c[i] != 3.0f * i;
    printf("vadd: %s\n", bad ? "FAIL" : "ok"); fails += !!bad;

    /* saxpy: y = 2*x + y, with y = b */
    float alpha = 2.0f; void *args2[] = { &alpha, &da, &db, &nn };
    CK(cuModuleGetFunction(&f, mod, "saxpy"));
    CK(cuLaunchKernel(f, 32, 1, 1, 128, 1, 1, 0, 0, args2, 0));
    CK(cuMemcpyDtoH(c, db, n * 4));
    bad = 0; for (int i = 0; i < n; i++) bad += c[i] != 4.0f * i;
    printf("saxpy: %s\n", bad ? "FAIL" : "ok"); fails += !!bad;

    /* reduce_sum over ones */
    for (int i = 0; i < n; i++) a[i] = 1.0f;
    float zero = 0, sum = 0; CUdeviceptr dsum;
    CK(cuMemAlloc(&dsum, 4)); CK(cuMemcpyHtoD(dsum, &zero, 4)); CK(cuMemcpyHtoD(da, a, n * 4));
    void *args3[] = { &da, &dsum, &nn };
    CK(cuModuleGetFunction(&f, mod, "reduce_sum"));
    CK(cuLaunchKernel(f, n / 256, 1, 1, 256, 1, 1, 0, 0, args3, 0));
    CK(cuMemcpyDtoH(&sum, dsum, 4));
    printf("reduce_sum: %s (%g)\n", sum == n ? "ok" : "FAIL", sum); fails += sum != n;

    /* matmul N x N: A = i+j, B = identity -> C == A */
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { a[i*N+j] = i + j; b[i*N+j] = i == j; }
    CK(cuMemcpyHtoD(da, a, N * N * 4)); CK(cuMemcpyHtoD(db, b, N * N * 4));
    int NN = N; void *args4[] = { &da, &db, &dc, &NN };
    CK(cuModuleGetFunction(&f, mod, "matmul"));
    CK(cuLaunchKernel(f, N / 16, N / 16, 1, 16, 16, 1, 0, 0, args4, 0));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(c, dc, N * N * 4));
    bad = 0; for (int i = 0; i < N * N; i++) bad += c[i] != a[i];
    printf("matmul: %s\n", bad ? "FAIL" : "ok"); fails += !!bad;

    CK(cuMemFree(da)); CK(cuMemFree(db)); CK(cuMemFree(dc)); CK(cuMemFree(dsum));
    CK(cuModuleUnload(mod)); CK(cuCtxDestroy(ctx));
    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL OK");
    return fails;
}
