// Column-major GEMM building blocks for the ignuda cuBLAS shim.
// op(A) is MxK, op(B) is KxN, C is MxN, all column-major with leading dims.
// transA/transB: 0 = N (as stored), 1 = T (transpose).
extern "C" __global__ void sgemm(
    int transA, int transB, int M, int N, int K,
    float alpha, const float *A, int lda, const float *B, int ldb,
    float beta, float *C, int ldc)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 0..M
    int col = blockIdx.y * blockDim.y + threadIdx.y; // 0..N
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = transA ? A[(size_t)row * lda + k] : A[(size_t)k * lda + row];
        float b = transB ? B[(size_t)k * ldb + col] : B[(size_t)col * ldb + k];
        acc += a * b;
    }
    size_t idx = (size_t)col * ldc + row;
    C[idx] = alpha * acc + (beta == 0.0f ? 0.0f : beta * C[idx]);
}

extern "C" __global__ void dgemm(
    int transA, int transB, int M, int N, int K,
    double alpha, const double *A, int lda, const double *B, int ldb,
    double beta, double *C, int ldc)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    double acc = 0.0;
    for (int k = 0; k < K; k++) {
        double a = transA ? A[(size_t)row * lda + k] : A[(size_t)k * lda + row];
        double b = transB ? B[(size_t)k * ldb + col] : B[(size_t)col * ldb + k];
        acc += a * b;
    }
    size_t idx = (size_t)col * ldc + row;
    C[idx] = alpha * acc + (beta == 0.0 ? 0.0 : beta * C[idx]);
}

extern "C" __global__ void saxpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[(size_t)i * incy] += alpha * x[(size_t)i * incx];
}

extern "C" __global__ void sscal(int n, float alpha, float *x, int incx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(size_t)i * incx] *= alpha;
}
