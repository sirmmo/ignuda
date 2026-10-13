extern "C" __global__ void vadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(float alpha, const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = alpha * x[i] + y[i];
}

extern "C" __global__ void reduce_sum(const float *in, float *out, int n) {
    __shared__ float tile[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    tile[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) tile[tid] += tile[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, tile[0]);
}

extern "C" __global__ void matmul(const float *A, const float *B, float *C, int N) {
    __shared__ float As[16][16], Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < N; t += 16) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}
