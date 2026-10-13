// Runtime-API test: fatbin embedded by nvcc, launched through libcudart -> libcuda.
#include <cstdio>
__global__ void scale(float *x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
int main() {
    const int n = 4096; float h[n], *d;
    for (int i = 0; i < n; i++) h[i] = i;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    scale<<<(n + 127) / 128, 128>>>(d, 0.5f, n);
    cudaError_t e = cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < n; i++) bad += h[i] != 0.5f * i;
    printf("runtime scale: %s (%s)\n", bad ? "FAIL" : "ok", cudaGetErrorString(e));
    cudaFree(d);
    return bad != 0;
}
