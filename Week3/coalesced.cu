#include <stdio.h>
#include <cuda_runtime.h>

// Renamed macros to avoid collision with function arguments
#define MAT_M 4096
#define MAT_N 4096
#define BLOCK_SIZE 256

__global__ void gemv_coalesced(const float *A, const float *x, float *y, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        // Standard row-major access
        // Note: For matrix A, this is technically not fully coalesced (stride N), 
        // but it is the standard baseline for row-per-thread GEMV.
        sum += A[i * N + j] * x[j];
    }
    y[i] = fmaxf(sum, 0.0f);  // ReLU
}

int main() {
    // Use the macros for setup
    int M = MAT_M;
    int N = MAT_N;

    float *h_A = new float[M * N];
    float *h_x = new float[N];
    float *h_y = new float[M];

    // Initialize
    for (int i = 0; i < M * N; i++) h_A[i] = 1.0f;
    for (int i = 0; i < N; i++) h_x[i] = 2.0f;

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int grid_size = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gemv_coalesced<<<grid_size, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * M * N;
    printf("Coalesced GEMV: %.3f ms, %.2f GFLOPS\n", ms, flops / (ms * 1e6));

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    delete[] h_A; delete[] h_x; delete[] h_y;
    return 0;
}
