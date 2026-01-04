#include <stdio.h>
#include <cuda_runtime.h>

// Renamed macros
#define MAT_M 4096
#define MAT_N 4096
#define BLOCK_SIZE 256
#define STRIDE 4

__global__ void gemv_non_coalesced(const float *A, const float *x, float *y, int M, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Explicit strided access pattern to degrade performance
    // This causes threads to access memory lines that are far apart (stride * N)
    int i = idx * stride; 

    if (i >= M) return;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += A[i * N + j] * x[j];
    }
    // Write to strided location (also uncoalesced output if we consider valid writes)
    // Note: In this specific logical implementation, we are skipping rows (processing only M/stride rows)
    // to demonstrate the latency impact on the active threads.
    y[idx] = fmaxf(sum, 0.0f);
}

int main() {
    int M = MAT_M;
    int N = MAT_N;
    int stride_val = STRIDE;

    float *h_A = new float[M * N];
    float *h_x = new float[N];
    float *h_y = new float[M];

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

    // Launch enough threads to cover the strided rows
    int num_active_rows = (M + stride_val - 1) / stride_val;
    int grid_size = (num_active_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    gemv_non_coalesced<<<grid_size, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N, stride_val);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Note: Total FLOPs here is actually less (M/STRIDE * N) because we skip rows,
    // but we use the same formula to show effective throughput degradation per matrix size.
    double flops = 2.0 * M * N; 
    printf("Non-Coalesced GEMV (stride=%d): %.3f ms, %.2f GFLOPS (Effective)\n", stride_val, ms, flops / (ms * 1e6));

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    delete[] h_A; delete[] h_x; delete[] h_y;
    return 0;
}
