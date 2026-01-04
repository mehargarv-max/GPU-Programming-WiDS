#include <stdio.h>
#include <cuda_runtime.h>

// Renamed macros
#define MAT_M 4096
#define MAT_N 4096
#define BLOCK_SIZE 256
#define TILE_SIZE 256

__global__ void gemv_shared(const float *A, const float *x, float *y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory to cache chunks of vector x
    extern __shared__ float s_x[];
    
    float sum = 0.0f;
    
    // Loop over tiles of the vector x
    for (int tile = 0; tile < N; tile += TILE_SIZE) {
        // 1. Load tile into shared memory (Coalesced load)
        int x_idx = tile + threadIdx.x;
        
        if (x_idx < N) 
            s_x[threadIdx.x] = x[x_idx];
        else 
            s_x[threadIdx.x] = 0.0f;
            
        __syncthreads();

        // 2. Compute partial dot product using shared memory
        if (row < M) {
            int tile_end = min(TILE_SIZE, N - tile);
            for (int j = 0; j < tile_end; j++) {
                sum += A[row * N + tile + j] * s_x[j];
            }
        }
        __syncthreads();
    }
    
    if (row < M) {
        y[row] = fmaxf(sum, 0.0f);
    }
}

int main() {
    int M = MAT_M;
    int N = MAT_N;

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

    int grid_size = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory size is size of one tile
    size_t shared_size = TILE_SIZE * sizeof(float);
    
    gemv_shared<<<grid_size, BLOCK_SIZE, shared_size>>>(d_A, d_x, d_y, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * M * N;
    printf("Shared Memory GEMV: %.3f ms, %.2f GFLOPS\n", ms, flops / (ms * 1e6));

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    delete[] h_A; delete[] h_x; delete[] h_y;
    return 0;
}
