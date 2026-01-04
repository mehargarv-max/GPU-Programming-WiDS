import numpy as np
import time

def gemv_numpy(A, x):
    """
    NumPy optimized GEMV with ReLU activation.
    Uses BLAS backend (via np.dot or @ operator).
    """
    # Matrix-Vector Multiplication followed by ReLU
    return np.maximum(A @ x, 0.0)

def benchmark_cpu(sizes):
    print(f"{'Size (N)':<15} | {'Time (ms)':<15} | {'Throughput (GFLOPS)':<20}")
    print("-" * 55)

    results = {}

    for N in sizes:
        M = N  # Square matrix assumption for this assignment
        
        # Initialize data with float32 (standard for GPU comparison)
        A = np.random.rand(M, N).astype(np.float32)
        x = np.random.rand(N).astype(np.float32)

        # Warm-up run (to load libraries/caches)
        _ = gemv_numpy(A, x)

        # Measurement run
        start_time = time.perf_counter()
        # Run multiple iterations for smaller sizes to get accurate timing
        iterations = 10 if N < 2048 else 1
        
        for _ in range(iterations):
            y = gemv_numpy(A, x)
            
        end_time = time.perf_counter()

        # Average time per iteration in milliseconds
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Calculate GFLOPS
        # Operations: 2 * M * N (Multiply + Add per element)
        # Note: ReLU is negligible compared to O(N^2) matmul
        flops = 2.0 * M * N
        gflops = (flops / (avg_time_ms / 1000)) / 1e9

        results[N] = (avg_time_ms, gflops)
        
        print(f"{N:<15} | {avg_time_ms:<15.3f} | {gflops:<20.2f}")

    return results

if __name__ == "__main__":
    print("Running CPU Baseline Benchmark (NumPy/BLAS)...")
    print("Task: Matrix-Vector Multiplication (GEMV) with ReLU\n")
    
    # Sizes matched to your report
    problem_sizes = [256, 1024, 4096, 8192]
    
    benchmark_data = benchmark_cpu(problem_sizes)
    
    print("\nBenchmark Complete.")