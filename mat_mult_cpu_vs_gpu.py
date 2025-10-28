import torch
import time
from datetime import datetime

# Check if MPS (Metal Performance Shaders) is available on M3
if torch.backends.mps.is_available():
    gpu_device = torch.device("mps")
    print("Using Metal Performance Shaders (GPU) on M3 Pro")
else:
    gpu_device = torch.device("cpu")
    print("MPS not available, GPU tests will use CPU")

cpu_device = torch.device("cpu")

# Test parameters
n_iterations = [100, 250, 500, 1000, 5000]
matrix_sizes = [5, 50, 100, 500, 1000, 2500, 5000]

def benchmark_cpu(size, n):
    """Benchmark matrix multiplication on CPU using PyTorch"""
    times = []
    
    for _ in range(n):
        mat1 = torch.rand(size, size, device=cpu_device)
        mat2 = torch.rand(size, size, device=cpu_device)
        
        start = time.time()
        result = torch.matmul(mat1, mat2)
        end = time.time()
        
        times.append(end - start)
    
    return sum(times)

def benchmark_gpu(size, n):
    """Benchmark matrix multiplication on GPU using PyTorch MPS"""
    times = []
    
    for _ in range(n):
        mat1 = torch.rand(size, size, device=gpu_device)
        mat2 = torch.rand(size, size, device=gpu_device)
        
        # Synchronize before timing
        if gpu_device.type == "mps":
            torch.mps.synchronize()
        
        start = time.time()
        result = torch.matmul(mat1, mat2)
        
        # Synchronize after computation to ensure completion
        if gpu_device.type == "mps":
            torch.mps.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    return sum(times)

# Open file for writing results
filename = "benchmark_results.txt"

with open(filename, 'w') as f:
    f.write("CPU vs GPU Matrix Multiplication Benchmark\n")
    f.write("=" * 60 + "\n")
    f.write(f"Device: M3 Pro Chip\n")
    f.write(f"Backend: PyTorch (consistent for both CPU and GPU)\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    
    # Run benchmarks for each iteration count
    for n in n_iterations:
        print(f"\nRunning benchmark for {n} iterations...")
        f.write(f"{'='*60}\n")
        f.write(f"Results for {n} iterations\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"{'Matrix Size':<15} {'CPU Time (s)':<20} {'GPU Time (s)':<20} {'Speedup':<15}\n")
        f.write(f"{'-'*70}\n")
        
        for size in matrix_sizes:
            print(f"  Testing matrix size: {size}x{size}")
            
            # CPU benchmark
            cpu_time = benchmark_cpu(size, n)
            
            # GPU benchmark
            gpu_time = benchmark_gpu(size, n)
            
            # Calculate speedup
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # Write results
            result_line = f"{size}x{size:<11} {cpu_time:<20.6f} {gpu_time:<20.6f} {speedup:<15.2f}x\n"
            f.write(result_line)
            print(f"    CPU: {cpu_time:.6f}s, GPU: {gpu_time:.6f}s, Speedup: {speedup:.2f}x")
        
        f.write("\n")
    
    # Summary section
    f.write(f"\n{'='*60}\n")
    f.write("Summary: All Tests Completed\n")
    f.write(f"{'='*60}\n")
    f.write(f"Iteration counts tested: {n_iterations}\n")
    f.write(f"Matrix sizes tested: {matrix_sizes}\n")
    f.write(f"Results saved to: {filename}\n")

print(f"\n{'='*60}")
print(f"Benchmark complete! Results saved to: {filename}")
print(f"{'='*60}")