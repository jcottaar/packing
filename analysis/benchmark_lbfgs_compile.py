"""
Benchmark script comparing original vs compiled L-BFGS implementations.

This script helps measure the overhead reduction from torch.compile.
"""
import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import lbfgs_torch_parallel as lbfgs_original


def create_test_problem(M, N, device='cpu'):
    """
    Create a simple quadratic test problem.

    Minimize: sum((x - target)^2) for each of M systems
    """
    target = torch.randn(M, N, device=device)

    def func(x):
        diff = x - target
        cost = (diff ** 2).sum(dim=1)  # (M,)
        grad = 2 * diff  # (M, N)
        return cost, grad

    return func, torch.randn(M, N, device=device)


def benchmark_version(lbfgs_func, func, x0, name, warmup=1, trials=5):
    """Benchmark a specific lbfgs implementation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    times = []

    # Warmup runs (important for torch.compile!)
    print(f"Warming up with {warmup} run(s)...")
    for i in range(warmup):
        x_test = x0.clone()
        start = time.perf_counter()
        result = lbfgs_func(func, x_test, max_iter=20, line_search_fn='strong_wolfe')
        elapsed = time.perf_counter() - start
        print(f"  Warmup {i+1}: {elapsed:.4f}s")

    # Actual benchmark runs
    print(f"\nRunning {trials} benchmark trials...")
    for i in range(trials):
        x_test = x0.clone()
        torch.cuda.synchronize() if x0.is_cuda else None

        start = time.perf_counter()
        result = lbfgs_func(func, x_test, max_iter=20, line_search_fn='strong_wolfe')
        torch.cuda.synchronize() if x0.is_cuda else None

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.4f}s")

    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults:")
    print(f"  Mean: {mean_time:.4f}s")
    print(f"  Min:  {min_time:.4f}s")
    print(f"  Max:  {max_time:.4f}s")
    print(f"  Std:  {(sum((t - mean_time)**2 for t in times) / len(times))**0.5:.4f}s")

    return mean_time


def main():
    """Run benchmark comparisons."""
    print("L-BFGS Compilation Benchmark")
    print("="*60)

    # Configuration
    M = 100  # Number of parallel systems
    N = 50   # Parameters per system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Batch size (M): {M}")
    print(f"  Parameters per system (N): {N}")
    print(f"  Device: {device}")
    print(f"  PyTorch version: {torch.__version__}")

    # Create test problem
    func, x0 = create_test_problem(M, N, device)

    # Benchmark original version
    time_original = benchmark_version(
        lbfgs_original.lbfgs,
        func, x0,
        "Original (with @torch.compile on helpers)",
        warmup=2,
        trials=5
    )

    # Try to import and benchmark optimized version
    try:
        import lbfgs_torch_parallel_optimized as lbfgs_opt

        # Optionally enable performance mode
        if device == 'cuda':
            print("\nEnabling TF32 performance mode...")
            lbfgs_opt.enable_performance_mode(use_tf32=True, matmul_precision='high')

        time_optimized = benchmark_version(
            lbfgs_opt.lbfgs,
            func, x0,
            "Optimized (mode='max-autotune')",
            warmup=1,  # May be slower due to more compilation
            trials=5
        )

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Original:        {time_original:.4f}s")
        print(f"Optimized:       {time_optimized:.4f}s")
        print(f"Speedup:         {time_original/time_optimized:.2f}x")
        print(f"Time saved:      {(time_original - time_optimized)*1000:.1f}ms per run")

    except ImportError:
        print("\nOptimized version not found. Skipping comparison.")
        print("(This is expected if you only modified the original file)")

    # Additional recommendations
    print(f"\n{'='*60}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. If overhead is still high, try larger batch sizes (M)")
    print("2. Use GPU if available (current:", device, ")")
    print("3. Profile with: torch.profiler to identify remaining bottlenecks")
    print("4. Consider reducing history_size if not needed")
    print("5. Check if your obj_func can be compiled with @torch.compile")


if __name__ == "__main__":
    main()
