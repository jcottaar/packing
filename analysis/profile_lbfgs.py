"""
Simple profiling utility for L-BFGS overhead analysis.

This helps identify whether the overhead is in:
- Objective function calls
- L-BFGS algorithmic overhead
- Line search overhead
"""
import torch
import time
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))


class ProfilerWrapper:
    """Wraps an objective function to track call time."""

    def __init__(self, func, name="objective"):
        self.func = func
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.times = []

    def __call__(self, x):
        self.call_count += 1
        start = time.perf_counter()
        result = self.func(x)
        elapsed = time.perf_counter() - start
        self.total_time += elapsed
        self.times.append(elapsed)
        return result

    def report(self):
        if self.call_count == 0:
            return "Not called"

        mean = self.total_time / self.call_count
        return (f"{self.name}:\n"
                f"  Calls: {self.call_count}\n"
                f"  Total: {self.total_time*1000:.2f}ms\n"
                f"  Mean:  {mean*1000:.4f}ms/call\n"
                f"  Min:   {min(self.times)*1000:.4f}ms\n"
                f"  Max:   {max(self.times)*1000:.4f}ms")


def profile_lbfgs(func, x0, **kwargs):
    """
    Profile an L-BFGS optimization run.

    Args:
        func: Objective function
        x0: Initial parameters
        **kwargs: Arguments to pass to lbfgs

    Returns:
        result, profile_dict
    """
    import lbfgs_torch_parallel as lbfgs_mod

    # Wrap objective function
    wrapped_func = ProfilerWrapper(func, "Objective Function")

    # Time the overall optimization
    start_total = time.perf_counter()
    result = lbfgs_mod.lbfgs(wrapped_func, x0, **kwargs)
    total_time = time.perf_counter() - start_total

    # Calculate overhead
    func_time = wrapped_func.total_time
    overhead_time = total_time - func_time
    overhead_pct = (overhead_time / total_time * 100) if total_time > 0 else 0

    profile_dict = {
        'total_time': total_time,
        'func_time': func_time,
        'func_calls': wrapped_func.call_count,
        'overhead_time': overhead_time,
        'overhead_pct': overhead_pct,
        'wrapped_func': wrapped_func,
    }

    return result, profile_dict


def print_profile_report(profile_dict):
    """Print a formatted profile report."""
    print("\n" + "="*60)
    print("L-BFGS PROFILE REPORT")
    print("="*60)

    print(f"\nTotal Optimization Time: {profile_dict['total_time']*1000:.2f}ms")
    print(f"\n{profile_dict['wrapped_func'].report()}")

    print(f"\nAlgorithm Overhead:")
    print(f"  Time:    {profile_dict['overhead_time']*1000:.2f}ms")
    print(f"  Percent: {profile_dict['overhead_pct']:.1f}%")

    print(f"\nBreakdown:")
    print(f"  {'Component':<25} {'Time (ms)':<12} {'Percent':<10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")

    total = profile_dict['total_time'] * 1000
    func_time = profile_dict['func_time'] * 1000
    overhead_time = profile_dict['overhead_time'] * 1000

    print(f"  {'Objective Function':<25} {func_time:>10.2f}   {func_time/total*100:>8.1f}%")
    print(f"  {'BFGS Overhead':<25} {overhead_time:>10.2f}   {overhead_time/total*100:>8.1f}%")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print("="*60)

    if profile_dict['overhead_pct'] > 50:
        print("⚠️  HIGH OVERHEAD DETECTED (>50%)")
        print("\nRecommendations:")
        print("1. ✅ Compilation will help significantly")
        print("2. Increase batch size M to amortize overhead")
        print("3. Consider using mode='max-autotune' for aggressive optimization")
        print("4. Use GPU if available")
    elif profile_dict['overhead_pct'] > 20:
        print("⚠️  MODERATE OVERHEAD (20-50%)")
        print("\nRecommendations:")
        print("1. ✅ Compilation should provide noticeable speedup")
        print("2. Objective function is reasonable, but overhead can be reduced")
        print("3. Try compiling both LBFGS and objective function")
    else:
        print("✅ LOW OVERHEAD (<20%)")
        print("\nRecommendations:")
        print("1. Objective function dominates - this is good!")
        print("2. Compilation may provide modest gains (10-30%)")
        print("3. Focus on optimizing objective function if more speed needed")
        print("4. Consider compiling objective function with @torch.compile")

    avg_func_time_ms = profile_dict['func_time'] / profile_dict['func_calls'] * 1000
    print(f"\nObjective Function Performance:")
    print(f"  Average call time: {avg_func_time_ms:.4f}ms")

    if avg_func_time_ms < 0.1:
        print("  ⚠️  Very fast function - overhead will dominate")
        print("     Consider: Larger batch size, compilation")
    elif avg_func_time_ms < 1.0:
        print("  ⚠️  Fast function - overhead is significant")
        print("     Recommendation: Compilation will help")
    else:
        print("  ✅ Function time is good - compilation optional")


def example_usage():
    """Example of how to use the profiler."""
    print("L-BFGS Profiler - Example Usage\n")

    # Create test problem
    M, N = 50, 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = torch.randn(M, N, device=device)

    def func(x):
        diff = x - target
        cost = (diff ** 2).sum(dim=1)
        grad = 2 * diff
        return cost, grad

    x0 = torch.randn(M, N, device=device)

    print(f"Configuration: M={M}, N={N}, device={device}\n")

    # Profile the optimization
    result, profile = profile_lbfgs(
        func, x0,
        lr=1.0,
        max_iter=20,
        line_search_fn='strong_wolfe'
    )

    # Print report
    print_profile_report(profile)

    print("\n" + "="*60)
    print("To use with your own problem:")
    print("="*60)
    print("""
from profile_lbfgs import profile_lbfgs, print_profile_report

result, profile = profile_lbfgs(
    my_objective_function,
    initial_parameters,
    lr=1.0,
    max_iter=20,
    line_search_fn='strong_wolfe'
)

print_profile_report(profile)
""")


if __name__ == "__main__":
    example_usage()
