#!/usr/bin/env python
"""
Standalone script for profiling LUT kernel with ncu/nsys.

Usage:
  # Basic ncu profiling (shows compute/memory utilization):
  ncu --target-processes all python profile_lut.py
  
  # Detailed ncu with report file:
  ncu --set full -o lut_profile python profile_lut.py
  
  # Timeline with nsys:
  nsys profile -o lut_timeline python profile_lut.py
  
  # Just the key metrics:
  ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed \
    python profile_lut.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))

import numpy as np
import cupy as cp
import kaggle_support as kgs
import pack_cuda
import pack_cost
import pack_cuda_lut

# Setup
pack_cuda.USE_FLOAT32 = True
pack_cuda._ensure_initialized()

# Create test solution (same as notebook)
N_individuals = 200000
xy = cp.meshgrid(cp.arange(10), cp.arange(10))
x = (xy[0].ravel()-5)/2
y = (xy[1].ravel()-5)/2
t = cp.random.default_rng(seed=42).uniform(0, 2*np.pi, size=x.shape[0])
xyt = cp.stack([x, y, t], axis=1)

sol = kgs.SolutionCollectionSquare()
sol.xyt = cp.tile(xyt, (N_individuals, 1, 1))
sol.xyt[:, :, :] += cp.random.default_rng(seed=43).uniform(0, 0.1, size=(N_individuals, sol.N_trees, 3))
sol.h = cp.tile(cp.array([[5.,0.,0.]], dtype=cp.float32), (N_individuals, 1))
sol.xyt = sol.xyt.astype(cp.float32)
sol.h = sol.h.astype(cp.float32)

print(f"Solution: {sol.N_solutions} ensembles x {sol.N_trees} trees")

# Build LUT
N_X, N_Y, N_THETA = 200, 200, 200
MAX_R = kgs.tree_max_radius

pack_cuda_lut.USE_TEXTURE = False  # Change to False to profile array mode
pack_cuda_lut.LUT_X = np.linspace(-2*MAX_R, 2*MAX_R, N_X)
pack_cuda_lut.LUT_Y = np.linspace(-2*MAX_R, 2*MAX_R, N_Y)
pack_cuda_lut.LUT_theta = np.linspace(-np.pi, np.pi, N_THETA)

# Generate LUT values
print("Generating LUT...")
dx_grid, dy_grid, dt_grid = cp.meshgrid(
    cp.asarray(pack_cuda_lut.LUT_X, dtype=cp.float32),
    cp.asarray(pack_cuda_lut.LUT_Y, dtype=cp.float32),
    cp.asarray(pack_cuda_lut.LUT_theta, dtype=cp.float32),
    indexing='ij'
)
N_lut = N_X * N_Y * N_THETA
xyt_lut = cp.zeros((N_lut, 2, 3), dtype=cp.float32)
xyt_lut[:, 1, 0] = dx_grid.ravel()
xyt_lut[:, 1, 1] = dy_grid.ravel()
xyt_lut[:, 1, 2] = dt_grid.ravel()

sol_lut = kgs.SolutionCollectionSquare()
sol_lut.xyt = xyt_lut
sol_lut.h = cp.tile(cp.array([[10., 0., 0.]], dtype=cp.float32), (N_lut, 1))

cost_fn = pack_cost.CollisionCostSeparation()
cost_fn.compute_cost_allocate(sol)


lut_costs, _, _ = cost_fn.compute_cost_allocate(sol_lut)


pack_cuda_lut.LUT_vals = lut_costs.get().reshape(N_X, N_Y, N_THETA).astype(np.float32)

pack_cuda_lut._ensure_initialized()

# Allocate output
cost_out = cp.empty(sol.N_solutions, dtype=cp.float32)

grad = cp.empty((sol.N_solutions, sol.N_trees, 3), dtype=cp.float32)
pack_cuda_lut.overlap_multi_ensemble(sol.xyt, cost_out, grad)
# # # Warmup
# # print("Warming up...")
# # for _ in range(3):
# #     pack_cuda_lut.overlap_multi_ensemble(sol.xyt, sol.xyt, cost_out)
# #     cp.cuda.Device().synchronize()

# # # Profile this section - ncu will capture these kernel calls
# # print("Running profiled kernels...")
# # cp.cuda.Device().synchronize()

# # # Mark region for profiler (optional, helps identify in timeline)
# # cp.cuda.nvtx.RangePush("LUT_kernel")
# # for _ in range(5):
# #     pack_cuda_lut.overlap_multi_ensemble(sol.xyt, sol.xyt, cost_out)
# # cp.cuda.Device().synchronize()
# # cp.cuda.nvtx.RangePop()

# # print("Done. Check profiler output.")
# # print(f"Result sample: {cost_out[:5].get()}")
