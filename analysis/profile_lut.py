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


cost_fn = pack_cost.CollisionCostSeparation()
cost_fn.use_lookup_table = True
cost_fn.compute_cost_allocate(sol)