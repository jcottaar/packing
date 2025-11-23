import numpy as np
import cupy as cp

# Test 1: Float64
print("=== Float64 ===")
import pack_cuda as pc
pc.USE_FLOAT32 = False

# Original test data that had non-zero overlap
xyt1 = np.array([
    [0.0, 0.0, 0.0],
    [0.8, 0.0, 0.0],  # This should overlap with xyt2
], dtype=np.float64)

xyt2 = np.array([
    [1.0, 0.0, 0.0],
], dtype=np.float64)

cost, grads = pc.overlap_list_total(xyt1, xyt2)
print(f"Cost dtype: {cost.dtype}")
print(f"Cost: {float(cost.get()):.12f}")
print(f"Grads[0]: {grads.get()[1]}")  # Check second polygon which overlaps
