import sys
import numpy as np
import cupy as cp

# Test 1: Fresh python with float64
print("=== Test 1: Float64 ===")
import pack_cuda as pc
pc.USE_FLOAT32 = False

xyt1 = np.array([[0.0, 0.0, 0.0], [1.5, 0.5, 0.3]], dtype=np.float64)
xyt2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

cost, grads = pc.overlap_list_total(xyt1, xyt2)
print(f"Cost: {float(cost.get()):.12f}")
print(f"Grads[0]: {grads.get()[0]}")
print(f"_initialized: {pc._initialized}")
