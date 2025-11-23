import numpy as np
import cupy as cp
import pack_cuda as pc

# Test data
xyt1 = np.array([
    [0.0, 0.0, 0.0],
    [1.5, 0.5, 0.3],
], dtype=np.float64)

xyt2 = np.array([
    [1.0, 0.0, 0.0],
], dtype=np.float64)

print("Testing float64:")
pc.USE_FLOAT32 = False
pc._initialized = False
cost_f64, grads_f64 = pc.overlap_list_total(xyt1, xyt2)
print(f"  Cost: {float(cost_f64.get()):.12f}")
print(f"  Grads[0]: {grads_f64.get()[0]}")

print("\nTesting float32:")
pc.USE_FLOAT32 = True
pc._initialized = False
cost_f32, grads_f32 = pc.overlap_list_total(xyt1, xyt2)
print(f"  Cost: {float(cost_f32.get()):.12f}")
print(f"  Grads[0]: {grads_f32.get()[0]}")

print("\nDifferences:")
print(f"  Cost diff: {abs(float(cost_f64.get()) - float(cost_f32.get())):.2e}")
print(f"  Grads diff (L2): {np.linalg.norm(grads_f64.get()[0] - grads_f32.get()[0]):.2e}")
