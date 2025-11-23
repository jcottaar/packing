import subprocess
import sys

# Test float64
print("=== Float64 ===")
result = subprocess.run([
    sys.executable, '-c', '''
import numpy as np
import cupy as cp
import pack_cuda as pc

pc.USE_FLOAT32 = False

xyt1 = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float64)
xyt2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

cost, grads = pc.overlap_list_total(xyt1, xyt2)
print(f"Cost: {float(cost.get()):.12f}")
print(f"Grads[1]: {grads.get()[1]}")
'''
], capture_output=True, text=True, cwd='/mnt/d/packing/code/core')
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Test float32
print("=== Float32 ===")
result = subprocess.run([
    sys.executable, '-c', '''
import numpy as np
import cupy as cp
import pack_cuda as pc

pc.USE_FLOAT32 = True

xyt1 = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float64)
xyt2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

cost, grads = pc.overlap_list_total(xyt1, xyt2)
print(f"Cost: {float(cost.get()):.12f}")
print(f"Grads[1]: {grads.get()[1]}")
'''
], capture_output=True, text=True, cwd='/mnt/d/packing/code/core')
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
