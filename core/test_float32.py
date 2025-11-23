#!/usr/bin/env python3
"""Test float32 mode"""
import pack_cuda as pc
import numpy as np

# Set float32 mode BEFORE initialization
pc.USE_FLOAT32 = True

print('Testing with float32:')
xyt = np.array([[0,0,0], [0.1,0.1,0.05]])
cost, grads = pc.overlap_list_total(xyt, xyt)
print(f'  Cost dtype: {cost.dtype}')
print(f'  Grads dtype: {grads.dtype}')
print(f'  Cost: {float(cost.get()):.8f}')
print(f'  Grads[0]: {grads.get()[0]}')
