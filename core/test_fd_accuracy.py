#!/usr/bin/env python3
"""Test finite difference accuracy for texture interpolation."""

import numpy as np
import cupy as cp
import kaggle_support as kgs
import pack_cost
import pack_cuda_lut

kgs.USE_FLOAT32 = True

# Test with both texture modes
for use_tex in [False, True]:
    pack_cuda_lut.USE_TEXTURE = use_tex
    pack_cuda_lut._initialized = False  # Force recompile
    
    print(f"\n{'='*60}")
    print(f"Testing USE_TEXTURE = {use_tex}")
    print(f"{'='*60}")
    
    # Build a small LUT
    cost_fn = pack_cost.CollisionCostOverlappingArea()
    lut = pack_cuda_lut.LookupTable.build_from_cost_function(
        cost_fn=cost_fn,
        N_x=20,
        N_y=20,
        N_theta=20,
        trim_zeros=False,
        verbose=False
    )
    
    # Pick a point in the middle of the LUT
    x_test = (lut.X_min + lut.X_max) / 2
    y_test = (lut.Y_min + lut.Y_max) / 2
    theta_test = 0.0
    
    # Create solutions with 2 trees at different step sizes
    step_sizes = [lut.grid_dx * f for f in [0.01, 0.125, 0.25, 0.5]]
    
    print(f"\nTesting at point ({x_test:.3f}, {y_test:.3f}, {theta_test:.3f})")
    print(f"Grid spacing: dx={lut.grid_dx:.4f}, dy={lut.grid_dy:.4f}, dtheta={lut.grid_dtheta:.4f}\n")
    
    for h in step_sizes:
        # Create solution: tree 0 at origin, tree 1 at test point
        xyt = np.zeros((1, 2, 3), dtype=np.float32)
        xyt[0, 1, :] = [x_test, y_test, theta_test]
        xyt_cp = cp.asarray(xyt)
        
        sol = kgs.SolutionCollectionSquare()
        sol.xyt = xyt_cp
        sol.h = cp.array([[10., 0., 0.]], dtype=cp.float32)
        
        # Compute gradient
        cost_lut = pack_cost.CollisionCostOverlappingArea(
            use_lookup_table=True,
            lut_N_x=20,
            lut_N_y=20,
            lut_N_theta=20
        )
        _, grad, _ = cost_lut.compute_cost_allocate(sol, evaluate_gradient=True)
        grad_np = grad.get()[0, 1, :]  # gradient for tree 1
        
        # Compute finite difference manually
        xyt_plus = xyt.copy()
        xyt_plus[0, 1, 0] += h
        sol.xyt = cp.asarray(xyt_plus)
        cost_plus, _, _ = cost_lut.compute_cost_allocate(sol, evaluate_gradient=False)
        
        sol.xyt = xyt_cp
        cost_base, _, _ = cost_lut.compute_cost_allocate(sol, evaluate_gradient=False)
        
        fd_grad_x = (cost_plus.get()[0] - cost_base.get()[0]) / h
        
        print(f"h = {h/lut.grid_dx:6.3f} * grid_dx:  ", end="")
        print(f"grad_x(kernel)={grad_np[0]:9.6f}  ", end="")
        print(f"grad_x(manual_FD)={fd_grad_x:9.6f}  ", end="")
        print(f"diff={abs(grad_np[0] - fd_grad_x):9.6e}")
