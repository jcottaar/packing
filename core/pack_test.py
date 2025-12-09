import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
from typeguard import typechecked
import pack_cost
import pack_basics
import pack_cuda
import pack_cuda_primitives_test
import matplotlib.pyplot as plt
import pack_vis_sol
import copy

CUDA_float32 = False
kgs.set_float32(CUDA_float32)
pack_cuda._ensure_initialized()

def run_all_tests():
    kgs.debugging_mode = 2    
    pack_cuda_primitives_test.run_all_tests()
    test_costs()
    print("All tests passed.")

def test_costs():
    print('Testing cost computation and gradients')
    costs_to_test = [pack_cost.CostDummy(), pack_cost.AreaCost(), pack_cost.CollisionCostSeparation(scaling=5., use_max=False),
                     pack_cost.BoundaryDistanceCost(use_kernel=False), pack_cost.BoundaryDistanceCost(use_kernel=True, scaling=5.), pack_cost.CollisionCostOverlappingArea(scaling=3.), 
                     pack_cost.CostCompound(scaling = 1.5, costs=[pack_cost.AreaCost(), pack_cost.CollisionCostSeparation()])]

    kgs.set_float32(False)
    sol_list = []
    sol_list.append(kgs.SolutionCollectionSquare())
    sol_list[-1].xyt = cp.array([pack_basics.place_random(10, 1.5, generator=np.random.default_rng(seed=0)).xyt], dtype=cp.float64)
    sol_list[-1].h = cp.array([[0.5, 0.2, 0.4]])
    sol_list.append(kgs.SolutionCollectionSquare())
    sol_list[-1].xyt = cp.array([pack_basics.place_random(10, 1.5, generator=np.random.default_rng(seed=2)).xyt], dtype=cp.float64)
    sol_list[-1].h = cp.array([[2., -0.1, -0.15]])
    sol_list.append(kgs.SolutionCollectionLattice())
    sol_list[-1].xyt = cp.array([[
        [0.0, 0.0, 0.0],      # Tree 0 at origin
        [1.5, 0.5, np.pi/4]   # Tree 1 offset and rotated
        ]])/4
    a_length = 2.5/3
    b_length = 2.5/2
    angle = np.pi / 3  # 90 degrees - square lattice
    sol_list[-1].h = cp.array([[a_length, b_length, angle]])
    sol_list.append(kgs.SolutionCollectionLattice())
    sol_list[-1].xyt = cp.array([[
        [0.0, 0.0, 0.0],      # Tree 0 at origin
        [1.0, 0.5, np.pi/4]   # Tree 1 offset and rotated
        ]])/4
    a_length = -2.5/6
    b_length = 2.5/6
    angle = np.pi / 3  # 90 degrees - square lattice
    sol_list[-1].h = cp.array([[a_length, b_length, angle]])
    
    for ii in range(len(sol_list)):
        pack_vis_sol.pack_vis_sol(sol_list[ii], solution_idx=0)
    plt.pause(0.001)

    for c in costs_to_test:
        kgs.set_float32(False)
        # Collect all outputs for vectorized check
        all_ref_outputs = []
        all_fast_outputs = []
        all_xyt = []
        all_bounds = []

        print(f"\nTesting {c.__class__.__name__}")

        for sol_single in sol_list:

            if (isinstance(c, pack_cost.BoundaryDistanceCost) or isinstance(c, pack_cost.BoundaryCost)) and sol_single.periodic:
                continue
            
            # Store for vectorized check
            all_xyt.append(sol_single.xyt)
            all_bounds.append(sol_single.h)
            
            # First, check that compute_cost and compute_cost_ref agree (new API: accept SolutionCollectionSquare)
            cost_ref, grad_ref, grad_bound_ref = c.compute_cost_ref(sol_single)
            sol_fast = copy.deepcopy(sol_single)
            kgs.set_float32(CUDA_float32)
            sol_fast.xyt = cp.array(sol_single.xyt,dtype=kgs.dtype_cp)
            sol_fast.h = cp.array(sol_single.h,dtype=kgs.dtype_cp)
            cost_fast, grad_fast, grad_bound_fast = c.compute_cost_allocate(sol_fast)
            cost_fast_no_grad = c.compute_cost_allocate(sol_fast, evaluate_gradient=False)[0]
            assert cp.all(cost_fast_no_grad == cost_fast)
            
            # Store all outputs
            all_ref_outputs.append((cost_ref, grad_ref, grad_bound_ref))
            all_fast_outputs.append((cost_fast, grad_fast, grad_bound_fast))
            
            # Show full precision
            print(cp.array2string(cp.asarray(cost_fast), precision=17, suppress_small=False))
            print(cp.array2string(cp.asarray(cost_ref), precision=17, suppress_small=False))
            if not isinstance(c, pack_cost.CostDummy):
                assert cost_ref > 0
            assert cp.allclose(cost_ref, cost_fast, rtol=1e-6), f"Cost mismatch: {cost_ref} vs {cost_fast}"
            assert cp.allclose(grad_ref, grad_fast, rtol=1e-4, atol=1e-4), f"Gradient mismatch: {grad_ref} vs {grad_fast}"
            assert cp.allclose(grad_bound_ref, grad_bound_fast, rtol=1e-4, atol=1e-4), f"Bound gradient mismatch: {grad_bound_ref} vs {grad_bound_fast}"

            kgs.set_float32(False)

            # Now check gradients via finite differences
            def _get_cost(obj, xyt_arr):
                # Create same type of solution as the one being tested
                sol_tmp = type(sol_single)()
                sol_tmp.xyt = cp.array(xyt_arr)
                sol_tmp.h = sol_single.h
                if CUDA_float32:
                    return obj.compute_cost_ref(sol_tmp)[0]
                else:
                    return obj.compute_cost_allocate(sol_tmp)[0]

            x0 = sol_single.xyt.copy()
            shape = x0.shape
            x_flat = x0.ravel()
            n = x_flat.size
            eps = 1e-6
            grad_num = cp.zeros(n, dtype=float)

            for i in range(n):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps

                c_plus = _get_cost(c, x_plus.reshape(shape))
                c_minus = _get_cost(c, x_minus.reshape(shape))
                grad_num[i] = (c_plus - c_minus) / (2.0 * eps)

            grad_fast_flat = cp.asarray(grad_fast).ravel()
            max_diff = cp.max(cp.abs(grad_num - grad_fast_flat)).get().item()
            assert cp.allclose(grad_num, grad_fast_flat, rtol=1e-4, atol=1e-4), f"Finite-diff gradient mismatch (max diff {max_diff})"

            # Check bound gradient via finite differences
            def _get_cost_bound(obj, bound_arr):
                # Create same type of solution as the one being tested
                sol_tmp = type(sol_single)()
                sol_tmp.xyt = sol_single.xyt
                sol_tmp.h = cp.array(bound_arr)
                if CUDA_float32:
                    return obj.compute_cost_ref(sol_tmp)[0]
                else:
                    return obj.compute_cost_allocate(sol_tmp)[0]

            b0 = sol_single.h.copy()
            bound_shape = b0.shape
            bound_flat = b0.ravel()
            n_bound = bound_flat.size
            grad_bound_num = cp.zeros(n_bound, dtype=float)

            for i in range(n_bound):
                b_plus = bound_flat.copy()
                b_minus = bound_flat.copy()
                b_plus[i] += eps
                b_minus[i] -= eps

                c_plus = _get_cost_bound(c, b_plus.reshape(bound_shape))
                c_minus = _get_cost_bound(c, b_minus.reshape(bound_shape))
                grad_bound_num[i] = (c_plus - c_minus) / (2.0 * eps)

            grad_bound_fast_flat = cp.asarray(grad_bound_fast).ravel()
            max_diff_bound = cp.max(cp.abs(grad_bound_num - grad_bound_fast_flat)).get().item()
            assert cp.allclose(grad_bound_num, grad_bound_fast_flat, rtol=1e-4, atol=1e-4), f"Finite-diff bound gradient mismatch (max diff {max_diff_bound})"
        
        for todo in ([[0,1]] if (isinstance(c, pack_cost.BoundaryDistanceCost) or isinstance(c, pack_cost.BoundaryCost)) else [[0,1],[2,3]]):
            # Vectorized check: call with all xyt and bounds for this cost function
            print(f"  Vectorized check for {c.__class__.__name__}")
            full_xyt = cp.concatenate(all_xyt[todo[0]:todo[-1]+1], axis=0)
            full_bounds = cp.concatenate(all_bounds[todo[0]:todo[-1]+1], axis=0)

            # Compute vectorized results using kgs.SolutionCollectionSquare
            full_sol = type(sol_list[todo[0]])()
            full_sol.xyt = full_xyt
            full_sol.h = full_bounds
            kgs.set_float32(False)
            print(full_sol.h)
            vec_cost_ref, vec_grad_ref, vec_grad_bound_ref = c.compute_cost_ref(full_sol)
            print(vec_cost_ref)
            kgs.set_float32(CUDA_float32)
            full_sol_fast = type(sol_list[todo[0]])()
            full_sol_fast.xyt = cp.array(full_xyt,dtype=kgs.dtype_cp)
            full_sol_fast.h = cp.array(full_bounds,dtype=kgs.dtype_cp)
            vec_cost_fast, vec_grad_fast, vec_grad_bound_fast = c.compute_cost_allocate(full_sol_fast)
            
            # Check each tree's results
            for i,i2 in enumerate(todo):
                # Get stored individual outputs
                stored_ref = all_ref_outputs[i2]
                stored_fast = all_fast_outputs[i2]
                
                # Extract from vectorized results
                vec_ref_cost_i = vec_cost_ref[i]
                vec_fast_cost_i = vec_cost_fast[i]
                vec_ref_grad_i = vec_grad_ref[i]
                vec_fast_grad_i = vec_grad_fast[i]
                vec_ref_grad_bound_i = vec_grad_bound_ref[i]
                vec_fast_grad_bound_i = vec_grad_bound_fast[i]
                
                # Compare vectorized with individual calls - must be exactly identical
                assert cp.array_equal(vec_ref_cost_i, stored_ref[0][0]), \
                    f"Vectorized ref cost mismatch for {c.__class__.__name__} tree {i}: {vec_ref_cost_i} vs {stored_ref[0]}"
                assert cp.array_equal(vec_fast_cost_i, stored_fast[0][0]), \
                    f"Vectorized fast cost mismatch for {c.__class__.__name__} tree {i}: {vec_fast_cost_i} vs {stored_fast[0]}"
                assert cp.array_equal(vec_ref_grad_i, stored_ref[1][0]), \
                    f"Vectorized ref grad mismatch for {c.__class__.__name__} tree {i}"
                assert cp.array_equal(vec_fast_grad_i, stored_fast[1][0]), \
                    f"Vectorized fast grad mismatch for {c.__class__.__name__} tree {i}"
                assert cp.array_equal(vec_ref_grad_bound_i, stored_ref[2][0]), \
                    f"Vectorized ref bound grad mismatch for {c.__class__.__name__} tree {i}"
                assert cp.array_equal(vec_fast_grad_bound_i, stored_fast[2][0]), \
                    f"Vectorized fast bound grad mismatch for {c.__class__.__name__} tree {i}"
            
            print(f"  ✓ Vectorized results exactly match individual calls")

        
        # # Timing comparisons
        # kgs.debugging_mode = 0
        # n_repeats = 100
        # import time
        # for evaluate_gradient in [False, True]:
        #     start = time.time()
        #     for _ in range(n_repeats):
        #         c.compute_cost(sol_fast, cost_fast, grad_fast, grad_bound_fast, evaluate_gradient=evaluate_gradient)
        #     end = time.time()
        #     time_taken = (end - start) / n_repeats
        #     print(f'Time taken for {c.__class__.__name__} single (evaluate_gradient={evaluate_gradient}): {time_taken*1000:.3f} ms')
        # kgs.debugging_mode = 2


if __name__ == "__main__":
    run_all_tests()