import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
from typeguard import typechecked
import pack_cost
import pack_basics
import pack_vis
import pack_cuda
import pack_cuda_primitives_test
pack_cuda.USE_FLOAT32 = True

def run_all_tests():
    pack_cuda_primitives_test.run_all_tests()
    test_costs()
    print("All tests passed.")

def test_costs():
    print('Testing cost computation and gradients')
    costs_to_test = [pack_cost.CostDummy(), pack_cost.AreaCost(), pack_cost.BoundaryDistanceCost(use_kernel=True), pack_cost.BoundaryDistanceCost(use_kernel=False), pack_cost.BoundaryCost(), pack_cost.CollisionCostOverlappingArea(scaling=3.), 
                     pack_cost.CostCompound(costs=[pack_cost.AreaCost(), pack_cost.BoundaryCost()])]

    tree_list = []
    tree_list.append(pack_basics.place_random(10, 1.5, generator=np.random.default_rng(seed=0)))
    tree_list.append(pack_basics.place_random(10, 1.5, generator=np.random.default_rng(seed=1)))
    pack_vis.visualize_tree_list(tree_list[0])
    pack_vis.visualize_tree_list(tree_list[1])

    bounds = cp.array([[0.5],[2.]])  # square bounds


    for c in costs_to_test:
        # Collect all outputs for vectorized check
        all_ref_outputs = []
        all_fast_outputs = []
        all_xyt = []
        all_bounds = []

        print(f"\nTesting {c.__class__.__name__}")

        for t, b in zip(tree_list, bounds):
            xyt_single = cp.array(t.xyt[None])
            b_single = b[None]
            sol_single = kgs.SolutionCollection()
            sol_single.xyt = xyt_single
            sol_single.h = b_single
            
            # Store for vectorized check
            all_xyt.append(xyt_single)
            all_bounds.append(b_single)
            
            # First, check that compute_cost and compute_cost_ref agree (new API: accept SolutionCollection)
            cost_ref, grad_ref, grad_bound_ref = c.compute_cost_ref(sol_single)
            sol_fast = kgs.SolutionCollection()
            sol_fast.xyt = cp.array(xyt_single,dtype=cp.float32)
            sol_fast.h = cp.array(b_single,dtype=cp.float32)
            cost_fast, grad_fast, grad_bound_fast = c.compute_cost(sol_fast)
            
            # Store all outputs
            all_ref_outputs.append((cost_ref, grad_ref, grad_bound_ref))
            all_fast_outputs.append((cost_fast, grad_fast, grad_bound_fast))
            
            # Show full precision
            print(cp.array2string(cp.asarray(cost_fast), precision=17, suppress_small=False))
            print(cp.array2string(cp.asarray(cost_ref), precision=17, suppress_small=False))
            if not isinstance(c, pack_cost.CostDummy):
                assert cost_ref > 0
            assert cp.allclose(cost_ref, cost_fast, rtol=1e-6), f"Cost mismatch: {cost_ref} vs {cost_fast}"
            assert cp.allclose(grad_ref, grad_fast, rtol=1e-2, atol=1e-2), f"Gradient mismatch: {grad_ref} vs {grad_fast}"
            assert cp.allclose(grad_bound_ref, grad_bound_fast, rtol=1e-2, atol=1e-2), f"Bound gradient mismatch: {grad_bound_ref} vs {grad_bound_fast}"
            print('back to 1e-4 later, also below')

            # Now check gradients via finite differences
            def _get_cost(obj, xyt_arr):
                sol_tmp = kgs.SolutionCollection()
                sol_tmp.xyt = cp.array(xyt_arr[None])
                sol_tmp.h = b_single
                return obj.compute_cost_ref(sol_tmp)[0]

            x0 = t.xyt.copy()
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
            assert cp.allclose(grad_num, grad_fast_flat, rtol=1e-2, atol=1e-2), f"Finite-diff gradient mismatch (max diff {max_diff})"

            # Check bound gradient via finite differences
            def _get_cost_bound(obj, bound_arr):
                sol_tmp = kgs.SolutionCollection()
                sol_tmp.xyt = xyt_single
                sol_tmp.h = cp.array(bound_arr[None])
                return obj.compute_cost_ref(sol_tmp)[0]

            b0 = b.copy()
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
            assert cp.allclose(grad_bound_num, grad_bound_fast_flat, rtol=1e-2, atol=1e-2), f"Finite-diff bound gradient mismatch (max diff {max_diff_bound})"
        
        # Vectorized check: call with all xyt and bounds for this cost function
        print(f"  Vectorized check for {c.__class__.__name__}")
        full_xyt = cp.concatenate(all_xyt, axis=0)
        full_bounds = cp.concatenate(all_bounds, axis=0)
        print(full_bounds)

        # Compute vectorized results using kgs.SolutionCollection
        full_sol = kgs.SolutionCollection()
        full_sol.xyt = full_xyt
        full_sol.h = full_bounds
        vec_cost_ref, vec_grad_ref, vec_grad_bound_ref = c.compute_cost_ref(full_sol)
        full_sol_fast = kgs.SolutionCollection()
        full_sol_fast.xyt = cp.array(full_xyt,dtype=cp.float32)
        full_sol_fast.h = cp.array(full_bounds,dtype=cp.float32)
        vec_cost_fast, vec_grad_fast, vec_grad_bound_fast = c.compute_cost(full_sol_fast)
        
        # Check each tree's results
        for i in range(len(tree_list)):
            # Get stored individual outputs
            stored_ref = all_ref_outputs[i]
            stored_fast = all_fast_outputs[i]
            
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

if __name__ == "__main__":
    run_all_tests()