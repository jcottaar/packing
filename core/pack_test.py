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
pack_cuda.USE_FLOAT32 = True

def run_all_tests():
    test_costs()
    print("All tests passed.")

def test_costs():
    #cost = pack_cost.PackingCost()
    #cost.collision_cost = pack_cost.CollisionCostOverlappingArea()
    #costs_to_test = [cost]

    print('add bounds grad check')
    costs_to_test = [pack_cost.CostDummy(), pack_cost.CollisionCostOverlappingArea(scaling=3.)]

    tree_list = pack_basics.place_random(10, 1.5)
    #tree_list = pack_basics.place_random(3, 0.1)
    #tree_list.xyt = [[5.,5.,5.],[1.,2,3.],[1.,2.,3.01]]
    pack_vis.visualize_tree_list(tree_list)

    bounds = cp.array([1.])

    for c in costs_to_test:
        # First, check that compute_cost and compute_cost_ref agree
        cost_ref, grad_ref, grad_bound_ref = c.compute_cost_ref(cp.array(tree_list.xyt), bounds)
        cost_fast, grad_fast, grad_fast_ref = c.compute_cost(cp.array(tree_list.xyt), bounds)        
        # Show full precision
        print(cp.array2string(cp.asarray(cost_fast), precision=17, suppress_small=False))
        print(cp.array2string(cp.asarray(cost_ref), precision=17, suppress_small=False))
        if not isinstance(c, pack_cost.CostDummy):
            assert cost_ref>0
        assert cp.allclose(cost_ref, cost_fast, rtol=1e-6), f"Cost mismatch: {cost_ref} vs {cost_fast}"
        assert cp.allclose(grad_ref, grad_fast, rtol=1e-4, atol=1e-4), f"Gradient mismatch: {grad_ref} vs {grad_fast}"

        # Now check gradients via finite differences
        def _get_cost(obj, xyt_arr):
            return obj.compute_cost_ref(cp.array(xyt_arr), bounds)[0]

        x0 = tree_list.xyt.copy()
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
        #print(grad_num, grad_fast_flat)
        assert cp.allclose(grad_num, grad_fast_flat, rtol=1e-4, atol=1e-6), f"Finite-diff gradient mismatch (max diff {max_diff})"