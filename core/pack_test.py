import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
from typeguard import typechecked
import pack_cost
import pack_basics

def run_all_tests():
    test_costs()
    print("All tests passed.")

def test_costs():
    cost = pack_cost.PackingCost()
    cost.collision_cost = pack_cost.CollisionCostOverlappingArea()
    costs_to_test = [cost]

    tree_list = pack_basics.place_random(2, 1)
    for c in costs_to_test:
        # First, check that compute_cost and compute_cost_ref agree
        cost_ref, grad_ref = c.compute_total_cost_ref(tree_list.xyt, include_gradients=True)
        cost_fast, grad_fast = c.compute_total_cost(tree_list.xyt, include_gradients=True)
        assert cost_ref>0
        assert np.allclose(cost_ref, cost_fast, rtol=1e-6), f"Cost mismatch: {cost_ref} vs {cost_fast}"
        assert np.allclose(grad_ref, grad_fast, rtol=1e-6), f"Gradient mismatch: {grad_ref} vs {grad_fast}"

        # Now check gradients via finite differences
        def _get_cost(obj, xyt_arr):
            return obj.compute_total_cost(xyt_arr, include_gradients=False)[0]

        x0 = tree_list.xyt.copy()
        shape = x0.shape
        x_flat = x0.ravel()
        n = x_flat.size
        eps = 1e-6
        grad_num = np.zeros(n, dtype=float)

        for i in range(n):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            c_plus = _get_cost(c, x_plus.reshape(shape))
            c_minus = _get_cost(c, x_minus.reshape(shape))
            grad_num[i] = (c_plus - c_minus) / (2.0 * eps)

        grad_fast_flat = np.asarray(grad_fast).ravel()
        max_diff = np.max(np.abs(grad_num - grad_fast_flat))
        assert np.allclose(grad_num, grad_fast_flat, rtol=1e-4, atol=1e-6), f"Finite-diff gradient mismatch (max diff {max_diff})"