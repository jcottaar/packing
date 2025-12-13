import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
from typeguard import typechecked
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
import pack_cost
import pack_vis_sol
import pack_dynamics
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import lap_batch
from itertools import product

def create_dimer():
    res = cp.array([[-0.15/2, 0.1+kgs.tree_centroid_offset[1], 0.],[0.15/2, -0.1-kgs.tree_centroid_offset[1], np.pi]], dtype=kgs.dtype_cp)
    res[:,:2]*=kgs.just_over_one
    return res

def snap_cell(sol, skip_assert=False):
    # Snaps crystal axis lengths to as small as possible
    cost = pack_cost.CollisionCostOverlappingArea()
    def _overlaps(sol):
        cost_val = cost.compute_cost_allocate(sol, evaluate_gradient=False)[0]
        return cost_val>0
    # implement line search on h[0,0] and h[0,1] to satisfy constraints, using _overlaps.
    import boolean_line_search
    def _snap_axis(axis_idx, max_iter=60):
        orig = sol.h[:, axis_idx].copy()
        sol_tmp = copy.deepcopy(sol)
        def f(factor):      
            nonlocal sol_tmp      
            sol_tmp.h[:, axis_idx] = orig * factor
            return _overlaps(sol_tmp)
        # lo: overlap (True), hi: no overlap (False)
        hi = 100.
        lo = 0.01
        factor = boolean_line_search.boolean_line_search_vectorized(f, lo, hi, sol.N_solutions, max_iter=max_iter)
        sol.h[:, axis_idx] = orig * factor

    _snap_axis(1)   
    if not skip_assert:
        assert cp.all(~_overlaps(sol))
        sol_tmp = copy.deepcopy(sol)
        sol_tmp.h[:,1]/=kgs.just_over_one
        assert(cp.all(_overlaps(sol_tmp)))
    _snap_axis(0)
    if not skip_assert:
        assert cp.all(~_overlaps(sol))
        sol_tmp = copy.deepcopy(sol)
        sol_tmp.h[:,0]/=kgs.just_over_one
        assert(cp.all(_overlaps(sol_tmp)))  
        sol_tmp = copy.deepcopy(sol)
        sol_tmp.h[:,1]/=kgs.just_over_one
        assert(cp.all(_overlaps(sol_tmp)))  

@kgs.profile_each_line
def try_tilings(sol, N_max=20, show_one=False):
    res = []
    for tile_x, tile_y, remove_top_row, remove_bottom_row, remove_left_col, remove_right_col in \
        product(range(2, N_max+1), range(2, N_max+1), [False, True], [False, True], [False, True], [False, True]):
        sol_here = kgs.SolutionCollectionSquare()
        # make an tile_x by tile_y tiling of sol, based on the crystal_axes of sol

        # Get crystal axes from the input solution
        crystal_axes = sol.get_crystal_axes_allocate()  # shape: (N_solutions, 2, 2)

        # Assuming sol has a single solution (N_solutions=1)
        axis_a = crystal_axes[0, 0, :]  # first lattice vector
        axis_b = crystal_axes[0, 1, :]  # second lattice vector

        # Get the original tree positions from sol
        xyt_orig = sol.xyt[0]  # shape: (N_trees, 3)
        N_trees_per_cell = xyt_orig.shape[0]

        # Create tiled solution with tile_x * tile_y * N_trees_per_cell trees
        N_trees_total = tile_x * tile_y * N_trees_per_cell

        # Vectorized tiling:
        # Create grid indices for all tiles
        i_indices = cp.arange(tile_x, dtype=kgs.dtype_cp)  # shape: (tile_x,)
        j_indices = cp.arange(tile_y, dtype=kgs.dtype_cp)  # shape: (tile_y,)

        # Create meshgrid for all tile positions
        i_grid, j_grid = cp.meshgrid(i_indices, j_indices, indexing='ij')  # both shape: (tile_x, tile_y)

        # Flatten to get all tile positions
        i_flat = i_grid.ravel()  # shape: (tile_x * tile_y,)
        j_flat = j_grid.ravel()  # shape: (tile_x * tile_y,)

        # Compute all translation vectors at once
        # translations shape: (tile_x * tile_y, 2)
        translations = i_flat[:, cp.newaxis] * axis_a + j_flat[:, cp.newaxis] * axis_b

        # Replicate original trees for all tiles
        # xyt_orig shape: (N_trees_per_cell, 3)
        # Repeat each tree pattern for each tile
        xyt_tiled = cp.tile(xyt_orig, (tile_x * tile_y, 1))  # shape: (N_trees_total, 3)

        # Add translations - each group of N_trees_per_cell trees gets the same translation
        # Repeat each translation N_trees_per_cell times
        translations_expanded = cp.repeat(translations, N_trees_per_cell, axis=0)  # shape: (N_trees_total, 2)
        xyt_tiled[:, 0] += translations_expanded[:, 0]
        xyt_tiled[:, 1] += translations_expanded[:, 1]

        # Reshape to (1, N_trees_total, 3) for SolutionCollectionSquare
        xyt_tiled = xyt_tiled[cp.newaxis, :, :]

        # remove the top row or column of trees; don't go back to translations, but just remove trees with max/min x or y value (with tolerance 1e-5)
        if remove_top_row or remove_bottom_row or remove_left_col or remove_right_col:
            # Extract x and y coordinates
            x_coords = xyt_tiled[0, :, 0]
            y_coords = xyt_tiled[0, :, 1]

            # Create mask for trees to keep (start with all True)
            keep_mask = cp.ones(xyt_tiled.shape[1], dtype=bool)

            tol = 1e-5

            if remove_top_row:
                # Remove trees with maximum y value
                y_max = cp.max(y_coords)
                keep_mask &= (y_coords < y_max - tol)

            if remove_bottom_row:
                # Remove trees with minimum y value
                y_min = cp.min(y_coords)
                keep_mask &= (y_coords > y_min + tol)

            if remove_right_col:
                # Remove trees with maximum x value
                x_max = cp.max(x_coords)
                keep_mask &= (x_coords < x_max - tol)

            if remove_left_col:
                # Remove trees with minimum x value
                x_min = cp.min(x_coords)
                keep_mask &= (x_coords > x_min + tol)

            # Filter trees based on mask
            xyt_tiled = xyt_tiled[:, keep_mask, :]
            N_trees_total = xyt_tiled.shape[1]

        # Set the tiled solution
        sol_here.xyt = xyt_tiled
        sol_here.snap()  # Compute the bounding square
        if show_one and tile_x==5 and tile_y==5:
            pack_vis_sol.pack_vis_sol(sol_here, solution_idx=0)
        res.append([sol_here.h[0,0].get(), N_trees_total, sol_here])
    best_per_tree = np.zeros(200)
    sol_per_tree = []
    for i_tree in range(200):
        # Find all solutions with at least i_tree+1 trees
        candidates = [(area, N_trees, sol_here) for area, N_trees, sol_here in res if N_trees >= i_tree+1]
        assert(candidates)
        if candidates:
            # Find the solution with minimal area
            min_area, min_N_trees, min_sol = min(candidates, key=lambda x: x[0])
            min_sol = copy.deepcopy(min_sol)
            best_per_tree[i_tree] = min_area
            min_sol.xyt = min_sol.xyt[:, :i_tree+1, :]
            assert(min_sol.N_trees == i_tree+1)
            sol_per_tree.append(min_sol)
    return best_per_tree, sol_per_tree

