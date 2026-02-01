"""
I/O Module for Packing Solutions

This module handles conversion between internal solution representations and external formats
(pandas DataFrames for Kaggle submissions). It also provides solution legalization functionality
to ensure solutions meet competition constraints (no overlaps, within boundaries).

Key Functions:
- solution_list_to_dataframe: Converts solution objects to submission format
- dataframe_to_solution_list: Parses submission DataFrames back to solution objects
- legalize: Iterative optimization to remove overlaps and satisfy constraints
"""

import pandas as pd
import numpy as np
import cupy as cp
import kaggle_support as kgs
import pack_cost
import pack_dynamics
import copy
import pack_metric


def legalize(sol, move_factor=10., tolerance_rel_change=1e-7, stop_on_cost_increase=False, 
             n_iter=20, target=1e-10, validate=True, verbose=True):
    """
    Legalize a solution by iteratively optimizing to remove overlaps and violations.
    
    Uses BFGS optimization with gradually decreasing step sizes and area cost scaling
    to push trees apart while minimizing boundary size. If validation fails, retries
    with stricter convergence criteria.
    
    Args:
        sol: SolutionCollection object to legalize (must be phenotype)
        move_factor: Multiplier for iteration count per optimization round
        tolerance_rel_change: Relative change threshold for convergence (0=strict)
        stop_on_cost_increase: Whether to halt optimization on cost increase
        n_iter: Maximum number of optimization rounds
        target: Target overlap cost threshold (default 1e-10)
        validate: Whether to validate solution via dataframe conversion
        verbose: Whether to print optimization progress
    
    Returns:
        SolutionCollection: Legalized solution with no overlaps
    
    Raises:
        Exception: If solution cannot be legalized after all attempts
    """
    assert sol.is_phenotype()
    
    # Prepare solution for optimization
    solx = copy.deepcopy(sol)
    solx.use_fixed_h = False
    solx.snap()
    
    # Set up cost function with separation constraint
    import pack_ga3
    ga = pack_ga3.baseline()
    cost = copy.deepcopy(ga.fitness_cost)
    cost.costs[2] = pack_cost.CollisionCostSeparation()
    
    # Create overlap-only cost for monitoring
    cost_overlap = copy.deepcopy(cost)
    cost_overlap.costs.pop(0)
    
    # Configure BFGS optimizer
    optimizer = pack_dynamics.OptimizerBFGS()
    optimizer.cost = copy.deepcopy(cost)
    optimizer.n_iterations = 20000
    optimizer.max_step = 1e-4
    optimizer.history_size = 10
    optimizer.tolerance_rel_change = tolerance_rel_change
    optimizer.use_line_search = False
    optimizer.stop_on_cost_increase = stop_on_cost_increase
    
    if verbose:
        print("Before optimization: ", cost.compute_cost_allocate(solx)[0].get().item(), 
              cost_overlap.compute_cost_allocate(solx)[0].get().item(), solx.h[0,0])
    
    # Iterative optimization with decreasing area cost weight
    for _ in range(n_iter):
        optimizer.cost.costs[0].scaling *= 0.5
        optimizer.max_step *= np.sqrt(0.5)
        solx = optimizer.run_simulation(solx)
        optimizer.n_iterations = np.round(200*move_factor).astype(int)
        
        if verbose:
            print("After optimization: ", cost.compute_cost_allocate(solx)[0].get().item(), 
                  cost_overlap.compute_cost_allocate(solx)[0].get().item(), solx.h[0,0])
        
        if cost_overlap.compute_cost_allocate(solx)[0].get().item() < target:
            break
    
    # Validate the legalized solution
    try:
        if validate:
            solution_list_to_dataframe([solx], compact=False)
        else:
            assert cost_overlap.compute_cost_allocate(solx)[0].get().item() < target
        return solx
    except Exception:
        if tolerance_rel_change == 0.:
            raise Exception('Could not legalize solution')
        else:
            return legalize(solx, move_factor=move_factor, tolerance_rel_change=0., 
                          stop_on_cost_increase=stop_on_cost_increase, n_iter=n_iter, 
                          target=target, validate=validate)


def solution_list_to_dataframe(sol_list, compact=True, compact_hi=1., return_scores=False, print_score=True):
    """
    Convert list of solutions to Kaggle submission DataFrame format.
    
    Transforms internal solution representation (centroid-centered, radians) to
    submission format (original coordinates, degrees, string-encoded). Optionally
    compacts solutions by uniformly scaling to maximize packing density while
    maintaining validity.
    
    Args:
        sol_list: List of SolutionCollection objects (each with N_solutions=1)
        compact: Whether to uniformly scale solutions for tighter packing
        compact_hi: Upper bound for compaction factor search
        return_scores: Whether to return individual scores along with DataFrame
        print_score: Whether to print total score to console
    
    Returns:
        pd.DataFrame: Submission DataFrame with columns ['id', 'x', 'y', 'deg']
        list (optional): Individual scores for each solution if return_scores=True
    
    Notes:
        - Coordinates are converted from centroid-centered to original frame
        - All numeric columns are string-encoded with 's' prefix for Kaggle format
        - Compaction uses boolean_line_search to find maximum valid scale factor
    """
    res_df_list = []
    scores = []
    score = 0.0

    for sol in sol_list:
        sol = copy.deepcopy(sol)
        assert sol.N_solutions == 1
        cols = ['x', 'y', 'deg']

        # Transform centroid-centered coordinates to original frame
        cx, cy = kgs.tree_centroid_offset
        cx, cy = -cx, -cy
        cos_theta = cp.cos(sol.xyt[0, :, 2])
        sin_theta = cp.sin(sol.xyt[0, :, 2])
        
        sol.xyt[0, :, 0] += cx * cos_theta - cy * sin_theta
        sol.xyt[0, :, 1] += cx * sin_theta + cy * cos_theta

        # Convert radians to degrees
        sol.xyt[0, :, 2] *= 360 / 2 / np.pi

        n = sol.N_trees

        # Create submission DataFrame with proper indexing
        submission = pd.DataFrame(
            index=[f'{n:03d}_{t}' for t in range(n)], columns=cols, 
            data=sol.xyt[0].get().astype(np.float64)).rename_axis('id')
        submission = submission.reset_index()

        # Optional compaction via binary search
        if compact and sol.N_trees > 1:
            def f(x):
                submission_copy = copy.deepcopy(submission)
                submission_copy['x'] *= x
                submission_copy['y'] *= x
                for col in submission_copy.columns[1:]:
                    submission_copy[col] = 's' + submission_copy[col].astype('string')
                try:
                    pack_metric.score(submission_copy, submission_copy, '', allow_error=False)
                    return False
                except Exception:
                    return True

            import boolean_line_search
            factor = boolean_line_search.boolean_line_search(f, 0.9, compact_hi)

            submission['x'] *= factor
            submission['y'] *= factor

        # Encode numeric columns as strings (Kaggle format requirement)
        for col in submission.columns[1:]:
            submission[col] = 's' + submission[col].astype('string')
        
        res_df_list.append(submission)
        score = pack_metric.score(submission, submission, '', allow_error=False)
        scores.append(score)

    # Merge all solutions into single submission DataFrame
    res_df = pd.concat(res_df_list, ignore_index=True)

    if print_score:
        print('Score of generated dataframe:', sum(scores))

    if return_scores:
        return res_df, scores
    else:
        return res_df


def dataframe_to_solution_list(df):
    """
    Convert Kaggle submission DataFrame back to internal solution objects.
    
    Inverse operation of solution_list_to_dataframe. Parses string-encoded
    coordinates, converts degrees to radians, and reverses centroid offset
    transformation. Groups DataFrame rows by tree count to reconstruct
    individual solutions.
    
    Args:
        df: pd.DataFrame with columns ['id', 'x', 'y', 'deg'] in Kaggle format
            where 'id' follows pattern 'NNN_T' (N=tree count, T=tree index)
    
    Returns:
        tuple: (sol_list, scores)
            - sol_list: List of SolutionCollectionSquare objects
            - scores: List of scores for each solution
    
    Notes:
        - String-encoded columns (prefix 's') are parsed back to floats
        - Creates SolutionCollectionSquare objects with snapped parameters
    """
    df = df.copy()
    
    sol_list = []
    scores = []

    # Extract tree count and index from ID column
    df['n_trees'] = df['id'].str.split('_').str[0].astype(int)
    df['tree_idx'] = df['id'].str.split('_').str[1].astype(int)

    for n_trees, group in df.groupby('n_trees', sort=False):
        scores.append(pack_metric.score(group, group, '', allow_error=False))
        group = group.sort_values('tree_idx')
        
        # Decode string-encoded numeric columns
        for col in ['x', 'y', 'deg']:
            if group[col].dtype == object and group[col].str.startswith('s').all():
                group[col] = group[col].str[1:].astype(float)

        # Extract position and rotation data
        x = group['x'].values
        y = group['y'].values
        deg = group['deg'].values

        # Convert degrees to radians
        theta = deg * 2 * np.pi / 360

        # Reverse centroid offset transformation
        cx, cy = kgs.tree_centroid_offset
        cx, cy = -cx, -cy
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x -= cx * cos_theta - cy * sin_theta
        y -= cx * sin_theta + cy * cos_theta

        # Create GPU array in solution format
        xyt = cp.array([x, y, theta]).T
        xyt = xyt[cp.newaxis, :, :]

        # Instantiate solution object
        sol = kgs.SolutionCollectionSquare(xyt=xyt)
        sol.snap()

        sol_list.append(sol)

    return sol_list, scores