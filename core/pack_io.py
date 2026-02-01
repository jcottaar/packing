import pandas as pd
import numpy as np
import cupy as cp
import kaggle_support as kgs
import pack_cost
import pack_dynamics
import copy
import pack_metric

def legalize(sol, do_plot=False, move_factor=10., tolerance_rel_change=1e-7, stop_on_cost_increase = False, n_iter=20, target=1e-10, validate=True, line_search=False, verbose=True, scaling=1.):
    assert sol.is_phenotype()
    solx = copy.deepcopy(sol)
    solx.use_fixed_h = False
    solx.snap()
    import pack_ga3
    ga = pack_ga3.baseline()
    cost = copy.deepcopy(ga.fitness_cost)
    cost.costs[2] = pack_cost.CollisionCostSeparation()
    #cost.costs[0].scaling*=0.1
    cost_overlap = copy.deepcopy(cost)
    cost_overlap.costs.pop(0)
    optimizer = pack_dynamics.OptimizerBFGS()
    optimizer.cost = copy.deepcopy(cost)
    optimizer.n_iterations = 20000
    optimizer.max_step = 1e-4
    optimizer.cost.costs[0].scaling*=scaling
    optimizer.max_step*=np.sqrt(scaling)    
    optimizer.history_size = 10
    optimizer.tolerance_rel_change = tolerance_rel_change
    optimizer.track_cost = do_plot
    optimizer.plot_cost = do_plot
    optimizer.use_line_search = line_search
    optimizer.stop_on_cost_increase = stop_on_cost_increase
    if verbose:
        print("Before optimization: ", cost.compute_cost_allocate(solx)[0].get().item(), cost_overlap.compute_cost_allocate(solx)[0].get().item(), solx.h[0,0])
    for _ in range(n_iter):
        optimizer.cost.costs[0].scaling*=0.5
        optimizer.max_step*=np.sqrt(0.5)    
        solx = optimizer.run_simulation(solx)
        optimizer.n_iterations = np.round(200*move_factor).astype(int)
        if verbose:
            print("After optimization: ", cost.compute_cost_allocate(solx)[0].get().item(), cost_overlap.compute_cost_allocate(solx)[0].get().item(), solx.h[0,0])
        if cost_overlap.compute_cost_allocate(solx)[0].get().item()<target:
            break   
    try:
        if validate:
            solution_list_to_dataframe([solx], compact=False)
        else:
            assert cost_overlap.compute_cost_allocate(solx)[0].get().item()<target
        return solx
    except:
        if tolerance_rel_change==0.:
            raise Exception('Could not legalize solution')
        else:
            return legalize(solx, do_plot=do_plot, move_factor=move_factor, tolerance_rel_change=0., stop_on_cost_increase=stop_on_cost_increase, n_iter=n_iter, target=target, validate=validate)


def solution_list_to_dataframe(sol_list, compact=True, compact_hi=1., return_scores=False, print_score=True):
    res_df_list = []
    scores = []
    score = 0.0

    for sol in sol_list:
        sol = copy.deepcopy(sol)
        assert sol.N_solutions == 1
        cols = ['x', 'y', 'deg']

        # Convert from centroid-centered coordinates back to original coordinates
        # The centroid offset (cx, cy) was subtracted when recentering, so we add it back
        # But we need to rotate the offset by each tree's angle first
        cx, cy = kgs.tree_centroid_offset
        cx,cy=-cx,-cy
        cos_theta = cp.cos(sol.xyt[0,:,2])
        sin_theta = cp.sin(sol.xyt[0,:,2])
        # Rotate the offset by each tree's angle and add to position
        sol.xyt[0,:,0] += cx * cos_theta - cy * sin_theta
        sol.xyt[0,:,1] += cx * sin_theta + cy * cos_theta

        sol.xyt[0,:,2]*=360/2/np.pi

        n = sol.N_trees

        submission = pd.DataFrame(
            index=[f'{n:03d}_{t}' for t in range(n)], columns=cols, data=sol.xyt[0].get().astype(np.float64)).rename_axis('id')
        submission = submission.reset_index()


        if compact and sol.N_trees>1:
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
                    #rint(err)
                    return True

            import boolean_line_search
            factor = boolean_line_search.boolean_line_search(f, 0.9, compact_hi)
            #print(sol.N_trees, factor)
            submission['x'] *= factor
            submission['y'] *= factor


        # To ensure everything is kept as a string, prepend an 's'
        for col in submission.columns[1:]:
            submission[col] = 's' + submission[col].astype('string')
        res_df_list.append(submission)
        score = pack_metric.score(submission, submission, '', allow_error=False)
        scores.append(score)

    # merge res_df_list into a single dataframe
    res_df = pd.concat(res_df_list, ignore_index=True)\

    if print_score:
        print('Score of generated dataframe:', sum(scores))

    if return_scores:
        return res_df, scores
    else:
        return res_df

def dataframe_to_solution_list(df):
    """
    Inverse of solution_list_to_dataframe.
    Converts a dataframe back to a list of solution objects.
    Does not apply compacting (assumes compact=False was used).
    """
    # Parse numeric columns by removing 's' prefix

    df = df.copy()
    
    sol_list = []
    scores = []

    # Group by the number of trees (extracted from id format 'NNN_T')
    df['n_trees'] = df['id'].str.split('_').str[0].astype(int)
    df['tree_idx'] = df['id'].str.split('_').str[1].astype(int)

    for n_trees, group in df.groupby('n_trees', sort=False):
        # Sort by tree index to ensure correct order
        scores.append(pack_metric.score(group, group, '', allow_error=False))
        group = group.sort_values('tree_idx')
        
        for col in ['x', 'y', 'deg']:
            if group[col].dtype == object and group[col].str.startswith('s').all():
                group[col] = group[col].str[1:].astype(float)

        # Extract x, y, deg values
        x = group['x'].values
        y = group['y'].values
        deg = group['deg'].values

        # Convert degrees back to radians
        theta = deg * 2 * np.pi / 360

        # Reverse the centroid offset transformation
        # Original: sol.xyt[0,:,0] += cx * cos_theta - cy * sin_theta
        # Reverse: sol.xyt[0,:,0] -= cx * cos_theta - cy * sin_theta
        cx, cy = kgs.tree_centroid_offset
        cx, cy = -cx, -cy
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x -= cx * cos_theta - cy * sin_theta
        y -= cx * sin_theta + cy * cos_theta

        # Create xyt array on GPU
        xyt = cp.array([x, y, theta]).T  # Shape: (n_trees, 3)
        xyt = xyt[cp.newaxis, :, :]  # Shape: (1, n_trees, 3)

        # Create a solution object (using the appropriate class)
        # We need to determine which solution class to use based on the data
        # For now, create a generic solution collection
        sol = kgs.SolutionCollectionSquare(xyt=xyt)

        sol.snap()

        sol_list.append(sol)
        

    return sol_list, scores