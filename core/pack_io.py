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
import pack_metric

def solution_list_to_dataframe(sol_list, compact=True):
    res_df_list = []
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
                except:
                    return True

            import boolean_line_search
            factor = boolean_line_search.boolean_line_search(f, 0.999, 1.001)
            #print(sol.N_trees, factor)
            submission['x'] *= factor
            submission['y'] *= factor


        # To ensure everything is kept as a string, prepend an 's'
        for col in submission.columns[1:]:
            submission[col] = 's' + submission[col].astype('string')
        res_df_list.append(submission)
        score += pack_metric.score(submission, submission, '', allow_error=False)

    # merge res_df_list into a single dataframe
    res_df = pd.concat(res_df_list, ignore_index=True)\

    print('Score of generated dataframe:', score)

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