import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '../core'))
import kaggle_support as kgs
import cupy as cp
from dataclasses import dataclass
import pack_cost
import copy
from typeguard import typechecked
from torch.utils.dlpack import to_dlpack
import torch

@dataclass
class OptimizerBFGS(kgs.BaseClass):

    # Hyperparameters
    cost = None
    n_iterations = 100
    history_size = 3
    max_step = 0.01
    tolerance_rel_change = 0.  # Relative change tolerance for convergence
    stop_on_cost_increase = False  # Stop optimization if cost increases
    use_line_search = False

    def __post_init__(self):
        super().__post_init__()
        self.cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=1.)])
        
    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):

        sol.check_constraints()
        sol = copy.deepcopy(sol)
        sol.prep_for_phenotype()

        sol_tmp = copy.deepcopy(sol)

        counter = 0

        x0 = cp.concatenate((sol_tmp.xyt.reshape(sol_tmp.N_solutions,-1), sol_tmp.h.reshape(sol_tmp.N_solutions,-1)),axis=1)
        tmp_x = cp.zeros_like(x0, dtype=kgs.dtype_cp)
        tmp_res = cp.zeros_like(x0, dtype=kgs.dtype_cp)
        x0 = torch.from_dlpack(x0.__dlpack__())
        
        tmp_xyt = copy.deepcopy(sol.xyt)
        tmp_h = copy.deepcopy(sol.h)
        tmp_cost = cp.zeros(sol.N_solutions, dtype=kgs.dtype_cp)
        tmp_grad = cp.zeros_like(sol.xyt, dtype=kgs.dtype_cp)
        tmp_grad_h = cp.zeros_like(sol.h, dtype=kgs.dtype_cp)
        
        def f_torch(x, is_main_loop):            
            nonlocal tmp_xyt, tmp_h, tmp_cost, tmp_grad, tmp_grad_h, tmp_res, tmp_x
            tmp_x[:x.shape[0]] = cp.from_dlpack(to_dlpack(x))
            nonlocal sol_tmp, counter
            counter+=1
            N_split = sol_tmp.xyt.shape[1]*3
            N = x.shape[0]

            # This has to go via tmp_xyt/tmp_h for contiguity
            tmp_xyt[:N, :] = tmp_x[:N,:N_split].reshape(N,-1,3)
            tmp_h[:N, :] = tmp_x[:N,N_split:].reshape(N,-1)
            sol_tmp.xyt = tmp_xyt[:N, :]
            sol_tmp.h = tmp_h[:N, :]

            self.cost.compute_cost(sol_tmp, tmp_cost[:N], tmp_grad[:N, :], tmp_grad_h[:N, :])
                
            res = cp.zeros_like(tmp_x[:N,:], dtype=kgs.dtype_cp)
            res[:,:N_split] = tmp_grad[:N, :].reshape(sol_tmp.N_solutions,-1)
            res[:,N_split:] = tmp_grad_h[:N, :].reshape(sol_tmp.N_solutions,-1)
            return torch.from_dlpack(tmp_cost[:N].__dlpack__()), torch.from_dlpack(res.__dlpack__())
        
        

        import lbfgs_torch_parallel
        results = lbfgs_torch_parallel.lbfgs(
            f_torch,x0,tolerance_grad=0, tolerance_change=0, tolerance_rel_change=self.tolerance_rel_change, max_iter=self.n_iterations, history_size=self.history_size, max_step=self.max_step,
            line_search_fn = 'strong_wolfe' if self.use_line_search else None, stop_on_cost_increase=self.stop_on_cost_increase)
        x_result = cp.from_dlpack(to_dlpack(results))
        sol.xyt = cp.ascontiguousarray(x_result[:,:sol.xyt.shape[1]*3].reshape(sol.N_solutions,-1,3))
        sol.h = cp.ascontiguousarray(x_result[:,sol.xyt.shape[1]*3:].reshape(sol.N_solutions,-1))

        sol.unprep_for_phenotype()
        return sol

def run_simulation_list(simulator, solution_list):
    """
    Run simulations on a list of SolutionCollection objects efficiently by grouping
    solutions with the same number of trees.
    
    Args:
        simulator: Any class with a run_simulation(sol: kgs.SolutionCollection) method
        solution_list: List of kgs.SolutionCollection objects (all same subclass)
    
    The function:
    1. Groups solutions by N_trees
    2. Merges each group into a single SolutionCollection (verifying compatibility)
    3. Runs simulation once per merged group
    4. Scatters results back to original solutions (modifies in-place)
    """
    if len(solution_list) == 0:
        return
    
    # Group solutions by N_trees
    from collections import defaultdict
    groups = defaultdict(list)
    
    for sol in solution_list:
        N_trees = sol.xyt.shape[1]
        groups[N_trees].append(sol)
    
    # Process each group
    for N_trees, group_sols in groups.items():
        if len(group_sols) == 1:
            # Single solution - just run directly and continue
            result = simulator.run_simulation(group_sols[0])
            group_sols[0].xyt[:] = result.xyt
            group_sols[0].h[:] = result.h
            continue
        
        # Verify compatibility if debugging
        if kgs.debugging_mode >= 2:
            # Check all solutions are same subclass
            first_type = type(group_sols[0])
            for sol in group_sols[1:]:
                if type(sol) is not first_type:
                    raise ValueError(f"Solution type mismatch in group: {first_type} vs {type(sol)}")
            
            # Check all properties except xyt and h match
            first_sol = group_sols[0]
            for i, sol in enumerate(group_sols[1:], 1):
                # Make copies and exclude xyt and h for comparison
                first_sol_copy = copy.deepcopy(first_sol)
                sol_copy = copy.deepcopy(sol)
                
                first_sol_copy.xyt = None
                first_sol_copy.h = None
                sol_copy.xyt = None
                sol_copy.h = None
                
                # Compare the copies
                if first_sol_copy != sol_copy:
                    raise ValueError(f"Solution {i} properties mismatch with first solution (excluding xyt and h)")
        
        # Merge: stack xyt and h along N_solutions dimension
        merged_xyt = cp.concatenate([sol.xyt for sol in group_sols], axis=0)
        merged_h = cp.concatenate([sol.h for sol in group_sols], axis=0)
        
        # Create merged solution using first solution as template
        merged_sol = copy.deepcopy(group_sols[0])
        merged_sol.xyt = merged_xyt
        merged_sol.h = merged_h
        
        # Run simulation on merged solution
        result = simulator.run_simulation(merged_sol)
        
        # Scatter results back to original solutions
        start_idx = 0
        for sol in group_sols:
            N_sol = sol.xyt.shape[0]
            sol.xyt[:] = result.xyt[start_idx:start_idx + N_sol]
            sol.h[:] = result.h[start_idx:start_idx + N_sol]
            start_idx += N_sol