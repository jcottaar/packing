import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '../core'))
import kaggle_support as kgs
import importlib
import matplotlib.pyplot as plt
importlib.reload(kgs)
import numpy as np
import cupy as cp
from dataclasses import dataclass, field, fields
import pack_cuda
import pack_vis
import pack_cost
import copy
from IPython.display import HTML, display, clear_output
from scipy import stats
from typeguard import typechecked

print('Preallocate for cost')

@dataclass
class Optimizer(kgs.BaseClass):
    # Minimizes the cost, not physics-based

    # Configuration    
    plot_interval = None

    # Hyperparameters
    cost = None
    dt = 0.1
    n_iterations = 100
    max_grad_norm = 10.0  # Clip gradients to prevent violent repulsion

    def __post_init__(self):
        super().__post_init__()
        self.cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-3), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=1.)])

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):
        # Initial configuration

        sol.check_constraints()
        sol = copy.deepcopy(sol)
        sol.snap()

        xyt = sol.xyt
        h = sol.h        
        
        n_ensembles = xyt.shape[0]
        n_trees = xyt.shape[1] 

        if self.plot_interval is not None:          
            fig, ax = plt.subplots(figsize=(8, 8))
            tree_list = kgs.TreeList()

        # Pre-allocate gradient arrays once (float32 for efficiency)
        total_cost = cp.zeros(n_ensembles, dtype=cp.float32)
        total_grad = cp.zeros_like(xyt, dtype=cp.float32)
        bound_grad = cp.zeros_like(h, dtype=cp.float32)

        t_total0 = np.float32(0.)   
        t_last_plot = np.float32(-np.inf)     
        for i_iteration in range(self.n_iterations):
            dt = self.dt
            # Reuse pre-allocated arrays
            total_cost[:] = 0
            total_grad[:] = 0
            bound_grad[:] = 0
            self.cost.compute_cost(sol, total_cost, total_grad, bound_grad)
            
            # Clip gradients per tree to prevent violent repulsion
            if self.max_grad_norm is not None:
                grad_norms = cp.sqrt(cp.sum(total_grad**2, axis=2))  # (n_ensembles, n_trees)
                grad_norms = cp.maximum(grad_norms, 1e-8)  # Avoid division by zero
                clip_factor = cp.minimum(1.0, self.max_grad_norm / grad_norms)  # (n_ensembles, n_trees)
                total_grad = total_grad * clip_factor[:, :, None]  # Apply to each component
            
            xyt -= dt * total_grad
            h -= dt * bound_grad
            t_total0 += dt
            
            if self.plot_interval is not None and t_total0 - t_last_plot >= self.plot_interval*0.999:
                t_last_plot = t_total0+0               
                ax.clear()
                ax.set_aspect('equal', adjustable='box')
                tree_list.xyt = cp.asnumpy(xyt[0])
                pack_vis.visualize_tree_list(tree_list, ax=ax, h=cp.asnumpy(h[0]))
                ax.set_title(f'Time: {t_total0:.2f}')
                display(fig)
                clear_output(wait=True)       
        return sol

@dataclass
class Dynamics(kgs.BaseClass):
    # Physics-based dynamics

    # Configuration    
    plot_interval = None

    # Hyperparameters
    cost0 = None # scales
    cost1 = None # doens't scale
    dt_list = None
    friction_list = None
    cost_0_scaling_list = None

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):
        # Initial configuration

        sol.check_constraints()
        sol = copy.deepcopy(sol)

        cost0 = copy.deepcopy(self.cost0)
        cost1 = copy.deepcopy(self.cost1)

        n_ensembles = sol.xyt.shape[0]
        n_trees = sol.xyt.shape[1]
        assert self.dt_list.shape == self.friction_list.shape == self.cost_0_scaling_list.shape
        assert self.dt_list.shape[0] == n_ensembles

        if self.plot_interval is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            tree_list = kgs.TreeList()

        t_total0 = np.float32(0.)      
        t_last_plot = np.float32(-np.inf)  
        velocity_xyt = cp.zeros_like(sol.xyt)
        velocity_h = cp.zeros_like(sol.h)
        prev_cost_0_scaling = 0.
        for i_iteration in range(self.dt_list.shape[1]):
            dt = self.dt_list[:, i_iteration]
            friction = self.friction_list[:, i_iteration]
            cost_0_scaling = self.cost_0_scaling_list[:, i_iteration]  
            if cost_0_scaling[0]>0 and prev_cost_0_scaling[0] == 0.: # assuming this applies to all...
                sol.snap()
            prev_cost_0_scaling = cost_0_scaling        
            total_cost0, total_grad0, bound_grad0 = cost0.compute_cost_allocate(sol)
            total_cost1, total_grad1, bound_grad1 = cost1.compute_cost_allocate(sol)
            total_cost = total_cost0 * cost_0_scaling + total_cost1
            total_grad = total_grad0 * cost_0_scaling[:, None, None] + total_grad1
            bound_grad = bound_grad0 * cost_0_scaling[:, None] + bound_grad1
            velocity_xyt += -dt[:,None,None]*friction[:,None,None]*velocity_xyt - dt[:,None,None]*total_grad
            velocity_h += 0*velocity_h - dt[:,None]*bound_grad
            sol.xyt += dt[:,None,None] * velocity_xyt
            sol.h += dt[:,None] * velocity_h
            t_total0 += dt[0]
            
            if self.plot_interval is not None and t_total0 - t_last_plot >= self.plot_interval*0.999:
                t_last_plot = t_total0+0                
                ax.clear()
                ax.set_aspect('equal', adjustable='box')
                tree_list.xyt = cp.asnumpy(sol.xyt[0])
                pack_vis.visualize_tree_list(tree_list, ax=ax, h=cp.asnumpy(sol.h[0]))
                ax.set_title(f'Time: {t_total0:.2f}')
                display(fig)
                clear_output(wait=True)       
        return sol

class DynamicsInitialize(Dynamics):
    n_rounds = 5
    duration_init = 10.
    duration_compact = 200.
    duration_final = 10.
    dt = 0.02
    friction_min = 0.1
    friction_max = 10.
    friction_periods = 10
    scaling_area_start = 0.3
    scaling_area_end = 0.001
    scaling_boundary = 5.
    scaling_overlap = 1. # recommend to keep this fixed
    use_boundary_distance = True
    use_separation_overlap = False

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):

        self.cost0 = pack_cost.AreaCost(scaling=1.)
        self.cost1 = pack_cost.CostCompound(costs = [pack_cost.BoundaryDistanceCost(scaling=self.scaling_boundary), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=self.scaling_overlap)])
        if not self.use_boundary_distance:
            self.cost1.costs[0] = pack_cost.BoundaryCost(scaling=self.scaling_boundary)
        if self.use_separation_overlap:
            self.cost1.costs[1] = pack_cost.CollisionCostSeparation(scaling=self.scaling_overlap)   
        t_total = np.float32(0.)
        dt = np.float32(self.dt)
        phase = 'init'
        t_this_phase = np.float32(0.)        
        rounds_done = 0
        self.dt_list = []
        self.friction_list = []
        self.cost_0_scaling_list = []
        while True:
            if phase == 'compact':
                frac = t_this_phase / self.duration_compact
                start = self.scaling_area_start
                end = self.scaling_area_end
                area_scaling = start * (end / start) ** frac
                cost_0_scaling = area_scaling
                friction = self.friction_max + (self.friction_min - self.friction_max) * (1+np.cos(frac*2*np.pi*self.friction_periods))/2
            else:
                cost_0_scaling = 0.
                friction = 1/dt
            self.dt_list.append(dt)
            self.friction_list.append(friction)
            self.cost_0_scaling_list.append(cost_0_scaling)        
                  
            t_this_phase += dt

            if phase == 'init' and t_this_phase >= self.duration_init:
                phase = 'compact'
                t_this_phase = 0.    
            elif phase == 'compact' and t_this_phase >= self.duration_compact:
                phase = 'final'
                t_this_phase = 0.
            elif phase == 'final' and t_this_phase >= self.duration_final:               
                rounds_done += 1
                if rounds_done >= self.n_rounds:
                    break
                
                phase = 'compact'
                t_this_phase = 0. 
        
        n_ensembles = sol.N_solutions
        self.dt_list = cp.array(self.dt_list)
        self.dt_list = cp.tile(self.dt_list[None, :], (n_ensembles, 1))
        self.friction_list = cp.array(self.friction_list)
        self.friction_list = cp.tile(self.friction_list[None, :], (n_ensembles, 1))
        self.cost_0_scaling_list = cp.array(self.cost_0_scaling_list)
        self.cost_0_scaling_list = cp.tile(self.cost_0_scaling_list[None, :], (n_ensembles, 1))
        sol = super().run_simulation(sol)
        self.dt_list = None
        self.friction_list = None
        self.cost_0_scaling_list = None
        return sol