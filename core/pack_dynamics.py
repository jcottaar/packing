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

@dataclass
class Dynamics(kgs.BaseClass):
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

        sol = copy.deepcopy(sol)
        xyt = sol.xyt
        h = sol.h

        assert len(xyt.shape)==3 and xyt.shape[2]==3
        n_ensembles = xyt.shape[0]
        n_trees = xyt.shape[1]
        assert h.shape == (n_ensembles, 1)
        assert self.dt_list.shape == self.friction_list.shape == self.cost_0_scaling_list.shape
        assert self.dt_list.shape[0] == n_ensembles

        #plt.ioff()
        if self.plot_interval is not None:
            #plt.ion()
            raise 'fix'
            fig, ax = plt.subplots(figsize=(8, 8))
            tree_list = kgs.TreeList()

        t_total0 = np.float32(0.)        
        velocity_xyt = cp.zeros_like(xyt)
        velocity_h = cp.zeros_like(h)
        for i_iteration in range(self.dt_list.shape[1]):
            dt = self.dt_list[:, i_iteration]
            friction = self.friction_list[:, i_iteration]
            cost_0_scaling = self.cost_0_scaling_list[:, i_iteration]          
            total_cost0, total_grad0, bound_grad0 = self.cost0.compute_cost(sol)
            total_cost1, total_grad1, bound_grad1 = self.cost1.compute_cost(sol)
            total_cost = total_cost0 * cost_0_scaling + total_cost1
            total_grad = total_grad0 * cost_0_scaling[:, None, None] + total_grad1
            bound_grad = bound_grad0 * cost_0_scaling[:, None] + bound_grad1
            #total_grad *= 10
            #bound_grad *= 10
            #print(t_total0, total_cost)
            velocity_xyt += -dt[:,None,None]*friction[:,None,None]*velocity_xyt - dt[:,None,None]*total_grad
            velocity_h += 0*velocity_h - dt[:,None]*bound_grad
            xyt += dt[:,None,None] * velocity_xyt
            h += dt[:,None] * velocity_h
            t_total0 += dt[0]
            
            if self.plot_interval is not None and t_total0 - t_last_plot >= self.plot_interval*0.999:
                t_last_plot = t_total                
                ax.clear()
                ax.set_aspect('equal', adjustable='box')
                tree_list.xyt = cp.asnumpy(xyt[0])
                pack_vis.visualize_tree_list(tree_list, ax=ax, h=cp.asnumpy(h[0,0]))
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

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):

        self.cost0 = pack_cost.AreaCost(scaling=1.)
        self.cost1 = pack_cost.CostCompound(costs = [pack_cost.BoundaryDistanceCost(scaling=self.scaling_boundary), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=self.scaling_overlap)])
        if not self.use_boundary_distance:
            self.cost1.costs[0] = pack_cost.BoundaryCost(scaling=self.scaling_boundary)
        
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
        return sol