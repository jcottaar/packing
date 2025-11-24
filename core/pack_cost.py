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
import pack_cuda

@dataclass
class Cost(kgs.BaseClass):
    scaling:float = field(init=True, default=1.0)

    @typechecked 
    def compute_cost_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        # xyt: (n_ensemble, n_trees, 3) array
        #bound: (n_ensemble, 1 or 3) array
        assert(xyt.shape[2]==3)  # x, y, theta
        N_ensembles = xyt.shape[0]
        assert(bound.shape == (N_ensembles,1) or bound.shape == (N_ensembles,3)) # 1: square bound, 3: periodic bound      
        cost,grad_xyt,grad_bound = self._compute_cost_ref(xyt,bound)
        return self.scaling*cost,self.scaling*grad_xyt,self.scaling*grad_bound
    
    def _compute_cost_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        N_ensembles = xyt.shape[0]
        cost = cp.zeros(N_ensembles)
        grad_xyt = cp.zeros_like(xyt)
        grad_bound = cp.zeros_like(bound)
        for i in range(N_ensembles):
            cost[i],grad_xyt[i],grad_bound[i] = self._compute_cost_single_ref(xyt[i], bound[i])
        return cost,grad_xyt,grad_bound
        
    def compute_cost(self, xyt:cp.ndarray, bound:cp.ndarray):
        # Subclass can implement faster version
        cost,grad_xyt,grad_bound =  self._compute_cost(xyt, bound)
        return self.scaling*cost,self.scaling*grad_xyt,self.scaling*grad_bound
    
    def _compute_cost(self, xyt:cp.ndarray, bound:cp.ndarray):
        # Subclass can implement faster version
        N_ensembles = xyt.shape[0]
        cost = cp.zeros(N_ensembles)
        grad_xyt = cp.zeros_like(xyt)
        grad_bound = cp.zeros_like(bound)
        for i in range(N_ensembles):
            cost[i],grad_xyt[i],grad_bound[i] = self._compute_cost_single(xyt[i], bound[i])
        return cost,grad_xyt,grad_bound
    
    def _compute_cost_single(self, xyt:cp.ndarray, bound:cp.ndarray):
        return self._compute_cost_single_ref(xyt, bound)

@dataclass
class CostDummy(Cost):
    # Dummy: always zero cost
    def _compute_cost_single_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        return cp.array(0.0), cp.zeros_like(xyt), cp.zeros_like(bound)

@dataclass    
class CollisionCost(Cost):
    
    @typechecked 
    def _compute_cost_single_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        # Compute collision cost for all pairs of trees
        assert(xyt.shape[1]==3)  # x, y, theta
        n_trees = xyt.shape[0]
        tree_list = kgs.TreeList()
        tree_list.xyt = xyt
        trees = tree_list.get_trees()
        total_cost = cp.array(0.0)
        total_grad = cp.zeros_like(xyt)
        for i in range(n_trees):
            this_xyt = xyt[i]            
            other_xyt = cp.delete(xyt, i, axis=0)
            this_tree = trees[i]
            other_trees = [t for t in trees if t is not this_tree]
            this_cost, this_grads = self._compute_cost_one_tree_ref(this_xyt, other_xyt, this_tree, other_trees)        
            total_cost += this_cost/2
            total_grad[i] += this_grads
        return total_cost, total_grad, cp.zeros_like(bound)
    
class CollisionCostOverlappingArea(CollisionCost):
    # Collision cost based on overlapping area of two trees
    def _compute_cost_one_tree_ref(self, xyt1:cp.ndarray, xyt2:cp.ndarray, tree1:Polygon, tree2:list):
        # Compute overlapping area between tree1 and the union of tree2 geometries.
        # tree2 is a list of shapely geometries; create a union before intersecting.
        area = cp.array(np.sum(shapely.area(tree1.intersection(tree2))))
        # Gradient computation is complex; use finite differences as a placeholder
        grad = cp.zeros_like(xyt1)
        if area>0:
            epsilon = 1e-6
            for j in range(3):  # x, y, theta
                xyt1_plus = xyt1.copy()
                xyt1_minus = xyt1.copy()
                xyt1_plus[j] += epsilon
                xyt1_minus[j] -= epsilon
                tree1_plus = kgs.create_tree(xyt1_plus[0].get().item(), xyt1_plus[1].get().item(), xyt1_plus[2].get().item()*360/2/np.pi)
                area_plus = np.sum(shapely.area(tree1_plus.intersection(tree2)))
                tree1_minus = kgs.create_tree(xyt1_minus[0].get().item(), xyt1_minus[1].get().item(), xyt1_minus[2].get().item()*360/2/np.pi)
                area_minus = np.sum(shapely.area(tree1_minus.intersection(tree2)))
                grad[j] = cp.array((area_plus - area_minus) / (2 * epsilon))
        return area, grad

    def _compute_cost(self, xyt:cp.ndarray, bound:cp.ndarray):
        cost,grad = pack_cuda.overlap_multi_ensemble(xyt, xyt)
        return cost,cp.array(grad),cp.zeros_like(bound)