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
class CollisionCost(kgs.BaseClass):
    scaling:float = field(default=1.0)
    
    @typechecked 
    def compute_cost_ref(self, xyt:cp.ndarray, include_gradients:bool):
        # Compute collision cost for all pairs of trees
        assert(xyt.shape[1]==3)  # x, y, theta
        n_trees = xyt.shape[0]
        tree_list = kgs.TreeList()
        tree_list.xyt = xyt
        trees = tree_list.get_trees()
        total_cost = cp.array(0.0)
        if include_gradients:
            total_grad = cp.zeros_like(xyt)
        for i in range(n_trees):
            this_xyt = xyt[i]            
            other_xyt = cp.delete(xyt, i, axis=0)
            this_tree = trees[i]
            other_trees = [t for t in trees if t is not this_tree]
            this_cost, this_grads = self._compute_cost_one_tree_ref(this_xyt, other_xyt, this_tree, other_trees, include_gradients)        
            total_cost += this_cost/2
            if include_gradients:
                total_grad[i] += this_grads
        if include_gradients:
            return self.scaling * total_cost, self.scaling * total_grad
        else:
            return self.scaling * total_cost, None
        
    @kgs.profile_each_line
    def compute_cost(self, xyt:cp.ndarray, include_gradients:bool):
        # Subclass can implement faster version
        #cost,grad =  self.compute_cost_ref(xyt, include_gradients)
        cost,grad = pack_cuda.overlap_list_total(xyt)
        return cost,grad
        
class CollisionCostDummy(CollisionCost):
    # Dummy: always zero cost
    def _compute_cost_one_tree_ref(self, xyt1:cp.ndarray, xyt2:cp.ndarray, tree1:Polygon, tree2:list, include_gradients:bool):
        # xyt1: (,3)
        # xyt2: (N,3)
        if include_gradients:
            return cp.array(0.0), cp.zeros_like(xyt1)
        else:
            return cp.array(0.0), None
    
class CollisionCostOverlappingArea(CollisionCost):
    # Collision cost based on overlapping area of two trees
    def _compute_cost_one_tree_ref(self, xyt1:cp.ndarray, xyt2:cp.ndarray, tree1:Polygon, tree2:list, include_gradients:bool):
        # Compute overlapping area between tree1 and the union of tree2 geometries.
        # tree2 is a list of shapely geometries; create a union before intersecting.
        area = cp.array(np.sum(shapely.area(tree1.intersection(tree2))))
        if include_gradients:
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
        else:
            return area, None
    
@dataclass
class PackingCost(kgs.BaseClass):
    collision_cost: CollisionCost = field(default=None)
    #edge_cost: EdgeCost = field(default=None)

    @typechecked
    def compute_total_cost_ref(self, xyt:cp.ndarray, include_gradients:bool=False):
        assert(xyt.shape[1]==3)  # x, y, theta
        cost_collision, grad_collision = self.collision_cost.compute_cost_ref(xyt, include_gradients)
        #cost_edge, grad_edge = self.edge_cost.compute_cost_ref(xyt, include_gradients=include_gradients)
        total_cost = cost_collision #+ cost_edge
        # If sub-costs follow the convention, grad_collision will be None when include_gradients is False
        if include_gradients:
            total_grad = grad_collision #+ grad_edge
            return total_cost, total_grad
        else:
            return total_cost, None
        
    
    def compute_total_cost(self, xyt:cp.ndarray, include_gradients:bool=False):
        cost_collision, grad_collision = self.collision_cost.compute_cost(xyt, include_gradients)
        #cost_edge, grad_edge = self.edge_cost.compute_cost_ref(xyt, include_gradients=include_gradients)
        total_cost = cost_collision #+ cost_edge
        # Return grad (or None) consistently
        return total_cost, grad_collision
