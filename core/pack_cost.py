import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
from typeguard import typechecked

@dataclass    
class CollisionCost(kgs.BaseClass):
    scaling:float = field(default=1.0)
    
    @typechecked 
    def compute_cost_ref(self, xyt:np.ndarray, include_gradients:bool):
        # Compute collision cost for all pairs of trees
        assert(xyt.shape[1]==3)  # x, y, theta
        n_trees = xyt.shape[0]
        total_cost = 0.0
        if include_gradients:
            total_grad = np.zeros_like(xyt)
        for i in range(n_trees):
            for j in range(i+1, n_trees):
                # Compute relative position in tree i's frame
                theta_i = xyt[i, 2]
                cos_theta = np.cos(theta_i)
                sin_theta = np.sin(theta_i)
                dx = xyt[j, 0] - xyt[i, 0]
                dy = xyt[j, 1] - xyt[i, 1]
                xyt2 = np.array([[
                    cos_theta * dx + sin_theta * dy,
                    -sin_theta * dx + cos_theta * dy,
                    xyt[j, 2] - xyt[i, 2]
                ]])
                cost_ij, grad_ij = self._compute_cost_one_pair_ref(xyt2, include_gradients=include_gradients)
                total_cost += cost_ij
                if include_gradients:
                    # Gradient w.r.t. xyt2 needs to be transformed back
                    grad_x2, grad_y2, grad_theta2 = grad_ij[0, :]
                    # Gradient w.r.t. tree j position and rotation
                    total_grad[j, 0] += cos_theta * grad_x2 - sin_theta * grad_y2
                    total_grad[j, 1] += sin_theta * grad_x2 + cos_theta * grad_y2
                    total_grad[j, 2] += grad_theta2
                    # Gradient w.r.t. tree i position and rotation
                    total_grad[i, 0] -= cos_theta * grad_x2 - sin_theta * grad_y2
                    total_grad[i, 1] -= sin_theta * grad_x2 + cos_theta * grad_y2
                    total_grad[i, 2] += (-sin_theta * dx + cos_theta * dy) * grad_x2 + (-cos_theta * dx - sin_theta * dy) * grad_y2 - grad_theta2
        if include_gradients:
            return self.scaling * total_cost, self.scaling * total_grad
        else:
            return self.scaling * total_cost, None
        
    def compute_cost(self, xyt:np.ndarray, include_gradients:bool):
        # Subclass can implement faster version
        return self.compute_cost_ref(xyt, include_gradients)
        
class CollisionCostDummy(CollisionCost):
    # Dummy: always zero cost
    def _compute_cost_one_pair_ref(self, xyt:np.ndarray, include_gradients:bool):
        if include_gradients:
            return 0.0, np.zeros_like(xyt)
        else:
            return 0.0, None
    
class CollisionCostOverlappingArea(CollisionCost):
    # Collision cost based on overlapping area of two trees
    def _compute_cost_one_pair_ref(self, xyt:np.ndarray, include_gradients:bool):
        tree1 = kgs.center_tree
        tree2 = kgs.create_tree(xyt[0,0], xyt[0,1], xyt[0,2]*360/(2*np.pi))
        intersection = tree1.intersection(tree2)
        area = intersection.area
        if include_gradients:
            # Gradient computation is complex; use finite differences as a placeholder
            grad = np.zeros_like(xyt)
            if area>0:
                eps = 1e-6
                for dim in range(3):
                    xyt_pos = xyt.copy()
                    xyt_neg = xyt.copy()
                    xyt_pos[0, dim] += eps
                    xyt_neg[0, dim] -= eps
                    tree2_pos = kgs.create_tree(xyt_pos[0,0], xyt_pos[0,1], xyt_pos[0,2]*360/(2*np.pi))
                    tree2_neg = kgs.create_tree(xyt_neg[0,0], xyt_neg[0,1], xyt_neg[0,2]*360/(2*np.pi))
                    area_pos = tree1.intersection(tree2_pos).area
                    area_neg = tree1.intersection(tree2_neg).area
                    grad[0, dim] = (area_pos - area_neg) / (2 * eps)
            return area, grad
        else:
            return area, None
    
@dataclass
class PackingCost(kgs.BaseClass):
    collision_cost: CollisionCost = field(default=None)
    #edge_cost: EdgeCost = field(default=None)

    @typechecked
    def compute_total_cost_ref(self, xyt:np.ndarray, include_gradients:bool=False):
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
        
    def compute_total_cost(self, xyt:np.ndarray, include_gradients:bool=False):
        cost_collision, grad_collision = self.collision_cost.compute_cost(xyt, include_gradients)
        #cost_edge, grad_edge = self.edge_cost.compute_cost_ref(xyt, include_gradients=include_gradients)
        total_cost = cost_collision #+ cost_edge
        # Return grad (or None) consistently
        return total_cost, grad_collision
