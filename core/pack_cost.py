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
        assert cost.shape == (N_ensembles,)
        assert grad_xyt.shape == xyt.shape
        assert grad_bound.shape == bound.shape 
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
        N_ensembles = xyt.shape[0]
        cost,grad_xyt,grad_bound =  self._compute_cost(xyt, bound)
        assert cost.shape == (N_ensembles,)
        assert grad_xyt.shape == xyt.shape
        assert grad_bound.shape == bound.shape 
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
class CostCompound(Cost):
    # Compound cost: sum of multiple costs
    costs:list = field(init=True, default_factory=list)

    def _compute_cost_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        N_ensembles = xyt.shape[0]
        total_cost = cp.zeros(N_ensembles)
        total_grad = cp.zeros_like(xyt)
        total_grad_bound = cp.zeros_like(bound)
        for c in self.costs:
            c_cost, c_grad, c_grad_bound = c.compute_cost_ref(xyt, bound)
            total_cost += c_cost
            total_grad += c_grad
            total_grad_bound += c_grad_bound
        return total_cost, total_grad, total_grad_bound

    def _compute_cost(self, xyt:cp.ndarray, bound:cp.ndarray):
        N_ensembles = xyt.shape[0]
        total_cost = cp.zeros(N_ensembles)
        total_grad = cp.zeros_like(xyt)
        total_grad_bound = cp.zeros_like(bound)
        for c in self.costs:
            c_cost, c_grad, c_grad_bound = c.compute_cost(xyt, bound)
            total_cost += c_cost
            total_grad += c_grad
            total_grad_bound += c_grad_bound
        return total_cost, total_grad, total_grad_bound

@dataclass
class CostDummy(Cost):
    # Dummy: always zero cost
    def _compute_cost_single_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        return cp.array(0.0), cp.zeros_like(xyt), cp.zeros_like(bound)

@dataclass    
class CollisionCost(Cost):
    
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
    

@dataclass
class BoundaryCost(Cost):
    # Cost for trees being out of bounds
    def _compute_cost_single_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        assert(bound.shape[0]==1) #other case todo
        # xyt is (n_trees, 3), bound is (1,); compute area outside square using union of all tree geometries
        # Use TreeList.get_trees() to build geometries and perform a single difference against the square (vectorized).
        b = float(bound[0].get().item())
        half = b / 2.0
        square = Polygon([(-half, -half), (half, -half), (half, half), (-half, half)])

        tree_list = kgs.TreeList()
        tree_list.xyt = xyt
        trees = tree_list.get_trees()

        n_trees = xyt.shape[0]
        area = n_trees*kgs.tree_area - np.sum(shapely.area(square.intersection(trees)))

        # Finite-difference gradient (central differences) w.r.t. each tree's x, y, theta
        grad = cp.zeros_like(xyt)
        epsilon = 1e-6
        for i in range(n_trees):
            # Extract base values as Python floats
            xi = xyt[i,0].get().item()
            yi = xyt[i,1].get().item()
            thetai = xyt[i,2].get().item()
            for j in range(3):
                if j == 0:
                    x_plus, x_minus = xi + epsilon, xi - epsilon
                    y_plus = y_minus = yi
                    th_plus = th_minus = thetai
                elif j == 1:
                    y_plus, y_minus = yi + epsilon, yi - epsilon
                    x_plus = x_minus = xi
                    th_plus = th_minus = thetai
                else:
                    th_plus, th_minus = thetai + epsilon, thetai - epsilon
                    x_plus = x_minus = xi
                    y_plus = y_minus = yi

                # create perturbed trees lists by replacing the i-th tree
                tree_plus = kgs.create_tree(x_plus, y_plus, th_plus*360/2/np.pi)
                tree_minus = kgs.create_tree(x_minus, y_minus, th_minus*360/2/np.pi)                

                area_plus = - square.intersection(tree_plus).area
                area_minus = - square.intersection(tree_minus).area

                grad_val = (area_plus - area_minus) / (2.0 * epsilon)
                grad[i, j] = cp.array(grad_val)

        # Compute gradient w.r.t. bound using finite differences
        grad_bound = cp.zeros_like(bound)
        b_plus = b + epsilon
        b_minus = b - epsilon
        half_plus = b_plus / 2.0
        half_minus = b_minus / 2.0
        square_plus = Polygon([(-half_plus, -half_plus), (half_plus, -half_plus), (half_plus, half_plus), (-half_plus, half_plus)])
        square_minus = Polygon([(-half_minus, -half_minus), (half_minus, -half_minus), (half_minus, half_minus), (-half_minus, half_minus)])
        
        area_plus = n_trees*kgs.tree_area - np.sum(shapely.area(square_plus.intersection(trees)))
        area_minus = n_trees*kgs.tree_area - np.sum(shapely.area(square_minus.intersection(trees)))
        grad_bound[0] = cp.array((area_plus - area_minus) / (2.0 * epsilon))

        return cp.array(area), grad, grad_bound
    
    def _compute_cost(self, xyt:cp.ndarray, bound:cp.ndarray):
        cost,grad,grad_h = pack_cuda.boundary_multi_ensemble(xyt, bound[:,0], compute_grad=True)
        return cost,grad,grad_h[:,None]

@dataclass 
class AreaCost(Cost):
    def _compute_cost_single_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        assert(bound.shape[0]==1) #other case todo
        cost = bound[0]**2
        grad_bound = cp.array([2.0*bound[0]])
        return cost, cp.zeros_like(xyt), grad_bound
    
    def _compute_cost(self, xyt:cp.ndarray, bound:cp.ndarray):
        cost = bound[:,0]**2
        grad_bound = cp.zeros_like(bound)
        grad_bound[:,0] = 2.0*bound[:,0]
        return cost, cp.zeros_like(xyt), grad_bound

@dataclass
class BoundaryDistanceCost(Cost):
    # Cost based on squared distance of vertices outside the square boundary
    # Per tree, use only the vertex with the maximum distance
    def _compute_cost_single_ref(self, xyt:cp.ndarray, bound:cp.ndarray):
        assert(bound.shape[0]==1) #other case todo
        # xyt is (n_trees, 3), bound is (1,); compute squared distance for each vertex outside square
        b = float(bound[0].get().item())
        half = b / 2.0
        
        tree_list = kgs.TreeList()
        tree_list.xyt = xyt
        trees = tree_list.get_trees()
        
        n_trees = xyt.shape[0]
        total_cost = 0.0
        
        # For each tree, find vertex with maximum distance outside square (vectorized over vertices)
        for i, tree in enumerate(trees):
            coords = np.array(tree.exterior.coords[:-1])  # skip closing vertex
            if coords.size == 0:
                continue
            vx = coords[:, 0]
            vy = coords[:, 1]
            dx = np.maximum(0.0, np.abs(vx) - half)
            dy = np.maximum(0.0, np.abs(vy) - half)
            dist_sq = dx**2 + dy**2
            max_dist_sq = float(np.max(dist_sq))
            total_cost += max_dist_sq
        
        # Compute gradients using finite differences
        grad = cp.zeros_like(xyt)
        epsilon = 1e-6
        
        for i in range(n_trees):
            xi = xyt[i,0].get().item()
            yi = xyt[i,1].get().item()
            thetai = xyt[i,2].get().item()
            
            for j in range(3):
                if j == 0:
                    x_plus, x_minus = xi + epsilon, xi - epsilon
                    y_plus = y_minus = yi
                    th_plus = th_minus = thetai
                elif j == 1:
                    y_plus, y_minus = yi + epsilon, yi - epsilon
                    x_plus = x_minus = xi
                    th_plus = th_minus = thetai
                else:
                    th_plus, th_minus = thetai + epsilon, thetai - epsilon
                    x_plus = x_minus = xi
                    y_plus = y_minus = yi
                
                # Compute cost for perturbed positions (vectorized over vertices)
                tree_plus = kgs.create_tree(x_plus, y_plus, th_plus*360/2/np.pi)
                tree_minus = kgs.create_tree(x_minus, y_minus, th_minus*360/2/np.pi)

                coords_plus = np.array(tree_plus.exterior.coords[:-1])
                if coords_plus.size == 0:
                    cost_plus = 0.0
                else:
                    vx_p = coords_plus[:, 0]
                    vy_p = coords_plus[:, 1]
                    dx_p = np.maximum(0.0, np.abs(vx_p) - half)
                    dy_p = np.maximum(0.0, np.abs(vy_p) - half)
                    dist_sq_p = dx_p**2 + dy_p**2
                    cost_plus = float(np.max(dist_sq_p))

                coords_minus = np.array(tree_minus.exterior.coords[:-1])
                if coords_minus.size == 0:
                    cost_minus = 0.0
                else:
                    vx_m = coords_minus[:, 0]
                    vy_m = coords_minus[:, 1]
                    dx_m = np.maximum(0.0, np.abs(vx_m) - half)
                    dy_m = np.maximum(0.0, np.abs(vy_m) - half)
                    dist_sq_m = dx_m**2 + dy_m**2
                    cost_minus = float(np.max(dist_sq_m))
                
                grad_val = (cost_plus - cost_minus) / (2.0 * epsilon)
                grad[i, j] = cp.array(grad_val)
        
        # Compute gradient w.r.t. bound using finite differences
        grad_bound = cp.zeros_like(bound)
        b_plus = b + epsilon
        b_minus = b - epsilon
        half_plus = b_plus / 2.0
        half_minus = b_minus / 2.0
        
        # Recompute cost for perturbed bounds (vectorized per-tree over vertices)
        cost_plus = 0.0
        for tree in trees:
            coords = np.array(tree.exterior.coords[:-1])
            if coords.size == 0:
                continue
            vx = coords[:, 0]
            vy = coords[:, 1]
            dx = np.maximum(0.0, np.abs(vx) - half_plus)
            dy = np.maximum(0.0, np.abs(vy) - half_plus)
            dist_sq = dx**2 + dy**2
            cost_plus += float(np.max(dist_sq))

        cost_minus = 0.0
        for tree in trees:
            coords = np.array(tree.exterior.coords[:-1])
            if coords.size == 0:
                continue
            vx = coords[:, 0]
            vy = coords[:, 1]
            dx = np.maximum(0.0, np.abs(vx) - half_minus)
            dy = np.maximum(0.0, np.abs(vy) - half_minus)
            dist_sq = dx**2 + dy**2
            cost_minus += float(np.max(dist_sq))
        
        grad_bound[0] = cp.array((cost_plus - cost_minus) / (2.0 * epsilon))
        
        return cp.array(total_cost), grad, grad_bound
    
    def _compute_cost_single(self, xyt:cp.ndarray, bound:cp.ndarray):
        assert(bound.shape[0]==1) #other case todo
        # xyt is (n_trees, 3), bound is (1,)
        # Use kgs.tree_vertices (precomputed center tree vertices) for efficient vectorized computation
        b = float(bound[0].get().item())
        half = b / 2.0
        
        n_trees = xyt.shape[0]
        
        epsilon = 1e-6
        
        # Extract positions
        x = xyt[:, 0:1]  # (n_trees, 1)
        y = xyt[:, 1:2]  # (n_trees, 1)
        theta = xyt[:, 2:3]  # (n_trees, 1)
        
        # Rotation matrices for all trees
        cos_t = cp.cos(theta)  # (n_trees, 1)
        sin_t = cp.sin(theta)  # (n_trees, 1)
        
        # kgs.tree_vertices is on GPU as CuPy array (n_vertices, 2)
        vx = kgs.tree_vertices[:, 0]  # (n_vertices,)
        vy = kgs.tree_vertices[:, 1]  # (n_vertices,)
        
        # Apply rotation and translation: R @ v + t for each tree
        # Rotated vertices: (n_trees, n_vertices) via broadcasting
        vx_rot = cos_t * vx - sin_t * vy  # (n_trees, n_vertices)
        vy_rot = sin_t * vx + cos_t * vy  # (n_trees, n_vertices)
        
        # Translate: add (x, y) for each tree
        vx_final = vx_rot + x  # (n_trees, n_vertices)
        vy_final = vy_rot + y  # (n_trees, n_vertices)
        
        # Compute boundary distance for each vertex
        dx = cp.maximum(0.0, cp.abs(vx_final) - half)  # (n_trees, n_vertices)
        dy = cp.maximum(0.0, cp.abs(vy_final) - half)  # (n_trees, n_vertices)
        dist_sq = dx**2 + dy**2  # (n_trees, n_vertices)
        
        # Max per tree
        max_dist_sq = cp.max(dist_sq, axis=1)  # (n_trees,)
        total_cost = cp.sum(max_dist_sq)
        
        # Compute gradients
        grad = cp.zeros_like(xyt)
        
        # Analytical gradients for x and y via backpropagation (vectorized)
        # For each tree, find which vertex has max distance
        max_indices = cp.argmax(dist_sq, axis=1)  # (n_trees,)
        
        # Vectorized extraction: get max-vertex values for all trees
        # max_indices shape: (n_trees,), need to gather from (n_trees, n_vertices)
        vx_max = vx_final[cp.arange(n_trees), max_indices]  # (n_trees,)
        vy_max = vy_final[cp.arange(n_trees), max_indices]  # (n_trees,)
        
        # Compute dx, dy for max vertex of each tree
        dx_max = cp.maximum(0.0, cp.abs(vx_max) - half)  # (n_trees,)
        dy_max = cp.maximum(0.0, cp.abs(vy_max) - half)  # (n_trees,)
        
        # Analytical gradients (vectorized over all trees)
        # d(dist_sq)/d(vx_final) = 2*dx * sign(vx_final) if dx > 0 else 0
        # d(dist_sq)/d(vy_final) = 2*dy * sign(vy_final) if dy > 0 else 0
        grad_vx_max = cp.where(dx_max > 0, 2.0 * dx_max * cp.sign(vx_max), 0.0)  # (n_trees,)
        grad_vy_max = cp.where(dy_max > 0, 2.0 * dy_max * cp.sign(vy_max), 0.0)  # (n_trees,)
        
        # d(vx_final)/d(x) = 1, d(vy_final)/d(y) = 1
        grad[:, 0] = grad_vx_max
        grad[:, 1] = grad_vy_max
        
        # Analytical gradient for theta via backpropagation (vectorized)
        # For max vertex of each tree, compute d(dist_sq)/d(theta)
        # Chain rule: d(dist_sq)/d(theta) = d(dist_sq)/d(vx_final) * d(vx_final)/d(theta) + d(dist_sq)/d(vy_final) * d(vy_final)/d(theta)
        # d(vx_final)/d(theta) = d(vx_rot)/d(theta) = -sin(theta)*vx - cos(theta)*vy
        # d(vy_final)/d(theta) = d(vy_rot)/d(theta) = cos(theta)*vx - sin(theta)*vy
        
        # Get the original (unrotated) vertex coordinates for max vertex of each tree
        vx_orig = vx[max_indices]  # (n_trees,)
        vy_orig = vy[max_indices]  # (n_trees,)
        
        # Derivatives of rotated vertices w.r.t. theta
        dvx_rot_dtheta = -sin_t[:, 0] * vx_orig - cos_t[:, 0] * vy_orig  # (n_trees,)
        dvy_rot_dtheta = cos_t[:, 0] * vx_orig - sin_t[:, 0] * vy_orig   # (n_trees,)
        
        # Chain through boundary distance
        grad_theta = grad_vx_max * dvx_rot_dtheta + grad_vy_max * dvy_rot_dtheta  # (n_trees,)
        grad[:, 2] = grad_theta
        
        # Analytical gradient for h (bound) via backpropagation (vectorized)
        # For the h gradient, we don't use the previous grad_vx/vy_max which include sign factors.
        # Instead: d(dist_sq)/d(half) = -2*dx - 2*dy (regardless of sign of vx/vy)
        # And d(dist_sq)/d(b) = d(dist_sq)/d(half) * d(half)/d(b) = -2*(dx_max + dy_max) * (1/2)
        
        grad_h = -(dx_max + dy_max)  # (n_trees,)
        grad_bound_val = cp.sum(grad_h)
        
        grad_bound = cp.array([grad_bound_val])
        
        return total_cost, grad, grad_bound

        