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
    def compute_cost_ref(self, sol:kgs.SolutionCollection):
        # xyt: (n_ensemble, n_trees, 3) array
        #bound: (n_ensemble, 1 or 3) array
        sol.check_constraints()                
        cost,grad_xyt,grad_bound = self._compute_cost_ref(sol)
        assert cost.shape == (sol.N_solutions,)
        assert grad_xyt.shape == sol.xyt.shape
        assert grad_bound.shape == sol.h.shape 
        return self.scaling*cost,self.scaling*grad_xyt,self.scaling*grad_bound
    
    def _compute_cost_ref(self, sol:kgs.SolutionCollection):        
        cost = cp.zeros(sol.N_solutions)
        grad_xyt = cp.zeros_like(sol.xyt)
        grad_bound = cp.zeros_like(sol.h)
        for i in range(sol.N_solutions):
            cost[i],grad_xyt[i],grad_bound[i] = self._compute_cost_single_ref(sol, sol.xyt[i], sol.h[i])
        return cost,grad_xyt,grad_bound
    
    def compute_cost_allocate(self, sol:kgs.SolutionCollection):
        # Allocates gradient arrays and calls compute_cost
        cost = cp.zeros(sol.N_solutions, dtype=sol.xyt.dtype)
        grad_xyt = cp.zeros_like(sol.xyt)
        grad_bound = cp.zeros_like(sol.h)
        self.compute_cost(sol, cost, grad_xyt, grad_bound)
        return cost, grad_xyt, grad_bound
    
    def compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        # Subclass can implement faster version with preallocated gradients
        self._compute_cost(sol, cost, grad_xyt, grad_bound)
        cost *= self.scaling
        grad_xyt *= self.scaling
        grad_bound *= self.scaling
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        # Subclass can implement faster version with preallocated gradients
        for i in range(sol.N_solutions):
            cost[i],grad_xyt[i],grad_bound[i] = self._compute_cost_single(sol, sol.xyt[i], sol.h[i])
    
    def _compute_cost_single(self, sol:kgs.SolutionCollection, xyt, h):
        return self._compute_cost_single_ref(sol, xyt, h)

@dataclass 
class CostCompound(Cost):
    # Compound cost: sum of multiple costs
    costs:list = field(init=True, default_factory=list)
    _temp_cost: cp.ndarray = field(init=False, default=None, repr=False)
    _temp_grad_xyt: cp.ndarray = field(init=False, default=None, repr=False)
    _temp_grad_bound: cp.ndarray = field(init=False, default=None, repr=False)
    _temp_shape: tuple = field(init=False, default=None, repr=False)

    def _compute_cost_ref(self, sol:kgs.SolutionCollection):
        total_cost = cp.zeros(sol.N_solutions)
        total_grad = cp.zeros_like(sol.xyt)
        total_grad_bound = cp.zeros_like(sol.h)
        for c in self.costs:
            c_cost, c_grad, c_grad_bound = c.compute_cost_ref(sol)
            total_cost += c_cost
            total_grad += c_grad
            total_grad_bound += c_grad_bound
        return total_cost, total_grad, total_grad_bound

    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        cost[:] = 0
        grad_xyt[:] = 0
        grad_bound[:] = 0
        
        # Check if we need to allocate or reallocate temporary arrays
        # Use dtype matching the output arrays to ensure consistency
        current_shape = (sol.N_solutions, sol.xyt.shape, sol.h.shape)
        if self._temp_shape != current_shape:
            self._temp_cost = cp.zeros(sol.N_solutions, dtype=cost.dtype)
            self._temp_grad_xyt = cp.zeros(sol.xyt.shape, dtype=grad_xyt.dtype)
            self._temp_grad_bound = cp.zeros(sol.h.shape, dtype=grad_bound.dtype)
            self._temp_shape = current_shape
        
        for c in self.costs:
            self._temp_cost[:] = 0
            self._temp_grad_xyt[:] = 0
            self._temp_grad_bound[:] = 0
            c.compute_cost(sol, self._temp_cost, self._temp_grad_xyt, self._temp_grad_bound)
            cost += self._temp_cost
            grad_xyt += self._temp_grad_xyt
            grad_bound += self._temp_grad_bound

@dataclass
class CostDummy(Cost):
    # Dummy: always zero cost
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection, xyt, h):
        return cp.array(0.0), cp.zeros_like(xyt), cp.zeros_like(h)

@dataclass    
class CollisionCost(Cost):
    
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection, xyt, h):
        # Compute collision cost for all pairs of trees
        n_trees = sol.N_trees
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
        return total_cost, total_grad, cp.zeros_like(h)
    
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

    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        pack_cuda.overlap_multi_ensemble(sol.xyt, sol.xyt, out_cost=cost, out_grads=grad_xyt)
        grad_bound[:] = 0

@dataclass
class CollisionCostSeparation(CollisionCost):
    # Collision cost based on minimum separation distance between trees
    # Cost = sum(separation_distance^2) over all pairwise overlaps
    # Only overlapping pairs contribute (separated pairs have zero cost)
    # Separation distance = minimum distance trees must move to no longer overlap

    use_max:bool = field(init=True, default=True)
    TEMP_use_kernel:bool = field(init=True, default=False)
    
    def _compute_separation_distance(
        self,
        poly1_world: Polygon,
        poly2_world: Polygon,
        local_coords1: np.ndarray,
        x1: float,
        y1: float,
        th1: float,
    ):
        """
        Compute minimum separation distance (penetration depth) between poly1 and poly2,
        AND the derivatives of that separation wrt the pose (x, y, theta) of poly1.

        Arguments
        ---------
        poly1_world : Polygon
            Convex piece of tree1 in world coordinates.
        poly2_world : Polygon
            Convex piece of tree2 in world coordinates.
        local_coords1 : (N, 2) ndarray
            Vertices of this piece in tree1's local coordinate frame (center-tree coords),
            in the SAME order used to build poly1_world.
        x1, y1 : float
            Translation of tree1.
        th1 : float
            Rotation (radians) of tree1.

        Returns
        -------
        sep : float
            Minimum penetration depth (0 if no overlap).
        dsep_dx : float
            Derivative of sep wrt translation x1 of poly1.
        dsep_dy : float
            Derivative of sep wrt translation y1 of poly1.
        dsep_dtheta : float
            Derivative of sep wrt rotation th1 of poly1.
        """

        overlap = poly1_world.intersection(poly2_world)
        if overlap.is_empty or overlap.area < 1e-10:
            return 0.0, 0.0, 0.0, 0.0        

        world_coords1 = np.array(poly1_world.exterior.coords[:-1])  # (N1, 2)
        coords2 = np.array(poly2_world.exterior.coords[:-1])  # (N2, 2)

        min_penetration = np.inf

        # Data for the active axis
        best_normal = None          # unit normal n
        best_case = None            # "pen1" or "pen2"
        best_v_idx = -1             # index of critical vertex in world_coords1 / local_coords1
        best_w_world = None         # opposing vertex (world coords) in poly2
        axis_from_poly1 = False     # did axis come from poly1's edges?

        # Helper: iterate over edge normals of a given vertex array
        def iter_axes(coords, is_poly1_axis: bool):
            n_verts = coords.shape[0]
            for i in range(n_verts):
                p1 = coords[i]
                p2 = coords[(i + 1) % n_verts]
                edge = p2 - p1

                # perpendicular vector
                normal = np.array([-edge[1], edge[0]], dtype=float)
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-10:
                    continue
                normal /= norm_len
                yield normal, is_poly1_axis

        # Precompute projections of poly1 on-demand, so we reuse them per axis
        # (we'll re-project for each axis anyway since normal changes)
        for normal, is_poly1_axis in iter_axes(world_coords1, True):
            # Poly1 axis
            proj1 = world_coords1 @ normal
            proj2 = coords2 @ normal

            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()

            if max1 < min2 or max2 < min1:
                continue  # separated on this axis

            pen1 = max1 - min2      # move poly1 along -normal
            pen2 = max2 - min1      # move poly1 along +normal
            penetration = min(pen1, pen2)
            if penetration < min_penetration:
                min_penetration = penetration
                best_normal = normal.copy()
                axis_from_poly1 = True

                if pen1 <= pen2:
                    best_case = "pen1"
                    best_v_idx = int(np.argmax(proj1))   # vertex in poly1 at max1
                    w_idx = int(np.argmin(proj2))        # vertex in poly2 at min2
                    best_w_world = coords2[w_idx].copy()
                else:
                    best_case = "pen2"
                    best_v_idx = int(np.argmin(proj1))   # vertex in poly1 at min1
                    w_idx = int(np.argmax(proj2))        # vertex in poly2 at max2
                    best_w_world = coords2[w_idx].copy()

        # Also check axes from poly2 edges
        for normal, is_poly1_axis in iter_axes(coords2, False):
            proj1 = world_coords1 @ normal
            proj2 = coords2 @ normal

            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()

            if max1 < min2 or max2 < min1:
                continue

            pen1 = max1 - min2
            pen2 = max2 - min1
            penetration = min(pen1, pen2)
            if penetration < min_penetration:
                min_penetration = penetration
                best_normal = normal.copy()
                axis_from_poly1 = False

                if pen1 <= pen2:
                    best_case = "pen1"
                    best_v_idx = int(np.argmax(proj1))
                    w_idx = int(np.argmin(proj2))
                    best_w_world = coords2[w_idx].copy()
                else:
                    best_case = "pen2"
                    best_v_idx = int(np.argmin(proj1))
                    w_idx = int(np.argmax(proj2))
                    best_w_world = coords2[w_idx].copy()

        # No valid penetrating axis found
        if (not np.isfinite(min_penetration) or best_normal is None or
            best_case is None or best_v_idx < 0 or best_w_world is None):
            return 0.0, 0.0, 0.0, 0.0

        sep = float(min_penetration)

        # -------- dsep/dx, dsep/dy from ∂p/∂v --------
        # For a fixed axis, penetration gradient wrt critical vertex is ±n
        if best_case == "pen1":
            # p = max1 - min2 => p ≈ (v - w) · n  at critical vertex
            grad_v = best_normal.copy()
        else:  # "pen2"
            # p = max2 - min1 => p ≈ (w - v) · n
            grad_v = -best_normal.copy()

        dsep_dx = float(grad_v[0])  # dv/dx = [1,0]
        dsep_dy = float(grad_v[1])  # dv/dy = [0,1]

        # -------- dsep/dtheta: vertex motion + axis rotation (if axis from poly1) --------
        # local coords of the critical vertex
        vx0, vy0 = local_coords1[best_v_idx]

        # dv/dtheta from local coords
        sv = np.sin(th1)
        cv = np.cos(th1)
        dvx_dt = -sv * vx0 - cv * vy0
        dvy_dt =  cv * vx0 - sv * vy0

        # term from vertex motion: ∂p/∂v · dv/dθ
        term_vertex = grad_v[0] * dvx_dt + grad_v[1] * dvy_dt

        # term from axis rotation (only if axis came from poly1)
        if axis_from_poly1:
            # critical vertex world coords
            v_world = world_coords1[best_v_idx]
            w_world = best_w_world

            if best_case == "pen1":
                # p = (v - w) · n => ∂p/∂n = (v - w)
                grad_n = v_world - w_world
            else:
                # p = (w - v) · n => ∂p/∂n = (w - v)
                grad_n = w_world - v_world

            # n(θ) = R(θ) n0 => dn/dθ = J n,  J = [[0, -1], [1, 0]]
            dn_dt = np.array([-best_normal[1], best_normal[0]], dtype=float)

            term_axis = grad_n[0] * dn_dt[0] + grad_n[1] * dn_dt[1]
        else:
            term_axis = 0.0

        dsep_dtheta = float(term_vertex + term_axis)

        return sep, dsep_dx, dsep_dy, dsep_dtheta

    
    def _compute_cost_one_tree_ref(self,
                               xyt1: cp.ndarray,
                               xyt2: cp.ndarray,
                               tree1: Polygon,
                               tree2: list):
        """
        Sum squared separation distance over all overlapping tree pairs.

        For each pair (this tree vs another tree), we:
        - Break both into convex pieces.
        - For each piece pair, compute SAT-based separation and gradients
            wrt (x1, y1, th1) using the helper.
        - Take the maximum separation over all piece pairs.
        - Add max_sep^2 to the cost, and use the gradient at the max piece pair.
        """

        total_sep_squared = 0.0
        total_grad = cp.zeros_like(xyt1)

        # Pose of this tree
        x1 = float(xyt1[0].get().item())
        y1 = float(xyt1[1].get().item())
        th1 = float(xyt1[2].get().item())

        # Transform convex breakdown pieces for this tree
        def transformed_pieces_for_tree1(xc, yc, th):
            c = np.cos(th)
            s = np.sin(th)
            R = np.array([[c, -s],
                        [s,  c]], dtype=float)
            offset = np.array([xc, yc], dtype=float)
            pieces = []
            for poly in kgs.convex_breakdown:
                local_coords = np.array(poly.exterior.coords[:-1])        # (N, 2)
                world_coords = local_coords @ R.T + offset                # (N, 2)
                poly_world = Polygon(world_coords)
                pieces.append((poly_world, local_coords))
            return pieces

        def transformed_pieces_other_tree(xc, yc, th):
            c = np.cos(th)
            s = np.sin(th)
            R = np.array([[c, -s],
                        [s,  c]], dtype=float)
            offset = np.array([xc, yc], dtype=float)
            pieces = []
            for poly in kgs.convex_breakdown:
                local_coords = np.array(poly.exterior.coords[:-1])        # not used for grads
                world_coords = local_coords @ R.T + offset
                poly_world = Polygon(world_coords)
                pieces.append((poly_world, world_coords))
            return pieces

        pieces_a = transformed_pieces_for_tree1(x1, y1, th1)

        # Iterate over each "other tree" pose
        for j in range(xyt2.shape[0]):
            xb = float(xyt2[j, 0].get().item())
            yb = float(xyt2[j, 1].get().item())
            thb = float(xyt2[j, 2].get().item())

            pieces_b = transformed_pieces_other_tree(xb, yb, thb)

            # For this pair of trees: max separation over all piece pairs
            max_sep = 0.0
            best_dsep_dx = 0.0
            best_dsep_dy = 0.0
            best_dsep_dtheta = 0.0

            for (pa_world, pa_local) in pieces_a:
                for (pb_world, pb_world2) in pieces_b:
                    if self.TEMP_use_kernel:
                        import pack_cuda_primitives_test
                        sep, grad = pack_cuda_primitives_test.sat_separation_with_grad_pose_fwd_bwd(
                            cp.array(pa_local),
                            cp.array(pb_world2),
                            x1, y1, np.cos(th1), np.sin(th1),
                            compute_gradients=True
                        )
                        dsep_dx, dsep_dy, dsep_dtheta = grad
                    else:
                        sep, dsep_dx, dsep_dy, dsep_dtheta = \
                            self._compute_separation_distance(
                                poly1_world=pa_world,
                                poly2_world=pb_world,
                                local_coords1=pa_local,
                                x1=x1,
                                y1=y1,
                                th1=th1,
                            )
                        import pack_cuda_primitives_test
                        sep2, grad2 = pack_cuda_primitives_test.sat_separation_with_grad_pose_fwd_bwd(
                            cp.array(pa_local),
                            cp.array(pb_world2),
                            x1, y1, np.cos(th1), np.sin(th1),
                            compute_gradients=True
                        )
                        print(sep, sep2.get().item())
                        print(grad2, (dsep_dx, dsep_dy, dsep_dtheta))
                        print('')

                    if self.use_max:
                        if sep > max_sep:
                            max_sep = sep
                            best_dsep_dx = dsep_dx
                            best_dsep_dy = dsep_dy
                            best_dsep_dtheta = dsep_dtheta
                    else:
                        # Sum contributions over all piece-pairs instead of taking the max.
                        if sep <= 0.0:
                            continue
                        # Add sep^2 contribution and its gradient 2*sep*dsep
                        total_sep_squared += sep ** 2
                        grad = cp.zeros_like(xyt1)
                        grad[0] = 2.0 * sep * float(dsep_dx)
                        grad[1] = 2.0 * sep * float(dsep_dy)
                        grad[2] = 2.0 * sep * float(dsep_dtheta)
                        total_grad += grad

            # If using max behavior, handle the accumulated max for this other-tree.
            if self.use_max:
                if max_sep <= 0.0:
                    continue

                # Cost contribution: sep^2
                total_sep_squared += max_sep ** 2

                # Gradient of sep^2: 2 * sep * dsep/dparam
                grad = cp.zeros_like(xyt1)
                grad[0] = 2.0 * max_sep * float(best_dsep_dx)
                grad[1] = 2.0 * max_sep * float(best_dsep_dy)
                grad[2] = 2.0 * max_sep * float(best_dsep_dtheta)

                total_grad += grad

        return cp.array(total_sep_squared), total_grad
    

@dataclass
class BoundaryCost(Cost):
    # Cost for trees being out of bounds
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection, xyt, h):        
        # Use TreeList.get_trees() to build geometries and perform a single difference against the square (vectorized).
        raise Error('TODO: deal with square offsets')
        b = float(h[0].get().item())
        half = b / 2.0
        square = Polygon([(-half, -half), (half, -half), (half, half), (-half, half)])

        tree_list = kgs.TreeList()
        tree_list.xyt = xyt.get()
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
        grad_bound = cp.zeros_like(h)
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
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        raise Error('TODO: deal with square offsets')
        grad_h_temp = cp.zeros(sol.N_solutions, dtype=sol.h.dtype)
        pack_cuda.boundary_multi_ensemble(sol.xyt, sol.h[:,0], out_cost=cost, out_grads=grad_xyt, out_grad_h=grad_h_temp)
        grad_bound[:] = 0
        grad_bound[:,0] = grad_h_temp

@dataclass 
class AreaCost(Cost):
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection, xyt, h):
        cost = h[0]**2
        # Build grad_bound on the GPU without implicitly converting via NumPy
        grad_bound = cp.empty(3, dtype=h.dtype)
        grad_bound[0] = 2.0 * h[0]
        grad_bound[1] = 0
        grad_bound[2] = 0
        return cost, cp.zeros_like(xyt), grad_bound
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        cost[:] = sol.h[:,0]**2
        grad_xyt[:] = 0
        grad_bound[:] = 0
        grad_bound[:,0] = 2.0*sol.h[:,0]

@dataclass
class BoundaryDistanceCost(Cost):
    use_kernel : bool = field(init=True, default=True)
    # Cost based on squared distance of vertices outside the square boundary
    # Per tree, use only the vertex with the maximum distance
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection, xyt,h):
        # xyt is (n_trees, 3), bound is (1,); compute squared distance for each vertex outside square
        b = float(h[0].get().item())
        half = b / 2.0
        
        tree_list = kgs.TreeList()
        xyt = xyt.get()
        xyt[:,0] = xyt[:,0] - h[1].get()
        xyt[:,1] = xyt[:,1] - h[2].get()        
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
            xi = xyt[i,0].item()
            yi = xyt[i,1].item()
            thetai = xyt[i,2].item()
            
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
        grad_bound = cp.zeros_like(h)
        
        # Gradient w.r.t. h[0] (square size)
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
        
        # Gradient w.r.t. h[1] (x-offset) and h[2] (y-offset)
        # Increasing offset shifts square right/up, equivalent to shifting trees left/down
        # So grad_h[1] = -grad_xyt summed over x, grad_h[2] = -grad_xyt summed over y
        grad_bound[1] = -cp.sum(grad[:, 0])
        grad_bound[2] = -cp.sum(grad[:, 1])
        
        return cp.array(total_cost), grad, grad_bound
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray):
        if self.use_kernel:
            pack_cuda.boundary_distance_multi_ensemble(sol.xyt, sol.h, out_cost=cost, out_grads=grad_xyt, out_grad_h=grad_bound)
        else:
            super()._compute_cost(sol, cost, grad_xyt, grad_bound)
            
    def _compute_cost_single(self, sol:kgs.SolutionCollection, xyt, h):
        # xyt is (n_trees, 3), h is (3,): [square_size, x_offset, y_offset]
        # Use kgs.tree_vertices (precomputed center tree vertices) for efficient vectorized computation
    
        b = float(h[0].get().item())
        half = b / 2.0
        offset_x = h[1]  # (scalar)
        offset_y = h[2]  # (scalar)
        
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
        
        # Translate: add (x, y) for each tree, then subtract offsets (to shift relative to offset square)
        vx_final = vx_rot + x - offset_x  # (n_trees, n_vertices)
        vy_final = vy_rot + y - offset_y  # (n_trees, n_vertices)
        
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
        grad_bound_0 = cp.sum(grad_h)
        
        # Gradient w.r.t. h[1] (x-offset) and h[2] (y-offset)
        # Since vx_final = vx_rot + x - offset_x, d(vx_final)/d(offset_x) = -1
        # And d(cost)/d(vx_final) = grad_vx_max for the max vertex of each tree
        # So d(cost)/d(offset_x) = -sum(grad_vx_max)
        grad_bound_1 = -cp.sum(grad_vx_max)
        grad_bound_2 = -cp.sum(grad_vy_max)
        
        grad_bound = cp.array([grad_bound_0, grad_bound_1, grad_bound_2])
        
        return total_cost, grad, grad_bound

        