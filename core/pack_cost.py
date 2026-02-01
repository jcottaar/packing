import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field
from typeguard import typechecked
from shapely.geometry import Polygon
import shapely
import pack_cuda
import pack_cuda_lut
import copy

@dataclass
class Cost(kgs.BaseClass):
    scaling:float = field(init=True, default=1.0)

    @typechecked 
    def compute_cost_ref(self, sol:kgs.SolutionCollection):
        # xyt: (n_ensemble, n_trees, 3) array
        #bound: (n_ensemble, 1 or 3) array
        if not sol.is_phenotype():
            sol_phenotype = sol.convert_to_phenotype()
            cost, grad_xyt_phenotype, grad_h_phenotype = self.compute_cost_ref(sol_phenotype)
            # backprop gradients from phenotype to genotype space
            grad_xyt = cp.zeros_like(sol.xyt)
            grad_bound = cp.zeros_like(sol.h)
            sol.backprop_phenotype(grad_xyt_phenotype, grad_h_phenotype, grad_xyt, grad_bound)
            return cost, grad_xyt, grad_bound

        sol.check_constraints()                
        cost,grad_xyt,grad_bound = self._compute_cost_ref(sol)
        assert cost.shape == (sol.N_solutions,)
        assert grad_xyt.shape == sol.xyt.shape
        assert grad_bound.shape == sol.h.shape 
        if sol.use_fixed_h:
            grad_bound[:] = 0
        return self.scaling*cost,self.scaling*grad_xyt,self.scaling*grad_bound
    
    def _compute_cost_ref(self, sol:kgs.SolutionCollection):        
        cost = cp.zeros(sol.N_solutions)
        grad_xyt = cp.zeros_like(sol.xyt)
        grad_bound = cp.zeros_like(sol.h)
        for i in range(sol.N_solutions):
            sol_tmp = copy.deepcopy(sol)
            sol_tmp.select_ids([i])
            cost[i],grad_xyt[i],grad_bound[i] = self._compute_cost_single_ref(sol_tmp)
        return cost,grad_xyt,grad_bound
    
    def compute_cost_allocate(self, sol:kgs.SolutionCollection, evaluate_gradient:bool=True):
        # Allocates gradient arrays and calls compute_cost
        cost = cp.zeros(sol.N_solutions, dtype=sol.xyt.dtype)
        # Always allocate arrays for gradients
        grad_xyt = cp.zeros_like(sol.xyt)
        grad_bound = cp.zeros_like(sol.h)
        self.compute_cost(sol, cost, grad_xyt, grad_bound, evaluate_gradient=evaluate_gradient)
        return cost, grad_xyt, grad_bound
    
    def compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient:bool=True):
        # Subclass can implement faster version with preallocated gradients
        if not sol.is_phenotype():            
            sol_phenotype = sol.convert_to_phenotype()
            # Allocate temp arrays for phenotype gradients
            grad_xyt_temp = cp.zeros_like(sol_phenotype.xyt)
            grad_h_temp = cp.zeros_like(sol_phenotype.h)
            self.compute_cost(sol_phenotype, cost, grad_xyt_temp, grad_h_temp, evaluate_gradient=evaluate_gradient)
            # backprop gradients from phenotype to genotype space
            if evaluate_gradient:
                sol.backprop_phenotype(grad_xyt_temp, grad_h_temp, grad_xyt, grad_bound)
            return
        self._compute_cost(sol, cost, grad_xyt, grad_bound, evaluate_gradient)
        if self.scaling != 1.0:
            cost *= self.scaling
            if evaluate_gradient:
                grad_xyt *= self.scaling
                if sol.use_fixed_h:
                    grad_bound[:] = 0
                else:
                    grad_bound *= self.scaling
        else:
            if sol.use_fixed_h:
                grad_bound[:] = 0
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient):
        # Subclass can implement faster version with preallocated gradients
        for i in range(sol.N_solutions):
            sol_tmp = copy.deepcopy(sol)
            sol_tmp.select_ids([i])
            cost[i],grad_xyt[i],grad_bound[i] = self._compute_cost_single(sol_tmp, evaluate_gradient)
    
    def _compute_cost_single(self, sol:kgs.SolutionCollection, evaluate_gradient):
        return self._compute_cost_single_ref(sol)

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

    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient):
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
            c.compute_cost(sol, self._temp_cost, self._temp_grad_xyt, self._temp_grad_bound, evaluate_gradient)
            cost += self._temp_cost
            grad_xyt += self._temp_grad_xyt
            grad_bound += self._temp_grad_bound

@dataclass
class CostDummy(Cost):
    # Dummy: always zero cost
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection):
        return cp.array(0.0), cp.zeros_like(sol.xyt[0]), cp.zeros_like(sol.h[0])

@dataclass
class CollisionCost(Cost):
    use_lookup_table: bool = field(init=True, default=False)
    lut_N_x: int = field(init=True, default=900)
    lut_N_y: int = field(init=True, default=900)
    lut_N_theta: int = field(init=True, default=900)
    lut_trim_zeros: bool = field(init=True, default=True)
    _lut: 'pack_cuda_lut.LookupTable' = field(init=False, default=None, repr=False)

    def _ensure_lut_initialized(self):
        """Initialize lookup table if use_lookup_table is True and not yet initialized."""
        if self.use_lookup_table and self._lut is None:
            import pack_cuda_lut
            print(f"Building lookup table for {self.__class__.__name__}...")
            
            # Create wrapper function that matches the expected signature
            def eval_fn(dx: np.ndarray, dy: np.ndarray, theta: float) -> np.ndarray:
                """Evaluate cost for array of (dx, dy) positions at given theta."""
                N = len(dx)
                
                # Create solution with 2 trees:
                # Tree 0: at origin (0, 0, 0)
                # Tree 1: at (dx, dy, theta)
                xyt = np.zeros((N, 2, 3), dtype=np.float32)
                xyt[:, 1, 0] = dx
                xyt[:, 1, 1] = dy
                xyt[:, 1, 2] = theta
                
                xyt_cp = cp.asarray(xyt)
                
                # Create solution collection
                sol = kgs.SolutionCollectionSquare()
                sol.xyt = xyt_cp
                # Large boundary to avoid clipping
                sol.h = cp.tile(cp.array([[10., 0., 0.]], dtype=cp.float32), (N, 1))
                sol.check_constraints()
                
                # Temporarily disable LUT to avoid infinite recursion
                saved_use_lut = self.use_lookup_table
                self.use_lookup_table = False
                
                try:
                    # Compute costs
                    costs, _, _ = self.compute_cost_allocate(sol, evaluate_gradient=False)
                    return costs.get()
                finally:
                    self.use_lookup_table = saved_use_lut
            
            self._lut = pack_cuda_lut.LookupTable.build_from_function(
                eval_fn=eval_fn,
                N_x=self.lut_N_x,
                N_y=self.lut_N_y,
                N_theta=self.lut_N_theta,
                trim_zeros=self.lut_trim_zeros,
                verbose=True
            )

    def __getstate__(self):
        """Exclude lookup table from pickle serialization (contains GPU resources)."""
        state = self.__dict__.copy()
        # Remove the unpicklable lookup table
        state['_lut'] = None
        return state

    def __setstate__(self, state):
        """Restore object state from pickle, lookup table will be lazily rebuilt if needed."""
        self.__dict__.update(state)
        # _lut is already None from __getstate__, will be rebuilt by _ensure_lut_initialized() if needed

    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection):
        # Compute collision cost for all pairs of trees
        assert(sol.N_solutions==1)
        xyt = sol.xyt[0]
        h = sol.h[0]
        n_trees = sol.N_trees

        if sol.periodic:
            # Periodic BC: use analytical gradients w.r.t. xyt (scatter pattern)
            # but finite differences for h (lattice parameters)
            tree_list = kgs.TreeList()
            tree_list.xyt = xyt
            trees = tree_list.get_trees()

            # Get crystal axes
            import copy
            sol_here = copy.deepcopy(sol)
            sol_here.h = h[None]
            crystal_axes = cp.zeros((1, 2, 2), dtype=xyt.dtype)
            sol_here.get_crystal_axes(crystal_axes)
            a_vec = crystal_axes[0, 0, :]
            b_vec = crystal_axes[0, 1, :]
            a_vec_np = a_vec.get()
            b_vec_np = b_vec.get()

            total_cost = cp.array(0.0)
            total_grad = cp.zeros_like(xyt)

            for i in range(n_trees):
                this_xyt = xyt[i]
                this_tree = trees[i]

                # Collect all other trees including periodic images (excluding ALL self-interactions)
                other_trees_all = []
                other_xyt_all = []

                # Loop over grid of periodic cells
                for dx in range(-sol.N_periodic, sol.N_periodic + 1):
                    for dy in range(-sol.N_periodic, sol.N_periodic + 1):
                        shift = dx * a_vec_np + dy * b_vec_np

                        for j in range(n_trees):
                            # Skip ALL self-interactions (including periodic)
                            if i == j:
                                continue

                            # Translate tree j by the lattice shift
                            tree_j_shifted = shapely.affinity.translate(trees[j], xoff=shift[0], yoff=shift[1])
                            other_trees_all.append(tree_j_shifted)

                            # Create shifted pose for xyt2 (needed by CollisionCostSeparation)
                            xyt_j_shifted = xyt[j].copy()
                            xyt_j_shifted[0] += cp.array(shift[0])
                            xyt_j_shifted[1] += cp.array(shift[1])
                            other_xyt_all.append(xyt_j_shifted)

                # Stack into array for xyt2
                if len(other_xyt_all) > 0:
                    other_xyt = cp.stack(other_xyt_all, axis=0)
                else:
                    other_xyt = cp.zeros((0, 3), dtype=xyt.dtype)

                # Use subclass's _compute_cost_one_tree_ref (overlap area or separation)
                # Get both cost and gradient
                this_cost, this_grads = self._compute_cost_one_tree_ref(this_xyt, other_xyt, this_tree, other_trees_all)
                total_cost += this_cost / 2
                total_grad[i] += this_grads

            # Handle self-interactions separately using finite differences
            # Build helper function to compute cost with given xyt for a specific tree i
            def _compute_cost_only_xyt(xyt_in, tree_idx):
                tree_list_tmp = kgs.TreeList()
                tree_list_tmp.xyt = xyt_in
                trees_tmp = tree_list_tmp.get_trees()

                this_tree_tmp = trees_tmp[tree_idx]
                other_trees_self = []
                other_xyt_self = []

                # Only include self-interactions (tree i with its periodic images)
                for dx in range(-sol.N_periodic, sol.N_periodic + 1):
                    for dy in range(-sol.N_periodic, sol.N_periodic + 1):
                        if dx == 0 and dy == 0:
                            continue  # Skip origin

                        shift = dx * a_vec_np + dy * b_vec_np
                        tree_i_shifted = shapely.affinity.translate(trees_tmp[tree_idx], xoff=shift[0], yoff=shift[1])
                        other_trees_self.append(tree_i_shifted)

                        xyt_i_shifted = xyt_in[tree_idx].copy()
                        xyt_i_shifted[0] += cp.array(shift[0])
                        xyt_i_shifted[1] += cp.array(shift[1])
                        other_xyt_self.append(xyt_i_shifted)

                cost_tmp = 0.0
                if len(other_xyt_self) > 0:
                    other_xyt_arr = cp.stack(other_xyt_self, axis=0)
                    this_cost_tmp, X = self._compute_cost_one_tree_ref(xyt_in[tree_idx], other_xyt_arr, this_tree_tmp, other_trees_self)
                    cost_tmp = this_cost_tmp / 2

                return cp.array(cost_tmp)

            # # Add self-interaction cost to total (sum over all trees)
            for i in range(n_trees):
                self_interaction_cost = _compute_cost_only_xyt(xyt, i)
                total_cost += self_interaction_cost

            # Compute self-interaction gradients using finite differences
            eps = 1e-6
            for i in range(n_trees):
                for j in [2]:  # theta only, x and y are 0
                    xyt_plus = xyt.copy()
                    xyt_minus = xyt.copy()
                    xyt_plus[i, j] += eps
                    xyt_minus[i, j] -= eps

                    cost_plus = _compute_cost_only_xyt(xyt_plus, i)
                    cost_minus = _compute_cost_only_xyt(xyt_minus, i)

                    total_grad[i, j] += (cost_plus - cost_minus) / (2.0 * eps)

            # Compute gradients w.r.t. h using finite differences
            grad_h = cp.zeros_like(h)

            def _compute_cost_only_h(h_in):
                # Recompute cost with different h
                tree_list_tmp = kgs.TreeList()
                tree_list_tmp.xyt = xyt
                trees_tmp = tree_list_tmp.get_trees()

                crystal_axes_tmp = cp.zeros((1, 2, 2), dtype=xyt.dtype)
                sol_tmp = copy.deepcopy(sol)
                sol_tmp.xyt = cp.array([xyt])
                sol_tmp.h = cp.array([h_in])
                sol_tmp.get_crystal_axes(crystal_axes_tmp)
                a_vec_tmp_np = crystal_axes_tmp[0, 0, :].get()
                b_vec_tmp_np = crystal_axes_tmp[0, 1, :].get()

                cost_tmp = 0.0
                for i in range(n_trees):
                    this_tree_tmp = trees_tmp[i]
                    other_trees_tmp = []
                    other_xyt_tmp = []

                    for dx in range(-sol.N_periodic, sol.N_periodic + 1):
                        for dy in range(-sol.N_periodic, sol.N_periodic + 1):
                            shift_tmp = dx * a_vec_tmp_np + dy * b_vec_tmp_np
                            for j in range(n_trees):
                                if dx == 0 and dy == 0 and i == j:
                                    continue
                                tree_j_tmp = shapely.affinity.translate(trees_tmp[j], xoff=shift_tmp[0], yoff=shift_tmp[1])
                                other_trees_tmp.append(tree_j_tmp)
                                xyt_j_tmp = xyt[j].copy()
                                xyt_j_tmp[0] += cp.array(shift_tmp[0])
                                xyt_j_tmp[1] += cp.array(shift_tmp[1])
                                other_xyt_tmp.append(xyt_j_tmp)

                    if len(other_xyt_tmp) > 0:
                        other_xyt_arr = cp.stack(other_xyt_tmp, axis=0)
                    else:
                        other_xyt_arr = cp.zeros((0, 3), dtype=xyt.dtype)

                    this_cost_tmp, _ = self._compute_cost_one_tree_ref(xyt[i], other_xyt_arr, this_tree_tmp, other_trees_tmp)
                    cost_tmp += this_cost_tmp / 2

                return cp.array(cost_tmp)

            for i in range(h.shape[0]):
                h_plus = h.copy()
                h_minus = h.copy()
                h_plus[i] += eps
                h_minus[i] -= eps

                cost_plus = _compute_cost_only_h(h_plus)
                cost_minus = _compute_cost_only_h(h_minus)

                grad_h[i] = (cost_plus - cost_minus) / (2.0 * eps)

            return total_cost, total_grad, grad_h
        else:
            # Non-periodic: use existing analytical approach
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
  
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient):
        if not sol.periodic:
            # Use lookup table if enabled
            if self.use_lookup_table:
                self._ensure_lut_initialized()
                import pack_cuda_lut
                # Pass LUT directly (fast - updates constants only if changed)
                if evaluate_gradient:
                    pack_cuda_lut.overlap_multi_ensemble(sol.xyt, cost, self._lut, grad_xyt)
                    grad_bound[:] = 0
                else:
                    pack_cuda_lut.overlap_multi_ensemble(sol.xyt, cost, self._lut)
            else:
                self._compute_cost_internal(sol, cost, grad_xyt, grad_bound, evaluate_gradient)
        else:
            crystal_axes = sol.get_crystal_axes_allocate().reshape(-1,4)
            self._compute_cost_internal(sol, cost, grad_xyt, grad_bound, evaluate_gradient, crystal_axes=crystal_axes)

            # do finite gradient over the above to compute grad_xyt from self interactions, excluded above
            # use only_self_interactions is true
            # Compute missing self-interaction costs (kernel does not provide gradients for these).
            eps = 1e-6

            tmp_grad_dummy = None
            tmp_bound_dummy = None

            if evaluate_gradient:
                # Compute theta-gradients for self-interactions by finite differences.
                # Vectorized over the ensemble: perturb each tree's theta across all solutions.
                n_trees = sol.N_trees
                cost_plus = cp.zeros(sol.N_solutions, dtype=cost.dtype)
                cost_minus = cp.zeros(sol.N_solutions, dtype=cost.dtype)

                for t in range(n_trees):
                    xyt_orig = sol.xyt.copy()
                    xyt_plus = sol.xyt.copy()
                    xyt_minus = sol.xyt.copy()
                    xyt_plus[:, t, 2] += eps
                    xyt_minus[:, t, 2] -= eps

                    # Only request costs for self-interactions
                    sol.xyt[...] = xyt_plus
                    self._compute_cost_internal(
                        sol,
                        cost_plus,
                        tmp_grad_dummy,
                        tmp_bound_dummy,
                        evaluate_gradient=False,
                        crystal_axes=crystal_axes,
                        only_self_interactions=True,
                    )
                    sol.xyt[...] = xyt_minus
                    self._compute_cost_internal(
                        sol,
                        cost_minus,
                        tmp_grad_dummy,
                        tmp_bound_dummy,
                        evaluate_gradient=False,
                        crystal_axes=crystal_axes,
                        only_self_interactions=True,
                    )
                    sol.xyt[...] = xyt_orig

                    grad_theta = (cost_plus - cost_minus) / (2.0 * eps)
                    grad_xyt[:, t, 2] += grad_theta/2

            if evaluate_gradient:

                # Now, find gradients w.r.t. h using finite differences
                # Vectorized finite-difference for grad w.r.t. h (no loop over solutions)
                eps = 1e-6
                grad_bound[:] = 0
                n_bounds = sol.h.shape[1]

                # Temporary arrays for compute_cost_internal outputs
                tmp_grad_xyt = cp.zeros_like(sol.xyt)
                tmp_grad_bound = cp.zeros_like(sol.h)

                for k in range(n_bounds):
                    # build perturbed h arrays for all solutions at once
                    h_orig = sol.h.copy()
                    h_plus = sol.h.copy()
                    h_minus = sol.h.copy()
                    h_plus[:, k] += eps
                    h_minus[:, k] -= eps

                    # prepare solution-like objects to compute crystal axes for each perturbed ensemble
                    #sol_plus = type(sol)()
                    #sol_plus.xyt = sol.xyt
                    #sol_plus.h = h_plus
                    sol.h[...] = h_plus
                    crystal_axes_plus = sol.get_crystal_axes_allocate().reshape(-1, 4)
                    cost_plus = cp.zeros(sol.N_solutions, dtype=cost.dtype)
                    self._compute_cost_internal(sol, cost_plus, tmp_grad_xyt, tmp_grad_bound, evaluate_gradient=False, crystal_axes=crystal_axes_plus)

                    # sol_minus = type(sol)()
                    # sol_minus.xyt = sol.xyt
                    # sol_minus.h = h_minus
                    sol.h[...] = h_minus 
                    crystal_axes_minus = sol.get_crystal_axes_allocate().reshape(-1, 4)
                    cost_minus = cp.zeros(sol.N_solutions, dtype=cost.dtype)
                    self._compute_cost_internal(sol, cost_minus, tmp_grad_xyt, tmp_grad_bound, evaluate_gradient=False, crystal_axes=crystal_axes_minus)

                    sol.h[...] = h_orig
                    

                    # central difference across the ensemble (vectorized)
                    grad_bound[:, k] = (cost_plus - cost_minus) / (2.0 * eps)
        
    
    
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

    def _compute_cost_internal(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient, crystal_axes=None,
                               only_self_interactions=False):
        # Use fast CUDA implementation
        if evaluate_gradient:
            pack_cuda.overlap_multi_ensemble(sol.xyt, sol.xyt, False, out_cost=cost, out_grads=grad_xyt, crystal_axes=crystal_axes, only_self_interactions=only_self_interactions, N_periodic=sol.N_periodic)
            grad_bound[:] = 0
        else:
            pack_cuda.overlap_multi_ensemble(sol.xyt, sol.xyt, False, out_cost=cost, crystal_axes=crystal_axes, only_self_interactions=only_self_interactions, N_periodic=sol.N_periodic)

@dataclass
class CollisionCostSeparation(CollisionCost):
    # Collision cost based on minimum separation distance between trees
    # Cost = sum(separation_distance^2) over all pairwise overlaps
    # Only overlapping pairs contribute (separated pairs have zero cost)
    # Separation distance = minimum distance trees must move to no longer overlap
    
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
                    sep, dsep_dx, dsep_dy, dsep_dtheta = \
                        self._compute_separation_distance(
                            poly1_world=pa_world,
                            poly2_world=pb_world,
                            local_coords1=pa_local,
                            x1=x1,
                            y1=y1,
                            th1=th1,
                        )

                    if sep <= 0.0:
                        continue
                    # Add sep^2 contribution and its gradient 2*sep*dsep
                    total_sep_squared += sep ** 2
                    grad = cp.zeros_like(xyt1)
                    grad[0] = 2.0 * sep * float(dsep_dx)
                    grad[1] = 2.0 * sep * float(dsep_dy)
                    grad[2] = 2.0 * sep * float(dsep_dtheta)
                    total_grad += grad

        return cp.array(total_sep_squared), total_grad
    
    def _compute_cost_internal(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient, crystal_axes=None,
                               only_self_interactions=False):
        # Use fast CUDA implementation
        if evaluate_gradient:
            pack_cuda.overlap_multi_ensemble(sol.xyt, sol.xyt, True, out_cost=cost, out_grads=grad_xyt, crystal_axes=crystal_axes, only_self_interactions=only_self_interactions, N_periodic=sol.N_periodic)
            grad_bound[:] = 0
        else:
            pack_cuda.overlap_multi_ensemble(sol.xyt, sol.xyt, True, out_cost=cost, crystal_axes=crystal_axes, only_self_interactions=only_self_interactions, N_periodic=sol.N_periodic)


@dataclass
class CollisionCostExactSeparation(CollisionCost):
    """
    Collision cost based on exact Minkowski difference separation distance.
    
    This class ONLY works with lookup tables - no reference implementation.
    Uses pack_minkowski.separation_distance to compute exact penetration depth
    between trees based on their Minkowski difference (configuration space obstacle).
    
    Tree 1 is always at rotation 0 (never rotated).
    """
    
    def _get_cache_filename(self):
        """Generate cache filename based on LUT parameters."""
        import hashlib
        
        # Create a string with all relevant parameters
        param_str = f"exact_sep_Nx{self.lut_N_x}_Ny{self.lut_N_y}_Nt{self.lut_N_theta}_trim{self.lut_trim_zeros}"
        
        # Add a hash of the tree geometry (from kgs.tree_vertices) to ensure cache validity
        tree_hash = hashlib.md5(kgs.tree_vertices.get().tobytes()).hexdigest()[:8]
        param_str += f"_tree{tree_hash}"
        
        return f"{param_str}.lut_cache"
    
    def _ensure_lut_initialized(self):
        """Initialize lookup table using pack_minkowski for exact separation computation."""
        if not self.use_lookup_table:
            raise ValueError(f"{self.__class__.__name__} requires use_lookup_table=True")
        
        if self._lut is None:
            import pack_cuda_lut
            import pack_minkowski as mink
            import os
            import pickle
            
            # Set up cache directory and filename
            cache_dir = kgs.temp_dir + '/lut_cache/'
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = cache_dir + self._get_cache_filename()
            
            # Try to load from cache
            if os.path.exists(cache_path):
                try:
                    print(f"Loading cached lookup table from {cache_path}...")
                    with open(cache_path, 'rb') as f:
                        self._lut = pickle.load(f)
                    print("Successfully loaded cached lookup table")
                    return
                except Exception as e:
                    print(f"Warning: Failed to load cache ({e}), rebuilding...")
            
            print(f"Building lookup table for {self.__class__.__name__} using Minkowski difference...")
            
            # Create wrapper function using pack_minkowski
            def eval_fn(dx: np.ndarray, dy: np.ndarray, theta: float) -> np.ndarray:
                """
                Evaluate exact separation distance for array of (dx, dy) positions at given theta.
                
                Uses Minkowski difference to compute penetration depth.
                Tree1 is at origin with rotation 0, Tree2 at (dx, dy) with rotation theta.
                
                Returns raw signed separation distances (positive = overlap, negative = clearance).
                The quadratic transform is applied later in _compute_cost().
                """
                
                # Tree positions for separation_distance
                # Tree1 always at origin with rotation 0
                tree1_pos = (0.0, 0.0)
                
                # Tree2 positions: array of (dx, dy)
                tree2_positions = np.column_stack([dx, dy])  # (N, 2)
                
                # Compute separation distances (vectorized)
                # Returns signed distance: positive for overlap, negative for clearance
                sep_distances = mink.separation_distance(tree1_pos, tree2_positions, theta)

                # Store raw separation distances in LUT (not squared)
                # _compute_cost() will apply the max(0, sep)^2 transform
                return sep_distances.astype(np.float32)
            
            lut = pack_cuda_lut.LookupTable.build_from_function(
                eval_fn=eval_fn,
                N_x=self.lut_N_x,
                N_y=self.lut_N_y,
                N_theta=self.lut_N_theta,
                trim_zeros=self.lut_trim_zeros,
                verbose=True
            )
            # Enable per-pair quadratic transform in kernel
            lut.apply_quadratic_transform = True
            self._lut = lut
            
            # Save to cache
            print(f"Saving lookup table to cache: {cache_path}...")
            with open(cache_path, 'wb') as f:
                pickle.dump(lut, f)
            print("Successfully cached lookup table")
    
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection):
        """Not implemented - this class requires lookup table."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a reference implementation. "
            "Set use_lookup_table=True to use this cost function."
        )
    
    def _compute_cost_internal(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, 
                               grad_bound:cp.ndarray, evaluate_gradient, crystal_axes=None,
                               only_self_interactions=False):
        """Not implemented - this class requires lookup table."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a GPU kernel implementation. "
            "Set use_lookup_table=True to use this cost function."
        )
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, 
                      grad_bound:cp.ndarray, evaluate_gradient):
        """
        Compute cost from exact Minkowski separation with quadratic penalty for overlap.
        
        Cost = sum over pairs of max(0, separation)^2 where separation > 0 means overlap.
        The LUT stores raw separation distances, and the kernel applies max(0, sep)^2 per-pair.
        """
        # Assert we're using lookup table
        assert self.use_lookup_table, f"{self.__class__.__name__} requires use_lookup_table=True"
        assert not sol.periodic, f"{self.__class__.__name__} does not support periodic boundaries yet"
        
        # Kernel applies quadratic transform per-pair (controlled by LUT.apply_quadratic_transform)
        # No post-processing needed - cost and gradients are already transformed
        super()._compute_cost(sol, cost, grad_xyt, grad_bound, evaluate_gradient)
        

@dataclass 
class AreaCost(Cost):
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection):
        if sol.use_fixed_h:
            return cp.array(0.0), cp.zeros_like(sol.xyt[0]), cp.zeros_like(sol.h[0])
        else:
            cost, grad_bound = sol.compute_cost_single_ref()
            return cost, cp.zeros_like(sol.xyt[0]), grad_bound
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient:bool):        
        grad_xyt[:] = 0
        if sol.use_fixed_h:
            cost[:] = 0
            grad_bound[:] = 0
        else:
            sol.compute_cost(sol, cost, grad_bound)        

@dataclass
class BoundaryDistanceCost(Cost):
    use_kernel : bool = field(init=True, default=True)
    # Cost based on squared distance of vertices outside the square boundary
    # Per tree, use only the vertex with the maximum distance
    def _compute_cost_single_ref(self, sol:kgs.SolutionCollection):
        assert(sol.N_solutions==1)
        xyt = sol.xyt[0]
        h = sol.h[0]

        assert not sol.periodic
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
    
    def _compute_cost(self, sol:kgs.SolutionCollection, cost:cp.ndarray, grad_xyt:cp.ndarray, grad_bound:cp.ndarray, evaluate_gradient):
        assert not sol.periodic
        if self.use_kernel:
            if evaluate_gradient:
                pack_cuda.boundary_distance_multi_ensemble(sol.xyt, sol.h, out_cost=cost, out_grads=grad_xyt, out_grad_h=grad_bound)
            else:
                pack_cuda.boundary_distance_multi_ensemble(sol.xyt, sol.h, out_cost=cost)
        else:
            super()._compute_cost(sol, cost, grad_xyt, grad_bound, evaluate_gradient)
            
    def _compute_cost_single(self, sol:kgs.SolutionCollection, evaluate_gradient):
        # xyt is (n_trees, 3), h is (3,): [square_size, x_offset, y_offset]
        # Use kgs.tree_vertices (precomputed center tree vertices) for efficient vectorized computation
        # evaluate_gradient is ignored
        assert(sol.N_solutions==1)
        xyt = sol.xyt[0]
        h = sol.h[0]
    
        b = float(h[0].get().item())
        half = b / 2.0
        offset_x = h[1]  # (scalar)
        offset_y = h[2]  # (scalar)
        
        n_trees = xyt.shape[0]
        
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

        