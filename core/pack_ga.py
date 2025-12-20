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
import pack_io
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import lap_batch


print('stop final relax at some point')



def compute_genetic_diversity(population_xyt: cp.ndarray, reference_xyt: cp.ndarray) -> cp.ndarray:
    """
    Compute the minimum-cost assignment distance between each individual in a population
    and a single reference configuration, considering all 8 symmetry transformations
    (4 rotations × 2 mirror states).
    
    Uses the Hungarian algorithm (via scipy) to find the optimal tree-to-tree mapping
    that minimizes total distance. The distance metric includes (x, y, theta) with equal weights.
    
    Parameters
    ----------
    population_xyt : cp.ndarray
        Shape (N_pop, N_trees, 3). Population of individuals, where each individual
        has N_trees trees with (x, y, theta) coordinates.
    reference_xyt : cp.ndarray
        Shape (N_trees, 3). Single reference configuration to compare against.
        
    Returns
    -------
    cp.ndarray
        Shape (N_pop,). Minimum assignment distance for each individual, taken over
        all 8 symmetry transformations.
        
    Notes
    -----
    The 8 transformations applied to each population individual are:
    - 0°, 90°, 180°, 270° rotations (about origin)
    - Each rotation with and without x-axis mirroring
    
    For each transformation, we compute the cost matrix between transformed trees
    and reference trees, then solve the linear assignment problem. The minimum
    cost across all 8 transformations is returned.
    
    Distance for each tree pair: sqrt((x1-x2)^2 + (y1-y2)^2 + angular_dist(theta1, theta2)^2)
    where angular_dist wraps to [-pi, pi].
    """
    N_pop, N_trees, _ = population_xyt.shape
    
    # Validate shapes
    assert reference_xyt.shape == (N_trees, 3), \
        f"Reference shape {reference_xyt.shape} doesn't match expected ({N_trees}, 3)"
    
    # Pre-compute the 8 transformation parameters
    # Each transformation is (rotation_angle, mirror_x)
    # rotation_angle: angle to rotate coordinates (0, pi/2, pi, 3pi/2)
    # mirror_x: whether to mirror across x-axis before rotation
    transformations = [
        (0.0,        False),  # Identity
        (np.pi/2,    False),  # 90° rotation
        (np.pi,      False),  # 180° rotation
        (3*np.pi/2,  False),  # 270° rotation
        (0.0,        True),   # Mirror only
        (np.pi/2,    True),   # Mirror + 90° rotation
        (np.pi,      True),   # Mirror + 180° rotation
        (3*np.pi/2,  True),   # Mirror + 270° rotation
    ]
    
    # Initialize result: will store minimum distance across all transformations
    min_distances = cp.full(N_pop, cp.inf, dtype=kgs.dtype_cp)
    
    # Reference coordinates (fixed, not transformed)
    ref_x = reference_xyt[:, 0]      # (N_trees,)
    ref_y = reference_xyt[:, 1]      # (N_trees,)
    ref_theta = reference_xyt[:, 2]  # (N_trees,)
    
    # Compute cost matrices for all 8 transformations on GPU
    all_cost_matrices = []
    for rot_angle, do_mirror in transformations:
        # ---------------------------------------------------------
        # Step 1: Apply transformation to population individuals (GPU)
        # ---------------------------------------------------------
        pop_x = population_xyt[:, :, 0].copy()
        pop_y = population_xyt[:, :, 1].copy()
        pop_theta = population_xyt[:, :, 2].copy()
        
        if do_mirror:
            pop_y = -pop_y
            pop_theta = -pop_theta
        
        if rot_angle != 0.0:
            cos_a = np.cos(rot_angle)
            sin_a = np.sin(rot_angle)
            new_x = pop_x * cos_a - pop_y * sin_a
            new_y = pop_x * sin_a + pop_y * cos_a
            pop_x = new_x
            pop_y = new_y
            pop_theta = pop_theta + rot_angle
        
        pop_theta = cp.remainder(pop_theta + np.pi, 2*np.pi) - np.pi
        
        # ---------------------------------------------------------
        # Step 2: Compute pairwise cost matrix (GPU)
        # ---------------------------------------------------------
        dx = pop_x[:, :, cp.newaxis] - ref_x[cp.newaxis, cp.newaxis, :]
        dy = pop_y[:, :, cp.newaxis] - ref_y[cp.newaxis, cp.newaxis, :]
        dtheta = pop_theta[:, :, cp.newaxis] - ref_theta[cp.newaxis, cp.newaxis, :]
        dtheta = cp.remainder(dtheta + np.pi, 2*np.pi) - np.pi
        cost_matrices = cp.sqrt(dx**2 + dy**2 + dtheta**2)
        
        all_cost_matrices.append(cost_matrices)
    
    # ---------------------------------------------------------
    # Step 3: Solve assignment problems on GPU using RAFT
    # ---------------------------------------------------------
    # Stack all cost matrices on GPU: shape (8, N_pop, N_trees, N_trees)
    # Then reshape to (8*N_pop, N_trees, N_trees) for batched solving
    stacked = cp.stack(all_cost_matrices, axis=0)  # (8, N_pop, N_trees, N_trees)
    batched = stacked.reshape(-1, N_trees, N_trees)    # (8*N_pop, N_trees, N_trees)
    
    # Solve all LAPs on GPU
    _, all_assignment_costs = lap_batch.solve_lap_batch(batched)  # (8*N_pop,)
    
    # Reshape back and take minimum across transformations
    all_costs_array = all_assignment_costs.reshape(8, N_pop)  # (8, N_pop)
    min_distances = all_costs_array.min(axis=0)
    
    if kgs.profiling:
        cp.cuda.Device().synchronize()
    
    return min_distances


# ============================================================
# Definition of population
# ============================================================

@dataclass
class Population(kgs.BaseClass):
    configuration: kgs.SolutionCollection = field(init=True, default=None)
    fitness: np.ndarray = field(init=True, default=None)
    parent_fitness: np.ndarray = field(init=True, default=None)
    # lineages: list = field(init=True, default=None)

    # Lineages is a list of lists, each list gives the history for an individual. Each element is a move, itself a list:
    # - First element describes the move (format up to the move itself, often including choices and perhaps some KPI)
    # - Second element is a list of fitness values:
    #  - Before the move
    #  - After the move
    #  - After rough relax
    #  - After fine relax

    def _check_constraints(self):
        self.configuration.check_constraints()
        assert self.fitness.shape == (self.configuration.N_solutions,)
        assert self.parent_fitness.shape == (self.configuration.N_solutions,)
        # assert len(self.lineages) == self.configuration.N_solutions

    def set_dummy_fitness(self):
        self.fitness = np.zeros(self.configuration.N_solutions)
        self.parent_fitness = np.zeros(self.configuration.N_solutions)

    def select_ids(self, inds):
        self.configuration.select_ids(inds)
        self.fitness = self.fitness[inds]
        self.parent_fitness = self.parent_fitness[inds]
        # self.lineages = [self.lineages[i] for i in inds]

    def create_empty(self, N_individuals, N_trees):
        configuration = self.configuration.create_empty(N_individuals, N_trees)
        population = type(self)(configuration=configuration)
        population.fitness = np.zeros(N_individuals, dtype=kgs.dtype_np)
        population.parent_fitness = np.zeros(N_individuals, dtype=kgs.dtype_np)
        # population.lineages = [ None for _ in range(N_individuals) ]
        return population

    def create_clone(self, idx: int, other: 'Population', parent_id: int):
        assert idx<self.configuration.N_solutions
        self.configuration.create_clone(idx, other.configuration, parent_id)
        self.fitness[idx] = other.fitness[parent_id]
        self.parent_fitness[idx] = other.fitness[parent_id]
        # self.lineages[idx] = copy.deepcopy(other.lineages[parent_id])

    def create_clone_batch(self, inds: np.ndarray, other: 'Population', parent_ids: np.ndarray):
        """Vectorized batch clone operation."""
        self.configuration.create_clone_batch(inds, other.configuration, parent_ids)
        self.fitness[inds] = other.fitness[parent_ids]
        self.parent_fitness[inds] = other.fitness[parent_ids]

    def merge(self, other:'Population'):
        self.configuration.merge(other.configuration)
        self.fitness = np.concatenate([self.fitness, other.fitness], axis=0)
        self.parent_fitness = np.concatenate([self.parent_fitness, other.parent_fitness], axis=0)
        # self.lineages = self.lineages + other.lineages


# ============================================================
# Population initializers
# ============================================================

@dataclass
class Initializer(kgs.BaseClass):
    seed: int = field(init=True, default=42)
    def initialize_population(self, N_individuals, N_trees):
        population = self._initialize_population(N_individuals, N_trees)
        population.set_dummy_fitness()
        assert population.configuration.N_solutions == N_individuals
        assert population.configuration.N_trees == N_trees
        population.check_constraints()
        return population        

ax=None

@dataclass
class InitializerRandomJiggled(Initializer):
    jiggler: pack_dynamics.DynamicsInitialize = field(init=True, default_factory=pack_dynamics.DynamicsInitialize)
    size_setup: float = field(init=True, default=0.65) # Will be scaled by sqrt(N_trees)    
    base_solution: kgs.SolutionCollection = field(init=True, default_factory=kgs.SolutionCollectionSquare)

    def _initialize_population(self, N_individuals, N_trees):
        self.check_constraints()
        size_setup_scaled = self.size_setup * np.sqrt(N_trees)
        xyt = np.random.default_rng(seed=self.seed).uniform(-0.5, 0.5, size=(N_individuals, N_trees, 3))
        xyt = xyt * [[[size_setup_scaled, size_setup_scaled, 2*np.pi]]]
        xyt = cp.array(xyt, dtype=kgs.dtype_np)    
        sol = copy.deepcopy(self.base_solution)
        sol.xyt = xyt   
        if not sol.use_fixed_h: 
            if isinstance(self.base_solution, kgs.SolutionCollectionSquare):
                sol.h = cp.array([[2*size_setup_scaled,0.,0.]]*N_individuals, dtype=kgs.dtype_np)           
            elif isinstance(self.base_solution, kgs.SolutionCollectionLatticeRectangle):
                sol.h = cp.array([[size_setup_scaled,size_setup_scaled]]*N_individuals, dtype=kgs.dtype_np)         
            elif isinstance(self.base_solution, kgs.SolutionCollectionLatticeFixed):
                sol.h = cp.array([[size_setup_scaled]]*N_individuals, dtype=kgs.dtype_np)  
                sol.aspect_ratios = cp.array([sol.aspect_ratios[0]]*N_individuals, dtype=kgs.dtype_cp)       
            else:
                assert(isinstance(self.base_solution, kgs.SolutionCollectionLattice))
                sol.h = cp.array([[size_setup_scaled,size_setup_scaled,np.pi/2]]*N_individuals, dtype=kgs.dtype_np)     
        else:
            sol.h = cp.tile(sol.fixed_h[cp.newaxis, :], (N_individuals, 1))          
        # NN=10
        # global ax
        # _,ax =  plt.subplots(NN,3,figsize=(24,8*NN))
        # for i in range(NN):
        #     pack_vis_sol.pack_vis_sol(sol, solution_idx=i, ax=ax[i,0])
        # sol.snap()
        # for i in range(NN):
        #     pack_vis_sol.pack_vis_sol(sol, solution_idx=i, ax=ax[i,1])
        sol = self.jiggler.run_simulation(sol)
        sol.snap()
        population = Population(configuration=sol)
        # population.lineages = [ [['InitializerRandomJiggled', [np.inf, np.inf, np.inf, 0., 0., 0.]]] for i in range(N_individuals) ]
        return population
    

# ============================================================
# Moves
# ============================================================
@dataclass
class Move(kgs.BaseClass):
    

    def do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                    inds_mate:np.ndarray, generator:np.random.Generator):
        """
        Vectorized move interface.

        Parameters
        ----------
        population : Population
            Target population where clones are already in place
        inds_to_do : np.ndarray
            Indices of individuals in population to modify (shape: N_moves,)
        old_pop : Population
            Source population to read from
        inds_mate : np.ndarray
            Indices of mate individuals in old_pop to use for crossover (shape: N_moves,)
        generator : np.random.Generator
            Random number generator
        """
        self._do_move_vec(population, inds_to_do, old_pop, inds_mate, generator)

    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """
        Default implementation: loop over individuals and call do_move for each.
        Clones are assumed to already be in population at indices inds_to_do.
        Subclasses can override this for better performance.
        """
        for ind_to_do, ind_mate in zip(inds_to_do, inds_mate):
            self._do_move(population, old_pop, ind_to_do, ind_mate, generator)    
    
class NoOp(Move):
    def _do_move(self, population, old_pop, individual_id, mate_id, generator):
        return None

@dataclass
class MoveSelector(Move):
    moves: list = field(init=True, default_factory=list) # each move is [Move, name, weight]
    _probabilities: np.ndarray = field(init=False, default=None)

    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """
        Vectorized move interface for MoveSelector.

        Loops over individuals, selecting which move type to use for each,
        then groups individuals by move type and calls do_move_vec on each group.
        """
        if self._probabilities is None:
            total_weight = sum([m[2] for m in self.moves])
            self._probabilities = np.array([m[2]/total_weight for m in self.moves], dtype=kgs.dtype_np)

        # First, select which move to use for each individual
        N_moves = len(inds_mate)
        chosen_move_ids = np.array([generator.choice(len(self.moves), p=self._probabilities) for _ in range(N_moves)])

        # Group individuals by chosen move type and execute moves
        for move_id in range(len(self.moves)):
            # Find all individuals that chose this move
            mask = chosen_move_ids == move_id
            if not np.any(mask):
                continue

            # Get indices for this move type
            batch_inds_to_do = inds_to_do[mask]
            batch_inds_mate = inds_mate[mask]

            # Call do_move_vec for this move type on the batch
            self.moves[move_id][0].do_move_vec(
                population, batch_inds_to_do, old_pop, batch_inds_mate, generator
            )

@dataclass
class MoveRandomTree(Move):
    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """Vectorized version: randomly reposition selected trees in multiple individuals."""
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_trees = new_xyt.shape[1]
        N_moves = len(inds_to_do)

        # Get h parameters (transfer to CPU once)
        h_params = new_h[inds_to_do].get()  # (N_moves, 3) on CPU

        # Generate random values and apply (loop to maintain exact RNG call order per individual)
        trees_to_mutate = []
        new_x_values = []
        new_y_values = []
        new_theta_values = []

        for i in range(N_moves):
            h_size = h_params[i, 0]
            tree_id = generator.integers(0, N_trees)
            x = generator.uniform(-h_size / 2, h_size / 2) + h_params[i, 1]
            y = generator.uniform(-h_size / 2, h_size / 2) + h_params[i, 2]
            theta = generator.uniform(-np.pi, np.pi)

            trees_to_mutate.append(tree_id)
            new_x_values.append(x)
            new_y_values.append(y)
            new_theta_values.append(theta)

        # Convert to GPU arrays
        new_x_gpu = cp.array(new_x_values)
        new_y_gpu = cp.array(new_y_values)
        new_theta_gpu = cp.array(new_theta_values)
        trees_to_mutate_gpu = cp.array(trees_to_mutate)
        inds_to_do_gpu = cp.array(inds_to_do)

        # Apply updates using fancy indexing (vectorized on GPU)
        new_xyt[inds_to_do_gpu, trees_to_mutate_gpu, 0] = new_x_gpu
        new_xyt[inds_to_do_gpu, trees_to_mutate_gpu, 1] = new_y_gpu
        new_xyt[inds_to_do_gpu, trees_to_mutate_gpu, 2] = new_theta_gpu  

@dataclass
class JiggleRandomTree(Move):
    max_xy_move: float = field(init=True, default=0.1)
    max_theta_move: float = field(init=True, default=np.pi)
    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """Vectorized version: jiggle random trees in multiple individuals."""
        new_xyt = population.configuration.xyt
        N_trees = new_xyt.shape[1]
        N_moves = len(inds_to_do)

        # Generate random choices (loop to maintain RNG order)
        trees_to_mutate = []
        offset_x = []
        offset_y = []
        offset_theta = []

        for i in range(N_moves):
            tree_id = generator.integers(0, N_trees)
            ox = generator.uniform(-self.max_xy_move, self.max_xy_move)
            oy = generator.uniform(-self.max_xy_move, self.max_xy_move)
            ot = generator.uniform(-self.max_theta_move, self.max_theta_move)

            trees_to_mutate.append(tree_id)
            offset_x.append(ox)
            offset_y.append(oy)
            offset_theta.append(ot)

        # Convert to GPU arrays
        offset_x_gpu = cp.array(offset_x)
        offset_y_gpu = cp.array(offset_y)
        offset_theta_gpu = cp.array(offset_theta)
        trees_to_mutate_gpu = cp.array(trees_to_mutate)
        inds_to_do_gpu = cp.array(inds_to_do)

        # Apply updates using fancy indexing (vectorized on GPU)
        new_xyt[inds_to_do_gpu, trees_to_mutate_gpu, 0] += offset_x_gpu
        new_xyt[inds_to_do_gpu, trees_to_mutate_gpu, 1] += offset_y_gpu
        new_xyt[inds_to_do_gpu, trees_to_mutate_gpu, 2] += offset_theta_gpu   

@dataclass
class JiggleCluster(Move):
    max_xy_move: float = field(init=True, default=0.1)
    max_theta_move: float = field(init=True, default=np.pi)
    min_N_trees: int = field(init=True, default=2)
    max_N_trees: int = field(init=True, default=20)
    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """Vectorized version: jiggle clusters of trees in multiple individuals."""
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_trees = new_xyt.shape[1]
        N_moves = len(inds_to_do)

        # Get h parameters (batch transfer)
        h_params = new_h[inds_to_do].get()  # (N_moves, 3)

        # Process each individual (RNG order matters, also different cluster sizes)
        for i, ind in enumerate(inds_to_do):
            h_size = h_params[i, 0]
            h_offset_x = h_params[i, 1]
            h_offset_y = h_params[i, 2]

            # Pick random center point
            center_x = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x
            center_y = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y

            # Get tree positions and compute distances
            tree_positions = new_xyt[ind, :, :2].get()  # (N_trees, 2)
            distances = (tree_positions[:, 0] - center_x)**2 + (tree_positions[:, 1] - center_y)**2

            # Find closest trees
            n_trees_to_jiggle = generator.integers(min(self.max_N_trees, N_trees), min(self.max_N_trees, N_trees) + 1)
            closest_tree_ids = np.argsort(distances)[:n_trees_to_jiggle]

            # Jiggle each selected tree
            for tree_id in closest_tree_ids:
                offset_x = generator.uniform(-self.max_xy_move, self.max_xy_move)
                offset_y = generator.uniform(-self.max_xy_move, self.max_xy_move)
                offset_theta = generator.uniform(-self.max_theta_move, self.max_theta_move)

                new_xyt[ind, tree_id, 0] += offset_x
                new_xyt[ind, tree_id, 1] += offset_y
                new_xyt[ind, tree_id, 2] += offset_theta

@dataclass
class Translate(Move):
    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """Vectorized version: translate all trees in multiple individuals."""
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_moves = len(inds_to_do)

        # Get h parameters (batch transfer)
        h_params = new_h[inds_to_do].get()  # (N_moves, 3)

        # Generate random offsets (loop to maintain RNG order)
        offset_x_list = []
        offset_y_list = []
        for i in range(N_moves):
            h_size = h_params[i, 0]
            offset_x_list.append(generator.uniform(-h_size / 2, h_size / 2))
            offset_y_list.append(generator.uniform(-h_size / 2, h_size / 2))

        # Convert to GPU and apply translations (vectorized)
        inds_to_do_gpu = cp.array(inds_to_do)
        offset_x_gpu = cp.array(offset_x_list)[:, cp.newaxis]  # (N_moves, 1)
        offset_y_gpu = cp.array(offset_y_list)[:, cp.newaxis]  # (N_moves, 1)
        h_sizes_gpu = cp.array(h_params[:, 0])[:, cp.newaxis]  # (N_moves, 1)

        # Apply translation with modulo (fully vectorized on GPU)
        new_xyt[inds_to_do_gpu, :, 0] = cp.mod(new_xyt[inds_to_do_gpu, :, 0] + offset_x_gpu, h_sizes_gpu) - h_sizes_gpu / 2
        new_xyt[inds_to_do_gpu, :, 1] = cp.mod(new_xyt[inds_to_do_gpu, :, 1] + offset_y_gpu, h_sizes_gpu) - h_sizes_gpu / 2
    
@dataclass
class Twist(Move):
    # Twist trees around a center. Angle of twist decreases linearly with distance from center
    min_radius: float = field(init=True, default=0.5)
    max_radius: float = field(init=True, default=2.)
    def _do_move_vec(self, population:Population, inds_to_do:np.ndarray, old_pop:Population,
                     inds_mate:np.ndarray, generator:np.random.Generator):
        """Vectorized version: twist trees around centers in multiple individuals."""
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_moves = len(inds_to_do)

        # Get h parameters (batch transfer)
        h_params = new_h[inds_to_do].get()  # (N_moves, 3)

        # Generate random parameters (loop to maintain RNG order)
        center_x_list = []
        center_y_list = []
        max_twist_angle_list = []
        radius_list = []

        for i in range(N_moves):
            h_size = h_params[i, 0]
            h_offset_x = h_params[i, 1]
            h_offset_y = h_params[i, 2]

            center_x_list.append(generator.uniform(-h_size / 2, h_size / 2) + h_offset_x)
            center_y_list.append(generator.uniform(-h_size / 2, h_size / 2) + h_offset_y)
            max_twist_angle_list.append(generator.uniform(-np.pi, np.pi))
            radius_list.append(generator.uniform(self.min_radius, self.max_radius))

        # Convert to GPU arrays with shape (N_moves, 1) for broadcasting
        inds_to_do_gpu = cp.array(inds_to_do)
        center_x_gpu = cp.array(center_x_list)[:, cp.newaxis]  # (N_moves, 1)
        center_y_gpu = cp.array(center_y_list)[:, cp.newaxis]  # (N_moves, 1)
        max_twist_angle_gpu = cp.array(max_twist_angle_list)[:, cp.newaxis]  # (N_moves, 1)
        radius_gpu = cp.array(radius_list)[:, cp.newaxis]  # (N_moves, 1)

        # Get tree positions (N_moves, N_trees)
        tree_x = new_xyt[inds_to_do_gpu, :, 0]
        tree_y = new_xyt[inds_to_do_gpu, :, 1]

        # Compute distances from center (fully vectorized on GPU)
        dx = tree_x - center_x_gpu
        dy = tree_y - center_y_gpu
        distances = cp.sqrt(dx**2 + dy**2)

        # Twist angle decreases linearly with distance
        twist_angles = max_twist_angle_gpu * cp.maximum(0, 1 - distances / radius_gpu)

        # Apply rotation around center point
        cos_angles = cp.cos(twist_angles)
        sin_angles = cp.sin(twist_angles)
        new_x = center_x_gpu + dx * cos_angles - dy * sin_angles
        new_y = center_y_gpu + dx * sin_angles + dy * cos_angles

        # Update all individuals at once (vectorized)
        new_xyt[inds_to_do_gpu, :, 0] = new_x
        new_xyt[inds_to_do_gpu, :, 1] = new_y
        new_xyt[inds_to_do_gpu, :, 2] += twist_angles


@dataclass
class Crossover(Move):
    """Replaces trees near a random point with transformed trees from a mate individual.
    
    Selects n trees closest to a random center point (using L-infinity distance for 
    square selection) and replaces them with the n closest trees from the mate,
    applying a random rotation (0/90/180/270°) and optional mirroring.
    """
    min_N_trees: int = field(init=True, default=4)
    max_N_trees: int = field(init=True, default=20)
    simple_mate_location: bool = field(init=True, default=False)

    def _do_move_vec(self, population: Population, inds_to_do: np.ndarray, old_pop: Population,
                     inds_mate: np.ndarray, generator: np.random.Generator):
        """Vectorized version: crossover trees from mates into multiple individuals."""
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_trees = new_xyt.shape[1]
        N_moves = len(inds_to_do)

        # Get h parameters (batch transfer)
        h_params = new_h[inds_to_do].get()  # (N_moves, 3)
        mate_h_params = old_pop.configuration.h[inds_mate].get()  # (N_moves, 3)

        # Process each individual (RNG order and variable crossover sizes)
        for i in range(N_moves):
            ind = inds_to_do[i]
            mate_id = inds_mate[i]

            h_size = h_params[i, 0]
            h_offset_x = h_params[i, 1]
            h_offset_y = h_params[i, 2]
            mate_h_size = mate_h_params[i, 0]

            # Pick random center point in individual's square
            offset_x = generator.uniform(-h_size / 2, h_size / 2)
            offset_y = generator.uniform(-h_size / 2, h_size / 2)
            center_x = offset_x + h_offset_x
            center_y = offset_y + h_offset_y

            # Pick mate center
            if self.simple_mate_location:
                mate_offset_x = generator.uniform(-h_size / 2, h_size / 2)
                mate_offset_y = generator.uniform(-h_size / 2, h_size / 2)
            else:
                mate_offset_x = abs(offset_x) * (mate_h_size / h_size) * generator.choice([-1, 1])
                mate_offset_y = abs(offset_y) * (mate_h_size / h_size) * generator.choice([-1, 1])
            mate_center_x = mate_offset_x + mate_h_params[i, 1]
            mate_center_y = mate_offset_y + mate_h_params[i, 2]

            # Find trees closest to centers (L-infinity distance)
            tree_positions = new_xyt[ind, :, :2].get()
            mate_positions = old_pop.configuration.xyt[mate_id, :, :2].get()

            distances_individual = np.maximum(
                np.abs(tree_positions[:, 0] - center_x),
                np.abs(tree_positions[:, 1] - center_y)
            )
            distances_mate = np.maximum(
                np.abs(mate_positions[:, 0] - mate_center_x),
                np.abs(mate_positions[:, 1] - mate_center_y)
            )

            # Select n closest trees
            n_trees_to_replace = generator.integers(
                min(self.min_N_trees, N_trees),
                min(self.max_N_trees, N_trees) + 1
            )
            individual_tree_ids = np.argsort(distances_individual)[:n_trees_to_replace]
            mate_tree_ids = np.argsort(distances_mate)[:n_trees_to_replace]

            # Replace centers with center of mass
            center_x = tree_positions[individual_tree_ids, 0].mean()
            center_y = tree_positions[individual_tree_ids, 1].mean()
            mate_center_x = mate_positions[mate_tree_ids, 0].mean()
            mate_center_y = mate_positions[mate_tree_ids, 1].mean()

            # Copy mate trees and apply transformation
            mate_trees = old_pop.configuration.xyt[mate_id, mate_tree_ids, :].copy()
            rotation_choice = generator.integers(0, 4)
            do_mirror = generator.integers(0, 2) == 1

            self._apply_transformation(
                mate_trees, mate_center_x, mate_center_y, center_x, center_y,
                rotation_choice, do_mirror
            )

            # Replace trees in individual
            new_xyt[ind, individual_tree_ids, :] = mate_trees

    def _apply_transformation(self, trees: cp.ndarray, 
                              src_center_x: float, src_center_y: float,
                              dst_center_x: float, dst_center_y: float,
                              rotation_choice: int, do_mirror: bool):
        """Apply rotation and mirroring, moving trees from src_center to dst_center."""
        # Get positions relative to source center (mate's center)
        dx = trees[:, 0] - src_center_x
        dy = trees[:, 1] - src_center_y
        theta = trees[:, 2]
        
        # Apply mirroring (across x-axis through center): y -> -y, theta -> pi - theta
        if do_mirror:
            dy = -dy
            theta = cp.pi - theta
        
        # Apply rotation (0°, 90°, 180°, or 270°)
        if rotation_choice != 0:
            rot_angle = rotation_choice * (np.pi / 2)
            cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
            dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
            theta = theta + rot_angle
        
        # Place at destination center (individual's center)
        trees[:, 0] = dst_center_x + dx
        trees[:, 1] = dst_center_y + dy
        trees[:, 2] = theta

# ============================================================
# Main GA algorithm
# ============================================================

@dataclass
class GA(kgs.BaseClass):
    # Configuration
    N_trees_to_do: np.ndarray = field(init=True, default=None)
    seed: int = field(init=True, default=42)
    plot_fitness_predictors: bool = field(init=True, default=False)
    plot_diversity_matrix: bool = field(init=True, default=False)
    plot_champions: bool = field(init=True, default=False)
    do_legalize: bool = field(init=True, default=True)

    # Hyperparameters
    population_size:int = field(init=True, default=4000)
    selection_size:list = field(init=True, default_factory=lambda: [int(4.*(x-1))+1 for x in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500]])
    n_generations:int = field(init=True, default=200)    
    fitness_cost: pack_cost.Cost = field(init=True, default=None)    
    initializer: Initializer = field(init=True, default_factory=InitializerRandomJiggled)
    rough_relaxers: list = field(init=True, default=None) # meant to prevent heavy overlaps
    fine_relaxers: list = field(init=True, default=None)  # meant to refine solutions
    h_schedule: list = field(init=True, default=None)
    reduce_h_threshold: float = field(init=True, default=-1.)
    reduce_h_amount: float = field(init=True, default=0.001)


    move: Move = field(init=True, default=None)

    # Outputs
    populations: list = field(init=True, default_factory=list)
    best_cost_per_generation = None
    best_individual_legalized = None
    scores = None

    def __post_init__(self):
        self.fitness_cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostSeparation(scaling=1.)])
        
        self.initializer.jiggler.n_rounds=0

        self.rough_relaxers = []
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostOverlappingArea(scaling=1.)
        relaxer.n_iterations = 120
        self.rough_relaxers.append(relaxer)

        self.fine_relaxers = []
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostSeparation(scaling=1.)
        relaxer.n_iterations = 30
        relaxer.max_step = 1e-2
        self.fine_relaxers.append(relaxer)
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostSeparation(scaling=1.)
        relaxer.n_iterations = 30
        relaxer.max_step = 1e-3
        self.fine_relaxers.append(relaxer)

        self.move = MoveSelector()
        self.move.moves = []
        self.move.moves.append( [MoveRandomTree(), 'MoveRandomTree', 1.0] )
        self.move.moves.append( [JiggleRandomTree(max_xy_move=0.05, max_theta_move=np.pi/6), 'JiggleTreeSmall', 1.0] ) 
        self.move.moves.append( [JiggleRandomTree(max_xy_move=0.1, max_theta_move=np.pi), 'JiggleTreeBig', 1.0] ) 
        self.move.moves.append( [JiggleCluster(max_xy_move=0.05, max_theta_move=np.pi/6), 'JiggleClusterSmall', 1.0] )
        self.move.moves.append( [JiggleCluster(max_xy_move=0.1, max_theta_move=np.pi), 'JiggleClusterBig', 1.0] )
        self.move.moves.append( [Translate(), 'Translate', 1.0] )
        self.move.moves.append( [Twist(), 'Twist', 1.0] )
        self.move.moves.append( [Crossover(), 'Crossover', 3.0] )

        super().__post_init__()
        # relaxer = pack_dynamics.Optimizer()
        # relaxer.cost = pack_cost.CollisionCostSeparation(scaling=1.)
        # relaxer.n_iterations *= 2
        # self.relaxers.append(relaxer)

    def _score(self, sol:kgs.SolutionCollection):
        costs = self.fitness_cost.compute_cost_allocate(sol)[0].get()
        for i in range(len(costs)):
            if np.isnan(costs[i]) or costs[i]>1e6:
                pack_vis_sol.pack_vis_sol(sol, solution_idx=i)
                plt.title(f'Invalid solution with cost {costs[i]}')
                raise AssertionError(f'Invalid solution with cost {costs[i]}')
        return costs


    def _relax_and_score(self, population:Population):
        sol = population.configuration

        # Track fitness at each stage
        fitness_initial = self._score(sol)

        for relaxer in self.rough_relaxers:
            sol = relaxer.run_simulation(sol)
        fitness_after_rough = self._score(sol)

        for relaxer in self.fine_relaxers:
            sol = relaxer.run_simulation(sol)
        fitness_final = self._score(sol)

        population.configuration = sol
        population.fitness = fitness_final

        if self.plot_fitness_predictors:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            # Plot 1: Parent fitness vs Final fitness
            plt.sca(ax[0])
            plt.scatter(population.parent_fitness, fitness_final)
            plt.xlabel('Parent Fitness')
            plt.ylabel('Final Fitness')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)

            # Plot 2: Fitness before rough relax vs Final fitness
            plt.sca(ax[1])
            plt.scatter(fitness_initial, fitness_final)
            plt.xlabel('Fitness Before Rough Relax')
            plt.ylabel('Final Fitness')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)

            # Plot 3: Fitness before fine relax vs Final fitness
            plt.sca(ax[2])
            plt.scatter(fitness_after_rough, fitness_final)
            plt.xlabel('Fitness Before Fine Relax')
            plt.ylabel('Final Fitness')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)

            plt.tight_layout()
            plt.pause(0.001)
            
    @kgs.profile_each_line
    def run(self):
        self.check_constraints()
        generator = np.random.default_rng(seed=self.seed)

        # Initialize populations
        for N_trees in self.N_trees_to_do:    
            self.initializer.seed = 200*self.seed + int(N_trees)
            population = self.initializer.initialize_population(self.population_size, N_trees)
            population.check_constraints()
            self._relax_and_score(population)    
            # global ax
            # NN=10
            # for i in range(NN):
            #     pack_vis_sol.pack_vis_sol(population.configuration, solution_idx=i, ax=ax[i,2])
            # plt.pause(0.001)
            self.populations.append(population)    

        self.best_cost_per_generation = np.zeros((self.n_generations, len(self.N_trees_to_do)))
        for i_gen in range(self.n_generations):

            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                if not self.h_schedule is None:
                    # Update h according to schedule
                    h_val = self.h_schedule[i_gen]
                    self.populations[i_N_trees].configuration.h[:, 0] = cp.array([h_val]*self.populations[i_N_trees].configuration.N_solutions, dtype=kgs.dtype_np)
                    self.populations[i_N_trees].configuration.fixed_h[0] = h_val
                    self.populations[i_N_trees].fitness = self._score(self.populations[i_N_trees].configuration)
                if np.min(self.populations[i_N_trees].fitness)<self.reduce_h_threshold:
                    # Reduce h if below threshold
                    self.populations[i_N_trees].configuration.h[:, 0] -= self.reduce_h_amount
                    self.populations[i_N_trees].configuration.fixed_h[0] -= self.reduce_h_amount
                    self.populations[i_N_trees].fitness = self._score(self.populations[i_N_trees].configuration)


            if i_gen>0:
                # Offspring generation
                for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                    old_pop = self.populations[i_N_trees]
                    old_pop.parent_fitness = old_pop.fitness.copy()
                    parent_size = old_pop.configuration.N_solutions
                    new_pop = old_pop.create_empty(self.population_size-parent_size, N_trees)

                    # Generate all parent and mate selections at once (vectorized)
                    N_offspring = new_pop.configuration.N_solutions

                    # Pick random parents
                    parent_ids = generator.integers(0, parent_size, size=N_offspring)

                    # Pick random mates (excluding parent) - fully vectorized
                    # Note: mate selection currently has weight=1 for all (0*np.arange(...)+1)
                    # This means uniform selection excluding the parent

                    # Strategy: pick random from [0, parent_size-1), then adjust if >= parent_id
                    mate_ids = generator.integers(0, parent_size - 1, size=N_offspring)
                    # If mate_id >= parent_id, increment by 1 to skip the parent
                    mate_ids = np.where(mate_ids >= parent_ids, mate_ids + 1, mate_ids)

                    # Clone parents into new_pop and set parent fitness (vectorized)
                    inds_to_do = np.arange(N_offspring)
                    new_pop.create_clone_batch(inds_to_do, old_pop, parent_ids)

                    # Apply moves using vectorized interface (clones already in place)
                    self.move.do_move_vec(new_pop, inds_to_do, old_pop, mate_ids, generator)



                        # new_pop.create_clone(i_ind, old_pop, parent_id)                    
                        # new_h = new_pop.configuration.h
                        # new_xyt = new_pop.configuration.xyt
                        # if generator.uniform() < self.p_move:
                        #     tree_to_mutate = generator.integers(0, N_trees)
                        #     h_size = new_h[i_ind, 0].get()  # Square size
                        #     h_offset_x = new_h[i_ind, 1].get()  # x offset
                        #     h_offset_y = new_h[i_ind, 2].get()  # y offset
                        #     new_xyt[i_ind, tree_to_mutate, 0] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x  # x
                        #     new_xyt[i_ind, tree_to_mutate, 1] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y  # y
                        #     new_xyt[i_ind, tree_to_mutate, 2] = generator.uniform(-np.pi, np.pi)  # theta                
                        # new_pop.lineages[i_ind].append(['dummy', [new_pop.fitness[i_ind],0.,0.,0.]])  # Placeholder for move description
                    self._relax_and_score(new_pop)
                    old_pop.merge(new_pop)
                    current_pop = old_pop
                    
                    current_pop.check_constraints()
                    self.populations[i_N_trees] = current_pop

            # Selection and diversity maintenance
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                best_id = np.argmin(self.populations[i_N_trees].fitness)                   
                print(f'Generation {i_gen}, Trees {N_trees}, Best cost: {self.populations[i_N_trees].fitness[best_id]:.8f}, Est: {100*self.populations[i_N_trees].fitness[best_id]/N_trees:.8f}, h: {self.populations[i_N_trees].configuration.h[best_id,0].get():.6f}')    
                #best_pop = copy.deepcopy(self.populations[i_N_trees])
                #best_pop.select_ids([best_id])
                #print(best_pop.configuration.h)
                #print(pack_cost.CollisionCostOverlappingArea().compute_cost_allocate(best_pop.configuration)[0].get(), 
                #      pack_cost.CollisionCostSeparation().compute_cost_allocate(best_pop.configuration)[0].get())
                self.best_cost_per_generation[i_gen, i_N_trees] = self.populations[i_N_trees].fitness[best_id]
                if (i_gen==0 or self.best_cost_per_generation[i_gen, i_N_trees]<self.best_cost_per_generation[i_gen-1, i_N_trees]) and self.plot_champions:
                    pack_vis_sol.pack_vis_sol(self.populations[i_N_trees].configuration, solution_idx=best_id)
                    plt.title(f'Generation: {i_gen}, cost: {self.best_cost_per_generation[i_gen, i_N_trees]}')
                    plt.pause(0.001)
                
                current_pop = self.populations[i_N_trees]
                current_pop.select_ids(np.argsort(current_pop.fitness))  # Sort by fitness
                current_xyt = current_pop.configuration.xyt  # (N_individuals, N_trees, 3)

                max_sel = np.max(self.selection_size)
                selected = np.zeros(self.population_size, dtype=bool)
                diversity = np.inf*np.ones(max_sel)
                for sel_size in self.selection_size:
                    selected_id = np.argmax(diversity[:sel_size])
                    selected[selected_id] = True
                    diversity = np.minimum(compute_genetic_diversity(cp.array(current_xyt[:max_sel]), cp.array(current_xyt[selected_id])).get(), diversity)
                    #print(sel_size, diversity)
                    assert(np.all(diversity[selected[:max_sel]]<1e-4))
                current_pop.select_ids(np.where(selected)[0])
                self.populations[i_N_trees] = current_pop
                self.populations[i_N_trees].check_constraints()
                if self.plot_diversity_matrix:
                    # Compute diversity matrix
                    diversity_matrix = np.zeros((self.populations[i_N_trees].configuration.N_solutions, self.populations[i_N_trees].configuration.N_solutions), dtype=kgs.dtype_np)
                    for i in range(self.populations[i_N_trees].configuration.N_solutions):
                        diversity_matrix[:,i] = compute_genetic_diversity(cp.array(self.populations[i_N_trees].configuration.xyt), cp.array(self.populations[i_N_trees].configuration.xyt[i])).get()

                    plt.figure(figsize=(6,6))
                    plt.imshow(diversity_matrix, cmap='viridis', vmin=0., vmax=np.max(diversity_matrix))
                    plt.colorbar(label='Diversity distance')
                    plt.title(f'Diversity matrix, Generation {i_gen}, Trees {N_trees}')
                    plt.pause(0.001)
    
        if self.do_legalize:
            # Final best individual legalization
            self.best_individual_legalized = []
            self.scores = []
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                pop = copy.deepcopy(self.populations[i_N_trees])
                best_id = np.argmin(pop.fitness)       
                pop.select_ids([best_id])
                sol = pop.configuration
                sol = pack_io.legalize(sol)
                self.best_individual_legalized.append(sol)
                self.scores.append((sol.h[0,0]**2/N_trees).get())

        
        