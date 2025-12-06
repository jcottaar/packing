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
import pack_dynamics
import copy

print('stop final relax at some point')


# ============================================================
# Genetic diversity measurement
# ============================================================

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
    min_distances = cp.full(N_pop, cp.inf, dtype=cp.float32)
    
    # Reference coordinates (fixed, not transformed)
    ref_x = reference_xyt[:, 0]      # (N_trees,)
    ref_y = reference_xyt[:, 1]      # (N_trees,)
    ref_theta = reference_xyt[:, 2]  # (N_trees,)
    
    for rot_angle, do_mirror in transformations:
        # ---------------------------------------------------------
        # Step 1: Apply transformation to population individuals
        # ---------------------------------------------------------
        # Extract coordinates
        pop_x = population_xyt[:, :, 0].copy()      # (N_pop, N_trees)
        pop_y = population_xyt[:, :, 1].copy()      # (N_pop, N_trees)
        pop_theta = population_xyt[:, :, 2].copy()  # (N_pop, N_trees)
        
        # Apply mirror first (reflect across x-axis: y -> -y, theta -> -theta)
        if do_mirror:
            pop_y = -pop_y
            pop_theta = -pop_theta
        
        # Apply rotation around origin
        # New coordinates: x' = x*cos(a) - y*sin(a), y' = x*sin(a) + y*cos(a)
        # New theta: theta' = theta + a
        if rot_angle != 0.0:
            cos_a = np.cos(rot_angle)
            sin_a = np.sin(rot_angle)
            new_x = pop_x * cos_a - pop_y * sin_a
            new_y = pop_x * sin_a + pop_y * cos_a
            pop_x = new_x
            pop_y = new_y
            pop_theta = pop_theta + rot_angle
        
        # Normalize theta to [-pi, pi] for consistent angular distance
        pop_theta = cp.remainder(pop_theta + np.pi, 2*np.pi) - np.pi
        
        # ---------------------------------------------------------
        # Step 2: Compute pairwise cost matrix for each individual
        # ---------------------------------------------------------
        # We need cost[i, j] = distance between pop_tree_i and ref_tree_j
        # for each individual in the population.
        
        # Expand dimensions for broadcasting:
        # pop: (N_pop, N_trees, 1), ref: (N_trees,) -> broadcast to (N_pop, N_trees, N_trees)
        dx = pop_x[:, :, cp.newaxis] - ref_x[cp.newaxis, cp.newaxis, :]  # (N_pop, N_trees, N_trees)
        dy = pop_y[:, :, cp.newaxis] - ref_y[cp.newaxis, cp.newaxis, :]  # (N_pop, N_trees, N_trees)
        
        # Angular distance: wrap difference to [-pi, pi]
        dtheta = pop_theta[:, :, cp.newaxis] - ref_theta[cp.newaxis, cp.newaxis, :]
        dtheta = cp.remainder(dtheta + np.pi, 2*np.pi) - np.pi  # Wrap to [-pi, pi]
        
        # Total distance (Euclidean in x,y,theta space with weight=1 for all)
        cost_matrices = cp.sqrt(dx**2 + dy**2 + dtheta**2)  # (N_pop, N_trees, N_trees)
        
        # ---------------------------------------------------------
        # Step 3: Solve assignment problem for each individual
        # ---------------------------------------------------------
        # Transfer all cost matrices to CPU in one go (single sync point),
        # then solve all assignments on CPU using scipy (faster for small N).
        
        cost_matrices_cpu = cost_matrices.get()  # Single GPU->CPU transfer
        transform_costs = np.empty(N_pop, dtype=np.float32)
        for i in range(N_pop):
            row_ind, col_ind = sp.optimize.linear_sum_assignment(cost_matrices_cpu[i])
            transform_costs[i] = cost_matrices_cpu[i, row_ind, col_ind].sum()
        
        # Single element-wise minimum update
        min_distances = cp.minimum(min_distances, cp.asarray(transform_costs))
    
    return min_distances


# ============================================================
# Definition of population
# ============================================================

@dataclass
class Population(kgs.BaseClass):
    configuration: kgs.SolutionCollection = field(init=True, default=None)
    fitness: np.ndarray = field(init=True, default=None)

    def _check_constraints(self):
        self.configuration.check_constraints()
        assert self.fitness.shape == (self.configuration.N_solutions,)

    def set_dummy_fitness(self):
        self.fitness = np.zeros(self.configuration.N_solutions)

    def select_ids(self, inds):
        self.configuration.xyt = self.configuration.xyt[inds]
        self.configuration.h = self.configuration.h[inds]
        self.fitness = self.fitness[inds]


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
        pass

@dataclass
class InitializerRandomJiggled(Initializer):
    jiggler: pack_dynamics.DynamicsInitialize = field(init=True, default_factory=pack_dynamics.DynamicsInitialize)
    size_setup: float = field(init=True, default=0.65) # Will be scaled by sqrt(N_trees)    

    def _initialize_population(self, N_individuals, N_trees):
        size_setup_scaled = self.size_setup * np.sqrt(N_trees)
        xyt = np.random.default_rng(seed=self.seed).uniform(-0.5, 0.5, size=(N_individuals, N_trees, 3))
        xyt = xyt * [[[size_setup_scaled, size_setup_scaled, np.pi]]]
        xyt = cp.array(xyt, dtype=np.float32)        
        h = cp.array([[2*size_setup_scaled,0.,0.]]*N_individuals, dtype=np.float32)
        sol = kgs.SolutionCollection(xyt=xyt, h=h)
        sol = self.jiggler.run_simulation(sol)
        population = Population(configuration=sol)
        return population



# ============================================================
# Main GA algorithm
# ============================================================

@dataclass
class GA(kgs.BaseClass):
    # Configuration
    N_trees_to_do: np.ndarray = field(init=True, default=None)
    seed: int = field(init=True, default=42)

    # Hyperparameters
    population_size:int = field(init=True, default=1000)
    selection_size:list[int] = field(init=True, default_factory=lambda: [1,2,3,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,500,1000])    
    n_generations:int = field(init=True, default=5000)    
    p_move: float = field(init=True, default=1.)
    fitness_cost: pack_cost.Cost = field(init=True, default=None)    
    initializer: Initializer = field(init=True, default_factory=InitializerRandomJiggled)
    relaxers: list[pack_dynamics.Optimizer] = field(init=True, default=None)

    # Outputs
    populations: list[Population] = field(init=True, default_factory=list)
    best_cost_per_generation = None

    def __post_init__(self):
        self.fitness_cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostSeparation(scaling=1.)])
        self.relaxers = []
        relaxer = pack_dynamics.Optimizer()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostOverlappingArea(scaling=1.)
        relaxer.n_iterations *= 2
        self.relaxers.append(relaxer)
        relaxer = pack_dynamics.Optimizer()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostSeparation(scaling=1.)
        relaxer.n_iterations *= 2
        self.relaxers.append(relaxer)
        # relaxer = pack_dynamics.Optimizer()
        # relaxer.cost = pack_cost.CollisionCostSeparation(scaling=1.)
        # relaxer.n_iterations *= 2
        # self.relaxers.append(relaxer)


    def _relax_and_score(self, population:Population):
        sol = population.configuration
        for relaxer in self.relaxers:
            sol = relaxer.run_simulation(sol)
        costs = self.fitness_cost.compute_cost_allocate(sol)[0]
        population.configuration = sol
        population.fitness = costs.get()

    def run(self):
        generator = np.random.default_rng(seed=self.seed)

        # Initialize populations
        for N_trees in self.N_trees_to_do:    
            self.initializer.seed = 200*self.seed + N_trees        
            population = self.initializer.initialize_population(self.population_size, N_trees)
            self._relax_and_score(population)
            self.populations.append(population)    

        self.best_cost_per_generation = np.zeros((self.n_generations, len(self.N_trees_to_do)))
        for i_gen in range(self.n_generations):

            # Selection and diversity maintenance
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                best_id = np.argmin(self.populations[i_N_trees].fitness)                   
                print(f'Generation {i_gen}, Trees {N_trees}, Best cost: {self.populations[i_N_trees].fitness[best_id]:.8f}, Est: {100*self.populations[i_N_trees].fitness[best_id]/N_trees:.8f}, h: {self.populations[i_N_trees].configuration.h[best_id,0].get():.6f}')    
                self.best_cost_per_generation[i_gen, i_N_trees] = self.populations[i_N_trees].fitness[best_id]                
                
                current_pop = self.populations[i_N_trees]
                current_pop.select_ids(np.argsort(current_pop.fitness))  # Sort by fitness
                current_xyt = current_pop.configuration.xyt  # (N_individuals, N_trees, 3)
                current_h = current_pop.configuration.h  # (N_individuals, 3) - [size, x_offset, y_offset]
                current_fitness = current_pop.fitness

                selected = np.zeros(self.population_size, dtype=bool)
                diversity = np.inf*np.ones(self.population_size)
                for sel_size in self.selection_size:
                    selected_id = np.argmax(diversity[:sel_size])
                    selected[selected_id] = True
                    diversity = np.minimum(compute_genetic_diversity(cp.array(current_xyt), cp.array(current_xyt[selected_id])).get(), diversity)
                    #print(sel_size, diversity)
                    assert(np.all(diversity[selected]<1e-4))
                current_pop.select_ids(np.where(selected)[0])
                self.populations[i_N_trees] = current_pop

            # Offspring generation
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                current_pop = self.populations[i_N_trees]
                parent_size = current_pop.configuration.N_solutions
                new_xyt = np.empty((self.population_size, N_trees, 3), dtype=np.float32)
                new_h = np.empty((self.population_size, 3), dtype=np.float32)
                new_xyt[:parent_size] = current_pop.configuration.xyt.get()
                new_h[:parent_size] = current_pop.configuration.h.get()
                for i_ind in np.arange(parent_size, self.population_size):
                    # Pick a random parent
                    parent_id = generator.integers(0, parent_size)
                    new_xyt[i_ind] = new_xyt[parent_id]
                    new_h[i_ind] = new_h[parent_id]
                    if generator.uniform() < self.p_move:
                        tree_to_mutate = generator.integers(0, N_trees)
                        h_size = new_h[i_ind, 0]  # Square size
                        h_offset_x = new_h[i_ind, 1]  # x offset
                        h_offset_y = new_h[i_ind, 2]  # y offset
                        new_xyt[i_ind, tree_to_mutate, 0] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x  # x
                        new_xyt[i_ind, tree_to_mutate, 1] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y  # y
                        new_xyt[i_ind, tree_to_mutate, 2] = generator.uniform(-np.pi, np.pi)  # theta
                current_pop.configuration.xyt = cp.array(new_xyt)
                current_pop.configuration.h = cp.array(new_h)
                self._relax_and_score(current_pop)
        