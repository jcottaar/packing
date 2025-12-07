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
import pack_vis
import pack_dynamics
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import lap_batch


print('stop final relax at some point')


@kgs.profile_each_line
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
    
    return min_distances


# ============================================================
# Definition of population
# ============================================================

@dataclass
class Population(kgs.BaseClass):
    configuration: kgs.SolutionCollection = field(init=True, default=None)
    fitness: np.ndarray = field(init=True, default=None)
    lineages: list = field(init=True, default=None)

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
        assert len(self.lineages) == self.configuration.N_solutions

    def set_dummy_fitness(self):
        self.fitness = np.zeros(self.configuration.N_solutions)

    def select_ids(self, inds):
        self.configuration.xyt = self.configuration.xyt[inds]
        self.configuration.h = self.configuration.h[inds]
        self.fitness = self.fitness[inds]
        self.lineages = [self.lineages[i] for i in inds]
    
    @classmethod
    def create_empty(cls, N_individuals, N_trees):
        xyt = cp.zeros((N_individuals, N_trees, 3), dtype=cp.float32)
        h = cp.zeros((N_individuals, 3), dtype=cp.float32)
        configuration = kgs.SolutionCollection(xyt=xyt, h=h)
        population = cls(configuration=configuration)
        population.fitness = np.zeros(N_individuals, dtype=np.float32)
        population.lineages = [ None for _ in range(N_individuals) ]
        return population

    def create_clone(self, idx: int, other: 'Population', parent_id: int):
        assert idx<self.configuration.N_solutions
        self.configuration.xyt[idx] = other.configuration.xyt[parent_id]
        self.configuration.h[idx] = other.configuration.h[parent_id]
        self.fitness[idx] = other.fitness[parent_id]
        self.lineages[idx] = copy.deepcopy(other.lineages[parent_id])

    def merge(self, other:'Population'):
        self.configuration.xyt = cp.concatenate([self.configuration.xyt, other.configuration.xyt], axis=0)
        self.configuration.h = cp.concatenate([self.configuration.h, other.configuration.h], axis=0)
        self.fitness = np.concatenate([self.fitness, other.fitness], axis=0)
        self.lineages = self.lineages + other.lineages


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
        population.lineages = [ [['InitializerRandomJiggled', [np.inf, np.inf, np.inf, 0., 0., 0.]]] for i in range(N_individuals) ]
        return population
    

# ============================================================
# Moves
# ============================================================
@dataclass
class Move(kgs.BaseClass):
    def do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):
        move_descriptor =  self._do_move(population, individual_id, mate_id, generator)
        # Check if any trees are within 1e-6 of each other -> error
        # moved = population.configuration.xyt[individual_id]  # (N_trees, 3) (cupy)
        # moved_arr = moved.get() if isinstance(moved, cp.ndarray) else np.array(moved)
        # x = moved_arr[:, 0][:, None]
        # y = moved_arr[:, 1][:, None]
        # theta = moved_arr[:, 2][:, None]
        # dx = x - x.T
        # dy = y - y.T
        # dtheta = theta - theta.T
        # dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        # pairwise_dist = np.sqrt(dx**2 + dy**2 + dtheta**2)
        # np.fill_diagonal(pairwise_dist, np.inf)
        # min_dist = pairwise_dist.min()
        # if min_dist < 1e-6:
        #     pairwise_dist[pairwise_dist<1e-6] = 1e-6
        #     plt.figure()
        #     plt.imshow(np.log(pairwise_dist))
        #     plt.colorbar()
        #     plt.pause(0.001)
        #     raise AssertionError(f"Two trees in individual {individual_id} are closer than 1e-6 (min_dist={min_dist})")
        return move_descriptor
    def _do_move(move:'Move', population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):
        raise NotImplementedError('Move subclass must implement do_move method')

@dataclass  
class MoveSelector(Move):
    moves: list = field(init=True, default_factory=list) # each move is [Move, name, weight]
    _probabilities: np.ndarray = field(init=False, default=None)

    def _do_move(self, population, individual_id, mate_id, generator):
        if self._probabilities is None:
            total_weight = sum([m[2] for m in self.moves])
            self._probabilities = np.array([m[2]/total_weight for m in self.moves], dtype=np.float32)
        chosen_id = generator.choice(len(self.moves), p=self._probabilities)
        move_descriptor = self.moves[chosen_id][0].do_move(population, individual_id, mate_id, generator)
        return [self.moves[chosen_id][1], move_descriptor]

@dataclass
class MoveRandomTree(Move):
    def _do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):                   
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        move_descriptor = []
        tree_to_mutate = generator.integers(0, new_xyt.shape[1])
        h_size = new_h[individual_id, 0].get()  # Square size
        h_offset_x = new_h[individual_id, 1].get()  # x offset
        h_offset_y = new_h[individual_id, 2].get()  # y offset
        move_descriptor.append(new_xyt[individual_id, tree_to_mutate].get())
        new_xyt[individual_id, tree_to_mutate, 0] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x  # x
        new_xyt[individual_id, tree_to_mutate, 1] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y  # y
        new_xyt[individual_id, tree_to_mutate, 2] = generator.uniform(-np.pi, np.pi)  # theta  
        move_descriptor.append(new_xyt[individual_id, tree_to_mutate].get()) 
        return move_descriptor  

@dataclass
class JiggleRandomTree(Move):
    max_xy_move: float = field(init=True, default=0.1)
    max_theta_move: float = field(init=True, default=np.pi)
    def _do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):                   
        new_xyt = population.configuration.xyt
        move_descriptor = []
        tree_to_mutate = generator.integers(0, new_xyt.shape[1])        
        move_descriptor.append(new_xyt[individual_id, tree_to_mutate].get())
        offset = [generator.uniform(-self.max_xy_move, self.max_xy_move),
                  generator.uniform(-self.max_xy_move, self.max_xy_move),
                    generator.uniform(-self.max_theta_move, self.max_theta_move)]
        new_xyt[individual_id, tree_to_mutate, 0] += offset[0]  # x
        new_xyt[individual_id, tree_to_mutate, 1] += offset[1]  # y
        new_xyt[individual_id, tree_to_mutate, 2] += offset[2]  # theta
        move_descriptor.append(offset)
        return move_descriptor   

@dataclass
class JiggleCluster(Move):
    max_xy_move: float = field(init=True, default=0.1)
    max_theta_move: float = field(init=True, default=np.pi)
    min_N_trees: int = field(init=True, default=2)
    max_N_trees: int = field(init=True, default=5)
    def _do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):                   
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_trees = new_xyt.shape[1]
        
        # Pick a random point inside the square defined by h
        h_size = new_h[individual_id, 0].get()  # Square size
        h_offset_x = new_h[individual_id, 1].get()  # x offset
        h_offset_y = new_h[individual_id, 2].get()  # y offset
        center_x = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x
        center_y = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y
        
        # Compute distances from all trees to the random point
        tree_positions = new_xyt[individual_id, :, :2].get()  # (N_trees, 2)
        distances = (tree_positions[:, 0] - center_x)**2 + (tree_positions[:, 1] - center_y)**2
        
        # Find min to max trees closest to that point
        n_trees_to_jiggle = generator.integers(min(self.max_N_trees, N_trees), min(self.max_N_trees, N_trees) + 1)
        closest_tree_ids = np.argsort(distances)[:n_trees_to_jiggle]
        
        move_descriptor = [(center_x, center_y), n_trees_to_jiggle]
        
        # Jiggle each selected tree independently
        for tree_id in closest_tree_ids:
            offset = [generator.uniform(-self.max_xy_move, self.max_xy_move),
                      generator.uniform(-self.max_xy_move, self.max_xy_move),
                      generator.uniform(-self.max_theta_move, self.max_theta_move)]
            new_xyt[individual_id, tree_id, 0] += offset[0]  # x
            new_xyt[individual_id, tree_id, 1] += offset[1]  # y
            new_xyt[individual_id, tree_id, 2] += offset[2]  # theta
            move_descriptor.append(offset)
        
        return move_descriptor

@dataclass
class Translate(Move):
    def _do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        h_size = new_h[individual_id, 0].get().item()  # Square size
        offset_x = generator.uniform(-h_size / 2, h_size / 2)
        offset_y = generator.uniform(-h_size / 2, h_size / 2)
        new_xyt[individual_id, :, 0] = cp.mod(new_xyt[individual_id, :, 0]+offset_x, h_size) - h_size/2
        new_xyt[individual_id, :, 1] = cp.mod(new_xyt[individual_id, :, 1]+offset_y, h_size) - h_size/2
        #new_xyt[individual_id, :, 0]        
        return [(offset_x, offset_y)]
    
@dataclass
class Twist(Move):
    # Twist trees around a center. Angle of twist decreases linearly with distance from center
    min_radius: float = field(init=True, default=0.)
    max_radius: float = field(init=True, default=2.)
    def _do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        
        # Pick a random center point inside the square defined by h
        h_size = new_h[individual_id, 0].get()  # Square size
        h_offset_x = new_h[individual_id, 1].get()  # x offset
        h_offset_y = new_h[individual_id, 2].get()  # y offset
        center_x = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x
        center_y = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y
        
        # Random twist angle at center and radius
        max_twist_angle = generator.uniform(-np.pi, np.pi)
        radius = generator.uniform(self.min_radius, self.max_radius)
        
        # Get tree positions
        tree_x = new_xyt[individual_id, :, 0]  # (N_trees,) on GPU
        tree_y = new_xyt[individual_id, :, 1]  # (N_trees,) on GPU
        
        # Compute distances from center
        dx = tree_x - center_x
        dy = tree_y - center_y
        distances = cp.sqrt(dx**2 + dy**2)
        
        # Twist angle decreases linearly with distance, zero at radius
        twist_angles = max_twist_angle * cp.maximum(0, 1 - distances / radius)
        
        # Apply rotation around center point
        cos_angles = cp.cos(twist_angles)
        sin_angles = cp.sin(twist_angles)
        new_x = center_x + dx * cos_angles - dy * sin_angles
        new_y = center_y + dx * sin_angles + dy * cos_angles
        
        new_xyt[individual_id, :, 0] = new_x
        new_xyt[individual_id, :, 1] = new_y
        new_xyt[individual_id, :, 2] += twist_angles  # Also rotate the trees themselves
        
        return [(center_x, center_y), max_twist_angle, radius]

@ dataclass
class Crossover(Move):
    # Replaces a number of trees closest to a given point with trees from another individual
    min_N_trees: int = field(init=True, default=4)
    max_N_trees: int = field(init=True, default=20)
    def _do_move(self, population:Population, individual_id:int, mate_id:int, generator:np.random.Generator):
        new_h = population.configuration.h
        new_xyt = population.configuration.xyt
        N_trees = new_xyt.shape[1]
        
        # Pick a random center point inside the square defined by h
        h_size = new_h[individual_id, 0].get()  # Square size
        h_offset_x = new_h[individual_id, 1].get()  # x offset
        h_offset_y = new_h[individual_id, 2].get()  # y offset
        center_x = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x
        center_y = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y
        
        # Compute distances from all trees (of individual) to the random point (L-infinity for square selection)
        tree_positions = new_xyt[individual_id, :, :2].get()  # (N_trees, 2)
        distances_individual = np.maximum(np.abs(tree_positions[:, 0] - center_x), np.abs(tree_positions[:, 1] - center_y))
        
        # Compute distances from all trees (of mate) to the random point (L-infinity for square selection)
        mate_positions = new_xyt[mate_id, :, :2].get()  # (N_trees, 2)
        distances_mate = np.maximum(np.abs(mate_positions[:, 0] - center_x), np.abs(mate_positions[:, 1] - center_y))
        
        # Find n trees closest to that point in individual (to be replaced)
        n_trees_to_replace = generator.integers(min(self.min_N_trees, N_trees), min(self.max_N_trees, N_trees) + 1)
        individual_tree_ids = np.argsort(distances_individual)[:n_trees_to_replace]
        
        # Find n trees closest to that point in mate (to be copied from)
        mate_tree_ids = np.argsort(distances_mate)[:n_trees_to_replace]
        
        # Get mate trees to copy (make a copy to apply transformations)
        mate_trees = new_xyt[mate_id, mate_tree_ids, :].copy()  # (n_trees_to_replace, 3) on GPU
        
        # Random 0/90/180/270 rotation around the center point
        rotation_choice = generator.integers(0, 4)  # 0, 1, 2, 3 -> 0, 90, 180, 270 degrees
        rot_angle = rotation_choice * (np.pi / 2)
        
        # Random mirroring (across x-axis through center)
        do_mirror = generator.integers(0, 2) == 1
        
        # Get positions relative to center
        dx = mate_trees[:, 0] - center_x
        dy = mate_trees[:, 1] - center_y
        theta = mate_trees[:, 2]
        
        # Apply mirroring first (across x-axis through center)
        # Position: y -> -y (relative to center)
        # Theta: theta -> pi - theta (flip the tree orientation)
        if do_mirror:
            dy = -dy
            theta = cp.pi - theta
        
        # Apply rotation
        if rotation_choice != 0:
            cos_a = np.cos(rot_angle)
            sin_a = np.sin(rot_angle)
            new_dx = dx * cos_a - dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            dx = new_dx
            dy = new_dy
            theta = theta + rot_angle
        
        # Convert back to absolute positions
        mate_trees[:, 0] = center_x + dx
        mate_trees[:, 1] = center_y + dy
        mate_trees[:, 2] = theta
        
        # Replace trees in individual with transformed trees from mate
        new_xyt[individual_id, individual_tree_ids, :] = mate_trees
        
        return [(center_x, center_y), n_trees_to_replace, rotation_choice, do_mirror]

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

    # Hyperparameters
    population_size:int = field(init=True, default=1000)
    selection_size:list = field(init=True, default_factory=lambda: [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500])
    n_generations:int = field(init=True, default=5000)    
    fitness_cost: pack_cost.Cost = field(init=True, default=None)    
    initializer: Initializer = field(init=True, default_factory=InitializerRandomJiggled)
    rough_relaxers: list = field(init=True, default=None) # meant to prevent heavy overlaps
    fine_relaxers: list = field(init=True, default=None)  # meant to refine solutions

    move: Move = field(init=True, default=None)

    # Outputs
    populations: list = field(init=True, default_factory=list)
    best_cost_per_generation = None

    def __post_init__(self):
        self.fitness_cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostSeparation(scaling=1.)])
        self.rough_relaxers = []
        relaxer = pack_dynamics.Optimizer()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostOverlappingArea(scaling=1.)
        relaxer.n_iterations *= 2
        self.rough_relaxers.append(relaxer)

        self.fine_relaxers = []
        relaxer = pack_dynamics.Optimizer()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostSeparation(scaling=1.)
        relaxer.n_iterations *= 2
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
        self.move.moves.append( [Crossover(), 'Crossover', 4.0] )
        # relaxer = pack_dynamics.Optimizer()
        # relaxer.cost = pack_cost.CollisionCostSeparation(scaling=1.)
        # relaxer.n_iterations *= 2
        # self.relaxers.append(relaxer)


    def _relax_and_score(self, population:Population):        
        sol = population.configuration
        costs = self.fitness_cost.compute_cost_allocate(sol)[0].get()
        for i in range(len(costs)):
            population.lineages[i][-1][1][3]= costs[i]
        for relaxer in self.rough_relaxers:
            sol = relaxer.run_simulation(sol)
        costs = self.fitness_cost.compute_cost_allocate(sol)[0].get()
        for i in range(len(costs)):
            population.lineages[i][-1][1][4]= costs[i]
        for relaxer in self.fine_relaxers:
            sol = relaxer.run_simulation(sol)
        costs = self.fitness_cost.compute_cost_allocate(sol)[0].get()
        for i in range(len(costs)):
            population.lineages[i][-1][1][5]= costs[i]
        population.configuration = sol
        population.fitness = costs
        if self.plot_fitness_predictors:
            fig,ax = plt.subplots(1,3,figsize=(12,4))
            y_vals = [x[-1][1][5] for x in population.lineages]
            for i_ax,ii in enumerate([0,3,4]):
                x_vals = [x[-1][1][ii] for x in population.lineages]
                plt.sca(ax[i_ax])
                plt.scatter(x_vals, y_vals)
                plt.grid(True)
            plt.pause(0.001)

    def run(self):
        self.check_constraints()
        generator = np.random.default_rng(seed=self.seed)

        # Initialize populations
        for N_trees in self.N_trees_to_do:    
            self.initializer.seed = 200*self.seed + N_trees        
            population = self.initializer.initialize_population(self.population_size, N_trees)
            population.check_constraints()
            self._relax_and_score(population)
            self.populations.append(population)    

        self.best_cost_per_generation = np.zeros((self.n_generations, len(self.N_trees_to_do)))
        for i_gen in range(self.n_generations):

            # Selection and diversity maintenance
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                best_id = np.argmin(self.populations[i_N_trees].fitness)                   
                print(f'Generation {i_gen}, Trees {N_trees}, Best cost: {self.populations[i_N_trees].fitness[best_id]:.8f}, Est: {100*self.populations[i_N_trees].fitness[best_id]/N_trees:.8f}, h: {self.populations[i_N_trees].configuration.h[best_id,0].get():.6f}')    
                self.best_cost_per_generation[i_gen, i_N_trees] = self.populations[i_N_trees].fitness[best_id]                
                if i_gen==0 or self.best_cost_per_generation[i_gen, i_N_trees]<self.best_cost_per_generation[i_gen-1, i_N_trees]:
                    tree_list = kgs.TreeList()
                    tree_list.xyt = self.populations[i_N_trees].configuration.xyt[best_id].get()
                    pack_vis.visualize_tree_list(tree_list)
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
                # Compute diversity matrix
                diversity_matrix = np.zeros((self.populations[i_N_trees].configuration.N_solutions, self.populations[i_N_trees].configuration.N_solutions), dtype=np.float32)
                for i in range(self.populations[i_N_trees].configuration.N_solutions):
                    diversity_matrix[:,i] = compute_genetic_diversity(cp.array(self.populations[i_N_trees].configuration.xyt), cp.array(self.populations[i_N_trees].configuration.xyt[i])).get()
                if self.plot_diversity_matrix:
                    plt.figure(figsize=(6,6))
                    plt.imshow(diversity_matrix, cmap='viridis', vmin=0., vmax=np.max(diversity_matrix))
                    plt.colorbar(label='Diversity distance')
                    plt.title(f'Diversity matrix, Generation {i_gen}, Trees {N_trees}')
                    plt.pause(0.001)

            

            # Offspring generation
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                old_pop = self.populations[i_N_trees]
                parent_size = old_pop.configuration.N_solutions
                new_pop = Population.create_empty(self.population_size-parent_size, N_trees)

                for i_ind in range(new_pop.configuration.N_solutions):
                    # Pick a random parent
                    parent_id = generator.integers(0, parent_size)
                    # Pick a random mate, but preferring champions
                    # pick a mate, where the worst (last) individual gets weight 1, the next best weight 2, etc. (DISABLED FOR NOW)
                    weights = 0*np.arange(parent_size, 0, -1, dtype=float)+1  # best has largest weight
                    weights[parent_id] = 0.0
                    probs = weights / weights.sum()
                    mate_id = generator.choice(parent_size, p=probs)

                    new_pop.create_clone(i_ind, old_pop, parent_id) 
                    move_descriptor = self.move.do_move(new_pop, i_ind, mate_id, generator)                  
                    new_pop.lineages[i_ind].append([move_descriptor, [old_pop.fitness[parent_id],old_pop.fitness[mate_id],diversity_matrix[parent_id, mate_id],0.,0.,0.]])  # Placeholder for move description



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
        