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
from pack_move import (Move, NoOp, MoveSelector, MoveRandomTree, JiggleRandomTree,
                       JiggleCluster, Translate, Twist, Crossover)


# ============================================================
# Definition of population
# ============================================================

@dataclass
class Population(kgs.BaseClass):
    configuration: kgs.SolutionCollection = field(init=True, default=None)
    fitness: np.ndarray = field(init=True, default=None)  # Shape: (N_solutions, N_components)
    parent_fitness: np.ndarray = field(init=True, default=None)  # Shape: (N_solutions, N_components)
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
        assert self.fitness.shape[0] == self.configuration.N_solutions
        assert self.fitness.ndim == 2  # Shape: (N_solutions, N_components)
        assert self.parent_fitness.shape[0] == self.configuration.N_solutions
        assert self.parent_fitness.ndim == 2  # Shape: (N_solutions, N_components)
        # assert len(self.lineages) == self.configuration.N_solutions

    def set_dummy_fitness(self, n_components=1):
        """Initialize fitness arrays with zeros.
        
        Parameters
        ----------
        n_components : int, optional
            Number of fitness components (default: 1)
        """
        self.fitness = np.zeros((self.configuration.N_solutions, n_components), dtype=kgs.dtype_np)
        self.parent_fitness = np.zeros((self.configuration.N_solutions, n_components), dtype=kgs.dtype_np)

    def select_ids(self, inds):
        self.configuration.select_ids(inds)
        self.fitness = self.fitness[inds]
        self.parent_fitness = self.parent_fitness[inds]
        # self.lineages = [self.lineages[i] for i in inds]

    def create_empty(self, N_individuals, N_trees):
        configuration = self.configuration.create_empty(N_individuals, N_trees)
        population = type(self)(configuration=configuration)
        # Initialize with same number of components as self
        n_components = self.fitness.shape[1]
        population.fitness = np.zeros((N_individuals, n_components), dtype=kgs.dtype_np)
        population.parent_fitness = np.zeros((N_individuals, n_components), dtype=kgs.dtype_np)
        # population.lineages = [ None for _ in range(N_individuals) ]
        return population

    def create_clone(self, idx: int, other: 'Population', parent_id: int):
        assert idx<self.configuration.N_solutions
        self.configuration.create_clone(idx, other.configuration, parent_id)
        self.fitness[idx] = other.fitness[parent_id]
        self.parent_fitness[idx] = other.fitness[parent_id]
        # self.lineages[idx] = copy.deepcopy(other.lineages[parent_id])

    def create_clone_batch(self, inds: cp.ndarray, other: 'Population', parent_ids: cp.ndarray):
        """Vectorized batch clone operation."""
        self.configuration.create_clone_batch(inds, other.configuration, parent_ids)
        # Convert indices to CPU for NumPy array indexing
        inds_cpu = inds.get() if isinstance(inds, cp.ndarray) else inds
        parent_ids_cpu = parent_ids.get() if isinstance(parent_ids, cp.ndarray) else parent_ids
        self.fitness[inds_cpu] = other.fitness[parent_ids_cpu]
        self.parent_fitness[inds_cpu] = other.fitness[parent_ids_cpu]

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
    fixed_h: cp.ndarray = field(init=True, default=None) # if not None, should be (3,) array

    def _initialize_population(self, N_individuals, N_trees):
        self.check_constraints()
        size_setup_scaled = self.size_setup * np.sqrt(N_trees)
        xyt = np.random.default_rng(seed=self.seed).uniform(-0.5, 0.5, size=(N_individuals, N_trees, 3))
        xyt = xyt * [[[size_setup_scaled, size_setup_scaled, 2*np.pi]]]
        xyt = cp.array(xyt, dtype=kgs.dtype_np)    
        sol = copy.deepcopy(self.base_solution)
        sol.xyt = xyt   
        if self.fixed_h is None:
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
            sol.h = cp.tile(self.fixed_h[cp.newaxis, :], (N_individuals, 1))                 
        sol = self.jiggler.run_simulation(sol)
        sol.snap()
        population = Population(configuration=sol)
        return population
    

# ============================================================
# Main GA algorithm
# ============================================================


@dataclass
class GA(kgs.BaseClass):
    # Abstract superclass for GA algorithms
    seed: int = field(init=True, default=42)    
    fitness_cost: pack_cost.Cost = field(init=True, default=None)     

    champions: list = field(init=True, default=None)
    best_costs_per_generation: list = field(init=True, default_factory=list)  
    do_legalize: bool = field(init=True, default=True)

    _cached_offspring = None

    def _check_constraints(self):
        super()._check_constraints()
        if self.fitness_cost is not None:
            self.fitness_cost.check_constraints()
        if self.champions is not None:
            for champion in self.champions:
                assert(champion.configuration.N_solutions==1)
                champion.check_constraints()
            assert(len(self.champions) == len(self.best_costs_per_generation))     
            for champion, cost in zip(self.champions, self.best_costs_per_generation):
                assert(np.all(cost[-1] == champion.fitness[0]))

    def initialize(self):
        self.check_constraints(debugging_mode_offset=2)        
        self.fitness_cost.check_constraints()
        self._initialize()

    def score(self, register_best=False):
        self._score(register_best)
        if register_best:
            assert(len(self.champions) == len(self.best_costs_per_generation))            

    def generate_offspring(self, mate_sol, mate_weights):
        if mate_sol is not None:
            assert(mate_sol.N_solutions == len(mate_weights))
        res = self._generate_offspring(mate_sol, mate_weights)
        self._cached_offspring = res
        return res
    
    def merge_offspring(self):
        self._merge_offspring()

    def get_list_for_simulation(self):
        return self._get_list_for_simulation()
           
    def apply_selection(self):
        self._apply_selection()

    def finalize(self):
        self._finalize()
        if self.do_legalize:
            for champion in self.champions:
                champion.configuration = pack_io.legalize(champion.configuration)
    
    def abbreviate(self):
        self._abbreviate()

@dataclass
class GAMulti(GA):
    # Configuration    
    ga_list: list = field(init=True, default=None)
    single_champion: bool = field(init=True, default=True)

    def _check_constraints(self):
        super()._check_constraints()
        if self.ga_list is not None:
            self.ga_list[0].check_constraints()
            if kgs.debugging_mode>=2:
                for ga in self.ga_list:
                    ga.check_constraints()
    def _initialize(self):
        for ga in self.ga_list:
            ga.initialize()
        assert self.single_champion
        self.best_costs_per_generation = [[]]
    def _score(self, register_best):
        for ga in self.ga_list:
            ga.score(register_best=register_best)
        if register_best:
            assert self.single_champion
            if kgs.debugging_mode>=2:
                for ga in self.ga_list:
                    assert(len(ga.champions) == 1)
            costs_per_ga = [ga.best_costs_per_generation[0][-1] for ga in self.ga_list]
            best_ga_idx = np.argmin(costs_per_ga)
            best_ga = self.ga_list[best_ga_idx]
            self.champions = [best_ga.champions[0]]
            self.best_costs_per_generation[0].append(best_ga.best_costs_per_generation[0][-1])
    def _generate_offspring(self, mate_sol, mate_weights):
        return sum([ga.generate_offspring(mate_sol, mate_weights) for ga in self.ga_list], [])
    def _merge_offspring(self):
        for ga in self.ga_list:
            ga.merge_offspring()
    def _get_list_for_simulation(self):
        return sum([ga.get_list_for_simulation() for ga in self.ga_list], [])
    def _apply_selection(self):
        for ga in self.ga_list:
            ga.apply_selection()
    def _finalize(self):
        for ga in self.ga_list:
            ga.finalize()
    def _abbreviate(self):
        for ga in self.ga_list:
            ga.abbreviate()

@dataclass
class GAMultiSimilar(GAMulti):
    # Configuration
    ga_base: GA = field(init=True, default=None)
    N: int = field(init=True, default=4)

    def _initialize(self):
        self.ga_list = []
        for i in range(self.N):
            ga_copy = copy.deepcopy(self.ga_base)
            ga_copy.seed = self.seed+i
            ga_copy.fitness_cost = self.fitness_cost
            ga_copy.initialize()
            self.ga_list.append(ga_copy)     
        super()._initialize()    

class GAMultiRing(GAMulti):
    # Configuration
    mate_distance: int = field(init=True, default=2)     
    def _generate_offspring(self, mate_sol, mate_weights):
        assert mate_sol is None
        # To each child GA, pass mate_sol as the merged champions of all GAs within mate_distance on the ring
        # Create mate_sol for each GA by merging populations from GAs within mate_distance
        offspring_list = []
        for i, ga in enumerate(self.ga_list):
            # Collect populations from GAs within mate_distance on the ring
            populations_to_merge = []
            for offset in range(1, self.mate_distance + 1):
                # Get indices on both sides of the ring
                left_idx = (i - offset) % len(self.ga_list)
                right_idx = (i + offset) % len(self.ga_list)
                populations_to_merge.append(self.ga_list[left_idx].population)
                populations_to_merge.append(self.ga_list[right_idx].population)
            
            # Merge all collected populations into a single solution collection
            merged_sol = populations_to_merge[0].configuration
            for pop in populations_to_merge[1:]:
                merged_sol.merge(pop.configuration)
            mate_weights = np.ones(merged_sol.N_solutions) / merged_sol.N_solutions
            
            # Generate offspring for this GA using the merged mate population
            offspring_list.extend(ga.generate_offspring(merged_sol, mate_weights))
        return sum([ga.generate_offspring(mate_sol, mate_weights) for ga in self.ga_list], [])

@dataclass
class GASinglePopulation(GA):
    # Configuration
    N_trees_to_do: int = field(init=True, default=None)
    population_size:int = field(init=True, default=4000) 
    initializer: Initializer = field(init=True, default_factory=InitializerRandomJiggled)
    move: Move = field(init=True, default=None)
    fixed_h: float = field(init=True, default=0.61)
    reduce_h_threshold: float = field(init=True, default=1e-5)
    reduce_h_amount: float = field(init=True, default=2e-3)
    reduce_h_per_individual: bool = field(init=True, default=True)

    # Results
    population: Population = field(init=True, default=None)    

    # Internal
    _generator: cp.random.Generator = field(init=False, default=None)

    def __post_init__(self):        
        self.initializer.jiggler.n_rounds=0        

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

    def _check_constraints(self):
        self.initializer.check_constraints()
        self.move.check_constraints()
        if self.population is not None:
            self.population.check_constraints()
        super()._check_constraints()

    def _initialize(self):
        self._generator = cp.random.default_rng(seed=self.seed)
        self.initializer.seed = 200*self.seed + self.N_trees_to_do # backwards compatibility        
        if self.fixed_h is not None:
            self.initializer.fixed_h = cp.array([self.fixed_h*np.sqrt(self.N_trees_to_do),0,0],dtype=kgs.dtype_cp)
            self.initializer.base_solution.use_fixed_h = True
        self.population = self.initializer.initialize_population(self.population_size, self.N_trees_to_do)
        #if self.fixed_h is not None:
            #self.population.configuration.use_fixed_h = True
            #self.population.configuration.h = cp.tile(cp.array([self.fixed_h,0,0],dtype=kgs.dtype_cp), (self.population.configuration.N_solutions, 1))  
            #self.population.configuration.snap()
            #self.fitness_cost.costs.pop(0) # remove area cost if fixed h        
        self.population.check_constraints()
        self.best_costs_per_generation = [[]]

    def _score(self, register_best):
        # Compute cost and reshape to (N_solutions, 1) for tuple-based fitness
        cost_values = self.fitness_cost.compute_cost_allocate(self.population.configuration, evaluate_gradient=False)[0].get()

        
        if self.population.configuration.use_fixed_h:
            if not self.reduce_h_per_individual:
                if np.min(cost_values) < self.reduce_h_threshold:
                    # Reduce h if below threshold
                    self.population.configuration.h[:, 0] -= self.reduce_h_amount
                    cost_values = self.fitness_cost.compute_cost_allocate(self.population.configuration, evaluate_gradient=False)[0].get()            
            else:
                # Reduce h per individual if below threshold
                for i in range(self.population.configuration.N_solutions):
                    if cost_values[i] < self.reduce_h_threshold:
                        self.population.configuration.h[i, 0] -= self.reduce_h_amount
                cost_values = self.fitness_cost.compute_cost_allocate(self.population.configuration, evaluate_gradient=False)[0].get()
            self.population.fitness = np.stack( (self.population.configuration.h[:,0].get(), cost_values)).T # Shape: (N_solutions, 2)
        else:
            self.population.fitness = cost_values.reshape((-1, 1))  # Shape: (N_solutions, 1)
        
        if register_best:
            best_idx = kgs.lexicographic_argmin(self.population.fitness)
            best_cost = self.population.fitness[best_idx]  # Shape: (N_components,)
            self.best_costs_per_generation[0].append(best_cost)
            if self.champions is None or kgs.lexicographic_less_than(best_cost, self.champions[0].fitness[0]):
                self.champions = [copy.deepcopy(self.population)]
                self.champions[0].select_ids([best_idx])

        self.check_constraints()

    def _get_list_for_simulation(self):
        return [self.population.configuration]
    
    def _finalize(self):
        self._generator = None

    def _abbreviate(self):
        self.population = None
    
    
    
@dataclass
class GASinglePopulationOld(GASinglePopulation):

    population_size:int = field(init=True, default=4000)
    selection_size:list = field(init=True, default_factory=lambda: [int(4.*(x-1))+1 for x in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500]])
    prob_mate_own: float = field(init=True, default=0.5)


    def _apply_selection(self):
        current_pop = self.population
        current_pop.select_ids(kgs.lexicographic_argsort(current_pop.fitness))  # Sort by fitness lexicographically
        current_xyt = current_pop.configuration.xyt  # (N_individuals, N_trees, 3)

        max_sel = np.max(self.selection_size)
        selected = np.zeros(self.population_size, dtype=bool)
        diversity = np.inf*np.ones(max_sel)
        for sel_size in self.selection_size:
            selected_id = np.argmax(diversity[:sel_size])
            selected[selected_id] = True
            diversity = np.minimum(kgs.compute_genetic_diversity(cp.array(current_xyt[:max_sel]), cp.array(current_xyt[selected_id])).get(), diversity)
            #print(sel_size, diversity)
            assert(np.all(diversity[selected[:max_sel]]<1e-4))
        current_pop.select_ids(np.where(selected)[0])
        self.population = current_pop
        self.population.check_constraints()

    def _generate_offspring(self, mate_sol, mate_weight):
        
        old_pop = self.population
        old_pop.parent_fitness = old_pop.fitness.copy()
        parent_size = old_pop.configuration.N_solutions
        new_pop = old_pop.create_empty(self.population_size-parent_size, self.N_trees_to_do)

        # Generate all parent and mate selections at once (vectorized)
        N_offspring = new_pop.configuration.N_solutions

        # Pick random parents
        parent_ids = self._generator.integers(0, parent_size, size=N_offspring)

        # Pick random mates (excluding parent) - fully vectorized
        # Note: mate selection currently has weight=1 for all (0*np.arange(...)+1)
        # This means uniform selection excluding the parent

        # ADD: pick a mate as below with probability self.prob_mate_own, else pick from mate_sol with the specified mate_weight. Vectorized!
        # Except if mate_sol is None

        # Strategy: pick random from [0, parent_size-1), then adjust if >= parent_id
        mate_ids = self._generator.integers(0, parent_size - 1, size=N_offspring)
        # If mate_id >= parent_id, increment by 1 to skip the parent
        mate_ids = np.where(mate_ids >= parent_ids, mate_ids + 1, mate_ids)

        # Clone parents into new_pop and set parent fitness (vectorized)
        inds_to_do = np.arange(N_offspring)
        new_pop.create_clone_batch(inds_to_do, old_pop, parent_ids)

        # Apply moves using vectorized interface (clones already in place)
        # Convert indices to GPU arrays
        inds_to_do_gpu = cp.array(inds_to_do)
        mate_ids_gpu = cp.array(mate_ids)
        self.move.do_move_vec(new_pop, inds_to_do_gpu, old_pop.configuration, mate_ids_gpu, self._generator)

        return [new_pop]
    
    def _merge_offspring(self):
        old_pop = self.population
        new_pop = self._cached_offspring[0]

        old_pop.merge(new_pop)
        self.population = old_pop
        
        self.population.check_constraints()
        

@dataclass
class Orchestrator(kgs.BaseClass):
    # Configuration
    ga: GA = field(init=True, default=None)
    fitness_cost: pack_cost.Cost = field(init=True, default=None)    
    rough_relaxers: list = field(init=True, default=None) # meant to prevent heavy overlaps
    fine_relaxers: list = field(init=True, default=None)  # meant to refine solutions
    n_generations: int = field(init=True, default=200)
    seed: int = field(init=True, default=42)

    # Intermediate
    _current_generation: int = field(init=False, default=0)

    
    def __post_init__(self):        
        self.fitness_cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                            pack_cost.BoundaryDistanceCost(scaling=1.), 
                            pack_cost.CollisionCostSeparation(scaling=1.)])
        
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
        relaxer.max_step = 1e-3 * np.sqrt(10)
        self.fine_relaxers.append(relaxer)
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = copy.deepcopy(self.fitness_cost)
        relaxer.cost.costs[2] = pack_cost.CollisionCostSeparation(scaling=1.)
        relaxer.n_iterations = 30
        relaxer.max_step = 1e-3
        self.fine_relaxers.append(relaxer)

        super().__post_init__()

    def _relax(self, sol_list):
        for relaxer in self.rough_relaxers:
            pack_dynamics.run_simulation_list(relaxer, sol_list)
        for relaxer in self.fine_relaxers:
            pack_dynamics.run_simulation_list(relaxer, sol_list)


    def _check_constraints(self):        
        self.fitness_cost.check_constraints()
        self.ga.check_constraints()
        return super()._check_constraints()
    
    def run(self):
        self.check_constraints(debugging_mode_offset=2)
        self.ga.fitness_cost = self.fitness_cost
        self.ga.seed = self.seed

        # Initialize
        self.ga.initialize()
        self._relax(self.ga.get_list_for_simulation())        

        for i_gen in range(self.n_generations):
            self._current_generation = i_gen
            if i_gen>0:
                offspring_list = self.ga.generate_offspring(None, None)
                self._relax([s.configuration for s in offspring_list])
                self.ga.merge_offspring()
            
            self.ga.score(register_best=True)            
            for s in self.ga.best_costs_per_generation:
                assert len(s) == self._current_generation + 1
            self.ga.apply_selection()
            # Format best costs as lists for display (max 6 decimals)
            best_costs_str = [[round(float(x), 6) for x in s[-1].flatten()] for s in self.ga.best_costs_per_generation]
            print(f'Generation {i_gen}: Best costs = {best_costs_str}')

            if kgs.debugging_mode>=2:
                self.check_constraints()
        
        self.ga.finalize()