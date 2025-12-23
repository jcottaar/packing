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
import pack_move


# ============================================================
# Definition of population
# ============================================================

@dataclass
class Population(kgs.BaseClass):
    genotype: kgs.SolutionCollection = field(init=True, default=None)
    phenotype: kgs.SolutionCollection = field(init=True, default=None)
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
        self.genotype.check_constraints()
        self.phenotype.check_constraints()
        assert self.fitness.shape[0] == self.genotype.N_solutions
        assert self.fitness.shape[0] == self.phenotype.N_solutions
        assert self.fitness.ndim == 2  # Shape: (N_solutions, N_components)
        assert self.parent_fitness.shape[0] == self.genotype.N_solutions
        assert self.parent_fitness.ndim == 2  # Shape: (N_solutions, N_components)
        # assert len(self.lineages) == self.configuration.N_solutions

    def set_dummy_fitness(self, n_components=1):
        """Initialize fitness arrays with zeros.
        
        Parameters
        ----------
        n_components : int, optional
            Number of fitness components (default: 1)
        """
        self.fitness = np.zeros((self.phenotype.N_solutions, n_components), dtype=kgs.dtype_np)
        self.parent_fitness = np.zeros((self.phenotype.N_solutions, n_components), dtype=kgs.dtype_np)

    def select_ids(self, inds):
        self.genotype.select_ids(inds)
        self.phenotype.select_ids(inds)
        self.fitness = self.fitness[inds]
        self.parent_fitness = self.parent_fitness[inds]
        # self.lineages = [self.lineages[i] for i in inds]

    def create_empty(self, N_individuals, N_trees):
        genotype = self.genotype.create_empty(N_individuals, N_trees)
        phenotype = self.phenotype.create_empty(N_individuals, N_trees)
        population = type(self)(phenotype=phenotype, genotype=genotype)
        # Initialize with same number of components as self
        n_components = self.fitness.shape[1]
        population.fitness = np.zeros((N_individuals, n_components), dtype=kgs.dtype_np)
        population.parent_fitness = np.zeros((N_individuals, n_components), dtype=kgs.dtype_np)
        # population.lineages = [ None for _ in range(N_individuals) ]
        return population

    def create_clone(self, idx: int, other: 'Population', parent_id: int):
        assert idx<self.genotype.N_solutions
        self.genotype.create_clone(idx, other.genotype, parent_id)
        self.fitness[idx] = other.fitness[parent_id]
        self.parent_fitness[idx] = other.fitness[parent_id]
        # self.lineages[idx] = copy.deepcopy(other.lineages[parent_id])

    def create_clone_batch(self, inds: cp.ndarray, other: 'Population', parent_ids: cp.ndarray):
        """Vectorized batch clone operation."""
        self.genotype.create_clone_batch(inds, other.genotype, parent_ids)
        # Convert indices to CPU for NumPy array indexing
        inds_cpu = inds.get() if isinstance(inds, cp.ndarray) else inds
        parent_ids_cpu = parent_ids.get() if isinstance(parent_ids, cp.ndarray) else parent_ids
        self.fitness[inds_cpu] = other.fitness[parent_ids_cpu]
        self.parent_fitness[inds_cpu] = other.fitness[parent_ids_cpu]

    def merge(self, other:'Population'):
        self.genotype.merge(other.genotype)
        self.phenotype.merge(other.phenotype)
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
        population.phenotype = copy.deepcopy(population.genotype)
        population.set_dummy_fitness()
        assert population.genotype.N_solutions == N_individuals
        assert population.genotype.N_trees == N_trees        
        population.check_constraints()
        return population        

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
        population = Population(genotype=sol)
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

    allow_reset: bool = field(init=True, default=True)
    allow_freeze: bool = field(init=True, default=True)
    reset_check_generations: int = field(init=True, default=None)
    reset_check_generations_ratio: float = field(init=True, default=0.1)
    reset_check_threshold: float = field(init=True, default=0.1)
    freeze_duration: int = field(init=True, default=100)
    

    best_costs_per_generation_ax = None
    champion_genotype_ax = None
    champion_phenotype_ax = None

    _cached_offspring: list = field(init=False, default=None)
    _last_reset_generation: int = field(init=False, default=0)
    _skip_next_selection: bool = field(init=False, default=False)
    _is_frozen2: bool = field(init=False, default=False)

    def _check_constraints(self):
        super()._check_constraints()
        if self.fitness_cost is not None:
            self.fitness_cost.check_constraints()
        if self.champions is not None:
            for champion in self.champions:
                assert(champion.phenotype.N_solutions==1)
                champion.check_constraints()
            assert(len(self.champions) == len(self.best_costs_per_generation))     
            for champion, cost in zip(self.champions, self.best_costs_per_generation):
                assert(np.all(cost[-1] == champion.fitness[0]))

    def initialize(self):
        self.check_constraints(debugging_mode_offset=2)        
        self.fitness_cost.check_constraints()
        self._initialize()

    def reset(self):
        self.champions = None
        self._reset()
        self._last_reset_generation = len(self.best_costs_per_generation[0])-1

    def score(self, register_best=False):
        if self._is_frozen2:
            if register_best:
                for c in self.best_costs_per_generation:
                    c.append(c[-1])
                effective_frozen_generations = self.freeze_duration + \
                    int(self.reset_check_generations_ratio * (len(self.best_costs_per_generation[0])))
                if len(self.best_costs_per_generation[0]) - self._last_reset_generation >= effective_frozen_generations:
                    self._is_frozen2 = False
                    self._last_reset_generation = len(self.best_costs_per_generation[0])-1
            return
        self._score(register_best)
        if register_best:
            assert(len(self.champions) == len(self.best_costs_per_generation))            
            if not self.reset_check_generations is None:
                assert len(self.best_costs_per_generation)==1
                effective_reset_check_generations = self.reset_check_generations + \
                    int(self.reset_check_generations_ratio * (len(self.best_costs_per_generation[0])))
                costs = self.best_costs_per_generation[0]
                if len(costs)>self._last_reset_generation+effective_reset_check_generations+2 and \
                        costs[-1][0]==costs[-effective_reset_check_generations][0] and \
                        costs[-1][1]>=self.reset_check_threshold*costs[-effective_reset_check_generations][1]:                    
                    if self.allow_reset:
                        self.reset()                        
                        self.best_costs_per_generation[0].pop(-1)
                        self._score(register_best)
                        self._skip_next_selection = True
                    elif self.allow_freeze:
                        self._is_frozen2 = True                        


    def generate_offspring(self, mate_sol, mate_weights):
        if self._is_frozen2:
            return []
        if mate_sol is not None:
            assert(mate_sol.N_solutions == len(mate_weights))
        res = self._generate_offspring(mate_sol, mate_weights)
        self._cached_offspring = res
        return res
    
    def merge_offspring(self):
        if self._is_frozen2:
            return
        self._merge_offspring()

    def get_list_for_simulation(self):
        if self._is_frozen2:
            return []
        return self._get_list_for_simulation()
           
    def apply_selection(self):
        if self._is_frozen2:
            return
        if self._skip_next_selection:
            self._skip_next_selection = False
            return
        self._apply_selection()

    def finalize(self):        
        self._finalize()
        if self.do_legalize:
            for champion in self.champions:
                champion.phenotype = pack_io.legalize(champion.phenotype)
    
    def abbreviate(self):
        self._abbreviate()

    def diagnostic_plots(self, plot_ax):
        if not self.best_costs_per_generation_ax is None:
            for xx in self.best_costs_per_generation_ax:
                ax = plot_ax[xx[2]]
                ax.clear()
                plt.sca(ax)
                to_plot = np.array([ [y[xx[0]] for y in x] for x in self.best_costs_per_generation])
                if xx[1]:
                    to_plot=np.log(to_plot)/np.log(10)
                plt.plot(to_plot.T)
                plt.grid(True)
                plt.xlabel('Generation')
                plt.ylabel('Best Cost')
        if not self.champion_genotype_ax is None:
            ax = plot_ax[ self.champion_genotype_ax ]
            plt.sca(ax)
            if len(self.best_costs_per_generation[0])<2 or kgs.lexicographic_less_than(self.best_costs_per_generation[0][-1], self.best_costs_per_generation[0][-2]):
                ax.clear()
                pack_vis_sol.pack_vis_sol(self.champions[0].genotype, ax=ax)
                plt.title('Champion Genotype')
        if not self.champion_phenotype_ax is None:
            ax = plot_ax[ self.champion_phenotype_ax ]
            plt.sca(ax)
            if len(self.best_costs_per_generation[0])<2 or kgs.lexicographic_less_than(self.best_costs_per_generation[0][-1], self.best_costs_per_generation[0][-2]):
                ax.clear()
                pack_vis_sol.pack_vis_sol(self.champions[0].phenotype, ax=ax)
                plt.title('Champion Phenotype')
        self._diagnostic_plots(plot_ax)
    

    def _diagnostic_plots(self, plot_ax):
        pass
    

@dataclass
class GAMulti(GA):
    # Configuration    
    ga_list: list = field(init=True, default=None)
    single_champion: bool = field(init=True, default=True)
    plot_diversity_ax = None
    plot_subpopulation_costs_per_generation_ax = None
    allow_reset_ratio: float = field(init=True, default=1.) # allow only the worst X% of subpopulations to reset
    diversity_reset_threshold: float = field(init=True, default=np.inf) # diversity required to avoid reset
    diversity_reset_check_frequency: int = field(init=True, default=5) # in generations

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
    def _reset(self):
        for ga in self.ga_list:
            ga.reset()
    def _score(self, register_best):
        for ga in self.ga_list:
            ga.score(register_best=register_best)
        if register_best:
            assert self.single_champion
            if kgs.debugging_mode>=2:
                for ga in self.ga_list:
                    assert(len(ga.champions) == 1)
            costs_per_ga = np.array([ga.best_costs_per_generation[0][-1] for ga in self.ga_list])
            best_ga_idx = kgs.lexicographic_argmin(costs_per_ga)
            best_ga = self.ga_list[best_ga_idx]
            self.champions = [best_ga.champions[0]]
            self.best_costs_per_generation[0].append(best_ga.champions[0].fitness[0])
            # Apply allow_reset_ratio - only allow worst X% of GAs to reset
            if self.allow_reset_ratio < 1.0:
                sorted_indices = kgs.lexicographic_argsort(costs_per_ga)
                n_allowed_to_reset = int(np.ceil(len(self.ga_list) * self.allow_reset_ratio))
                # Worst performers are at the end of the sorted list
                for i, idx in enumerate(sorted_indices):
                    if i < len(self.ga_list) - n_allowed_to_reset:
                        # Better performers - disable reset
                        self.ga_list[idx].allow_reset = False
                    else:
                        # Worst performers - enable reset
                        self.ga_list[idx].allow_reset = True
            # Check for diversity-based resets
            if self.diversity_reset_threshold < np.inf and len(self.best_costs_per_generation[0]) % self.diversity_reset_check_frequency == 0:
                n_ga = len(self.ga_list)
                if n_ga > 1:
                    champion_xyts = cp.concatenate([ga.champions[0].genotype.xyt for ga in self.ga_list], axis=0)
                    champion_xyts_cp = cp.array(champion_xyts)
                    diversity_matrix = kgs.compute_genetic_diversity_matrix(champion_xyts_cp, champion_xyts_cp).get()
                    to_reset = set()
                    for i in range(n_ga):
                        if i in to_reset:
                            continue
                        for j in range(i + 1, n_ga):
                            if j in to_reset:
                                continue
                            if diversity_matrix[i, j] >= self.diversity_reset_threshold*self.ga_list[0].N_trees_to_do:
                                continue
                            cost_i = costs_per_ga[i]
                            cost_j = costs_per_ga[j]
                            if kgs.lexicographic_less_than(cost_i, cost_j):
                                worse_idx = j
                            elif kgs.lexicographic_less_than(cost_j, cost_i):
                                worse_idx = i
                            else:
                                worse_idx = max(i, j)
                            to_reset.add(worse_idx)
                    for idx in sorted(to_reset):
                        ga_to_reset = self.ga_list[idx]
                        if ga_to_reset.best_costs_per_generation and ga_to_reset.best_costs_per_generation[0]:
                            ga_to_reset.best_costs_per_generation[0].pop(-1)
                        ga_to_reset.reset()
                        ga_to_reset.score(register_best=True)
                        ga_to_reset._skip_next_selection = True
                        costs_per_ga[idx] = ga_to_reset.champions[0].fitness[0]

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
    def _diagnostic_plots(self, plot_ax):
        if not self.plot_subpopulation_costs_per_generation_ax is None:
            for xx in self.plot_subpopulation_costs_per_generation_ax:
                ax = plot_ax[xx[2]]
                ax.clear()
                plt.sca(ax)
                to_plot = np.array([ [y[xx[0]] for y in x.best_costs_per_generation[0]] for x in self.ga_list])
                if xx[1]:
                    to_plot=np.log(to_plot)/np.log(10)
                im = plt.imshow(to_plot.T, aspect='auto', cmap='viridis', interpolation='none')
                if not hasattr(ax, '_colorbar') or ax._colorbar is None:
                    ax._colorbar = plt.colorbar(im, ax=ax, label='Best cost')
                else:
                    ax._colorbar.update_normal(im)
                # Mark frozen subpopulations with red X at the top
                for i, ga in enumerate(self.ga_list):
                    if ga._is_frozen2:
                        plt.plot(i, -0.5, 'rx', markersize=10, markeredgewidth=2)
                plt.xlabel('Subpopulation')
                plt.ylabel('Generation')
        if self.plot_diversity_ax is not None:
            ax = plot_ax[self.plot_diversity_ax]
            ax.clear()
            plt.sca(ax)
            champions_pop = copy.deepcopy(self.ga_list[0].champions[0].genotype)
            for ga in self.ga_list[1:]:
                champions_pop.merge(ga.champions[0].genotype)
            # Compute diversity matrix
            N_sols = champions_pop.N_solutions
            diversity_matrix = kgs.compute_genetic_diversity_matrix(cp.array(champions_pop.xyt), cp.array(champions_pop.xyt)).get()
            im = plt.imshow(diversity_matrix, cmap='viridis', vmin=0., vmax=np.max(diversity_matrix), interpolation='none')
            if not hasattr(ax, '_colorbar') or ax._colorbar is None:
                ax._colorbar = plt.colorbar(im, ax=ax, label='Diversity distance')
            else:
                ax._colorbar.update_normal(im)
            plt.title('Diversity Matrix Across GA Subpopulations')
            plt.xlabel('Individual')
            plt.ylabel('Individual')

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

@dataclass
class GAMultiRing(GAMultiSimilar):
    # Configuration
    mate_distance: int = field(init=True, default=4)     
    def _generate_offspring(self, mate_sol, mate_weights):
        assert mate_sol is None
        #if self.champions is None:
        #    return super()._generate_offspring(mate_sol, mate_weights)
        # Sort ga_list by cost value using lexicographic sort
        # Get the best cost for each GA
        costs_per_ga = np.array([ga.champions[0].fitness[0] for ga in self.ga_list])
        sorted_indices = kgs.lexicographic_argsort(costs_per_ga)
        sorted_ga_list = [self.ga_list[i] for i in sorted_indices]
        
        # To each child GA, pass mate_sol as the merged champions of all GAs within mate_distance
        # Each GA gets the mate_distance nearest neighbors in the sorted list
        offspring_list = []
        for i, ga in enumerate(sorted_ga_list):
            # Collect the mate_distance nearest populations
            populations_to_merge = []
            
            # Collect neighbors, expanding outward from current position
            offset = 1
            while len(populations_to_merge) < self.mate_distance and offset < len(sorted_ga_list):
                # Add left neighbor if available
                if i - offset >= 0 and len(populations_to_merge) < self.mate_distance:
                    populations_to_merge.append(sorted_ga_list[i - offset].population)
                # Add right neighbor if available
                if i + offset < len(sorted_ga_list) and len(populations_to_merge) < self.mate_distance:
                    populations_to_merge.append(sorted_ga_list[i + offset].population)
                offset += 1
            
            # Merge all collected populations into a single solution collection
            if len(populations_to_merge)>0:
                merged_sol = copy.deepcopy(populations_to_merge[0].genotype)
                for pop in populations_to_merge[1:]:
                    merged_sol.merge(pop.genotype)
                mate_weights = np.ones(merged_sol.N_solutions) / merged_sol.N_solutions
            else:
                merged_sol, mate_weights = None, None
            
            # Generate offspring for this GA using the merged mate population
            offspring_list.extend(ga.generate_offspring(merged_sol, mate_weights))
        # offspring_list is out of order - this is OK (child GAs cache their own offspring)
        return offspring_list

@dataclass
class GASinglePopulation(GA):
    # Configuration
    N_trees_to_do: int = field(init=True, default=None)
    population_size:int = field(init=True, default=4000) 
    initializer: Initializer = field(init=True, default_factory=InitializerRandomJiggled)
    move: pack_move.Move = field(init=True, default=None)
    fixed_h: float = field(init=True, default=0.61)
    reduce_h_threshold: float = field(init=True, default=1e-5)
    reduce_h_amount: float = field(init=True, default=2e-3)
    reduce_h_per_individual: bool = field(init=True, default=True)

    plot_diversity_ax = None

    # Results
    population: Population = field(init=True, default=None)    

    # Internal
    _generator: cp.random.Generator = field(init=False, default=None)

    def __post_init__(self):        
        self.initializer.jiggler.n_rounds=0        

        self.move = pack_move.MoveSelector()
        self.move.moves = []
        self.move.moves.append( [pack_move.MoveRandomTree(), 'MoveRandomTree', 1.0] )
        self.move.moves.append( [pack_move.JiggleRandomTree(max_xy_move=0.05, max_theta_move=np.pi/6), 'JiggleTreeSmall', 1.0] ) 
        self.move.moves.append( [pack_move.JiggleRandomTree(max_xy_move=0.1, max_theta_move=np.pi), 'JiggleTreeBig', 1.0] ) 
        self.move.moves.append( [pack_move.JiggleCluster(max_xy_move=0.05, max_theta_move=np.pi/6), 'JiggleClusterSmall', 1.0] )
        self.move.moves.append( [pack_move.JiggleCluster(max_xy_move=0.1, max_theta_move=np.pi), 'JiggleClusterBig', 1.0] )
        self.move.moves.append( [pack_move.Translate(), 'Translate', 1.0] )
        self.move.moves.append( [pack_move.Twist(), 'Twist', 1.0] )
        self.move.moves.append( [pack_move.Crossover(), 'Crossover', 2.0] )
        self.move.moves.append( [pack_move.CrossoverStripe(), 'CrossoverStripe', 2.0] )

        super().__post_init__()

    def _check_constraints(self):
        self.initializer.check_constraints()
        self.move.check_constraints()
        if self.population is not None:
            self.population.check_constraints()
        super()._check_constraints()

    def _initialize(self):
        self._generator = cp.random.default_rng(seed=self.seed)
        self.best_costs_per_generation = [[]]

    def _reset(self):
        self.initializer.seed = self._generator.integers(0, 2**30).get().item()
        if self.fixed_h is not None:
            self.initializer.fixed_h = cp.array([self.fixed_h*np.sqrt(self.N_trees_to_do),0,0],dtype=kgs.dtype_cp)
            self.initializer.base_solution.use_fixed_h = True
        self.population = self.initializer.initialize_population(len(self.selection_size), self.N_trees_to_do)    
        self.population.check_constraints()        

    def _score(self, register_best):
        # Compute cost and reshape to (N_solutions, 1) for tuple-based fitness
        cost_values = self.fitness_cost.compute_cost_allocate(self.population.phenotype, evaluate_gradient=False)[0].get()

        
        if self.population.phenotype.use_fixed_h:
            if not self.reduce_h_per_individual:
                if np.min(cost_values) < self.reduce_h_threshold:
                    # Reduce h if below threshold
                    self.population.genotype.h[:, 0] -= self.reduce_h_amount
                    self.population.phenotype.h[:, 0] -= self.reduce_h_amount
                    cost_values = self.fitness_cost.compute_cost_allocate(self.population.phenotype, evaluate_gradient=False)[0].get()            
            else:
                # Reduce h per individual if below threshold
                for i in range(self.population.phenotype.N_solutions):
                    if cost_values[i] < self.reduce_h_threshold:
                        self.population.genotype.h[i, 0] -= self.reduce_h_amount
                        self.population.phenotype.h[i, 0] -= self.reduce_h_amount
                cost_values = self.fitness_cost.compute_cost_allocate(self.population.phenotype, evaluate_gradient=False)[0].get()
            self.population.fitness = np.stack( (self.population.phenotype.h[:,0].get(), cost_values)).T # Shape: (N_solutions, 2)
        else:
            self.population.fitness = cost_values.reshape((-1, 1))  # Shape: (N_solutions, 1)
        
        if register_best:
            best_idx = kgs.lexicographic_argmin(self.population.fitness)
            best_cost = self.population.fitness[best_idx]  # Shape: (N_components,)
            update_champion = self.champions is None or kgs.lexicographic_less_than(best_cost, self.champions[0].fitness[0])
            if update_champion:
                self.champions = [copy.deepcopy(self.population)]
                self.champions[0].select_ids([best_idx])                
            self.best_costs_per_generation[0].append(self.champions[0].fitness[0])
            assert(np.all(self.champions[0].fitness[0]==self.best_costs_per_generation[0][-1]))

        self.check_constraints()

    def _get_list_for_simulation(self):
        return [self.population]
    
    def _finalize(self):
        self._generator = None

    def _abbreviate(self):
        self.population = None

    def _diagnostic_plots(self, plot_ax):
        if self.plot_diversity_ax is not None:
            ax = plot_ax[self.plot_diversity_ax]
            ax.clear()
            plt.sca(ax)
            pop = self.population.genotype
            # Compute diversity matrix
            N_sols = pop.N_solutions
            diversity_matrix = kgs.compute_genetic_diversity_matrix(cp.array(pop.xyt), cp.array(pop.xyt)).get()
            im = plt.imshow(diversity_matrix, cmap='viridis', vmin=0., vmax=np.max(diversity_matrix), interpolation='none')
            if not hasattr(ax, '_colorbar') or ax._colorbar is None:
                ax._colorbar = plt.colorbar(im, ax=ax, label='Diversity distance')
            else:
                ax._colorbar.update_normal(im)
            plt.title('Diversity Matrix Across single population')
            plt.xlabel('Individual')
            plt.ylabel('Individual')
    
    
    
@dataclass
class GASinglePopulationTournament(GASinglePopulation):
    """Tournament-based selection GA with explicit champion exploitation.
    
    Selection strategy:
    - Keep top `selection_fraction` (default 50%) of population as parents
    - Generate `champion_fraction` (default 10%) offspring from champion
    - Generate remaining offspring via tournament selection (N=tournament_size)
    
    Designed for use with GAMultiRing (32 islands, ring topology).
    """
    
    population_size: int = field(init=True, default=500)
    champion_fraction: float = field(init=True, default=0.1)  # 10% from champion
    tournament_size: int = field(init=True, default=4)  # Tournament N=4
    selection_fraction: float = field(init=True, default=0.5)  # Keep top 50%
    prob_mate_own: float = field(init=True, default=0.5)  # For ring migration
    
    # Internal state
    _cached_champion_pop: Population = field(init=False, default=None)
    
    # Computed from selection_fraction * population_size (used by _reset)
    @property
    def selection_size(self):
        return list(range(int(self.population_size * self.selection_fraction)))

    def _apply_selection(self):
        """Keep top selection_fraction of population as parents."""
        current_pop = self.population
        sorted_ids = kgs.lexicographic_argsort(current_pop.fitness)
        n_keep = int(self.population_size * self.selection_fraction)
        current_pop.select_ids(sorted_ids[:n_keep])
        self.population = current_pop
        self.population.check_constraints()

    def _tournament_select(self, parent_size: int, n_offspring: int) -> cp.ndarray:
        """Perform tournament selection to choose parents.
        
        For each offspring, draw tournament_size candidates and select the best.
        Returns array of parent indices (shape: n_offspring).
        """
        # Draw tournament candidates: (n_offspring, tournament_size)
        candidates = self._generator.integers(0, parent_size, (n_offspring, self.tournament_size))
        
        # Get fitness for all candidates - need to find best in each tournament
        # fitness shape: (parent_size, n_components)
        fitness = self.population.fitness
        
        # For each tournament, find the winner (lexicographically smallest fitness)
        winners = cp.zeros(n_offspring, dtype=cp.int64)
        candidates_cpu = candidates.get()
        for i in range(n_offspring):
            tournament_fitness = fitness[candidates_cpu[i]]  # (tournament_size, n_components)
            best_in_tournament = kgs.lexicographic_argmin(tournament_fitness)
            winners[i] = candidates_cpu[i, best_in_tournament]
        
        return winners

    def _generate_offspring(self, mate_sol, mate_weights):
        old_pop = self.population
        old_pop.check_constraints()
        old_pop.parent_fitness = old_pop.fitness.copy()
        parent_size = old_pop.genotype.N_solutions  # This is the surviving 50%
        
        # Save the champion for elitism (will be inserted in _merge_offspring)
        best_idx = kgs.lexicographic_argmin(old_pop.fitness)
        self._cached_champion_pop = copy.deepcopy(old_pop)
        self._cached_champion_pop.select_ids([best_idx])
        
        # Create new population with full size
        new_pop = old_pop.create_empty(self.population_size, self.N_trees_to_do)
        
        # Layout:
        # - indices 0 to n_champion-1: champion offspring (mutated clones of champion)
        # - indices n_champion to end: tournament offspring
        # Note: position 0 will be overwritten with unmutated champion in _merge_offspring
        
        n_champion = int(self.population_size * self.champion_fraction)
        n_tournament = self.population_size - n_champion
        
        # === Determine parent for each offspring ===
        # Champion offspring: parent is always the champion
        # Tournament offspring: parent selected via tournament
        all_parent_ids = cp.zeros(self.population_size, dtype=cp.int64)
        
        if n_champion > 0:
            all_parent_ids[:n_champion] = best_idx
        
        if n_tournament > 0:
            all_parent_ids[n_champion:] = self._tournament_select(parent_size, n_tournament)
        
        # Clone all parents at once
        all_inds = cp.arange(self.population_size)
        new_pop.create_clone_batch(all_inds, old_pop, all_parent_ids)
        
        # === Determine mates for all offspring (same logic for champion and tournament) ===
        if mate_sol is None or mate_sol.N_solutions == 0:
            use_own = cp.ones(self.population_size, dtype=bool)
        else:
            use_own = self._generator.random(self.population_size) < self.prob_mate_own
        
        # Own-population mates (always mate with champion)
        inds_use_own = cp.where(use_own)[0]
        if len(inds_use_own) > 0:
            mate_ids_own = cp.full(len(inds_use_own), best_idx, dtype=cp.int64)
            self.move.do_move_vec(new_pop, inds_use_own, old_pop.genotype,
                                  mate_ids_own, self._generator)
        
        # External-population mates
        inds_use_external = cp.where(~use_own)[0]
        if len(inds_use_external) > 0:
            mate_prob = cp.asarray(mate_weights) / cp.sum(mate_weights)
            cum_prob = cp.cumsum(mate_prob)
            random_vals = self._generator.random(len(inds_use_external))
            mate_ids_external = cp.searchsorted(cum_prob, random_vals)
            self.move.do_move_vec(new_pop, inds_use_external, mate_sol.genotype,
                                  mate_ids_external, self._generator)
        
        return [new_pop]

    def _merge_offspring(self):
        """Replace population with offspring, preserving the champion (elitism)."""
        new_pop = self._cached_offspring[0]
        
        # Insert unmutated champion at position 0 (overwrite the first champion offspring)
        new_pop.create_clone_batch(cp.array([0]), self._cached_champion_pop, cp.array([0]))
        
        self.population = new_pop
        self.population.check_constraints()


@dataclass
class GASinglePopulationOld(GASinglePopulation):

    population_size:int = field(init=True, default=4000)
    selection_size:list = field(init=True, default_factory=lambda: [int(4.*(x-1))+1 for x in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500]])
    prob_mate_own: float = field(init=True, default=0.5)


    def _apply_selection(self):
        current_pop = self.population
        current_pop.select_ids(kgs.lexicographic_argsort(current_pop.fitness))  # Sort by fitness lexicographically
        current_xyt = current_pop.genotype.xyt  # (N_individuals, N_trees, 3)

        max_sel = np.max(self.selection_size)
        selected = np.zeros(self.population_size, dtype=bool)
        diversity = np.inf*np.ones(max_sel)
        
        # Pre-compute full diversity matrix once (max_sel × max_sel)
        diversity_matrix = kgs.compute_genetic_diversity_matrix(
            cp.array(current_xyt[:max_sel]), 
            cp.array(current_xyt[:max_sel])
        ).get()
        
        for sel_size in self.selection_size:
            selected_id = np.argmax(diversity[:sel_size])
            selected[selected_id] = True
            # Index into pre-computed matrix instead of recomputing
            diversity = np.minimum(diversity_matrix[:, selected_id], diversity)
            #print(sel_size, diversity)
            assert(np.all(diversity[selected[:max_sel]]<1e-4))
        current_pop.select_ids(np.where(selected)[0])
        self.population = current_pop
        self.population.check_constraints()

    def _generate_offspring(self, mate_sol, mate_weights):
        
        old_pop = self.population
        old_pop.check_constraints()
        old_pop.parent_fitness = old_pop.fitness.copy()
        parent_size = old_pop.genotype.N_solutions
        new_pop = old_pop.create_empty(self.population_size-parent_size, self.N_trees_to_do)

        # Generate all parent and mate selections at once (vectorized)
        N_offspring = new_pop.genotype.N_solutions

        # Pick random parents (all from old_pop)
        parent_ids = self._generator.integers(0, parent_size, size=N_offspring)

        # Decide which offspring use own population vs external mate population
        if mate_sol is None or mate_sol.N_solutions == 0:
            use_own = np.ones(N_offspring, dtype=bool)
        else:
            use_own = self._generator.random(N_offspring) < self.prob_mate_own

        # Split offspring into two groups
        inds_use_own = np.where(use_own)[0]
        inds_use_external = np.where(~use_own)[0]

        # Process offspring using own population as mate source
        if len(inds_use_own) > 0:
            parent_ids_own = parent_ids[inds_use_own]
            # Pick random mates (excluding parent) - fully vectorized
            mate_ids_own = self._generator.integers(0, parent_size - 1, size=len(inds_use_own))
            # If mate_id >= parent_id, increment by 1 to skip the parent
            mate_ids_own = np.where(mate_ids_own >= parent_ids_own, mate_ids_own + 1, mate_ids_own)
            
            # Clone parents into new_pop
            new_pop.create_clone_batch(inds_use_own, old_pop, parent_ids_own)
            
            # Apply moves with mates from own population
            self.move.do_move_vec(new_pop, cp.array(inds_use_own), old_pop.genotype, cp.array(mate_ids_own), self._generator)

        # Process offspring using external population as mate source
        if len(inds_use_external) > 0:
            parent_ids_external = parent_ids[inds_use_external]
            mate_size = mate_sol.N_solutions
            
            # Normalize mate_weight to get probability distribution
            mate_prob = cp.asarray(mate_weights) / cp.sum(mate_weights)
            
            # Sample mates using weighted selection (CuPy doesn't have choice, use cumsum + searchsorted)
            cum_prob = cp.cumsum(mate_prob)
            random_vals = self._generator.random(len(inds_use_external))
            mate_ids_external = cp.searchsorted(cum_prob, random_vals)
            
            # Clone parents into new_pop
            new_pop.create_clone_batch(inds_use_external, old_pop, parent_ids_external)
            
            # Apply moves with mates from external population
            self.move.do_move_vec(new_pop, cp.array(inds_use_external), mate_sol, mate_ids_external, self._generator)

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
    genotype_at: int = field(init=True, default=1)  # 0:before relax, 1:after rough relax, 2:after fine relax(=phenotype)
    seed: int = field(init=True, default=42)
    

    # Diagnostics
    diagnostic_plot: bool = field(init=True, default=False)

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
        # pack_vis_sol.pack_vis_sol(sol_list[0].genotype)  # Sanity check
        # plt.title('Genotype')
        # pack_vis_sol.pack_vis_sol(sol_list[0].phenotype)  # Sanity check
        # plt.title('Phenotype')
        for s in sol_list:
            s.phenotype.xyt[:] = s.genotype.xyt[:]
            s.phenotype.h[:] = s.genotype.h[:]
        conf_list = [s.phenotype for s in sol_list]
        if self.genotype_at == 0:
            for s in sol_list:
                s.genotype.xyt[:] = s.phenotype.xyt[:]
                s.genotype.h[:] = s.phenotype.h[:]
        for relaxer in self.rough_relaxers:
            pack_dynamics.run_simulation_list(relaxer, conf_list)
        if self.genotype_at == 1:
            for s in sol_list:
                s.genotype.xyt[:] = s.phenotype.xyt[:]
                s.genotype.h[:] = s.phenotype.h[:]
        for relaxer in self.fine_relaxers:
            pack_dynamics.run_simulation_list(relaxer, conf_list)
        if self.genotype_at == 2:
            for s in sol_list:
                s.genotype.xyt[:] = s.phenotype.xyt[:]
                s.genotype.h[:] = s.phenotype.h[:]


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
        self.ga.reset()
        #self._relax(self.ga.get_list_for_simulation())    
        self.ga.score(register_best=True)

        if self.diagnostic_plot:
            #plt.ion() 
            plot_fig, plot_ax = plt.subplots(2,3, figsize=(18,12))

        for i_gen in range(self.n_generations):
            self._current_generation = i_gen

            offspring_list = self.ga.generate_offspring(None, None)
            self._relax(offspring_list)
            self.ga.merge_offspring()
            
            self.ga.score(register_best=True)            
            for s in self.ga.best_costs_per_generation:
                assert len(s) == self._current_generation + 2
            self.ga.apply_selection()
            # Format best costs as lists for display (max 6 decimals)
            best_costs_str = [[round(float(x), 6) for x in s[-1].flatten()] for s in self.ga.best_costs_per_generation]
            if self.diagnostic_plot:
                self.ga.diagnostic_plots(plot_ax)                
                from IPython.display import clear_output, display
                clear_output(wait=True)  # Clear previous output                
                plt.suptitle(f'Generation {i_gen}: Best costs = {best_costs_str}')
                display(plot_fig)
            else:
                print(f'Generation {i_gen}: Best costs = {best_costs_str}')

            if kgs.debugging_mode>=2:
                self.check_constraints()
        
        self.ga.finalize()