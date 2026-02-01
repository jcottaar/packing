import pandas as pd
import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field
import pack_cost
import pack_vis_sol
import pack_dynamics
import pack_io
import copy
import matplotlib.pyplot as plt
import lap_batch
import pack_move
import os
from IPython.display import clear_output, display

# ============================================================
# Definition of population
# ============================================================

@dataclass
class Population(kgs.BaseClass):
    genotype: kgs.SolutionCollection = field(init=True, default=None)
    phenotype: kgs.SolutionCollection = field(init=True, default=None)
    fitness: np.ndarray = field(init=True, default=None)  # Shape: (N_solutions, N_components)
    parent_fitness: np.ndarray = field(init=True, default=None)  # Shape: (N_solutions, N_components)

    def _check_constraints(self):
        self.genotype.check_constraints()
        self.phenotype.check_constraints()
        assert self.phenotype.N_trees == self.genotype.N_trees
        assert self.fitness.shape[0] == self.genotype.N_solutions
        assert self.fitness.shape[0] == self.phenotype.N_solutions
        assert self.fitness.ndim == 2  # Shape: (N_solutions, N_components)
        assert self.parent_fitness.shape[0] == self.genotype.N_solutions
        assert self.parent_fitness.ndim == 2  # Shape: (N_solutions, N_components)

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

    def create_empty(self, N_individuals, N_trees):
        genotype = self.genotype.create_empty(N_individuals, N_trees)
        phenotype = self.phenotype.create_empty(N_individuals, N_trees)
        population = type(self)(phenotype=phenotype, genotype=genotype)
        n_components = self.fitness.shape[1]
        population.fitness = np.zeros((N_individuals, n_components), dtype=kgs.dtype_np)
        population.parent_fitness = np.zeros((N_individuals, n_components), dtype=kgs.dtype_np)
        return population

    def create_clone(self, idx: int, other: 'Population', parent_id: int):
        assert idx<self.genotype.N_solutions
        self.genotype.create_clone(idx, other.genotype, parent_id)
        self.phenotype.create_clone(idx, other.phenotype, parent_id)
        self.fitness[idx] = other.fitness[parent_id]
        self.parent_fitness[idx] = other.fitness[parent_id]

    def create_clone_batch(self, inds: cp.ndarray, other: 'Population', parent_ids: cp.ndarray):
        """Vectorized batch clone operation."""
        self.genotype.create_clone_batch(inds, other.genotype, parent_ids)
        self.phenotype.create_clone_batch(inds, other.phenotype, parent_ids)
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


# ============================================================
# Population initializers
# ============================================================

@dataclass
class Initializer(kgs.BaseClass):
    seed: int = field(init=True, default=42)
    def initialize_population(self, N_individuals, N_trees):
        population = self._initialize_population(N_individuals, N_trees)
        population.phenotype = copy.deepcopy(population.genotype.convert_to_phenotype())
        population.set_dummy_fitness()
        assert population.genotype.N_solutions == N_individuals
        assert population.genotype.N_trees == N_trees        
        assert population.phenotype.N_solutions == N_individuals
        assert population.phenotype.N_trees == N_trees        
        population.check_constraints()
        return population        

@dataclass
class InitializerRandom(Initializer): 
    base_solution: kgs.SolutionCollection = field(init=True, default_factory=kgs.SolutionCollectionSquare)
    fixed_h: cp.ndarray = field(init=True, default=None) # if not None, should be (3,) array    
    ref_sol_crystal_type: str = field(init=True, default=None)
    ref_sol_axis1_offset: object = field(init=True, default=None)
    ref_sol_axis2_offset: object = field(init=True, default=None)
    ref_sol: kgs.SolutionCollection = field(init=True, default=None)
    ref_rotate: float = field(init=True, default=0.) # in radians

    new_tree_placer: bool = field(init=True, default=False)

    def _initialize_population(self, N_individuals, N_trees):
        self.check_constraints()
        sol = self.base_solution.create_empty(N_individuals, N_trees)
        size_setup_scaled = float(cp.asnumpy(self.fixed_h[0]))
        generator = np.random.default_rng(seed=self.seed)
        if not self.new_tree_placer:
            xyt = generator.uniform(-0.5, 0.5, size=sol.xyt.shape)
            xyt = xyt * [[[size_setup_scaled, size_setup_scaled, 2*np.pi]]]
            xyt = cp.array(xyt, dtype=kgs.dtype_np)
            sol.xyt = xyt   
            sol.h = cp.tile(self.fixed_h[cp.newaxis, :], (N_individuals, 1))                 
            sol.canonicalize()
            assert self.ref_sol_crystal_type is None and self.ref_sol is None            
        else:
            assert self.fixed_h is not None
            sol.h = cp.tile(self.fixed_h[cp.newaxis, :], (N_individuals, 1))      
            if self.ref_sol_crystal_type is not None:
                axis1_offset = self.ref_sol_axis1_offset(generator)
                axis2_offset = self.ref_sol_axis2_offset(generator)
                self.ref_sol = kgs.create_tiled_solution(self.ref_sol_crystal_type, 25, make_symmetric=not isinstance(self.base_solution, kgs.SolutionCollectionSquare), 
                                                        axis1_offset=axis1_offset, axis2_offset=axis2_offset)
            if self.ref_sol is not None:
                ref_sol_use = copy.deepcopy(self.ref_sol)
                if self.ref_rotate is None:
                    ref_rotate = generator.uniform(0., 2*np.pi)
                else:
                    ref_rotate = self.ref_rotate
                ref_sol_use.rotate(cp.array([ref_rotate], dtype=kgs.dtype_cp))
                ref_sol_use.canonicalize()
                expected_score = 0.317 + 0.206/np.sqrt(N_trees)
                expected_h = cp.array([[np.sqrt(expected_score*N_trees),0.,0.]], dtype=kgs.dtype_cp)
                to_keep = ~sol.edge_spacer.check_valid(ref_sol_use.xyt, expected_h)[0]
                ref_sol_use.xyt = ref_sol_use.xyt[:,to_keep,:]
                N1 = ref_sol_use.xyt.shape[1]
                assert N1<=sol.xyt.shape[1]
                sol.xyt[:,:ref_sol_use.xyt.shape[1],:] = cp.tile(ref_sol_use.xyt[0:1,:,:], (N_individuals,1,1))
            else:
                raise 'no ref sol branch todo'
            # now add trees up to N_trees using vectorized rejection sampling
            N_remaining = sol.xyt.shape[1] - N1
            if N_remaining > 0:
                # Track which trees still need placement for each individual
                needs_placement = np.ones((N_individuals, N_remaining), dtype=bool)
                
                while np.any(needs_placement):
                    # Generate candidates for all trees that need placement
                    for i_individual in range(N_individuals):
                        n_to_place = np.sum(needs_placement[i_individual])
                        if n_to_place == 0:
                            continue
                        
                        # Generate random candidates
                        h_value = float(sol.h[i_individual, 0].get())
                        candidates_xy = generator.uniform(-0.5 * h_value, 0.5 * h_value, (n_to_place, 2))
                        candidates_theta = generator.uniform(0., 2 * np.pi, (n_to_place, 1))
                        candidates = np.concatenate([candidates_xy, candidates_theta], axis=1)
                        candidates_cp = cp.array(candidates[np.newaxis, :, :], dtype=kgs.dtype_cp)
                        
                        # Check validity
                        valid_mask = sol.edge_spacer.check_valid(candidates_cp, sol.h[i_individual:i_individual+1])[0].get()
                     
                        # Place valid candidates
                        if np.any(valid_mask):
                            tree_indices = np.where(needs_placement[i_individual])[0]
                            valid_tree_indices = tree_indices[valid_mask]
                            sol.xyt[i_individual, N1 + valid_tree_indices, :] = candidates_cp[0, valid_mask, :]
                            needs_placement[i_individual, valid_tree_indices] = False
        sol.canonicalize()
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
    best_ever: list = field(init=True, default=None)
    best_costs_per_generation: list = field(init=True, default_factory=list)  
    do_legalize: bool = field(init=True, default=False)

    allow_reset: bool = field(init=True, default=True)
    stop_check_generations: int = field(init=True, default=0)
    stop_check_generations_scale: int = field(init=True, default=50) # scale with N_trees
    reset_check_generations: int = field(init=True, default=None)
    reset_check_generations_ratio: float = field(init=True, default=0.1)
    reset_check_threshold: float = field(init=True, default=0.1)
    always_allow_mate_with_better: bool = field(init=True, default=False)
    
    make_own_fig = None # input to plt.subplots()
    make_own_fig_size = None

    best_costs_per_generation_ax = None
    champion_genotype_ax = None
    champion_phenotype_ax = None

    _cached_offspring: list = field(init=False, default=None)
    _last_reset_generation: int = field(init=False, default=0)
    _skip_next_selection: bool = field(init=False, default=False)
    _allow_mate_with_better: bool = field(init=True, default=True)    
    _stopped: bool = field(init=False, default=False)

    _fig = None
    _ax = None
    _always_plot_trees = True

    def _check_constraints(self):
        super()._check_constraints()
        if self.fitness_cost is not None:
            self.fitness_cost.check_constraints()
        if self.champions is not None:
            assert(len(self.champions)==1) # more than one champion is legacy functionality, not currently used
            for champion in self.champions:
                assert(champion.phenotype.N_solutions==1)
                champion.check_constraints()
            assert(len(self.champions) == len(self.best_costs_per_generation))  
            for champion, cost in zip(self.champions, self.best_costs_per_generation):
                assert(np.all(cost[-1] == champion.fitness[0]))
        if self.best_ever is not None:
            for champion in self.best_ever:
                assert(champion.phenotype.N_solutions==1)
                champion.check_constraints()
            assert(len(self.best_ever) == len(self.best_costs_per_generation))
            

    def initialize(self, generator):
        self.check_constraints(debugging_mode_offset=2)        
        self.fitness_cost.check_constraints()
        self._initialize(generator)

    def reset(self, generator):
        self.champions = None
        self._reset(generator)
        self._last_reset_generation = len(self.best_costs_per_generation[0])-1

    def score(self, generator, register_best=False):        
        self._score(generator, register_best)
        if register_best:
            assert(len(self.champions) == len(self.best_costs_per_generation))     
            if self.best_ever is None:
                self.best_ever = copy.deepcopy(self.champions)
            for i,c in enumerate(self.champions):
                b = self.best_ever[i]
                if kgs.lexicographic_less_than(c.fitness[0], b.fitness[0]):
                    self.best_ever[i] = copy.deepcopy(c)
            if self.stop_check_generations is not None and len(self.champions)>0:
                assert len(self.best_costs_per_generation)==1
                effective_stop_check_generations = self.stop_check_generations + self.champions[0].phenotype.N_trees * self.stop_check_generations_scale 
                costs = self.best_costs_per_generation[0]          
                if len(costs)>effective_stop_check_generations+2 and \
                        costs[-1][0]==costs[-effective_stop_check_generations][0] and \
                        costs[-1][1]>=self.reset_check_threshold*costs[-effective_stop_check_generations][1]:     
                    self._stopped = True        
            if self.reset_check_generations is not None:
                assert len(self.best_costs_per_generation)==1
                effective_reset_check_generations = self.reset_check_generations + \
                    int(self.reset_check_generations_ratio * (len(self.best_costs_per_generation[0])))
                costs = self.best_costs_per_generation[0]
                self._allow_mate_with_better = self.always_allow_mate_with_better
                if len(costs)>self._last_reset_generation+effective_reset_check_generations//2+2 and \
                        costs[-1][0]==costs[-effective_reset_check_generations//2][0] and \
                        costs[-1][1]>=self.reset_check_threshold*costs[-effective_reset_check_generations//2][1]:     
                    self._allow_mate_with_better = True
                if len(costs)>self._last_reset_generation+effective_reset_check_generations+2 and \
                        costs[-1][0]==costs[-effective_reset_check_generations][0] and \
                        costs[-1][1]>=self.reset_check_threshold*costs[-effective_reset_check_generations][1]:                    
                    if self.allow_reset:
                        self.reset(generator)                        
                        self.best_costs_per_generation[0].pop(-1)
                        self._score(generator, register_best)
                        self._skip_next_selection = True                  


    def generate_offspring(self, mate_sol, mate_weights, mate_costs, generator):        
        filtered_mate_sol = mate_sol
        filtered_mate_weights = mate_weights
        filtered_mate_costs = mate_costs
        if mate_sol is not None:
            assert(mate_sol.N_solutions == len(mate_weights))
            if self.champions is not None and len(self.champions) > 1:
                raise ValueError("Cannot supply mate population when GA has multiple champions.")
            if not self._allow_mate_with_better:
                champion_cost = self.champions[0].fitness[0]
                mate_costs_np = np.asarray(mate_costs)
                allowed_mask = np.array([
                    not kgs.lexicographic_less_than(cost, champion_cost)
                    for cost in mate_costs_np
                ], dtype=bool)
                if not np.any(allowed_mask):
                    filtered_mate_sol = None
                    filtered_mate_weights = None
                    filtered_mate_costs = None
                else:
                    allowed_idx = np.where(allowed_mask)[0]
                    filtered_mate_sol = copy.deepcopy(mate_sol)
                    filtered_mate_sol.select_ids(allowed_idx)
                    filtered_mate_costs = mate_costs_np[allowed_idx]
                    filtered_mate_weights = np.asarray(mate_weights)[allowed_idx]                    
        res = self._generate_offspring(filtered_mate_sol, filtered_mate_weights, filtered_mate_costs, generator)        
        self._cached_offspring = res
        return res
    
    def merge_offspring(self):
        self._merge_offspring()
        self._cached_offspring = None

    def get_list_for_simulation(self):
        return self._get_list_for_simulation()
           
    def apply_selection(self):
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
        self.fitness_cost = None
        self._abbreviate()

    def diagnostic_plots(self, i_gen, plot_ax):
        if self.make_own_fig:
            if self._fig is None:
                self._fig, self._ax = plt.subplots(*self.make_own_fig, figsize=self.make_own_fig_size, squeeze=False)
            plot_ax = self._ax
            plt.tight_layout()
        if self.best_costs_per_generation_ax is not None:
            for xx in self.best_costs_per_generation_ax:
                ax = plot_ax[xx[2]]
                ax.clear()
                plt.sca(ax)
                to_plot = np.array([[y[xx[0]] for y in x] for x in self.best_costs_per_generation])
                if xx[1]:
                    to_plot = np.log10(to_plot)
                plt.plot(to_plot.T, alpha=0.5)
                avg_line = np.mean(to_plot, axis=0)
                plt.plot(avg_line, color='black', linewidth=2.0)
                plt.grid(True)
                plt.xlabel('Generation')
                plt.ylabel('Best Cost')
        if self.champion_genotype_ax is not None:
            ax = plot_ax[ self.champion_genotype_ax ]
            plt.sca(ax)
            if self._always_plot_trees or len(self.best_costs_per_generation[0])<2 or kgs.lexicographic_less_than(self.best_costs_per_generation[0][-1], self.best_costs_per_generation[0][-2]):
                ax.clear()
                pack_vis_sol.pack_vis_sol(self.champions[0].genotype, ax=ax)
                plt.title(f'Champion Genotype ({self.champions[0].phenotype.N_trees} trees)')
        if self.champion_phenotype_ax is not None:
            ax = plot_ax[ self.champion_phenotype_ax ]
            plt.sca(ax)
            if self._always_plot_trees or len(self.best_costs_per_generation[0])<2 or kgs.lexicographic_less_than(self.best_costs_per_generation[0][-1], self.best_costs_per_generation[0][-2]):
                ax.clear()
                pack_vis_sol.pack_vis_sol(self.champions[0].phenotype, ax=ax)
                plt.title(f'Champion Phenotype ({self.champions[0].phenotype.N_trees} trees)')
        self._always_plot_trees = False
        self._diagnostic_plots(i_gen, plot_ax)
        if self.make_own_fig:
            plt.suptitle(f'Generation {i_gen}, {self.champions[0].phenotype.N_trees} trees')
            display(self._fig)
    

    def _diagnostic_plots(self, i_gen, plot_ax):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fig'] = None
        state['_ax'] = None
        state['_always_plot_trees'] = True
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fig = None
        self._ax = None
    

@dataclass
class GAMulti(GA):
    # Configuration    
    ga_list: list = field(init=True, default=None)    

    allow_reset_ratio: float = field(init=True, default=0.95) # allow only the worst X% of subpopulations to reset
    diversity_reset_threshold: float = field(init=True, default=0.01/40) # diversity required to avoid reset
    diversity_reset_check_frequency: int = field(init=True, default=5) # in generations

    plot_diversity_ax = None
    plot_subpopulation_costs_per_generation_ax = None

    def _check_constraints(self):
        super()._check_constraints()
        if self.ga_list is not None:
            self.ga_list[0].check_constraints()
            if kgs.debugging_mode>=2:
                for ga in self.ga_list:
                    ga.check_constraints()
    def _initialize(self, generator):          
        for ga in self.ga_list:
            ga.seed = generator.integers(0, 2**30).get().item()
            ga.fitness_cost = self.fitness_cost
            ga.initialize(generator)
        self.best_costs_per_generation = [[]]
    def _reset(self, generator):
        for ga in self.ga_list:
            ga.reset(generator)
    def _score(self, generator, register_best):
        for ga in self.ga_list:
            ga.score(generator, register_best=register_best)
        if register_best:            
            if kgs.debugging_mode>=2:
                for ga in self.ga_list:
                    assert(len(ga.champions) == 1)
            costs_per_ga = np.array([ga.best_costs_per_generation[0][-1] for ga in self.ga_list])
            best_ga_idx = kgs.lexicographic_argmin(costs_per_ga)
            best_ga = self.ga_list[best_ga_idx]
            if self.champions is None or kgs.lexicographic_less_than(best_ga.champions[0].fitness[0], self.champions[0].fitness[0]):
                self.champions = [best_ga.champions[0]]
            self.best_costs_per_generation[0].append(self.champions[0].fitness[0])
            # Apply allow_reset_ratio - only allow worst X% of GAs to reset
            if self.allow_reset_ratio < 1.0:
                sorted_indices = kgs.lexicographic_argsort(costs_per_ga)
                n_allowed_to_reset = int(np.floor(len(self.ga_list) * self.allow_reset_ratio))
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
                    for idx in sorted(to_reset)[::-1]:
                        ga_to_reset = self.ga_list[idx]
                        if ga_to_reset.best_costs_per_generation and ga_to_reset.best_costs_per_generation[0]:
                            ga_to_reset.best_costs_per_generation[0].pop(-1)
                        ga_to_reset.reset(generator)
                        ga_to_reset.score(generator, register_best=True)
                        ga_to_reset._skip_next_selection = True
                        costs_per_ga[idx] = ga_to_reset.champions[0].fitness[0]

    def _generate_offspring(self, mate_sol, mate_weights, mate_costs, generator):
        return sum([
            ga.generate_offspring(mate_sol, mate_weights, mate_costs, generator)
            for ga in self.ga_list
        ], [])
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
    def _diagnostic_plots(self, i_gen, plot_ax):
        if self.plot_subpopulation_costs_per_generation_ax is not None:
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
                for i, ga in enumerate(self.ga_list):
                    if ga._allow_mate_with_better:
                        plt.plot(i, -0.5, 'gx', markersize=10, markeredgewidth=2)
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
            diversity_matrix = kgs.compute_genetic_diversity_matrix(cp.array(champions_pop.xyt), cp.array(champions_pop.xyt)).get()
            im = plt.imshow(diversity_matrix, cmap='viridis', vmin=0., vmax=np.max(diversity_matrix), interpolation='none')
            if not hasattr(ax, '_colorbar') or ax._colorbar is None:
                ax._colorbar = plt.colorbar(im, ax=ax, label='Diversity distance')
            else:
                ax._colorbar.update_normal(im)
            plt.title('Diversity Matrix Across GA Subpopulations')
            plt.xlabel('Individual')
            plt.ylabel('Individual')            
        for ga in self.ga_list:
            ga.diagnostic_plots(i_gen, plot_ax)

@dataclass
class GAMultiSimilar(GAMulti):
    # Configuration
    ga_base: GA = field(init=True, default=None)
    N: int = field(init=True, default=4)

    def _initialize(self, generator):
        if self.ga_list is None:
            self.ga_list = []        
            for i in range(self.N):
                ga_copy = copy.deepcopy(self.ga_base)            
                self.ga_list.append(ga_copy)     
        super()._initialize(generator)    

@dataclass
class GAMultiIsland(GAMultiSimilar):
    """Base class for island-based GA with configurable connectivity patterns.
    
    Subclasses must implement _get_connectivity_matrix() to define which islands
    can mate with each other.
    """
    
    def _get_connectivity_matrix(self) -> np.ndarray:
        """Return boolean connectivity matrix where [i,j]=True means island i can mate with island j.
        
        Returns:
            np.ndarray: Boolean matrix of shape (n_islands, n_islands)
        """
        raise NotImplementedError("Subclasses must implement _get_connectivity_matrix")
    
    def _generate_offspring(self, mate_sol, mate_weights, mate_costs, generator):
        assert mate_sol is None
        
        # Get connectivity matrix from subclass
        connectivity_matrix = self._get_connectivity_matrix()
        n_ga = len(self.ga_list)
        assert connectivity_matrix.shape == (n_ga, n_ga), f"Connectivity matrix shape {connectivity_matrix.shape} doesn't match number of islands {n_ga}"
        
        # Precompute the population objects and fitness arrays to use for mating
        # Always use champions for mating
        population_sources = []
        for ga in self.ga_list:
            source_pop = ga.champions[0]
            population_sources.append((source_pop, source_pop.fitness))
        
        # Generate offspring for each island based on connectivity
        offspring_list = []
        for i, ga in enumerate(self.ga_list):
            # Collect populations from connected islands
            populations_to_merge = []
            for j in range(n_ga):
                if connectivity_matrix[i, j]:  # Island i can mate with island j
                    populations_to_merge.append(population_sources[j])
            
            # Merge all connected populations into a single solution collection
            if len(populations_to_merge) > 0:
                first_pop, first_costs = populations_to_merge[0]
                merged_sol = copy.deepcopy(first_pop.genotype)
                merged_costs = [first_costs]
                for pop, costs in populations_to_merge[1:]:
                    merged_sol.merge(pop.genotype)
                    merged_costs.append(costs)
                mate_weights = np.ones(merged_sol.N_solutions) / merged_sol.N_solutions
                mate_costs = np.concatenate(merged_costs, axis=0)
            else:
                merged_sol, mate_weights, mate_costs = None, None, None
            
            # Generate offspring for this island using the merged mate population
            offspring_list.extend(ga.generate_offspring(merged_sol, mate_weights, mate_costs, generator))
        
        return offspring_list


@dataclass
class GAMultiRing(GAMultiIsland):
    # Configuration
    mate_distance: int = field(init=True, default=6)
    star_topology: bool = field(init=True, default=False)  # Connect all islands to island 0
    asymmetric_star: bool = field(init=True, default=False)  # If True with star_topology, hub receives but doesn't send
    small_world_rewiring: float = field(init=True, default=0.0)  # Probability of rewiring edges
    def _get_connectivity_matrix(self) -> np.ndarray:
        """Return connectivity matrix for ring topology with mate_distance.
        
        Each island can mate with islands at distances 1 to mate_distance in both directions,
        with wraparound (periodic boundaries). Optionally convert to star topology or apply
        small-world rewiring.
        """
        n_ga = len(self.ga_list)
        connectivity_matrix = np.zeros((n_ga, n_ga), dtype=bool)
        
        # Always start with ring topology (if we have more than 1 island)
        if n_ga > 1:
            for i in range(n_ga):
                # Add connections to neighbors within mate_distance
                for offset in range(1, min(self.mate_distance + 1, n_ga // 2 + 1)):
                    # Left neighbor (with wraparound)
                    left_idx = (i - offset) % n_ga
                    connectivity_matrix[i, left_idx] = True
                    
                    # Right neighbor (with wraparound) 
                    right_idx = (i + offset) % n_ga
                    connectivity_matrix[i, right_idx] = True
        
        # Optionally ADD star topology connections on top of ring
        if self.star_topology:
            # Star topology: hub (island 0) connects with all others
            for i in range(1, n_ga):
                if self.asymmetric_star:
                    # Asymmetric: hub receives from all, but doesn't send back
                    connectivity_matrix[0, i] = True   # hub receives from island i
                    connectivity_matrix[i, 0] = False  # island i does NOT receive from hub
                else:
                    # Symmetric: bidirectional connections with hub
                    connectivity_matrix[i, 0] = True
                    connectivity_matrix[0, i] = True
        
        # Apply small-world rewiring if specified (only to original ring edges)
        if self.small_world_rewiring > 0.0 and n_ga > 2:
            rng = np.random.default_rng(self.seed)  # Use deterministic rewiring
            for i in range(n_ga):
                for j in range(i + 1, n_ga):
                    if connectivity_matrix[i, j] and rng.random() < self.small_world_rewiring:
                        # Rewire edge (i,j) to (i,k) where k is random
                        connectivity_matrix[i, j] = False
                        connectivity_matrix[j, i] = False
                        # Find a random target that's not i and not already connected
                        candidates = [k for k in range(n_ga) if k != i and not connectivity_matrix[i, k]]
                        if candidates:
                            k = rng.choice(candidates)
                            connectivity_matrix[i, k] = True
                            connectivity_matrix[k, i] = True
        return connectivity_matrix

ref_solution = None

@dataclass
class GASinglePopulation(GA):
    # Configuration
    N_trees_to_do: int = field(init=True, default=None)
    population_size:int = field(init=True, default=1500) 
    initializer: Initializer = field(init=True, default_factory=InitializerRandom)
    move: pack_move.Move = field(init=True, default=None)
    fixed_h: float = field(init=True, default=-1.)
    ref_score_scale: float = field(init=True, default=1.1)
    reduce_h_threshold: float = field(init=True, default=1e-5/40) # scaled by N_trees
    reduce_h_amount: float = field(init=True, default=2e-3/np.sqrt(40)) # scaled by sqrt(N_trees)
    reduce_h_per_individual: bool = field(init=True, default=False)    

    plot_diversity_ax = None
    plot_diversity_alt_ax = None
    plot_population_fitness_ax = None

    remove_population_after_abbreviate: bool = field(init=True, default=True)

    _initial_population_size: int = field(init=False, default=None) # set by subclass in initialize()
    
    # Results
    population: Population = field(init=True, default=None)

    def __post_init__(self):        
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
        self.move.moves.append( [pack_move.Crossover2(), 'Crossover2', 2.0] )

        super().__post_init__()

    def _check_constraints(self):
        if self.initializer is not None:
            self.initializer.check_constraints()
        self.move.check_constraints()
        if self.population is not None:
            self.population.check_constraints()
        super()._check_constraints()

    def _initialize(self, generator):
        self.best_costs_per_generation = [[]]

    def _reset(self, generator):
        self.initializer.seed = generator.integers(0, 2**30).get().item()
        if self.fixed_h is not None:
            if self.fixed_h == -1.:                
                ref_score = 0.317 + 0.206/np.sqrt(self.N_trees_to_do)
                ref_h = np.sqrt(ref_score*self.N_trees_to_do*np.sqrt(self.ref_score_scale))         
                self.initializer.fixed_h = cp.array([ref_h,0,0],dtype=kgs.dtype_cp)
            else:
                self.initializer.fixed_h = cp.array([self.fixed_h*np.sqrt(self.N_trees_to_do),0,0],dtype=kgs.dtype_cp)
            self.initializer.base_solution.use_fixed_h = True
        self.population = self.initializer.initialize_population(self._initial_population_size, self.N_trees_to_do)    
        self.population.check_constraints()        

    def _score(self, generator, register_best):
        # Compute cost and reshape to (N_solutions, 1) for tuple-based fitness
        cost_values = self.fitness_cost.compute_cost_allocate(self.population.phenotype, evaluate_gradient=False)[0].get()

        
        if self.population.phenotype.use_fixed_h:
            reduce_h_amount = self.reduce_h_amount*np.sqrt(self.N_trees_to_do)
            if not self.reduce_h_per_individual:
                while np.min(cost_values) < self.reduce_h_threshold*self.N_trees_to_do:
                    # Reduce h if below threshold
                    self.population.genotype.h[:, 0] -= reduce_h_amount
                    self.population.phenotype.h[:, 0] -= reduce_h_amount
                    cost_values = self.fitness_cost.compute_cost_allocate(self.population.phenotype, evaluate_gradient=False)[0].get()            
            else:
                # Reduce h per individual if below threshold
                any_reduced = True
                while any_reduced:
                    any_reduced = False
                    for i in range(self.population.phenotype.N_solutions):
                        if cost_values[i] < self.reduce_h_threshold*self.N_trees_to_do:
                            self.population.genotype.h[i, 0] -= reduce_h_amount
                            self.population.phenotype.h[i, 0] -= reduce_h_amount
                            any_reduced = True
                    cost_values = self.fitness_cost.compute_cost_allocate(self.population.phenotype, evaluate_gradient=False)[0].get()
            self.population.fitness = np.stack( (self.population.phenotype.h[:,0].get()**2/self.N_trees_to_do, cost_values)).T # Shape: (N_solutions, 2)
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
        pass

    def _abbreviate(self):
        if self.remove_population_after_abbreviate:
            self.population = None
    

@dataclass
class GASinglePopulationDiversity(GASinglePopulation):

    population_size:int = field(init=True, default=1500)
    generate_extra: float = field(init=True, default=0.4)  # Make this equal to Orchestrator.filter_before_rought
    prob_mate_own: float = field(init=True, default=0.7)
    
    # Parameters for generating selection_size (used only if selection_size is None)
    # These are ratios that remain constant when scaling population_size
    survival_rate: float = field(init=True, default=0.074)  # Fraction of population that survives (37/500)
    elitism_fraction: float = field(init=True, default=0.25)  # Fraction of survivors that are elite (18/37)
    search_depth: float = field(init=True, default=1.)  # How deep to look for diversity (max_tier/pop_size)
    diversity_criterion: float = field(init=True, default=0.2)  # will scale with N_trees
    diversity_criterion_scaling: float = field(init=True, default=0.) # will scale with N_trees
    diversity_to_elite_only: bool = field(init=True, default=False)
    lap_config: lap_batch.LAPConfig = field(init=True, default_factory=lambda: lap_batch.LAPConfig(algorithm='min_cost_row'))

    def _initialize(self, generator):
        # Convoluted way to find _initial_population_size (due to legacy)
        total_survivors = max(2, int(self.population_size * self.survival_rate))
        n_elite = max(1, int(total_survivors * self.elitism_fraction))
        n_diversity_tiers = total_survivors - n_elite
        max_tier = max(n_elite + 1, int(self.population_size * self.search_depth))
        
        elite = list(range(1, n_elite + 1))
        if n_diversity_tiers > 0:
            tiers = np.geomspace(n_elite + 1, max_tier, n_diversity_tiers).astype(int)
            tiers = list(np.unique(tiers))
        else:
            tiers = []
        selection_size = elite + tiers
        self._initial_population_size = len(selection_size)
        super()._initialize(generator)

    def _apply_selection(self):
        current_pop = self.population
        current_pop.select_ids(kgs.lexicographic_argsort(current_pop.fitness))  # Sort by fitness lexicographically
        current_xyt = current_pop.genotype.xyt  # (N_individuals, N_trees, 3)

        # Based on best individuals that meet diversity criterion
        
        # Calculate how many to select
        total_survivors = max(2, int(self.population_size * self.survival_rate))
        n_elite = max(1, int(total_survivors * self.elitism_fraction))
        n_diversity = total_survivors - n_elite
        
        # Limit search space to best pop_size*search_depth individuals
        max_search = max(total_survivors + 1, int(self.population_size * self.search_depth))
        max_search = min(max_search, self.population_size)  # Don't exceed population
        
        # Step 1: Select N best individuals (elite)
        selected = np.zeros(self.population_size, dtype=bool)
        selected[:n_elite] = True
        
        # Step 2: Select M more individuals based on diversity
        if n_diversity > 0:
            # Get xyt for search space
            search_xyt = current_xyt[:max_search]  # (max_search, N_trees, 3)
            diversity_threshold = self.diversity_criterion * self.N_trees_to_do
            
            # Initialize diversity array by computing to all elite individuals
            diversity_matrix = kgs.compute_genetic_diversity_matrix(
                cp.array(search_xyt),
                cp.array(current_xyt[:n_elite]),
                lap_config=self.lap_config
            ).get()
            diversity = diversity_matrix.min(axis=1)  # (max_search,)
            diversity[:n_elite] = 0.0  # Elite have zero diversity to themselves
            
            # For each diversity pick
            for i in range(n_diversity):
                # Find candidates that meet diversity criterion and are not yet selected
                meets_criterion = (diversity >= diversity_threshold) & (~selected[:max_search])
                
                # Pick the best among those that meet criterion, or just the best if none meet
                if np.any(meets_criterion):
                    # Pick best (lowest index = best fitness) that meets criterion
                    selected_id = np.where(meets_criterion)[0][0]
                else:
                    # Pick best that's not yet selected
                    unselected = np.where(~selected[:max_search])[0]
                    if len(unselected) > 0:
                        selected_id = unselected[0]
                    else:
                        # All searched individuals are selected, can't add more
                        break
                
                selected[selected_id] = True
                
                # Update diversity by computing to the newly selected individual only
                if not self.diversity_to_elite_only:
                    diversity = np.minimum(
                        kgs.compute_genetic_diversity(
                            cp.array(search_xyt),
                            cp.array(current_xyt[selected_id]),
                            lap_config=self.lap_config
                        ).get(),
                        diversity
                    )

                diversity_threshold += self.diversity_criterion_scaling * self.N_trees_to_do
        
        current_pop.select_ids(np.where(selected)[0])

        self.population = current_pop
        self.population.check_constraints()

    def _generate_offspring(self, mate_sol, mate_weights, mate_costs, generator):
        
        old_pop = self.population
        old_pop.check_constraints()
        old_pop.parent_fitness = old_pop.fitness.copy()
        parent_size = old_pop.genotype.N_solutions
        new_pop = old_pop.create_empty(int(self.population_size/self.generate_extra)-parent_size, self.N_trees_to_do)

        # Generate all parent and mate selections at once (vectorized)
        N_offspring = new_pop.genotype.N_solutions

        # Pick random parents (all from old_pop)
        parent_ids = generator.integers(0, parent_size, size=N_offspring)

        # Merge old_pop.genotype and mate_sol into a single mate collection
        # Layout: [0, parent_size) are from old_pop, [parent_size, ...) are from mate_sol
        merged_mate_sol = copy.deepcopy(old_pop.genotype)
        if mate_sol is not None and mate_sol.N_solutions > 0:
            merged_mate_sol.merge(mate_sol)
            use_own = generator.random(N_offspring) < self.prob_mate_own
        else:
            use_own = np.ones(N_offspring, dtype=bool)

        # Split offspring into two groups
        inds_use_own = np.where(use_own)[0]
        inds_use_external = np.where(~use_own)[0]

        # === Prepare mate selection for own population offspring ===
        if len(inds_use_own) > 0:
            parent_ids_own = parent_ids[inds_use_own]
            # Pick random mates (excluding parent) - fully vectorized
            mate_ids_own = generator.integers(0, parent_size - 1, size=len(inds_use_own))
            # If mate_id >= parent_id, increment by 1 to skip the parent
            mate_ids_own = np.where(mate_ids_own >= parent_ids_own, mate_ids_own + 1, mate_ids_own)
        else:
            parent_ids_own = None
            mate_ids_own = None

        # === Prepare mate selection for external population offspring ===
        if len(inds_use_external) > 0:
            parent_ids_external = parent_ids[inds_use_external]
            
            # Normalize mate_weight to get probability distribution
            mate_prob = cp.asarray(mate_weights) / cp.sum(mate_weights)
            
            # Sample mates using weighted selection (CuPy doesn't have choice, use cumsum + searchsorted)
            cum_prob = cp.cumsum(mate_prob)
            random_vals = generator.random(len(inds_use_external))
            mate_ids_external = cp.searchsorted(cum_prob, random_vals)
            
            # Offset external mate indices to point into merged collection
            mate_ids_external = mate_ids_external + parent_size
        else:
            parent_ids_external = None
            mate_ids_external = None

        # === Clone all parents into new_pop ===
        if len(inds_use_own) > 0:
            new_pop.create_clone_batch(inds_use_own, old_pop, parent_ids_own)
        if len(inds_use_external) > 0:
            new_pop.create_clone_batch(inds_use_external, old_pop, parent_ids_external)

        # === Apply moves (merged into single call) ===
        all_inds = []
        all_mate_ids = []
        
        if len(inds_use_own) > 0:
            all_inds.append(inds_use_own)
            # Ensure mate_ids_own is NumPy array
            mate_ids_own_np = mate_ids_own.get() if isinstance(mate_ids_own, cp.ndarray) else np.asarray(mate_ids_own)
            all_mate_ids.append(mate_ids_own_np)
        
        if len(inds_use_external) > 0:
            all_inds.append(inds_use_external)
            # Ensure mate_ids_external is NumPy array
            mate_ids_external_np = mate_ids_external.get() if isinstance(mate_ids_external, cp.ndarray) else np.asarray(mate_ids_external)
            all_mate_ids.append(mate_ids_external_np)
        
        if len(all_inds) > 0:
            combined_inds = cp.array(np.concatenate(all_inds))
            combined_mate_ids = cp.array(np.concatenate(all_mate_ids))
            self.move.do_move_vec(new_pop, combined_inds, merged_mate_sol, combined_mate_ids, generator)

        new_pop.genotype.canonicalize()

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
    filter_before_rough: float = field(init=True, default=0.4)  # fraction of solutions to keep before rough relax
    rough_relaxers: list = field(init=True, default=None) # meant to prevent heavy overlaps    
    fine_relaxers: list = field(init=True, default=None)  # meant to refine solutions
    n_generations: int = field(init=True, default=200)
    genotype_at: int = field(init=True, default=1)  # 0:before relax, 1:after rough relax, 2:after fine relax(=phenotype)
    seed: int = field(init=True, default=42)
    
    # Diagnostics
    diagnostic_plot: bool = field(init=True, default=False)
    plot_every: int = field(init=True, default=1)

    # Intermediate
    _current_generation: int = field(init=False, default=0)
    _is_finalized: bool = field(init=False, default=False)
    _generator: cp.random.Generator = field(init=False, default=None)

    
    def __post_init__(self):        
        self.fitness_cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                            pack_cost.BoundaryDistanceCost(scaling=1.), 
                            pack_cost.CollisionCostExactSeparation(scaling=1., use_lookup_table=True)])
        
        self.rough_relaxers = []
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = self.fitness_cost
        relaxer.max_step = 1e-1
        relaxer.n_iterations = 80
        self.rough_relaxers.append(relaxer)

        self.fine_relaxers = []
        cost_fine =  self.fitness_cost
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = cost_fine
        relaxer.n_iterations = 30
        relaxer.max_step = 1e-3 * np.sqrt(10)
        self.fine_relaxers.append(relaxer)
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = cost_fine
        relaxer.n_iterations = 30
        relaxer.max_step = 1e-3
        self.fine_relaxers.append(relaxer)
        relaxer = pack_dynamics.OptimizerBFGS()
        relaxer.cost = cost_fine
        relaxer.n_iterations = 30 
        relaxer.max_step = 1e-3 / np.sqrt(10)
        self.fine_relaxers.append(relaxer)

        super().__post_init__()

    def _relax(self, sol_list):
        assert self.genotype_at==1
        for s in sol_list:
            fitness = self.fitness_cost.compute_cost_allocate(s.genotype, evaluate_gradient=False)[0].get()
            sorted_ids = kgs.lexicographic_argsort(fitness)
            n_keep = int(s.genotype.N_solutions*self.filter_before_rough)
            s.select_ids(sorted_ids[:n_keep])
        conf_list = [s.genotype for s in sol_list]        
        for relaxer in self.rough_relaxers:
            pack_dynamics.run_simulation_list(relaxer, conf_list)
        for s in sol_list:
            s.genotype.canonicalize()
            s.phenotype = s.genotype.convert_to_phenotype()
        conf_list = [s.phenotype for s in sol_list]        
        for relaxer in self.fine_relaxers:
            pack_dynamics.run_simulation_list(relaxer, conf_list)
        s.genotype.snap()



    def _check_constraints(self):        
        self.fitness_cost.check_constraints()
        self.ga.check_constraints()
        return super()._check_constraints()
    
    def run(self):
        self.check_constraints(debugging_mode_offset=2)

        while self._current_generation<self.n_generations and not self.ga._stopped:
            if self._current_generation==0:
                # Initialize generator if not already present (e.g., loaded from checkpoint)
                if self._generator is None:
                    self._generator = cp.random.default_rng(seed=self.seed)
                
                self.ga.fitness_cost = self.fitness_cost
                self.ga.seed = self.seed

                # Initialize
                self.ga.initialize(self._generator)
                self.ga.reset(self._generator)
                self.ga.score(self._generator, register_best=True)


            self._current_generation = self._current_generation

            offspring_list = self.ga.generate_offspring(None, None, None, self._generator)
            self._relax(offspring_list)
            self.ga.merge_offspring()
            
            self.ga.score(self._generator, register_best=True)            
            for s in self.ga.best_costs_per_generation:
                assert len(s) == self._current_generation + 2
            self.ga.apply_selection()
            # Format best costs as lists for display (max 6 decimals)
            best_costs_str = [[round(float(x), 6) for x in s[-1].flatten()] for s in self.ga.best_costs_per_generation]
            if self.diagnostic_plot:
                if self._current_generation % self.plot_every == 0:                    
                    self.ga.diagnostic_plots(self._current_generation, None)        
                    clear_output(wait=True)  # Clear previous output                
            else:
                print(f'Generation {self._current_generation}: Best costs = {best_costs_str}')

            self._current_generation += 1

            if kgs.debugging_mode>=2:
                self.check_constraints()
        
        if not self._is_finalized:
            self._generator = None
            self.ga.finalize()
            self.ga.abbreviate()
            self._generator = None
            self._is_finalized = True

def baseline():
    runner = Orchestrator(n_generations=60000)
    runner.ga = GAMultiRing(N=16)

    ga_base = GASinglePopulationDiversity(N_trees_to_do=-1) # N_trees_to_do to be set by client    
    ga_base.reset_check_generations = 100
    ga_base.reset_check_threshold = 0.5

    runner.ga.ga_base = ga_base
    runner.ga.do_legalize = True

    runner.ga.make_own_fig = (2,3)
    runner.ga.make_own_fig_size = (18,12)
    runner.ga.best_costs_per_generation_ax = ( (0,False,(0,0)),)# ,(1,True,(1,0)))
    runner.ga.plot_subpopulation_costs_per_generation_ax = ( (0,False,(0,1)) ,(1,True,(1,1)))
    runner.ga.champion_genotype_ax = (1,2)
    runner.ga.champion_phenotype_ax = (0,2)
    runner.ga.plot_diversity_ax = (1,0)
    runner.plot_every = 20

    return runner

def baseline_symmetry_90():
    runner = baseline()
    runner.ga.ga_base.initializer.base_solution = kgs.SolutionCollectionSquareSymmetric90()
    runner.ga.ga_base.move.moves = [m for m in runner.ga.ga_base.move.moves if m[1] not in ['Translate']]
    runner.ga.ga_base.move.moves.pop(-2) # remove square crossover
    runner.ga.ga_base.move.moves.append( [pack_move.Crossover2(distance_function = 'square90'), 'CrossoverSquare', 2.0] )
    runner.ga.ga_base.move.moves.append( [pack_move.Crossover2(distance_function = 'square90', decouple_mate_location=True), 
                                          'CrossoverSquareDecoupled', 2.0] )
    runner.ga.stop_check_generations_scale = 25
    return runner

def baseline_symmetry_180():
    runner = baseline()
    runner.ga.ga_base.initializer.base_solution = kgs.SolutionCollectionSquareSymmetric180()
    runner.ga.ga_base.move.moves = [m for m in runner.ga.ga_base.move.moves if m[1] not in ['Translate']]
    runner.ga.ga_base.move.moves.pop(-2) # remove square crossover
    runner.ga.ga_base.move.moves.append( [pack_move.Crossover2(distance_function = 'square180', max_N_trees_ratio = 0.45), 'CrossoverSquare', 2.0] )
    runner.ga.ga_base.move.moves.append( [pack_move.Crossover2(distance_function = 'square180', decouple_mate_location=True, max_N_trees_ratio = 0.45), 
                                          'CrossoverSquareDecoupled', 2.0] )
    runner.ga.stop_check_generations_scale = 35
    return runner

def baseline_tesselated(adapt_moves=True):
    runner = baseline()
    runner.ga.ga_base.move.moves = [m for m in runner.ga.ga_base.move.moves if m[1] not in ['Translate']]
    runner.ga.ga_base.move.moves.pop(-2) # remove square crossover
    runner.ga.ga_base.move.moves.append( [pack_move.Crossover2(distance_function = 'square', max_N_trees_ratio = 0.45), 'CrossoverSquare', 2.0] )
    runner.ga.ga_base.move.moves.append( [pack_move.Crossover2(distance_function = 'square', decouple_mate_location=True, max_N_trees_ratio = 0.45), 
                                          'CrossoverSquareDecoupled', 2.0] )
    runner.ga.stop_check_generations_scale = 20
    runner.ga.ga_base.reset_check_generations_ratio = 0.

    runner.ga.ga_base.initializer.ref_sol_crystal_type = 'Perfect dimer'
    runner.ga.ga_base.initializer.ref_sol_axis1_offset = lambda r:r.uniform(-1.,1.)
    runner.ga.ga_base.initializer.ref_sol_axis2_offset = lambda r:r.uniform(-1.,1.)

    runner.ga.ga_base.initializer.new_tree_placer = True
    runner.ga.ga_base.initializer.base_solution.edge_spacer = kgs.EdgeSpacerBasic(dist_x = 0.75, dist_y = 0.5, dist_corner = 0.)

    if adapt_moves:
        runner.ga.ga_base.initializer.base_solution.filter_move_locations_with_edge_spacer = True
        for m in runner.ga.ga_base.move.moves:
            if isinstance(m[0], pack_move.Crossover2):
                m[0].do_90_rotation = False
                m[0].use_edge_clearance_when_decoupled = False
                if m[0].distance_function == 'stripe':
                    m[0].respect_edge_spacer_filter = False
                if m[0].decouple_mate_location:
                    m[0].max_N_trees_ratio = 0.25

    return runner

def baseline_symmetry_180_tesselated(adapt_moves=True):
    runner = baseline_symmetry_180()
    runner.ga.ga_base.initializer.ref_sol_crystal_type = 'Perfect dimer'
    runner.ga.ga_base.initializer.ref_sol_axis1_offset = lambda r:r.choice([0.,0.5]).item()
    runner.ga.ga_base.initializer.ref_sol_axis2_offset = 'set!' # must be configured by client
    runner.ga.stop_check_generations_scale = 10
    runner.ga.ga_base.reset_check_generations_ratio = 0.

    runner.ga.ga_base.initializer.new_tree_placer = True
    runner.ga.ga_base.initializer.base_solution.edge_spacer = kgs.EdgeSpacerBasic(dist_x = 0.75, dist_y = 0.5, dist_corner = 0.)

    if adapt_moves:
        runner.ga.ga_base.initializer.base_solution.filter_move_locations_with_edge_spacer = True
        for m in runner.ga.ga_base.move.moves:
            if isinstance(m[0], pack_move.Crossover2):
                m[0].do_90_rotation = False
                m[0].use_edge_clearance_when_decoupled = False
                if m[0].distance_function == 'stripe':
                    m[0].respect_edge_spacer_filter = False
                if m[0].decouple_mate_location:
                    m[0].max_N_trees_ratio = 0.25
    return runner