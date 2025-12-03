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
    n_generations:int = field(init=True, default=5000)
    tournament_size:int = field(init=True, default=20)
    n_elites:int = field(init=True, default=10)
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
            for (i_N_trees, N_trees) in enumerate(self.N_trees_to_do):
                best_id = np.argmin(self.populations[i_N_trees].fitness)                   
                print(f'Generation {i_gen}, Trees {N_trees}, Best cost: {self.populations[i_N_trees].fitness[best_id]:.8f}, Est: {100*self.populations[i_N_trees].fitness[best_id]/N_trees:.8f}, h: {self.populations[i_N_trees].configuration.h[best_id,0].get():.6f}')    
                self.best_cost_per_generation[i_gen, i_N_trees] = self.populations[i_N_trees].fitness[best_id]
                import matplotlib.pyplot as plt
                #plt.figure()
                #plt.plot(np.sort(self.populations[i_N_trees].fitness))
                #plt.pause(0.0001)
                
                current_pop = self.populations[i_N_trees]
                current_xyt = current_pop.configuration.xyt.get()  # (N_individuals, N_trees, 3)
                current_h = current_pop.configuration.h.get()  # (N_individuals, 3) - [size, x_offset, y_offset]
                current_fitness = current_pop.fitness

                # Elitism: find the best n_elites individuals
                elite_indices = np.argsort(current_fitness)[:self.n_elites]

                # Tournament selection for each new individual
                new_xyt = np.empty_like(current_xyt)
                new_h = np.empty_like(current_h)
                
                # Copy elites unmutated
                for i_ind, elite_idx in enumerate(elite_indices):
                    new_xyt[i_ind] = current_xyt[elite_idx]
                    new_h[i_ind] = current_h[elite_idx]
                
                # Fill remaining slots with tournament selection + mutation
                for i_ind in range(self.n_elites, self.population_size):
                    # Pick tournament_size individuals at random
                    tournament_ids = generator.choice(self.population_size, size=self.tournament_size, replace=False)
                    # Select the one with the best (lowest) fitness
                    winner_idx = tournament_ids[np.argmin(current_fitness[tournament_ids])]
                    
                    # Copy the winner's configuration
                    new_xyt[i_ind] = current_xyt[winner_idx]
                    new_h[i_ind] = current_h[winner_idx]
                    
                    # Pick a random tree and give it a new random position within the square
                    if generator.uniform()<self.p_move:
                        tree_to_mutate = generator.integers(0, N_trees)
                        h_size = new_h[i_ind, 0]  # Square size
                        h_offset_x = new_h[i_ind, 1]  # x offset
                        h_offset_y = new_h[i_ind, 2]  # y offset
                        new_xyt[i_ind, tree_to_mutate, 0] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_x  # x
                        new_xyt[i_ind, tree_to_mutate, 1] = generator.uniform(-h_size / 2, h_size / 2) + h_offset_y  # y
                        new_xyt[i_ind, tree_to_mutate, 2] = generator.uniform(-np.pi, np.pi)  # theta

                # Update existing population's configuration in-place
                current_pop.configuration.xyt = cp.array(new_xyt, dtype=np.float32)
                current_pop.configuration.h = cp.array(new_h, dtype=np.float32)
                
                # Relax and score the population
                self._relax_and_score(current_pop)

                
        