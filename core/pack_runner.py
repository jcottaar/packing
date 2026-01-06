import numpy as np
import copy
import time
from dataclasses import dataclass, field
import pack_ga3
import pack_cost
import pack_dynamics
import kaggle_support as kgs


@dataclass
class Runner(kgs.BaseClass):
    """Runner for hyperparameter analysis of pack_ga.GA()"""
    # Inputs
    label: str = field(init=False, default='')
    seed: int = 0
    base_ga: pack_ga3.Orchestrator = field(default_factory=pack_ga3.Orchestrator)
    modifier_dict: dict = field(default_factory=dict)
    use_missing_value: bool = False

    # Outputs
    modifier_values: dict = field(default_factory=dict)
    configured_ga: pack_ga3.Orchestrator = None
    result_ga: pack_ga3.Orchestrator = None
    best_costs: np.ndarray = None
    runtime_seconds: float = None
    exception: str = None

    def run(self):
        start_time = time.time()
        try:
            # Set up modified GA
            rng = np.random.default_rng(seed=self.seed)
            ga = copy.deepcopy(self.base_ga)
            self.modifier_values = {'seed': self.seed}
            ga.seed = self.seed

            # Apply modifiers
            for key, value in self.modifier_dict.items():
                if not self.use_missing_value:
                    self.modifier_values[key] = value.random_function(rng)
                else:
                    self.modifier_values[key] = value.missing_value
                value.modifier_function(ga, key, self.modifier_values[key])

            self.configured_ga = copy.deepcopy(ga)
            print(self.modifier_values)

            # Run GA
            ga.run()

            self.result_ga = copy.deepcopy(ga)
            # Extract costs from best_costs_per_generation (list of lists of fitness tuples)
            # Structure: best_costs_per_generation[i_config][i_generation] = fitness_tuple
            # We want output shape: (n_generations, n_configs)
            n_generations = len(ga.ga.best_costs_per_generation[0]) if ga.ga.best_costs_per_generation else 0
            self.best_costs = np.array([[float(ga.ga.best_costs_per_generation[i_config][i_gen][0]) 
                                        for i_config in range(len(ga.ga.best_costs_per_generation))]
                                       for i_gen in range(n_generations)])

            # Record runtime
            self.runtime_seconds = time.time() - start_time
            print(f"Runtime: {self.runtime_seconds:.1f}s")

        except Exception as e:
            self.runtime_seconds = time.time() - start_time
            print('ERROR!')
            import traceback
            self.exception = traceback.format_exc()
            print(self.exception)


def pm(missing_value, random_function, modifier_function):
    """Create PropertyModifier with given parameters"""
    res = PropertyModifier()
    res.missing_value = missing_value
    res.random_function = random_function
    res.modifier_function = modifier_function
    return res


@dataclass
class PropertyModifier:
    """Modifier for a single hyperparameter"""
    default_value = 0
    missing_value = 0  # value to assume if missing in older output
    random_function = 0  # gets RNG as input, returns a new value
    modifier_function = 0  # gets GA, name, and value as input, modifies GA


# ============================================================
# Modifier functions for GA hyperparameters
# ============================================================


def set_orchestrator_prop(ga, name, value):
    """Generic setter for GA properties - sets ga.{name} = value"""
    # Handle special case where modifier name differs from property name
    setattr(ga, name, value)

def set_ga_prop(ga, name, value):
    """Generic setter for GA properties - sets ga.{name} = value"""
    # Handle special case where modifier name differs from property name
    setattr(ga.ga, name, value)

def set_ga_base_ga_prop(ga,name,value):
    """Generic setter for GA properties - sets ga.ga.{name} = value"""
    # Handle special case where modifier name differs from property name
    setattr(ga.ga.ga_base, name, value)

def scale_population_size(ga, name, value):
    """Scale population size by given factor"""
    ga.ga.ga_base.population_size = int(ga.ga.ga_base.population_size * value)
    #ga.ga.ga_base.selection_size = [int( (s-1) * value)+1 for s in ga.ga.ga_base.selection_size]
    #ga.n_generations = int(ga.n_generations / value)
    # now make sure selection_size is unique, i.e. 1,2,2,3,3,4,4,5,20 must become 1,2,3,4,5,6,7,8,40
    # seen = set()
    # unique_selection = []
    # for s in ga.ga.ga_base.selection_size:
    #     while s in seen:
    #         s += 1
    #     seen.add(s)
    #     unique_selection.append(s)
    # ga.ga.ga_base.selection_size = unique_selection
    # ga.ga.N = int(ga.ga.N / value)
    # print(ga.ga.N, ga.ga.ga_base.population_size, ga.ga.ga_base.selection_size)

def set_n_selection_size(ga, name, value):
    """Set number of selection sizes to use"""
    ga.ga.ga_base.selection_size = ga.ga.ga_base.selection_size[:value]

def disable_stripe_crossover(ga, name, value):
    """Disable stripe crossover in multi-ring GA"""
    print(ga.ga.ga_base.move.moves[-1])
    if value:
        ga.ga.ga_base.move.moves.pop(-1)

def scale_rough_iterations(ga, name, value):
    for r in ga.rough_relaxers:
        r.n_iterations = int(r.n_iterations * value)


def scale_fine_iterations(ga, name, value):
    for r in ga.fine_relaxers:
        r.n_iterations = int(r.n_iterations * value)

def set_auction(ga, name, value):
    if value:
        ga.ga.ga_base.lap_config.algorithm='auction'

def set_fixed_scaling(ga, name, value):
    if value:
        ga.ga.ga_base.diversity_criterion_scaling = 0.
    else:
        ga.ga.ga_base.diversity_criterion_scaling = ga.ga.ga_base.diversity_criterion/10.
        ga.ga.ga_base.diversity_criterion = 0.

def make_single(ga, name, value):
    if value:
        ga.ga = ga.ga.ga_base
        ga.n_generations //= 2
        ga.ga.do_legalize = True
        ga.ga.reset_check_generations = 10000000

def set_alternative_selection(ga, name, value):
    if value:
        ga.ga.ga_base.alternative_selection = True
        ga.ga.ga_base.search_depth = 1.
    else:
        ga.ga.ga_base.alternative_selection = False
        ga.ga.ga_base.search_depth = 0.5

def generate_extra(ga, name, value):
    ga.ga.ga_base.generate_extra = value
    ga.filter_before_rough = value

def remove_fine_1(ga, name, value):
    if value:
        ga.fine_relaxers.pop(0)

def remove_fine_2(ga, name, value):
    if value:
        ga.fine_relaxers.pop(1)

def remove_fine_3(ga, name, value):
    if value:
        ga.fine_relaxers.pop(2)

def alter_diversity(ga, name, value):
    if value == 1:
        ga.ga.ga_base.lap_config.algorithm = 'min_cost_row'
    elif value == 2:
        ga.ga.ga_base.lap_config.algorithm = 'min_cost_col'

def set_size_setup(ga, name, value):
    ga.ga.ga_base.initializer.use_fixed_h_for_size_setup = value

def use_lookup_table_rough(ga, name, value):
    ga.rough_relaxers[0].cost.costs[2].use_lookup_table = value
def use_lookup_table_fine(ga, name, value):
    for r in ga.fine_relaxers:
        r.cost.costs[2].use_lookup_table = value

def simple_mate_location(ga, name, value):
    ga.ga.ga_base.move.moves[-2][0].simple_mate_location = value

minkowski = pack_cost.CollisionCostExactSeparation()
minkowski.use_lookup_table = True
def use_minkowski_rough(ga, name, value):
    if value:
        ga.rough_relaxers[0].cost.costs[2] = minkowski
def use_minkowski_fine(ga, name, value):
    if value:
        for r in ga.fine_relaxers:
            r.cost.costs[2] = minkowski
def set_minkowski_cost(ga, name, value):
    if not value:
        ga.fitness_cost = copy.deepcopy(ga.fitness_cost)
        ga.fitness_cost.costs[2] = pack_cost.CollisionCostSeparation()

def set_reset_approach(ga, name, value):
    match value:
        case 1:
            pass
        case 2:
            ga.ga.allow_reset_ratio = 0.75
        case 3:
            ga.ga.allow_reset_ratio = 0.5
        case 4:
            ga.ga.allow_reset_based_on_local_champion = True


def set_connectivity_pattern(ga, name, value):
    """Switch between different connectivity patterns for multi-island GA"""
    # Save current ga properties
    old_ga = ga.ga
    current_ga_base = old_ga.ga_base
    mate_distance = ga.ga.mate_distance
    
    match value:
        case 1:  # Default ring
            new_ga = pack_ga3.GAMultiRing(N=32)
            new_ga.mate_distance = mate_distance
        case 2:  # Ring with symmetric star topology
            new_ga = pack_ga3.GAMultiRing(N=32)
            new_ga.mate_distance = mate_distance
            new_ga.star_topology = True
            new_ga.asymmetric_star = False
        case 3:  # Ring with asymmetric star topology
            new_ga = pack_ga3.GAMultiRing(N=32)
            new_ga.mate_distance = mate_distance
            new_ga.star_topology = True
            new_ga.asymmetric_star = True
        case 4:  # Ring with small world rewiring
            new_ga = pack_ga3.GAMultiRing(N=32)
            new_ga.mate_distance = mate_distance
            new_ga.small_world_rewiring = 0.1
        case 5:  # Hypercube (N must be power of 2)
            new_ga = pack_ga3.GAMultiHypercube(N=32)
        case 6:  # Tree (N must be 2^k-1, so use 31)
            new_ga = pack_ga3.GAMultiTree(N=31)
        case _:
            raise ValueError(f"Unknown connectivity pattern: {value}")
    
    # Copy all properties from old GA to new GA, excluding class-specific ones
    for attr_name in dir(old_ga):
        if (not attr_name.startswith('_') and 
            hasattr(new_ga, attr_name) and 
            not callable(getattr(old_ga, attr_name)) and
            attr_name not in ['N', 'mate_distance', 'star_topology', 'asymmetric_star', 'small_world_rewiring']):
            try:
                setattr(new_ga, attr_name, getattr(old_ga, attr_name))
            except (AttributeError, TypeError):
                # Skip properties that can't be set (read-only, etc.)
                pass
    
    # Ensure ga_base is properly set
    new_ga.ga_base = current_ga_base
    
    # Replace the GA
    ga.ga = new_ga
            
def remove_move_by_name(ga, name, value):
    if not value:
        len1 = len(ga.ga.ga_base.move.moves)
        ga.ga.ga_base.move.moves = [m for m in ga.ga.ga_base.move.moves if m[1] != name]
        assert len(ga.ga.ga_base.move.moves) == len1-1, f"Move '{name}' not found to remove."

def set_jiggle_max_trees(ga, name, value):
    import pack_move
    any_set = False
    for move in ga.ga.ga_base.move.moves:
        if isinstance(move[0], pack_move.JiggleCluster):
            any_set = True
            move[0].max_N_trees = value
    assert any_set, "No 'Jiggle' moves found to set JiggleMaxTrees."

def set_jitter(ga, name, value):
    import pack_move
    any_set = False
    for move in ga.ga.ga_base.move.moves:
        if hasattr(move[0], 'jitter'):
            any_set = True
            move[0].jitter = value
    assert any_set, "No moves with 'jitter' attribute found to set jitter."

def set_rough_relax_max_step(ga, name, value):
    if value>0:
        for r in ga.rough_relaxers:
            r.max_step = value

def scale_N_and_pop_size(ga, name, value):
    ga.ga.N = int(ga.ga.N * value)
    ga.ga.ga_base.population_size = int(ga.ga.ga_base.population_size / value)




# ============================================================
# Example runner configurations
# ============================================================

def baseline_runner(fast_mode=False):
    """Baseline configuration with no modifications"""
    res = Runner()
    res.label = 'Baseline'


    runner = pack_ga3.baseline_symmetry_180()
    runner.n_generations = 2000 if not fast_mode else 2
    runner.diagnostic_plot = False
    runner.ga.do_legalize = not fast_mode
    runner.ga.stop_check_generations = 1000000
    #runner.ga.ga_base.alternative_selection = True
    #runner.ga.ga_base.search_depth = 1.
    
    

    # # run some code here to compare runner and runner2, highlighting where they differ. I remember there's a package for this
    # from deepdiff import DeepDiff
    # print(DeepDiff(runner,runner2,ignore_order=True).pretty())

    res.base_ga = runner

    res.modifier_dict['mate_distance'] = pm(6, lambda r:r.choice([4,6,8]).item(), set_ga_prop)
    res.modifier_dict['scale_N'] = pm(1., lambda r:r.choice([0.5,1.0,2.0]).item(), scale_N_and_pop_size)
    res.modifier_dict['reset_approach'] = pm(1, lambda r:1, set_reset_approach) #
    res.modifier_dict['reset_check_generations'] = pm(100, lambda r:100, set_ga_base_ga_prop) 
    res.modifier_dict['diversity_reset_threshold'] = pm(-1., lambda r:0.01/40, set_ga_prop) 
    res.modifier_dict['scale_rough_iterations'] = pm(1.0, lambda r:1., scale_rough_iterations) 
    res.modifier_dict['connectivity_pattern'] = pm(1, lambda r:r.choice([1,6]).item(), set_connectivity_pattern)
    res.modifier_dict['prob_mate_own'] = pm(0.7, lambda r:r.uniform(0.6,0.8), set_ga_base_ga_prop)
    # res.modifier_dict['JiggleMaxTrees'] = pm(20, lambda r:r.choice([4,5,6,7,8,9,10]).item(), set_jiggle_max_trees)
    # res.modifier_dict['MoveRandomTree'] = pm(True, lambda r:r.choice([True, False]).item(), remove_move_by_name)
    # res.modifier_dict['JiggleTreeSmall'] = pm(True, lambda r:r.choice([True, False]).item(), remove_move_by_name)
    # res.modifier_dict['JiggleTreeBig'] = pm(True, lambda r:r.choice([True, False]).item(), remove_move_by_name)
    # res.modifier_dict['JiggleClusterSmall'] = pm(True, lambda r:r.choice([True, False]).item(), remove_move_by_name)
    # res.modifier_dict['JiggleClusterBig'] = pm(True, lambda r:r.choice([True, False]).item(), remove_move_by_name)
    # res.modifier_dict['Twist'] = pm(True, lambda r:r.choice([True, False]).item(), remove_move_by_name)
    res.modifier_dict['rough_relax_max_step'] = pm(-1e-1, lambda r:1e-1, set_rough_relax_max_step)


    #jitter
    

    
    
     
    #res.modifier_dict['make_single'] = pm(False, lambda r:False, make_single)
    #res.modifier_dict['use_minkowski_rough'] = pm(False, lambda r:r.choice([False,True]).item(), use_minkowski_rough)
    #res.modifier_dict['use_lookup_table_fine'] = pm(False, lambda r:r.choice([False,True]).item(), use_lookup_table_fine)
    # res.modifier_dict['elitism_fraction'] = pm(0.25, lambda r:r.choice([0., 0.25]).item(), set_ga_base_ga_prop)
    # res.modifier_dict['diversity_to_elite_only'] = pm(False, lambda r:r.choice([False, True]).item(), set_ga_base_ga_prop)
    # res.modifier_dict['scale_population_size'] = pm(1.0, lambda r:r.choice([1.0, 2.0]).item(), scale_population_size)
    # res.modifier_dict['reduce_h_per_individual'] = pm(False, lambda r:r.choice([False, True]).item(), set_ga_base_ga_prop)
    # res.modifier_dict['use_minkowski_for_overall_cost'] = pm(True, lambda r:r.choice([False, True]).item(), set_minkowski_cost)
    # res.modifier_dict['simple_mate_location'] = pm(True, lambda r:r.choice([False, True]).item(), simple_mate_location)



    return res

