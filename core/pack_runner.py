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




# ============================================================
# Example runner configurations
# ============================================================

def baseline_runner(fast_mode=False):
    """Baseline configuration with no modifications"""
    res = Runner()
    res.label = 'Baseline'


    runner = pack_ga3.baseline()
    runner.n_generations = 600 if not fast_mode else 2
    runner.diagnostic_plot = False
    runner.ga.do_legalize = True

    #runner.ga.ga_base.alternative_selection = True
    #runner.ga.ga_base.search_depth = 1.
    
    

    # # run some code here to compare runner and runner2, highlighting where they differ. I remember there's a package for this
    # from deepdiff import DeepDiff
    # print(DeepDiff(runner,runner2,ignore_order=True).pretty())

    res.base_ga = runner
    
    res.modifier_dict['use_fixed_h_for_size_setup'] = pm(False, lambda r:r.choice([False,True]).item(), set_size_setup)
    res.modifier_dict['always_allow_mate_with_better'] = pm(False, lambda r:False, set_ga_prop)
    res.modifier_dict['allow_reset_ratio'] = pm(0.5, lambda r:r.uniform(0.3,0.7), set_ga_prop)
    res.modifier_dict['diversity_reset_threshold'] = pm(5./40, lambda r:r.uniform(3./40, 10./40), set_ga_prop)
    res.modifier_dict['mate_distance'] = pm(6, lambda r:r.choice([4,6,8]).item(), set_ga_prop)
    res.modifier_dict['generate_extra'] = pm(0.4, lambda r:r.uniform(0.1,0.4), generate_extra)
    res.modifier_dict['genotype_at'] = pm(1, lambda r:(r.choice([2,1]).item()), set_orchestrator_prop)
    res.modifier_dict['remove_fine_1'] = pm(False, lambda r:False, remove_fine_1)
    res.modifier_dict['remove_fine_2'] = pm(False, lambda r:r.choice([False,True]).item(), remove_fine_2)

    res.modifier_dict['reduce_h_threshold'] = pm(1e-5/40, lambda r:r.uniform(0.5e-5/40, 1.5e-5/40), set_ga_base_ga_prop)
    res.modifier_dict['reduce_h_amount'] = pm(2e-3/np.sqrt(40), lambda r:r.uniform(1e-3/np.sqrt(40), 4e-3/np.sqrt(40)), set_ga_base_ga_prop)
    res.modifier_dict['reduce_h_per_individual'] = pm(False, lambda r:r.choice([False,True]).item(), set_ga_base_ga_prop)

    res.modifier_dict['population_size'] = pm(1., lambda r:r.uniform(0.75,1.25), scale_population_size)    
    res.modifier_dict['prob_mate_own'] = pm(1., lambda r:r.uniform(0.5,0.9), set_ga_base_ga_prop)
    res.modifier_dict['survival_rate'] = pm(0.074, lambda r:r.uniform(0.07,0.15), set_ga_base_ga_prop)
    res.modifier_dict['elitism_fraction'] = pm(0.25, lambda r:r.uniform(0.1,0.4), set_ga_base_ga_prop)
    res.modifier_dict['search_depth'] = pm(1., lambda r:r.uniform(0.5,1.), set_ga_base_ga_prop)
    res.modifier_dict['diversity_criterion'] = pm(0.2, lambda r:r.uniform(0.1,0.3), set_ga_base_ga_prop)
    res.modifier_dict['alter_diversity'] = pm(0, lambda r:r.choice([0,1,2]).item(), alter_diversity)


    return res

