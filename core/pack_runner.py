import numpy as np
import copy
import time
from dataclasses import dataclass, field
import pack_ga2
import pack_cost
import pack_dynamics
import kaggle_support as kgs


@dataclass
class Runner(kgs.BaseClass):
    """Runner for hyperparameter analysis of pack_ga.GA()"""
    # Inputs
    label: str = field(init=False, default='')
    seed: int = 0
    base_ga: pack_ga2.Orchestrator = field(default_factory=pack_ga2.Orchestrator)
    modifier_dict: dict = field(default_factory=dict)
    use_missing_value: bool = False

    # Outputs
    modifier_values: dict = field(default_factory=dict)
    configured_ga: pack_ga2.Orchestrator = None
    result_ga: pack_ga2.Orchestrator = None
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



# ============================================================
# Example runner configurations
# ============================================================

def baseline_runner(fast_mode=False):
    """Baseline configuration with no modifications"""
    res = Runner()
    res.label = 'Baseline'

    res.modifier_dict['n_generations'] = pm(200, lambda r:r.integers(200,601).item(), set_orchestrator_prop)
    res.modifier_dict['reduce_h_per_individual'] = pm(False, lambda r:r.choice([False,True]).item(), set_ga_prop)
    res.modifier_dict['fixed_h'] = pm(0.605576, lambda r:r.uniform(0.6,0.618), set_ga_prop)

    res.base_ga.ga = pack_ga2.GASinglePopulationOld()
    
    runner = res.base_ga

    if fast_mode:         
        runner.n_generations = 5
        runner.ga.population_size = 100
        runner.ga.selection_size = [1,2,5,10]
        runner.ga.do_legalize = False
        res.modifier_dict['n_generations'] = pm(200, lambda r:r.integers(5,6).item(), set_orchestrator_prop)

    return res

