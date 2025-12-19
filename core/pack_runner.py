import numpy as np
import copy
import time
from dataclasses import dataclass, field
import pack_ga
import pack_cost
import pack_dynamics
import kaggle_support as kgs


@dataclass
class Runner(kgs.BaseClass):
    """Runner for hyperparameter analysis of pack_ga.GA()"""
    # Inputs
    label: str = field(init=False, default='')
    seed: int = 0
    base_ga: pack_ga.GA = field(default_factory=pack_ga.GA)
    modifier_dict: dict = field(default_factory=dict)
    use_missing_value: bool = False

    # Outputs
    modifier_values: dict = field(default_factory=dict)
    configured_ga: pack_ga.GA = None
    result_ga: pack_ga.GA = None
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
            self.best_costs = ga.best_cost_per_generation

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

def scale_population_size(ga, name, value):
    ga.population_size = int(ga.population_size * value)
    ga.selection_size = [int((s-1) * value)+1 for s in ga.selection_size]

def set_genetic_diversity(ga, name, value):
    if not value:
        ga.selection_size = list(np.arange(len(ga.selection_size))+1)

def set_alt_diversity(ga, name, value):
    if value:
        ga.selection_size = [np.round(x).astype(int)+1 for x in np.arange(0/32,0.999,1/32)*ga.population_size]
        print(ga.selection_size)

def set_no_jiggle(ga, name, value):
    if value:
        ga.initializer.jiggler.n_rounds = 0

def scale_rough_iterations(ga, name, value):
    if len(ga.rough_relaxers) > 0:
        ga.rough_relaxers[0].n_iterations = int(ga.rough_relaxers[0].n_iterations * value)

def scale_fine_iterations(ga, name, value):
    for relaxer in ga.fine_relaxers:
        relaxer.n_iterations = int(relaxer.n_iterations * value)

def set_number_of_rough_steps(ga, name, value):
    """Set the number of rough relaxer steps to use (keeps first N relaxers)"""
    if value < len(ga.rough_relaxers):
        ga.rough_relaxers = ga.rough_relaxers[:value]

def set_number_of_fine_steps(ga, name, value):
    """Set the number of fine relaxer steps to use (keeps first N relaxers)"""
    if value < len(ga.fine_relaxers):
        ga.fine_relaxers = ga.fine_relaxers[:value]

def set_rough_bfgs(ga, name, value):
    """Replace rough relaxer with BFGS optimizer if True"""
    if value and len(ga.rough_relaxers) > 0:
        old_cost = ga.rough_relaxers[0].cost
        old_n_iterations = ga.rough_relaxers[0].n_iterations
        ga.rough_relaxers[0] = pack_dynamics.OptimizerBFGS()
        ga.rough_relaxers[0].cost = old_cost
        ga.rough_relaxers[0].n_iterations = 400
        ga.rough_relaxers[0].track_cost = False
        ga.rough_relaxers[0].plot_cost = False

def disable_move(ga, name, value):
    """Disable a specific move by setting its weight to 0 if value is True"""
    if value:
        # Extract move name from modifier name (e.g., 'disable_Crossover' -> 'Crossover')
        move_name = name.replace('disable_', '')
        for move_item in ga.move.moves:
            if move_item[1] == move_name:
                move_item[2] = 0.0  # Set weight to 0
                break

def set_JiggleClusterSmallMaxN(ga, name, value):
    """Set max_N_trees for JiggleClusterSmall move"""
    for move_item in ga.move.moves:
        if move_item[1] == 'JiggleClusterSmall':
            move_item[0].max_N_trees = value
            break

def set_JiggleClusterBigMaxN(ga, name, value):
    """Set max_N_trees for JiggleClusterBig move"""
    for move_item in ga.move.moves:
        if move_item[1] == 'JiggleClusterBig':
            move_item[0].max_N_trees = value
            break

def set_TwistMinRadius(ga, name, value):
    """Set min_radius for Twist move"""
    for move_item in ga.move.moves:
        if move_item[1] == 'Twist':
            move_item[0].min_radius = value
            break

def set_TwistMaxRadius(ga, name, value):
    """Set max_radius for Twist move"""
    for move_item in ga.move.moves:
        if move_item[1] == 'Twist':
            move_item[0].max_radius = value
            break

def set_CrossoverMaxNtrees(ga, name, value):
    """Set max_N_trees for Crossover move"""
    for move_item in ga.move.moves:
        if move_item[1] == 'Crossover':
            move_item[0].max_N_trees = value
            break

def set_CrossoverSimpleMate(ga, name, value):
    """Set simple_mate_location for Crossover move"""
    for move_item in ga.move.moves:
        if move_item[1] == 'Crossover':
            move_item[0].simple_mate_location = value
            break

def set_CrossoverP(ga,name,value):
    for move_item in ga.move.moves:
        if move_item[1] == 'Crossover':
            print(move_item[2])
            #assert(move_item[2]==4.)
            move_item[2] = value
            break

def disable_init(ga,name,value):
    if value:
        ga.initializer.jiggler.duration_init /=10000
        ga.initializer.jiggler.duration_compact /=10000
        ga.initializer.jiggler.duration_final/=10000


def set_used_fixed_h(ga,name,value):
    ga.initializer.base_solution.use_fixed_h = bool(value)


def set_fixed_h(ga,name,value):
    import cupy as cp    
    ga.initializer.base_solution.fixed_h = cp.array([value,0.,0.], dtype=kgs.dtype_cp)

def set_h_schedule(ga,name,value):
    n_gens = ga.n_generations
    end_part = np.round(n_gens/3).astype(int)
    print(n_gens)
    end_val = ga.initializer.base_solution.fixed_h[0]
    h_schedule = list(np.linspace(end_val+value, end_val, n_gens-end_part)) + [end_val]*end_part
    ga.h_schedule = h_schedule
    print('h_schedule', ga.h_schedule)

def set_ga_prop(ga, name, value):
    """Generic setter for GA properties - sets ga.{name} = value"""
    # Handle special case where modifier name differs from property name
    setattr(ga, name, value)


# ============================================================
# Example runner configurations
# ============================================================

def baseline_runner(fast_mode=False):
    """Baseline configuration with no modifications"""
    res = Runner()
    res.label = 'Baseline'
    
    #res.modifier_dict['scale_population'] = pm(1., lambda r:4., scale_population_size)
    #res.modifier_dict['genetic_diversity'] = pm(True, lambda r:r.choice([True]), set_genetic_diversity)
    #res.modifier_dict['alt_diversity'] = pm(False, lambda r:r.choice([False]), set_alt_diversity)
    #res.modifier_dict['no_jiggle'] = pm(False, lambda  r:r.choice([True]), set_no_jiggle)
    #res.modifier_dict['bfgs_for_rough'] = pm(False, lambda r:r.choice([True]), set_rough_bfgs)
    #res.modifier_dict['scale_rough_iterations'] = pm(1., lambda r:0.3, scale_rough_iterations)
    #res.modifier_dict['scale_fine_iterations'] = pm(1., lambda r:0.3, scale_fine_iterations)
    #res.modifier_dict['rough_steps'] = pm(1, lambda r:r.choice([1]), set_number_of_rough_steps)
    #res.modifier_dict['fine_steps'] = pm(3, lambda r:r.choice([2]), set_number_of_fine_steps)
    #res.modifier_dict['JiggleClusterSmallMaxN'] = pm(5, lambda r:20, set_JiggleClusterSmallMaxN)
    #res.modifier_dict['JiggleClusterBigMaxN'] = pm(5, lambda r:20, set_JiggleClusterBigMaxN)
    #res.modifier_dict['TwistMinRadius'] = pm(0., lambda r:0.5, set_TwistMinRadius)
    #res.modifier_dict['TwistMaxRadius'] = pm(0., lambda r:2., set_TwistMaxRadius)
    #res.modifier_dict['CrossoverMaxNtrees'] = pm(20, lambda r:20, set_CrossoverMaxNtrees)
    #res.modifier_dict['CrossoverSimpleMate'] = pm(False, lambda r:False, set_CrossoverSimpleMate)
    #res.modifier_dict['CrossoverP'] = pm(0.4, lambda r:3., set_CrossoverP)
    #res.modifier_dict['disable_init'] = pm(False, lambda r:r.choice([False,True]), disable_init)

    res.modifier_dict['set_used_fixed_h'] = pm(True, lambda r:r.choice([False,True]), set_used_fixed_h)
    res.modifier_dict['set_fixed_h'] = pm(3.7, lambda r:r.uniform(3.77,3.8), set_fixed_h)
    res.modifier_dict['reduce_h_threshold'] = pm(1e-4, lambda r:10**r.uniform(-5,-4), set_ga_prop)
    res.modifier_dict['reduce_h_amount'] = pm(1e-3, lambda r:r.uniform(1e-3, 2e-3), set_ga_prop)
    
    #res.modifier_dict['set_h_schedule'] = pm(0., lambda r:r.uniform(0,0.15), set_h_schedule)
    # # Add modifiers to disable each move with 20% probability
    # # Get move names from base_ga
    # for move_item in res.base_ga.move.moves:
    #     move_name = move_item[1]  # Extract move name
    #     res.modifier_dict[f'disable_{move_name}'] = pm(
    #         False,
    #         lambda r: r.uniform() < 0.2,  # 20% chance to disable
    #         disable_move
    #     )

    # del res.modifier_dict['disable_JiggleTreeSmall']

    runner = res.base_ga
    runner.n_generations = 300

    # runner.fine_relaxers[0] = pack_dynamics.OptimizerBFGS()
    # runner.fine_relaxers[0].cost = pack_ga.GA().fine_relaxers[0].cost
    # runner.fine_relaxers[0].track_cost = False
    # runner.fine_relaxers[0].plot_cost = False  
    # runner.fine_relaxers.append(copy.deepcopy(runner.fine_relaxers[0]))
    # runner.fine_relaxers[1].max_step = 1e-3
    # runner.fine_relaxers.append(copy.deepcopy(runner.fine_relaxers[0]))
    # runner.fine_relaxers[2].max_step = 1e-4
    # runner.fine_relaxers[0].use_line_search = False
    if fast_mode:         
        runner.n_generations = 5
        runner.population_size = 100
        runner.selection_size = [1,2,5,10]

    return res


