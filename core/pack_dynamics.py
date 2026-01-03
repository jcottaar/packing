import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '../core'))
import kaggle_support as kgs
import importlib
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
from dataclasses import dataclass, field, fields
import pack_cuda
import pack_vis_sol
import pack_cost
import copy
from IPython.display import HTML, display, clear_output
from scipy import stats
from typeguard import typechecked
from torch.utils.dlpack import to_dlpack, from_dlpack

@dataclass
class OptimizerBFGS(kgs.BaseClass):
    # Configuration    
    track_cost = False  # Record cost history
    plot_cost = False  # Plot cost history at end

    # Hyperparameters
    cost = None
    n_iterations = 100
    history_size = 3
    max_step = 0.01
    tolerance_rel_change = 0.  # Relative change tolerance for convergence
    stop_on_cost_increase = False  # Stop optimization if cost increases
    use_line_search = False

    def __post_init__(self):
        super().__post_init__()
        self.cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=1.)])
        
    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):

        sol.check_constraints()
        sol = copy.deepcopy(sol)
        sol.prep_for_phenotype()
        #sol.snap()

        #print('snapped', cp.min(self.cost.compute_cost_allocate(sol)[0]))

        sol_tmp = copy.deepcopy(sol)

        if self.track_cost:
            cost_history = []

        counter = 0

        x0 = cp.concatenate((sol_tmp.xyt.reshape(sol_tmp.N_solutions,-1), sol_tmp.h.reshape(sol_tmp.N_solutions,-1)),axis=1)
        tmp_x = cp.zeros_like(x0, dtype=kgs.dtype_cp)
        tmp_res = cp.zeros_like(x0, dtype=kgs.dtype_cp)
        x0 = from_dlpack(x0.toDlpack())
        
        tmp_xyt = copy.deepcopy(sol.xyt)
        tmp_h = copy.deepcopy(sol.h)
        tmp_cost = cp.zeros(sol.N_solutions, dtype=kgs.dtype_cp)
        tmp_grad = cp.zeros_like(sol.xyt, dtype=kgs.dtype_cp)
        tmp_grad_h = cp.zeros_like(sol.h, dtype=kgs.dtype_cp)
        
        def f_torch(x, is_main_loop):            
            nonlocal tmp_xyt, tmp_h, tmp_cost, tmp_grad, tmp_grad_h, tmp_res, tmp_x
            tmp_x[:x.shape[0]] = cp.from_dlpack(to_dlpack(x))
            nonlocal sol_tmp, counter
            counter+=1
            N_split = sol_tmp.xyt.shape[1]*3
            N = x.shape[0]

            # This has to go via tmp_xyt/tmp_h for contiguity
            tmp_xyt[:N, :] = tmp_x[:N,:N_split].reshape(N,-1,3)
            tmp_h[:N, :] = tmp_x[:N,N_split:].reshape(N,-1)
            sol_tmp.xyt = tmp_xyt[:N, :]
            sol_tmp.h = tmp_h[:N, :]

            self.cost.compute_cost(sol_tmp, tmp_cost[:N], tmp_grad[:N, :], tmp_grad_h[:N, :])

            if self.track_cost:
                cost_history.append(tmp_cost[:N].get())
                
            res = cp.zeros_like(tmp_x[:N,:], dtype=kgs.dtype_cp)
            res[:,:N_split] = tmp_grad[:N, :].reshape(sol_tmp.N_solutions,-1)
            res[:,N_split:] = tmp_grad_h[:N, :].reshape(sol_tmp.N_solutions,-1)
            if kgs.profiling:
                cp.cuda.Device().synchronize()
            return from_dlpack(tmp_cost[:N].toDlpack()), from_dlpack(res.toDlpack())
        
        

        import lbfgs_torch_parallel
        results = lbfgs_torch_parallel.lbfgs(
            f_torch,x0,tolerance_grad=0, tolerance_change=0, tolerance_rel_change=self.tolerance_rel_change, max_iter=self.n_iterations, history_size=self.history_size, max_step=self.max_step,
            line_search_fn = 'strong_wolfe' if self.use_line_search else None, stop_on_cost_increase=self.stop_on_cost_increase)
        x_result = cp.from_dlpack(to_dlpack(results))
        sol.xyt = cp.ascontiguousarray(x_result[:,:sol.xyt.shape[1]*3].reshape(sol.N_solutions,-1,3))
        sol.h = cp.ascontiguousarray(x_result[:,sol.xyt.shape[1]*3:].reshape(sol.N_solutions,-1))

        if self.plot_cost and self.track_cost:
            cost_history = np.array(cost_history)  # Convert list to array: (n_actual_iterations, N_solutions)
            plt.figure(figsize=(8,8))
            plt.plot(cost_history)
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            ax = plt.gca()
            ax.set_yscale('log')
            plt.pause(0.001)
            

        #print('after', cp.min(self.cost.compute_cost_allocate(sol)[0]))

        sol.unprep_for_phenotype()
        return sol

@dataclass
class Optimizer(kgs.BaseClass):
    # Minimizes the cost, not physics-based

    # Configuration
    plot_interval = None
    use_lookahead = False  # Enable second-order lookahead (predictor-corrector)
    track_cost = False  # Record cost history
    plot_cost = False  # Plot cost history at end
    adaptive_step_size = False  # Enable adaptive step size based on cost changes

    # Hyperparameters
    cost = None
    dt = 0.05
    n_iterations = 1200
    max_grad_norm = 10.0  # Clip gradients to prevent violent repulsion

    def __post_init__(self):
        super().__post_init__()
        self.cost = pack_cost.CostCompound(costs = [pack_cost.AreaCost(scaling=1e-2), 
                                        pack_cost.BoundaryDistanceCost(scaling=1.), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=1.)])

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):
        # Initial configuration

        sol.check_constraints()
        sol.prep_for_phenotype()
        sol = copy.deepcopy(sol)
        #sol.snap()

        xyt = sol.xyt
        h = sol.h        
        
        n_ensembles = xyt.shape[0]
        n_trees = xyt.shape[1] 

        if self.plot_interval is not None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Pre-allocate gradient arrays once (float32 for efficiency)
        total_cost = cp.zeros(n_ensembles, dtype=kgs.dtype_cp)
        total_grad = cp.zeros_like(xyt, dtype=kgs.dtype_cp)
        bound_grad = cp.zeros_like(h, dtype=kgs.dtype_cp)

        t_total0 = kgs.dtype_np(0.)   
        t_last_plot = kgs.dtype_np(-np.inf)

        if self.track_cost:
            cost_history = np.zeros((self.n_iterations, n_ensembles), dtype=kgs.dtype_np)

        # For second-order lookahead, store previous gradients
        if self.use_lookahead:
            prev_grad = cp.zeros_like(xyt, dtype=kgs.dtype_cp)
            prev_bound_grad = cp.zeros_like(h, dtype=kgs.dtype_cp)

        # For adaptive step size, track previous cost and state
        if self.adaptive_step_size:
            prev_cost = cp.zeros(n_ensembles, dtype=kgs.dtype_cp)
            prev_xyt = cp.zeros_like(xyt, dtype=kgs.dtype_cp)
            prev_h = cp.zeros_like(h, dtype=kgs.dtype_cp)
            new_cost = cp.zeros(n_ensembles, dtype=kgs.dtype_cp)
            temp_grad = cp.zeros_like(xyt, dtype=kgs.dtype_cp)
            temp_grad_h = cp.zeros_like(h, dtype=kgs.dtype_cp)
            # Initialize dt as a scalar to allow per-iteration modification
            current_dt = self.dt
            # Compute initial cost
            self.cost.compute_cost(sol, prev_cost, total_grad, bound_grad)

        for i_iteration in range(self.n_iterations):            
            if self.adaptive_step_size:
                dt = current_dt
            else:
                dt = self.dt
            #print(dt)

            # Save state before step (for adaptive step size)
            if self.adaptive_step_size:
                prev_xyt[:] = xyt
                prev_h[:] = h
                prev_cost_value = prev_cost.copy()

            # Compute cost and gradient at current position
            self.cost.compute_cost(sol, total_cost, total_grad, bound_grad)

            # Clip gradients per tree to prevent violent repulsion
            # if self.max_grad_norm is not None:
            #     grad_norms = cp.sqrt(cp.sum(total_grad**2, axis=2))  # (n_ensembles, n_trees)
            #     grad_norms = cp.maximum(grad_norms, 1e-8)  # Avoid division by zero
            #     clip_factor = cp.minimum(1.0, self.max_grad_norm / grad_norms)  # (n_ensembles, n_trees)
            #     total_grad = total_grad * clip_factor[:, :, None]  # Apply to each component

            # Store cost before taking step (for adaptive and non-adaptive cases)
            if self.adaptive_step_size:
                prev_cost[:] = total_cost

            if self.use_lookahead and i_iteration > 0:
                # Second-order lookahead: use gradient + 0.5 * (gradient - prev_gradient)
                # This is a predictor-corrector scheme: x_{n+1} = x_n - dt * (1.5*g_n - 0.5*g_{n-1})
                # Equivalent to Adams-Bashforth 2-step method
                effective_grad = 1.5 * total_grad - 0.5 * prev_grad
                effective_bound_grad = 1.5 * bound_grad - 0.5 * prev_bound_grad
                xyt -= dt * effective_grad
                h -= dt * effective_bound_grad
            else:
                xyt -= dt * total_grad
                h -= dt * bound_grad

            # Adaptive step size: check if cost increased or decreased after taking step
            if self.adaptive_step_size:
                # Compute new cost after the step (reuse pre-allocated arrays)
                self.cost.compute_cost(sol, new_cost, temp_grad, temp_grad_h)

                # Compare with previous cost
                cost_increased = new_cost > prev_cost_value
                cost_decreased = new_cost <= prev_cost_value

                if cp.any(cost_increased):
                    # Restore previous state for all (global dt reduction)
                    xyt[:] = prev_xyt
                    h[:] = prev_h
                    # Reduce dt by half
                    current_dt *= 0.5
                    # Cost stays at prev_cost_value (step was rejected)
                    total_cost[:] = prev_cost_value
                elif cp.any(cost_decreased):
                    # Accept new state, cost decreased, increase dt
                    current_dt *= 1.05
                    total_cost[:] = new_cost
                else:
                    # Cost stayed the same, accept state but don't change dt
                    total_cost[:] = new_cost

            if self.track_cost:
                cost_history[i_iteration, :] = total_cost.get()

            if self.use_lookahead:
                # Store current gradients for next iteration
                prev_grad[:] = total_grad
                prev_bound_grad[:] = bound_grad

            t_total0 += dt
            
            if self.plot_interval is not None and t_total0 - t_last_plot >= self.plot_interval*0.999:
                t_last_plot = t_total0+0
                ax.clear()
                pack_vis_sol.pack_vis_sol(sol, solution_idx=0, ax=ax)
                ax.set_title(f'Time: {t_total0:.2f}, cost: {total_cost[0].get().item():.12f}, {sol.h[0,0].get()}, {bound_grad[0,0].get()}')
                display(fig)
                clear_output(wait=True)       
        if self.plot_cost and self.track_cost:
            plt.figure(figsize=(8,8))
            plt.plot(cost_history)
            plt.xlabel('Iteration')
            ax = plt.gca()
            ax.set_yscale('log')
            plt.pause(0.001)
        sol.unprep_for_phenotype()
        return sol

@dataclass
class Dynamics(kgs.BaseClass):
    # Physics-based dynamics

    # Configuration    
    plot_interval = None
    seed = None  # Random seed for reproducible Langevin noise (None = non-reproducible)

    # Hyperparameters
    cost0 = None # scales
    cost1 = None # doens't scale
    dt_list = None
    friction_list = None
    temperature_list = None  # Langevin temperature (0 = deterministic)
    cost_0_scaling_list = None
    mass_h = 1.0  # Mass for boundary parameter h (single value)

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):
        # Initial configuration

        sol.check_constraints()
        sol = copy.deepcopy(sol)

        cost0 = copy.deepcopy(self.cost0)
        cost1 = copy.deepcopy(self.cost1)

        n_ensembles = sol.xyt.shape[0]
        n_trees = sol.xyt.shape[1]
        assert self.dt_list.shape == self.friction_list.shape == self.cost_0_scaling_list.shape
        assert self.temperature_list.shape == self.dt_list.shape
        assert self.dt_list.shape[0] == n_ensembles

        # Set up random generator for reproducible Langevin noise
        if self.seed is not None:
            rng = cp.random.Generator(cp.random.XORWOW(seed=self.seed))
        else:
            rng = cp.random.Generator(cp.random.XORWOW())

        if self.plot_interval is not None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Pre-allocate gradient arrays once (float32 for efficiency)
        total_cost0 = cp.zeros(n_ensembles, dtype=kgs.dtype_cp)
        total_grad0 = cp.zeros_like(sol.xyt, dtype=kgs.dtype_cp)
        bound_grad0 = cp.zeros_like(sol.h, dtype=kgs.dtype_cp)
        total_cost1 = cp.zeros(n_ensembles, dtype=kgs.dtype_cp)
        total_grad1 = cp.zeros_like(sol.xyt, dtype=kgs.dtype_cp)
        bound_grad1 = cp.zeros_like(sol.h, dtype=kgs.dtype_cp)

        t_total0 = kgs.dtype_np(0.)      
        t_last_plot = kgs.dtype_np(-np.inf)  
        velocity_xyt = cp.zeros_like(sol.xyt)
        velocity_h = cp.zeros_like(sol.h)
        prev_cost_0_scaling = 0.
        for i_iteration in range(self.dt_list.shape[1]):
            dt = self.dt_list[:, i_iteration]
            friction = self.friction_list[:, i_iteration]
            cost_0_scaling = self.cost_0_scaling_list[:, i_iteration]  
            if cost_0_scaling[0]>0 and prev_cost_0_scaling[0] == 0.: # assuming this applies to all...
                sol.snap()
            prev_cost_0_scaling = cost_0_scaling
            
            # Velocity Verlet with exact exponential friction
            # Step 1: Half-step position update with OLD velocity
            sol.xyt += 0.5 * dt[:, None, None] * velocity_xyt
            sol.h += 0.5 * dt[:, None] * velocity_h
            
            # Step 2: Compute forces at midpoint
            cost0.compute_cost(sol, total_cost0, total_grad0, bound_grad0)
            cost1.compute_cost(sol, total_cost1, total_grad1, bound_grad1)
            total_cost = total_cost0 * cost_0_scaling + total_cost1
            total_grad = total_grad0 * cost_0_scaling[:, None, None] + total_grad1
            bound_grad = bound_grad0 * cost_0_scaling[:, None] + bound_grad1
            
            # Step 3: Exact exponential friction decay for stability (handles large gamma*dt)
            decay_xyt = cp.exp(-friction[:, None, None] * dt[:, None, None])
            decay_h = cp.exp(-friction[:, None] * dt[:, None])
            # Force coefficient: (1 - exp(-gamma*dt)) / gamma, with limit dt as gamma->0
            force_coef_xyt = cp.where(friction[:, None, None] > 1e-8, 
                                       (1 - decay_xyt) / friction[:, None, None],
                                       dt[:, None, None])
            force_coef_h = cp.where(friction[:, None] > 1e-8,
                                     (1 - decay_h) / friction[:, None],
                                     dt[:, None])
            
            # Step 4: Velocity update with exact exponential friction
            velocity_xyt = decay_xyt * velocity_xyt - force_coef_xyt * total_grad
            velocity_h = decay_h * velocity_h - force_coef_h * bound_grad / self.mass_h
            
            # Step 4b: Langevin noise injection (fluctuation-dissipation: sigma = sqrt(T/m * (1 - exp(-2*gamma*dt))))
            temperature = self.temperature_list[:, i_iteration]
            # Noise amplitude for exact exponential friction discretization
            # Divide by mass for proper equipartition: (1/2)*m*<v^2> = (1/2)*T
            noise_var_xyt = temperature[:, None, None] * (1 - decay_xyt**2)
            noise_var_h = (temperature[:, None] / self.mass_h) * (1 - decay_h**2)
            noise_xyt = cp.sqrt(cp.maximum(noise_var_xyt, 0.)) * rng.standard_normal(velocity_xyt.shape, dtype=velocity_xyt.dtype)
            noise_h = cp.sqrt(cp.maximum(noise_var_h, 0.)) * rng.standard_normal(velocity_h.shape, dtype=velocity_h.dtype)
            velocity_xyt += noise_xyt
            velocity_h += noise_h
            
            # Step 5: Half-step position update with NEW velocity
            sol.xyt += 0.5 * dt[:, None, None] * velocity_xyt
            sol.h += 0.5 * dt[:, None] * velocity_h
            t_total0 += dt[0]
            
            if self.plot_interval is not None and t_total0 - t_last_plot >= self.plot_interval*0.999:
                t_last_plot = t_total0+0
                ax.clear()
                pack_vis_sol.pack_vis_sol(sol, solution_idx=0, ax=ax)
                ax.set_title(f'Time: {t_total0:.2f}')
                display(fig)
                clear_output(wait=True)       
        return sol

class DynamicsInitialize(Dynamics):
    n_rounds = 5
    duration_init = 10./10000
    duration_compact = 150./10000
    duration_final = 10./10000
    dt = 0.04
    friction_min = 0.18
    friction_max = 0.
    friction_periods = 3
    friction_high = 15.8  # Tuned to match original 1/dt Euler behavior with exponential friction
    scaling_area_start = 0.6
    scaling_area_end = 0.002
    scaling_boundary = 50.
    scaling_overlap = 10. # recommend to keep this fixed
    use_boundary_distance = True
    use_separation_overlap = True

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):

        self.cost0 = pack_cost.AreaCost(scaling=1.)
        self.cost1 = pack_cost.CostCompound(costs = [pack_cost.BoundaryDistanceCost(scaling=self.scaling_boundary), 
                                        pack_cost.CollisionCostOverlappingArea(scaling=self.scaling_overlap)])
        if not self.use_boundary_distance:
            self.cost1.costs[0] = pack_cost.BoundaryCost(scaling=self.scaling_boundary)
        if self.use_separation_overlap:
            self.cost1.costs[1] = pack_cost.CollisionCostSeparation(scaling=self.scaling_overlap)   
        if sol.periodic:
            self.cost1.costs.pop(0)
        t_total = kgs.dtype_np(0.)
        dt = kgs.dtype_np(self.dt)
        phase = 'init'
        t_this_phase = kgs.dtype_np(0.)        
        rounds_done = 0
        self.dt_list = []
        self.friction_list = []
        self.temperature_list = []
        self.cost_0_scaling_list = []
        while True:
            if phase == 'compact':
                frac = t_this_phase / self.duration_compact
                start = self.scaling_area_start
                end = self.scaling_area_end
                area_scaling = start * (end / start) ** frac
                cost_0_scaling = area_scaling
                friction = self.friction_max + (self.friction_min - self.friction_max) * (1+np.cos(frac*2*np.pi*self.friction_periods))/2
            else:
                cost_0_scaling = 0.
                friction = self.friction_high  # Use high friction (exact exp decay handles this stably)
            self.dt_list.append(dt)
            self.friction_list.append(friction)
            self.temperature_list.append(0.)  # No thermal noise by default
            self.cost_0_scaling_list.append(cost_0_scaling)        
                  
            t_this_phase += dt

            if phase == 'init' and t_this_phase >= self.duration_init:
                phase = 'compact'
                t_this_phase = 0.    
            elif phase == 'compact' and t_this_phase >= self.duration_compact:
                phase = 'final'
                t_this_phase = 0.
            elif phase == 'final' and t_this_phase >= self.duration_final:               
                rounds_done += 1
                if rounds_done >= self.n_rounds:
                    break
                
                phase = 'compact'
                t_this_phase = 0. 
        
        n_ensembles = sol.N_solutions
        self.dt_list = cp.array(self.dt_list)
        self.dt_list = cp.tile(self.dt_list[None, :], (n_ensembles, 1))
        self.friction_list = cp.array(self.friction_list)
        self.friction_list = cp.tile(self.friction_list[None, :], (n_ensembles, 1))
        self.temperature_list = cp.array(self.temperature_list)
        self.temperature_list = cp.tile(self.temperature_list[None, :], (n_ensembles, 1))
        self.cost_0_scaling_list = cp.array(self.cost_0_scaling_list)
        self.cost_0_scaling_list = cp.tile(self.cost_0_scaling_list[None, :], (n_ensembles, 1))
        sol = super().run_simulation(sol)
        self.dt_list = None
        self.friction_list = None
        self.temperature_list = None
        self.cost_0_scaling_list = None
        return sol


@dataclass
class DynamicsAnneal(Dynamics):
    """
    Single-round simulated annealing with exponential temperature decay.
    Temperature decays as T(t) = T_start * exp(-t / tau).
    
    Parameters can vary per-individual for use in genetic algorithms.
    """
    # Hyperparameters (per-individual arrays of size N_solutions)
    cost = None  # Cost function (set in __post_init__)
    dt: float = 0.04
    total_time: float = 20.
    friction = 1.  # Shape (N_solutions,) - friction coefficient per individual
    T_start = 0.05   # Shape (N_solutions,) - starting temperature per individual
    tau = 1.       # Shape (N_solutions,) - exponential decay time constant per individual
    seed: int = None  # Random seed for reproducible Langevin noise

    def __post_init__(self):
        super().__post_init__()
        self.cost = pack_cost.CostCompound(costs=[
            pack_cost.AreaCost(scaling=1e-2),
            pack_cost.BoundaryDistanceCost(scaling=1.),
            pack_cost.CollisionCostOverlappingArea(scaling=1.)
        ])

    @typechecked
    def run_simulation(self, sol: kgs.SolutionCollection):
        n_ensembles = sol.N_solutions
        
        # Use cost as cost1, no cost0 (no force balance scheduling)
        self.cost0 = pack_cost.CostDummy()  # Empty, no scaling cost
        self.cost1 = self.cost
        
        # Convert scalar parameters to arrays if needed
        friction = np.atleast_1d(self.friction)
        T_start = np.atleast_1d(self.T_start)
        tau = np.atleast_1d(self.tau)
        
        # Broadcast to n_ensembles if single values provided
        if friction.shape[0] == 1:
            friction = np.full(n_ensembles, friction[0])
        if T_start.shape[0] == 1:
            T_start = np.full(n_ensembles, T_start[0])
        if tau.shape[0] == 1:
            tau = np.full(n_ensembles, tau[0])
        
        assert friction.shape == (n_ensembles,), f"friction shape {friction.shape} != ({n_ensembles},)"
        assert T_start.shape == (n_ensembles,), f"T_start shape {T_start.shape} != ({n_ensembles},)"
        assert tau.shape == (n_ensembles,), f"tau shape {tau.shape} != ({n_ensembles},)"
        
        # Build schedule lists
        dt = kgs.dtype_np(self.dt)
        n_steps = int(np.ceil(self.total_time / dt))
        
        self.dt_list = []
        self.friction_list = []
        self.temperature_list = []
        self.cost_0_scaling_list = []
        
        for i_step in range(n_steps):
            t = i_step * dt
            # Temperature: T(t) = T_start * exp(-t / tau)
            # Handle tau=0 or very small tau (instant decay to 0)
            temperature = np.where(tau > 1e-8, T_start * np.exp(-t / tau), 0.)
            
            self.dt_list.append(np.full(n_ensembles, dt, dtype=kgs.dtype_np))
            self.friction_list.append(friction.astype(kgs.dtype_np))
            self.temperature_list.append(temperature.astype(kgs.dtype_np))
            self.cost_0_scaling_list.append(np.zeros(n_ensembles, dtype=kgs.dtype_np))  # No cost0 scaling
        
        # Convert to cupy arrays with shape (n_ensembles, n_steps)
        self.dt_list = cp.array(np.stack(self.dt_list, axis=1))
        self.friction_list = cp.array(np.stack(self.friction_list, axis=1))
        self.temperature_list = cp.array(np.stack(self.temperature_list, axis=1))
        self.cost_0_scaling_list = cp.array(np.stack(self.cost_0_scaling_list, axis=1))
        
        # Run the simulation
        sol = super().run_simulation(sol)
        
        # Clean up
        self.dt_list = None
        self.friction_list = None
        self.temperature_list = None
        self.cost_0_scaling_list = None
        return sol


@dataclass
class OptimizerGraph(Optimizer):
    """
    Variant of `Optimizer` that builds and executes a CuPy CUDA graph for the
    loop body on each iteration. The graph is recreated for every iteration
    (not reused across runs) so this is only suitable for measuring raw
    graph execution time. The total graph execution time for the run is
    accumulated in `self.last_graph_exec_time` (seconds).
    """

    last_graph_exec_time: float = 0.0

    @typechecked
    def run_simulation(self, sol:kgs.SolutionCollection):
        # Initial configuration (same setup as Optimizer)

        sol.check_constraints()
        sol = copy.deepcopy(sol)
        sol.snap()

        xyt = sol.xyt
        h = sol.h

        n_ensembles = xyt.shape[0]
        n_trees = xyt.shape[1]

        # Pre-allocate gradient arrays once (float32 for efficiency)
        total_cost = cp.zeros(n_ensembles, dtype=kgs.dtype_cp)
        total_grad = cp.zeros_like(xyt, dtype=kgs.dtype_cp)
        bound_grad = cp.zeros_like(h, dtype=kgs.dtype_cp)

        # warmup cost compute
        pack_cost.skip_allocations = False
        self.cost.compute_cost(sol, total_cost, total_grad, bound_grad)
        pack_cost.skip_allocations = True

        # Build a single CUDA graph that captures the entire loop.
        # Use stream.begin_capture() / end_capture() (CuPy API).
        stream = cp.cuda.Stream()
        with stream:
            stream.begin_capture()
            # Unroll the iteration loop inside the capture so the graph
            # contains the whole sequence of operations.
            for i_iteration in range(self.n_iterations):
                dt = self.dt
                # Compute cost/grad and apply the update. Omit plotting and
                # max_grad_norm handling as requested.
                self.cost.compute_cost(sol, total_cost, total_grad, bound_grad)
                xyt -= dt * total_grad
                h -= dt * bound_grad
            graph = stream.end_capture()

        pack_cost.skip_allocations = False

        # Time the graph execution (launch + synchronize)
        t0 = time.perf_counter()
        graph.launch(stream)
        stream.synchronize()
        t1 = time.perf_counter()
        self.last_graph_exec_time = (t1 - t0)

        return sol
    

    # ...existing code...

def run_simulation_list(simulator, solution_list):
    """
    Run simulations on a list of SolutionCollection objects efficiently by grouping
    solutions with the same number of trees.
    
    Args:
        simulator: Any class with a run_simulation(sol: kgs.SolutionCollection) method
        solution_list: List of kgs.SolutionCollection objects (all same subclass)
    
    The function:
    1. Groups solutions by N_trees
    2. Merges each group into a single SolutionCollection (verifying compatibility)
    3. Runs simulation once per merged group
    4. Scatters results back to original solutions (modifies in-place)
    """
    if len(solution_list) == 0:
        return
    
    # Group solutions by N_trees
    from collections import defaultdict
    groups = defaultdict(list)
    
    for sol in solution_list:
        N_trees = sol.xyt.shape[1]
        groups[N_trees].append(sol)
    
    # Process each group
    for N_trees, group_sols in groups.items():
        if len(group_sols) == 1:
            # Single solution - just run directly and continue
            result = simulator.run_simulation(group_sols[0])
            group_sols[0].xyt[:] = result.xyt
            group_sols[0].h[:] = result.h
            continue
        
        # Verify compatibility if debugging
        if kgs.debugging_mode >= 2:
            # Check all solutions are same subclass
            first_type = type(group_sols[0])
            for sol in group_sols[1:]:
                if type(sol) != first_type:
                    raise ValueError(f"Solution type mismatch in group: {first_type} vs {type(sol)}")
            
            # Check all properties except xyt and h match
            first_sol = group_sols[0]
            for i, sol in enumerate(group_sols[1:], 1):
                # Make copies and exclude xyt and h for comparison
                first_sol_copy = copy.deepcopy(first_sol)
                sol_copy = copy.deepcopy(sol)
                
                first_sol_copy.xyt = None
                first_sol_copy.h = None
                sol_copy.xyt = None
                sol_copy.h = None
                
                # Compare the copies
                if first_sol_copy != sol_copy:
                    raise ValueError(f"Solution {i} properties mismatch with first solution (excluding xyt and h)")
        
        # Merge: stack xyt and h along N_solutions dimension
        merged_xyt = cp.concatenate([sol.xyt for sol in group_sols], axis=0)
        merged_h = cp.concatenate([sol.h for sol in group_sols], axis=0)
        
        # Create merged solution using first solution as template
        merged_sol = copy.deepcopy(group_sols[0])
        merged_sol.xyt = merged_xyt
        merged_sol.h = merged_h
        
        # Run simulation on merged solution
        result = simulator.run_simulation(merged_sol)
        
        # Scatter results back to original solutions
        start_idx = 0
        for sol in group_sols:
            N_sol = sol.xyt.shape[0]
            sol.xyt[:] = result.xyt[start_idx:start_idx + N_sol]
            sol.h[:] = result.h[start_idx:start_idx + N_sol]
            start_idx += N_sol