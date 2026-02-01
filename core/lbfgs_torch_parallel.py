"""L-BFGS Parallel Optimizer with Strong Wolfe Line Search.

This module provides a batched (parallel) implementation of the Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm. It can optimize
multiple independent systems simultaneously on GPU, with optional strong Wolfe line search.

It is based on the PyTorch L-BFGS implementation, modified to support batches.

Key features:
- Batched processing: optimizes M independent systems in parallel
- Pre-allocated buffers to minimize memory allocations
- Strong Wolfe line search with cubic interpolation
- Two-loop recursion for approximate Hessian computation
- GPU-accelerated using PyTorch

This code is released under CC BY-SA 4.0, meaning you can freely use and adapt it
(including commercially), but must give credit to the original author (Jeroen Cottaar)
and keep it under this license.
"""

# mypy: allow-untyped-defs
from typing import Optional
import torch
from torch import Tensor


__all__ = ["lbfgs"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """Scalar cubic interpolation between two points.
    
    Finds the minimum of a cubic polynomial interpolating two points with
    function values and derivatives. Used in line search.
    
    Args:
        x1: First point position (scalar)
        f1: Function value at x1 (scalar)
        g1: Derivative at x1 (scalar)
        x2: Second point position (scalar)
        f2: Function value at x2 (scalar)
        g2: Derivative at x2 (scalar)
        bounds: Optional tuple (xmin, xmax) to constrain result
    
    Returns:
        Interpolated minimum position (scalar), clamped to bounds if provided
    """
    # Determine bounds for interpolation
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Compute cubic interpolation coefficients
    import math
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    
    # Check if interpolation is valid
    if d2_square >= 0:
        d2 = math.sqrt(d2_square)
        
        # Compute minimum position based on point ordering
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        # Fallback to midpoint if interpolation fails
        return (xmin_bound + xmax_bound) / 2.0



def _cubic_interpolate_batch(x1, f1, g1, x2, f2, g2, xmin_bound, xmax_bound):
    """Vectorized cubic interpolation for batched line search.
    
    Performs cubic interpolation independently for M systems in parallel.
    
    Args:
        x1: First point positions, shape (M,)
        f1: Function values at x1, shape (M,)
        g1: Derivatives at x1, shape (M,)
        x2: Second point positions, shape (M,)
        f2: Function values at x2, shape (M,)
        g2: Derivatives at x2, shape (M,)
        xmin_bound: Lower bounds for interpolation, shape (M,)
        xmax_bound: Upper bounds for interpolation, shape (M,)
    
    Returns:
        Interpolated minimum positions, shape (M,), clamped to bounds
    """
    # Compute cubic interpolation coefficients
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2

    # Check which systems have valid interpolation
    valid = d2_square >= 0
    d2 = torch.sqrt(torch.clamp(d2_square, min=0))

    # Compute minimum position based on point ordering
    x1_le_x2 = x1 <= x2
    min_pos_1 = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    min_pos_2 = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
    min_pos = torch.where(x1_le_x2, min_pos_1, min_pos_2)

    # Clamp valid results to bounds, use midpoint for invalid
    result_valid = torch.clamp(min_pos, min=xmin_bound, max=xmax_bound)
    result_invalid = (xmin_bound + xmax_bound) / 2.0

    return torch.where(valid, result_valid, result_invalid)



def _strong_wolfe_batched(
    obj_func,
    x,
    t,
    d,
    f,
    g,
    gtd,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25
):
    """Batched strong Wolfe line search for multiple systems.
    
    Performs strong Wolfe conditions line search independently for M systems,
    using cubic interpolation to find acceptable step sizes. This is the 
    bracketing + zoom approach.
    
    Args:
        obj_func: Objective function that takes x (M, N) and returns (f, g)
                  where f has shape (M,) and g has shape (M, N)
        x: Current parameter values, shape (M, N)
        t: Initial step sizes, shape (M,)
        d: Search directions, shape (M, N)
        f: Current function values, shape (M,)
        g: Current gradients, shape (M, N)
        gtd: Current directional derivatives (g·d), shape (M,)
        c1: Armijo condition constant (default: 1e-4)
        c2: Curvature condition constant (default: 0.9)
        tolerance_change: Minimum bracket width (default: 1e-9)
        max_ls: Maximum line search iterations (default: 25)
    
    Returns:
        f_new: New function values, shape (M,)
        g_new: New gradients, shape (M, N)
        t: Final step sizes, shape (M,)
        ls_func_evals: Number of function evaluations per system, shape (M,)
    """
    M = x.shape[0]  # Number of systems
    device = x.device
    dtype = x.dtype

    # Initialize tracking for each system
    d_norm = d.abs().max(dim=1).values  # (M,)
    g = g.clone()

    # Evaluate initial step for all systems
    x_new = x + t.unsqueeze(1) * d
    f_new, g_new = obj_func(x_new, True)
    ls_func_evals = torch.ones(M, dtype=torch.long, device=device)
    gtd_new = (g_new * d).sum(dim=1)  # (M,)

    # Track active systems (still searching)
    active = torch.ones(M, dtype=torch.bool, device=device)

    # Track active systems (still searching)
    active = torch.ones(M, dtype=torch.bool, device=device)

    # Bracketing phase state (per system)
    t_prev = torch.zeros(M, dtype=dtype, device=device)
    f_prev = f.clone()
    g_prev = g.clone()
    gtd_prev = gtd.clone()

    # Bracket state: stores interval [t_low, t_high] for each system
    bracket_t = torch.zeros((M, 2), dtype=dtype, device=device)  # Step sizes
    bracket_f = torch.zeros((M, 2), dtype=dtype, device=device)  # Function values
    bracket_g = torch.zeros((M, 2, x.shape[1]), dtype=dtype, device=device)  # Gradients
    bracket_gtd = torch.zeros((M, 2), dtype=dtype, device=device)  # Directional derivatives
    bracket_size = torch.zeros(M, dtype=torch.long, device=device)  # 0, 1, or 2

    done = torch.zeros(M, dtype=torch.bool, device=device)
    ls_iter = 0

    # ========== Bracketing phase ==========
    # Find an interval [t_low, t_high] containing acceptable step
    while active.any() and ls_iter < max_ls:
        # Check Armijo condition (sufficient decrease)
        armijo_fail = f_new > (f + c1 * t * gtd)
        not_decreasing = (ls_iter > 1) & (f_new >= f_prev)
        bracket_condition = armijo_fail | not_decreasing

        # Systems that just found a bracket
        newly_bracketed = active & bracket_condition & (bracket_size == 0)
        if newly_bracketed.any():
            idx = newly_bracketed
            bracket_size[idx] = 2
            bracket_t[idx, 0] = t_prev[idx]
            bracket_t[idx, 1] = t[idx]
            bracket_f[idx, 0] = f_prev[idx]
            bracket_f[idx, 1] = f_new[idx]
            bracket_g[idx, 0] = g_prev[idx]
            bracket_g[idx, 1] = g_new[idx]
            bracket_gtd[idx, 0] = gtd_prev[idx]
            bracket_gtd[idx, 1] = gtd_new[idx]
            active[idx] = False

        # Check strong Wolfe conditions (curvature condition)
        wolfe_satisfied = (gtd_new.abs() <= -c2 * gtd) & active
        if wolfe_satisfied.any():
            idx = wolfe_satisfied
            bracket_size[idx] = 1
            bracket_t[idx, 0] = t[idx]
            bracket_f[idx, 0] = f_new[idx]
            bracket_g[idx, 0] = g_new[idx]
            done[idx] = True
            active[idx] = False

        # Check for positive curvature (need to bracket)
        pos_curv = (gtd_new >= 0) & active
        if pos_curv.any():
            idx = pos_curv
            bracket_size[idx] = 2
            bracket_t[idx, 0] = t_prev[idx]
            bracket_t[idx, 1] = t[idx]
            bracket_f[idx, 0] = f_prev[idx]
            bracket_f[idx, 1] = f_new[idx]
            bracket_g[idx, 0] = g_prev[idx]
            bracket_g[idx, 1] = g_new[idx]
            bracket_gtd[idx, 0] = gtd_prev[idx]
            bracket_gtd[idx, 1] = gtd_new[idx]
            active[idx] = False

        if not active.any():
            break

        # Interpolate new step size for active systems
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10

        if active.any():
            t[active] = _cubic_interpolate_batch(
                t_prev[active], f_prev[active], gtd_prev[active],
                t[active], f_new[active], gtd_new[active],
                min_step[active], max_step[active]
            )

        # Update previous state
        t_prev = t.clone()
        f_prev = f_new.clone()
        g_prev = g_new.clone()
        gtd_prev = gtd_new.clone()

        # Evaluate new points for active systems
        x_new[active] = x[active] + t[active].unsqueeze(1) * d[active]
        f_new_batch, g_new_batch = obj_func(x_new[active], False)
        f_new[active] = f_new_batch
        g_new[active] = g_new_batch
        ls_func_evals[active] += 1
        gtd_new[active] = (g_new[active] * d[active]).sum(dim=1)

        ls_iter += 1

    # Systems that hit max iterations without bracketing
    no_bracket = (bracket_size == 0) & (ls_iter == max_ls)
    if no_bracket.any():
        idx = no_bracket
        bracket_size[idx] = 2
        bracket_t[idx, 0] = 0
        bracket_t[idx, 1] = t[idx]
        bracket_f[idx, 0] = f[idx]
        bracket_f[idx, 1] = f_new[idx]
        bracket_g[idx, 0] = g[idx]
        bracket_g[idx, 1] = g_new[idx]

    # ========== Zoom phase ==========
    # Refine the bracket to find acceptable step satisfying Wolfe conditions
    active = (bracket_size == 2) & ~done  # Only zoom for bracketed, not done systems
    insuf_progress = torch.zeros(M, dtype=torch.bool, device=device)

    # Find low/high positions in bracket (low = better function value)
    low_pos = (bracket_f[:, 0] <= bracket_f[:, 1]).long()
    high_pos = 1 - low_pos

    while active.any() and ls_iter < max_ls:
        # Check if bracket is too small to continue
        bracket_width = (bracket_t[:, 1] - bracket_t[:, 0]).abs()
        too_small = (bracket_width * d_norm < tolerance_change) & active
        active[too_small] = False

        if not active.any():
            break

        # Compute new trial value via cubic interpolation
        t_new = torch.zeros(M, dtype=dtype, device=device)
        if active.any():
            xmin = bracket_t.min(dim=1).values
            xmax = bracket_t.max(dim=1).values
            t_new[active] = _cubic_interpolate_batch(
                bracket_t[active, 0], bracket_f[active, 0], bracket_gtd[active, 0],
                bracket_t[active, 1], bracket_f[active, 1], bracket_gtd[active, 1],
                xmin[active], xmax[active]
            )

        # Check if we're making sufficient progress
        eps = 0.1 * bracket_width
        dist_to_bounds = torch.stack([
            bracket_t.max(dim=1).values - t_new,
            t_new - bracket_t.min(dim=1).values
        ], dim=1).min(dim=1).values

        close_to_boundary = (dist_to_bounds < eps) & active

        # Adjust if insufficient progress or out of bounds
        bracket_min = bracket_t.min(dim=1).values
        bracket_max = bracket_t.max(dim=1).values

        need_adjust = close_to_boundary & (
            insuf_progress | 
            (t_new >= bracket_max) | 
            (t_new <= bracket_min)
        )

        if need_adjust.any():
            # Move 0.1*eps away from nearest boundary
            dist_to_max = (t_new - bracket_max).abs()
            dist_to_min = (t_new - bracket_min).abs()
            closer_to_max = dist_to_max < dist_to_min

            t_new_adjusted = torch.where(
                closer_to_max, 
                bracket_max - eps, 
                bracket_min + eps
            )
            t_new[need_adjust] = t_new_adjusted[need_adjust]
            insuf_progress[need_adjust] = False

        # Mark insufficient progress for systems close but not adjusted
        mark_insuf = close_to_boundary & ~need_adjust
        insuf_progress[mark_insuf] = True
        insuf_progress[~close_to_boundary] = False

        # Evaluate new points for active systems
        x_new[active] = x[active] + t_new[active].unsqueeze(1) * d[active]
        f_new_batch, g_new_batch = obj_func(x_new[active], False)
        f_new[active] = f_new_batch
        g_new[active] = g_new_batch
        ls_func_evals[active] += 1
        gtd_new[active] = (g_new[active] * d[active]).sum(dim=1)

        ls_iter += 1

        # Update brackets based on Wolfe conditions
        low_f = bracket_f[torch.arange(M, device=device), low_pos]
        todo1 = active & (
            (f_new > (f + c1 * t_new * gtd)) | 
            (f_new >= low_f)
        )

        # Update high bracket for systems that fail Armijo
        if todo1.any():
            hp_idx = high_pos[todo1].unsqueeze(1)  # (num_todo1, 1)
            idx_expanded = torch.where(todo1)[0].unsqueeze(1)  # (num_todo1, 1)
            scatter_idx = torch.cat([idx_expanded, hp_idx], dim=1)  # (num_todo1, 2)

            bracket_t[scatter_idx[:, 0], scatter_idx[:, 1]] = t_new[todo1]
            bracket_f[scatter_idx[:, 0], scatter_idx[:, 1]] = f_new[todo1]
            bracket_g[scatter_idx[:, 0], scatter_idx[:, 1]] = g_new[todo1]
            bracket_gtd[scatter_idx[:, 0], scatter_idx[:, 1]] = gtd_new[todo1]

            # Update low/high positions
            idx = todo1
            m01 = bracket_f[:, 0] <= bracket_f[:, 1]
            low_pos[idx] = torch.where(m01[idx], 0, 1)
            high_pos[idx] = 1 - low_pos[idx]

        todo2 = active & ~todo1

        # Check if Wolfe conditions are satisfied
        wolfe_satisfied = todo2 & (gtd_new.abs() <= -c2 * gtd)
        if wolfe_satisfied.any():
            done[wolfe_satisfied] = True
            active[wolfe_satisfied] = False

        # Update low bracket for remaining systems
        update_low = todo2 & ~wolfe_satisfied

        if update_low.any():
            lp_update = low_pos[update_low]
            hp_update = high_pos[update_low]

            # Check if old high should become new low (swap needed)
            bracket_t_diff = (
                bracket_t[update_low].gather(1, hp_update.unsqueeze(1)).squeeze(1) -
                bracket_t[update_low].gather(1, lp_update.unsqueeze(1)).squeeze(1)
            )

            swap_needed = gtd_new[update_low] * bracket_t_diff >= 0

            if swap_needed.any():
                swap_mask = torch.zeros(M, dtype=torch.bool, device=device)
                swap_mask[update_low] = swap_needed

                lp_swap = low_pos[swap_mask]
                hp_swap = high_pos[swap_mask]

                # Copy low to high for swapped systems
                for m_idx, m in enumerate(torch.where(swap_mask)[0]):
                    lp = lp_swap[m_idx].item()
                    hp = hp_swap[m_idx].item()
                    bracket_t[m, hp] = bracket_t[m, lp]
                    bracket_f[m, hp] = bracket_f[m, lp]
                    bracket_g[m, hp] = bracket_g[m, lp]
                    bracket_gtd[m, hp] = bracket_gtd[m, lp]

        # Update low bracket with new point for all todo2 systems
        idx = todo2
        if idx.any():
            lp = low_pos[idx]
            bracket_t[idx, lp] = t_new[idx]
            bracket_f[idx, lp] = f_new[idx]
            bracket_g[idx, lp] = g_new[idx]
            bracket_gtd[idx, lp] = gtd_new[idx]

    # ========== Extract final results ==========
    size_1_mask = bracket_size == 1
    size_2_mask = ~size_1_mask

    # For size == 1 (already satisfied Wolfe), use position 0
    if size_1_mask.any():
        t[size_1_mask] = bracket_t[size_1_mask, 0]
        f_new[size_1_mask] = bracket_f[size_1_mask, 0]
        g_new[size_1_mask] = bracket_g[size_1_mask, 0]

    # For size == 2 (bracketed), use low_pos position
    if size_2_mask.any():
        low_idx = low_pos[size_2_mask].unsqueeze(1)
        t[size_2_mask] = bracket_t[size_2_mask].gather(1, low_idx).squeeze(1)
        f_new[size_2_mask] = bracket_f[size_2_mask].gather(1, low_idx).squeeze(1)
        
        low_idx_expanded = low_idx.unsqueeze(2).expand(-1, -1, g_new.shape[1])
        g_new[size_2_mask] = bracket_g[size_2_mask].gather(
            1, 
            low_idx_expanded
        ).squeeze(1)

    return f_new, g_new, t, ls_func_evals



def lbfgs(
    func,
    x0: Tensor,
    lr: float = 1.0,
    max_iter: int = 20,
    max_eval: Optional[int] = None,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    tolerance_rel_change: float = 0.,
    stop_on_cost_increase: bool = False,
    history_size: int = 100,
    line_search_fn: Optional[str] = None,
    max_step: Optional[float] = 0.5,
):
    """Minimize multiple objective functions using batched L-BFGS algorithm.

    This function implements the Limited-memory BFGS optimization algorithm
    for M independent optimization problems in parallel. It supports optional
    strong Wolfe line search and step size clamping for constrained problems.

    The algorithm maintains a limited history of gradient differences to
    approximate the inverse Hessian, using the two-loop recursion formula.
    All operations are vectorized across the M systems.

    Args:
        func: Objective function taking x (M, N) and returning (cost, gradient).
              cost must have shape (M,), gradient must have shape (M, N).
        x0: Initial parameters, shape (M, N) where M is batch size, N is 
            parameters per system. Modified in-place with final result.
        lr: Learning rate / initial step size (default: 1.0)
        max_iter: Maximum optimization iterations (default: 20)
        max_eval: Maximum function evaluations per system (default: max_iter * 1.25)
        tolerance_grad: Terminate if max gradient element < this (default: 1e-7)
        tolerance_change: Terminate if parameter/loss change < this (default: 1e-9)
        tolerance_rel_change: Terminate if relative loss change < this (default: 0.)
        stop_on_cost_increase: Terminate if cost increases (default: False)
        history_size: Number of gradient pairs to store (default: 100)
        line_search_fn: Line search method: 'strong_wolfe' or None (default: None)
        max_step: Maximum parameter change per step, None to disable (default: 0.5)

    Returns:
        Optimized parameters (same tensor as x0, modified in-place)

    Example:
        >>> # Optimize 2 systems with 3 parameters each
        >>> x = torch.tensor([[1.5, 2.0, 1.0], [3.0, 4.0, 2.0]])
        >>> def rosenbrock(x):
        ...     cost = (100 * (x[:, 1] - x[:, 0]**2)**2 + 
        ...             (1 - x[:, 0])**2).sum(dim=1)
        ...     # ... compute gradient ...
        ...     return cost, grad
        >>> result = lbfgs(rosenbrock, x, max_iter=100)
    """
    if max_eval is None:
        max_eval = max_iter * 5 // 4

    M, N = x0.shape  # M systems, N parameters per system
    device = x0.device
    dtype = x0.dtype

    # Working copy of parameters
    x = x0.detach().clone()

    # Evaluate initial function and gradient
    loss, flat_grad = func(x, True)  # loss: (M,), flat_grad: (M, N)
    func_evals = torch.ones(M, dtype=torch.long, device=device)

    # Track which systems are still optimizing
    active = torch.ones(M, dtype=torch.bool, device=device)

    # Check if any already converged
    grad_norm = flat_grad.abs().max(dim=1).values  # (M,)
    converged = grad_norm <= tolerance_grad
    active &= ~converged

    if not active.any():
        x0.copy_(x)
        return x0

    # ========== Initialize history buffers ==========
    # These store gradient/step differences for L-BFGS approximation
    old_dirs = torch.zeros(M, history_size, N, dtype=dtype, device=device)  # y vectors
    old_stps = torch.zeros(M, history_size, N, dtype=dtype, device=device)  # s vectors
    ro = torch.zeros(M, history_size, dtype=dtype, device=device)  # 1/(y·s) values
    hist_len = torch.zeros(M, dtype=torch.long, device=device)  # Actual history length
    H_diag = torch.ones(M, dtype=dtype, device=device)  # Diagonal Hessian approximation

    # ========== Pre-allocate reusable buffers ==========
    # These eliminate allocations in the hot loop for better performance
    hist_arange = torch.arange(history_size, device=device)
    hist_mask = torch.zeros(M, history_size, dtype=torch.bool, device=device)
    q_buffer = torch.zeros(M, N, dtype=dtype, device=device)
    r_buffer = torch.zeros(M, N, dtype=dtype, device=device)
    al_buffer = torch.zeros(M, history_size, dtype=dtype, device=device)
    prev_flat_grad = torch.zeros(M, N, dtype=dtype, device=device)
    prev_loss_iter = torch.zeros(M, dtype=dtype, device=device)
    t_buffer = torch.zeros(M, dtype=dtype, device=device)
    grad_sum_buffer = torch.zeros(M, dtype=dtype, device=device)
    ones_M_buffer = torch.ones(M, dtype=dtype, device=device)
    ls_evals_buffer = torch.zeros(M, dtype=torch.long, device=device)
    y_buffer = torch.zeros(M, N, dtype=dtype, device=device)
    s_buffer = torch.zeros(M, N, dtype=dtype, device=device)
    ys_buffer = torch.zeros(M, dtype=dtype, device=device)
    param_change_buffer = torch.zeros(M, dtype=dtype, device=device)
    loss_change_buffer = torch.zeros(M, dtype=dtype, device=device)
    append_mask_buffer = torch.zeros(M, dtype=torch.bool, device=device)
    shift_mask_buffer = torch.zeros(M, dtype=torch.bool, device=device)

    n_iter = 0
    prev_loss = loss.clone()

    # ========== Main optimization loop ==========
    while n_iter < max_iter and active.any():
        n_iter += 1

        # Compute search direction using L-BFGS two-loop recursion
        if n_iter == 1:
            # First iteration: use steepest descent
            d = -flat_grad.clone()
            H_diag.fill_(1.0)
        else:
            # L-BFGS two-loop recursion for search direction
            
            # Create mask for valid history entries
            torch.less(hist_arange.unsqueeze(0), hist_len.unsqueeze(1), out=hist_mask)

            # Initialize q = -gradient
            torch.neg(flat_grad, out=q_buffer)
            q = q_buffer

            # Storage for alpha values
            al_buffer.zero_()
            al = al_buffer

            # First loop: backward through history (compute alphas)
            for i in range(history_size - 1, -1, -1):
                mask_i = hist_mask[:, i]  # Systems with history at position i

                if mask_i.any():
                    # alpha[i] = rho[i] * s[i]^T * q
                    al[:, i] = (old_stps[:, i] * q).sum(dim=1) * ro[:, i]

                    # q = q - alpha[i] * y[i]
                    q = q - old_dirs[:, i] * al[:, i].unsqueeze(1) * mask_i.unsqueeze(1)

            # Multiply by initial Hessian approximation
            torch.mul(q, H_diag.unsqueeze(1), out=r_buffer)
            r = r_buffer

            # Second loop: forward through history (compute direction)
            for i in range(history_size):
                mask_i = hist_mask[:, i]

                if mask_i.any():
                    # beta[i] = rho[i] * y[i]^T * r
                    be_i = (old_dirs[:, i] * r).sum(dim=1) * ro[:, i]

                    # r = r + (alpha[i] - beta[i]) * s[i]
                    update = old_stps[:, i] * (al[:, i] - be_i).unsqueeze(1)
                    r = r + update * mask_i.unsqueeze(1)

            d = r

        # Store previous gradient and loss for history update
        prev_flat_grad.copy_(flat_grad)
        prev_loss_iter.copy_(loss)

        # Compute initial step size
        if n_iter == 1:
            # First iteration: adaptive step based on gradient magnitude
            grad_sum_buffer[:] = flat_grad.abs().sum(dim=1)
            t_buffer.copy_(ones_M_buffer)
            torch.div(t_buffer, grad_sum_buffer, out=t_buffer)
            torch.minimum(ones_M_buffer, t_buffer, out=t_buffer)
            t_buffer.mul_(lr)
            t_buffer[~active] = 0  # Zero out inactive systems
            t = t_buffer
        else:
            t_buffer.fill_(lr)
            t = t_buffer

        # Apply trust region constraint (clamp maximum step size)
        if max_step is not None:
            param_change_buffer[:] = (d * t.unsqueeze(1)).abs().max(dim=1).values
            needs_scaling = param_change_buffer > max_step
            
            if needs_scaling.any():
                scale_factor = max_step / param_change_buffer[needs_scaling]
                t[needs_scaling] *= scale_factor

        # Check directional derivative for convergence
        gtd = (flat_grad * d).sum(dim=1)  # (M,)
        early_stop = gtd > -tolerance_change
        active &= ~early_stop

        if not active.any():
            break

        # ========== Perform line search ==========
        if line_search_fn == 'strong_wolfe':
            # Strong Wolfe line search for active systems
            loss_active, flat_grad_active, t_active, ls_evals_active = \
                _strong_wolfe_batched(
                    func, x[active], t[active], d[active],
                    loss[active], flat_grad[active], gtd[active]
                )
            
            loss[active] = loss_active
            flat_grad[active] = flat_grad_active
            t[active] = t_active
            ls_evals_buffer.zero_()
            ls_evals_buffer[active] = ls_evals_active
            x[active] = x[active] + t[active].unsqueeze(1) * d[active]
        else:
            # Simple gradient descent step
            x = x + t.unsqueeze(1) * d
            
            if n_iter != max_iter:
                loss, flat_grad = func(x, True)
            
            ls_evals_buffer.fill_(1)

        func_evals += ls_evals_buffer

        # ========== Update L-BFGS history ==========
        if n_iter > 1:
            # Compute gradient and step differences
            torch.sub(flat_grad, prev_flat_grad, out=y_buffer)  # y = grad_new - grad_old
            torch.mul(d, t.unsqueeze(1), out=s_buffer)  # s = t * d
            y = y_buffer
            s = s_buffer

            # Compute y^T s for curvature condition
            ys_buffer[:] = (y * s).sum(dim=1)
            ys = ys_buffer

            # Only update history if curvature condition is satisfied
            update_mask = active & (ys > 1e-10)

            if update_mask.any():
                # Update diagonal Hessian approximation: H0 = (y^T s) / (y^T y)
                H_diag[update_mask] = (
                    ys[update_mask] / 
                    (y[update_mask] * y[update_mask]).sum(dim=1)
                )

                curr_len = hist_len[update_mask]

                # Systems with room to append (not at history limit)
                append_mask_buffer.copy_(update_mask)
                append_mask_buffer[update_mask] = curr_len < history_size
                append_mask = append_mask_buffer

                # Systems that need to shift (at history limit)
                torch.logical_and(update_mask, ~append_mask, out=shift_mask_buffer)
                shift_mask = shift_mask_buffer

                # Append new history entry for systems with room
                if append_mask.any():
                    append_positions = hist_len[append_mask]
                    append_indices = torch.where(append_mask)[0]
                    scatter_idx = torch.stack([append_indices, append_positions], dim=1)

                    old_dirs[scatter_idx[:, 0], scatter_idx[:, 1]] = y[append_mask]
                    old_stps[scatter_idx[:, 0], scatter_idx[:, 1]] = s[append_mask]
                    ro[scatter_idx[:, 0], scatter_idx[:, 1]] = 1.0 / ys[append_mask]

                    hist_len[append_mask] += 1

                # Shift history and append for systems at limit
                if shift_mask.any():
                    # Shift all entries left by 1
                    old_dirs[shift_mask, :-1] = old_dirs[shift_mask, 1:].clone()
                    old_stps[shift_mask, :-1] = old_stps[shift_mask, 1:].clone()
                    ro[shift_mask, :-1] = ro[shift_mask, 1:].clone()

                    # Add new entry at end
                    old_dirs[shift_mask, history_size - 1] = y[shift_mask]
                    old_stps[shift_mask, history_size - 1] = s[shift_mask]
                    ro[shift_mask, history_size - 1] = 1.0 / ys[shift_mask]

        # ========== Check convergence conditions ==========
        
        # Gradient convergence
        grad_norm = flat_grad.abs().max(dim=1).values  # (M,)
        converged_grad = grad_norm <= tolerance_grad
        active &= ~converged_grad

        # Function evaluation limit
        converged_eval = func_evals >= max_eval
        active &= ~converged_eval

        # Parameter change convergence
        torch.mul(d, t.unsqueeze(1), out=s_buffer)
        param_change_buffer[:] = s_buffer.abs().max(dim=1).values
        converged_param = (param_change_buffer <= tolerance_change) & active
        active &= ~converged_param

        # Loss change convergence
        torch.sub(loss, prev_loss_iter, out=loss_change_buffer)
        loss_change_buffer.abs_()
        converged_loss = (loss_change_buffer < tolerance_change) & active
        active &= ~converged_loss

        # Relative loss change convergence
        if tolerance_rel_change > 0 and n_iter > history_size:
            rel_change = loss_change_buffer / (prev_loss_iter.abs() + 1e-12)
            converged_rel = (rel_change < tolerance_rel_change) & active
            active &= ~converged_rel

        # Stop on cost increase
        if stop_on_cost_increase:
            cost_increase = (loss > prev_loss) & active
            active &= ~cost_increase

    # Copy final result to input tensor
    x0.copy_(x)

    return x0