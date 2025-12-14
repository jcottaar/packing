# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import Tensor

import kaggle_support as kgs


__all__ = ["lbfgs"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    import math
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = math.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe_batched(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    """Batched strong Wolfe line search.

    Args:
        obj_func: Function that takes x (M, N) and returns (f, g) with shapes (M,) and (M, N)
        x: (M, N) current parameters
        t: (M,) initial step sizes
        d: (M, N) search directions
        f: (M,) current function values
        g: (M, N) current gradients
        gtd: (M,) directional derivatives

    Returns:
        f_new: (M,) new function values
        g_new: (M, N) new gradients
        t: (M,) final step sizes
        ls_func_evals: (M,) number of function evaluations per system
    """
    M = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Initialize tracking for each system
    d_norm = d.abs().max(dim=1).values  # (M,)
    g = g.clone()

    # Evaluate initial step for all systems
    x_new = x + t.unsqueeze(1) * d
    f_new, g_new = obj_func(x_new)
    ls_func_evals = torch.ones(M, dtype=torch.long, device=device)
    gtd_new = (g_new * d).sum(dim=1)  # (M,)

    # Track active systems (still searching)
    active = torch.ones(M, dtype=torch.bool, device=device)

    # Bracketing phase state (per system)
    t_prev = torch.zeros(M, dtype=dtype, device=device)
    f_prev = f.clone()
    g_prev = g.clone()
    gtd_prev = gtd.clone()

    # Bracket state: use tensors for values that all systems have
    bracket_t = torch.zeros((M, 2), dtype=dtype, device=device)
    bracket_f = torch.zeros((M, 2), dtype=dtype, device=device)
    bracket_g = torch.zeros((M, 2, x.shape[1]), dtype=dtype, device=device)
    bracket_gtd = torch.zeros((M, 2), dtype=dtype, device=device)
    bracket_size = torch.zeros(M, dtype=torch.long, device=device)  # 0, 1, or 2

    done = torch.zeros(M, dtype=torch.bool, device=device)
    ls_iter = 0

    # Bracketing phase
    while active.any() and ls_iter < max_ls:
        # Check Armijo condition
        armijo_fail = f_new > (f + c1 * t * gtd)
        not_decreasing = (ls_iter > 1) & (f_new >= f_prev)
        bracket_condition = armijo_fail | not_decreasing

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

        # Check Wolfe conditions
        wolfe_satisfied = (gtd_new.abs() <= -c2 * gtd) & active
        if wolfe_satisfied.any():
            idx = wolfe_satisfied
            bracket_size[idx] = 1
            bracket_t[idx, 0] = t[idx]
            bracket_f[idx, 0] = f_new[idx]
            bracket_g[idx, 0] = g_new[idx]
            done[idx] = True
            active[idx] = False

        # Check positive curvature
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

        # Interpolate for active systems
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10

        # Cubic interpolation (vectorized over active systems)
        for m in range(M):
            if active[m]:
                t[m] = _cubic_interpolate(
                    t_prev[m].item(), f_prev[m].item(), gtd_prev[m].item(),
                    t[m].item(), f_new[m].item(), gtd_new[m].item(),
                    bounds=(min_step[m].item(), max_step[m].item())
                )

        # Update previous state
        t_prev = t.clone()
        f_prev = f_new.clone()
        g_prev = g_new.clone()
        gtd_prev = gtd_new.clone()

        # Evaluate new points for active systems
        x_new[active] = x[active] + t[active].unsqueeze(1) * d[active]
        f_new_batch, g_new_batch = obj_func(x_new[active])
        f_new[active] = f_new_batch
        g_new[active] = g_new_batch
        ls_func_evals[active] += 1
        gtd_new[active] = (g_new[active] * d[active]).sum(dim=1)

        ls_iter += 1

    # Systems that hit max_ls without bracketing
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

    # Zoom phase
    active = (bracket_size == 2) & ~done  # Only zoom for bracketed, not done systems
    insuf_progress = torch.zeros(M, dtype=torch.bool, device=device)

    # Find low/high positions in bracket
    low_pos = (bracket_f[:, 0] <= bracket_f[:, 1]).long()
    high_pos = 1 - low_pos

    while active.any() and ls_iter < max_ls:
        # Check bracket size
        bracket_width = (bracket_t[:, 1] - bracket_t[:, 0]).abs()
        too_small = (bracket_width * d_norm < tolerance_change) & active
        active[too_small] = False

        if not active.any():
            break

        # Compute new trial value via cubic interpolation
        t_new = torch.zeros(M, dtype=dtype, device=device)
        for m in range(M):
            if active[m]:
                t_new[m] = _cubic_interpolate(
                    bracket_t[m, 0].item(), bracket_f[m, 0].item(), bracket_gtd[m, 0].item(),
                    bracket_t[m, 1].item(), bracket_f[m, 1].item(), bracket_gtd[m, 1].item()
                )

        # Check progress
        eps = 0.1 * bracket_width
        dist_to_bounds = torch.stack([
            bracket_t.max(dim=1).values - t_new,
            t_new - bracket_t.min(dim=1).values
        ], dim=1).min(dim=1).values

        close_to_boundary = (dist_to_bounds < eps) & active

        for m in range(M):
            if close_to_boundary[m]:
                if insuf_progress[m] or t_new[m] >= bracket_t[m].max() or t_new[m] <= bracket_t[m].min():
                    # Move to 0.1 away from nearest boundary
                    if abs(t_new[m] - bracket_t[m].max()) < abs(t_new[m] - bracket_t[m].min()):
                        t_new[m] = bracket_t[m].max() - eps[m]
                    else:
                        t_new[m] = bracket_t[m].min() + eps[m]
                    insuf_progress[m] = False
                else:
                    insuf_progress[m] = True
            else:
                insuf_progress[m] = False

        # Evaluate new points for active systems
        x_new[active] = x[active] + t_new[active].unsqueeze(1) * d[active]
        f_new_batch, g_new_batch = obj_func(x_new[active])
        f_new[active] = f_new_batch
        g_new[active] = g_new_batch
        ls_func_evals[active] += 1
        gtd_new[active] = (g_new[active] * d[active]).sum(dim=1)

        ls_iter += 1

        # Update brackets based on conditions
        for m in range(M):
            if not active[m]:
                continue

            lp = low_pos[m].item()
            hp = high_pos[m].item()

            if f_new[m] > (f[m] + c1 * t_new[m] * gtd[m]) or f_new[m] >= bracket_f[m, lp]:
                # Update high bracket
                bracket_t[m, hp] = t_new[m]
                bracket_f[m, hp] = f_new[m]
                bracket_g[m, hp] = g_new[m]
                bracket_gtd[m, hp] = gtd_new[m]
                # Recompute low/high
                if bracket_f[m, 0] <= bracket_f[m, 1]:
                    low_pos[m] = 0
                    high_pos[m] = 1
                else:
                    low_pos[m] = 1
                    high_pos[m] = 0
            else:
                if gtd_new[m].abs() <= -c2 * gtd[m]:
                    # Wolfe conditions satisfied
                    done[m] = True
                    active[m] = False
                elif gtd_new[m] * (bracket_t[m, hp] - bracket_t[m, lp]) >= 0:
                    # Old high becomes new low
                    bracket_t[m, hp] = bracket_t[m, lp]
                    bracket_f[m, hp] = bracket_f[m, lp]
                    bracket_g[m, hp] = bracket_g[m, lp]
                    bracket_gtd[m, hp] = bracket_gtd[m, lp]

                # New point becomes new low
                bracket_t[m, lp] = t_new[m]
                bracket_f[m, lp] = f_new[m]
                bracket_g[m, lp] = g_new[m]
                bracket_gtd[m, lp] = gtd_new[m]

    # Extract final results from brackets
    for m in range(M):
        if bracket_size[m] == 1:
            t[m] = bracket_t[m, 0]
            f_new[m] = bracket_f[m, 0]
            g_new[m] = bracket_g[m, 0]
        else:
            lp = low_pos[m].item()
            t[m] = bracket_t[m, lp]
            f_new[m] = bracket_f[m, lp]
            g_new[m] = bracket_g[m, lp]

    return f_new, g_new, t, ls_func_evals


@kgs.profile_each_line
def lbfgs(
    func,
    x0: Tensor,
    lr: float = 1.0,
    max_iter: int = 20,
    max_eval: Optional[int] = None,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    history_size: int = 100,
    line_search_fn: Optional[str] = None,
):
    """Minimize a function using L-BFGS algorithm (batched version).

    A batched functional interface to L-BFGS optimization.

    Args:
        func: Callable that takes x (Tensor MxN) and returns (cost, gradient) tuple.
              cost should be (M,) and gradient should be (M, N).
        x0: Initial parameter tensor (M, N) where M is batch size, N is parameters per system.
            Will be modified in-place.
        lr: Learning rate (default: 1.0)
        max_iter: Maximum number of iterations per optimization (default: 20)
        max_eval: Maximum number of function evaluations (default: max_iter * 1.25)
        tolerance_grad: Termination tolerance on gradient norm (default: 1e-7)
        tolerance_change: Termination tolerance on function/parameter changes (default: 1e-9)
        history_size: Number of previous gradients to store (default: 100)
        line_search_fn: Line search method, either 'strong_wolfe' or None (default: None)

    Returns:
        Optimized parameters (same tensor as x0, modified in-place)

    Example:
        >>> x = torch.tensor([[1.5, 2.0], [3.0, 4.0]])  # 2 systems, 2 params each
        >>> def f(x):
        ...     cost = (x ** 2).sum(dim=1)  # shape (M,)
        ...     grad = 2 * x  # shape (M, N)
        ...     return cost, grad
        >>> result = lbfgs(f, x)
    """
    if max_eval is None:
        max_eval = max_iter * 5 // 4

    M, N = x0.shape
    device = x0.device
    dtype = x0.dtype

    # Working copy of parameters (M, N)
    x = x0.detach().clone()

    # Evaluate initial function and gradient
    loss, flat_grad = func(x)  # loss: (M,), flat_grad: (M, N)
    func_evals = torch.ones(M, dtype=torch.long, device=device)

    # Track which systems are still active
    active = torch.ones(M, dtype=torch.bool, device=device)

    # Check if any already optimal
    grad_norm = flat_grad.abs().max(dim=1).values  # (M,)
    converged = grad_norm <= tolerance_grad
    active &= ~converged

    if not active.any():
        x0.copy_(x)
        return x0

    # Initialize history for each system (lists of tensors)
    old_dirs = [[] for _ in range(M)]  # List of M lists, each containing history
    old_stps = [[] for _ in range(M)]
    ro = [[] for _ in range(M)]
    H_diag = torch.ones(M, dtype=dtype, device=device)

    n_iter = 0
    prev_loss = loss.clone()

    while n_iter < max_iter and active.any():
        n_iter += 1

        # Compute search direction for all systems
        d = torch.zeros_like(flat_grad)

        for m in range(M):
            if not active[m]:
                continue

            if n_iter == 1:
                # Steepest descent direction
                d[m] = -flat_grad[m]
                H_diag[m] = 1.0
            else:
                # L-BFGS two-loop recursion
                q = -flat_grad[m]
                num_old = len(old_dirs[m])
                al = [None] * num_old

                # First loop
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[m][i].dot(q) * ro[m][i]
                    q = q + old_dirs[m][i] * (-al[i])

                # Multiply by initial Hessian approximation
                r = q * H_diag[m]

                # Second loop
                for i in range(num_old):
                    be_i = old_dirs[m][i].dot(r) * ro[m][i]
                    r = r + old_stps[m][i] * (al[i] - be_i)

                d[m] = r

        # Store previous gradient and loss
        prev_flat_grad = flat_grad.clone()
        prev_loss_iter = loss.clone()

        # Compute step length
        if n_iter == 1:
            # Vectorized: t = min(1.0, 1.0 / ||g||_1) * lr for each system
            grad_sum = flat_grad.abs().sum(dim=1)  # (M,)
            t = torch.minimum(torch.ones_like(grad_sum), 1.0 / grad_sum) * lr
            t[~active] = 0  # Zero out inactive systems
        else:
            t = torch.full((M,), lr, dtype=dtype, device=device)

        # Check directional derivative and mark converged
        gtd = (flat_grad * d).sum(dim=1)  # (M,)
        early_stop = gtd > -tolerance_change
        active &= ~early_stop

        if not active.any():
            break

        # Line search (batched for all active systems)
        if line_search_fn == 'strong_wolfe':
            # Batched strong Wolfe line search
            loss_active, flat_grad_active, t_active, ls_evals_active = _strong_wolfe_batched(
                func, x[active], t[active], d[active],
                loss[active], flat_grad[active], gtd[active]
            )
            loss[active] = loss_active
            flat_grad[active] = flat_grad_active
            t[active] = t_active
            ls_evals = torch.zeros(M, dtype=torch.long, device=device)
            ls_evals[active] = ls_evals_active
            x[active] = x[active] + t[active].unsqueeze(1) * d[active]
        else:
            # Simple step
            x = x + t.unsqueeze(1) * d
            if n_iter != max_iter:
                loss, flat_grad = func(x)
            ls_evals = torch.ones(M, dtype=torch.long, device=device)

        func_evals += ls_evals

        # Update history for active systems
        if n_iter > 1:
            for m in range(M):
                if not active[m]:
                    continue

                y = flat_grad[m] - prev_flat_grad[m]
                s = d[m] * t[m]
                ys = y.dot(s)

                if ys > 1e-10:
                    if len(old_dirs[m]) == history_size:
                        old_dirs[m].pop(0)
                        old_stps[m].pop(0)
                        ro[m].pop(0)

                    old_dirs[m].append(y)
                    old_stps[m].append(s)
                    ro[m].append(1.0 / ys.item())
                    H_diag[m] = ys / y.dot(y)

        # Check convergence for each system
        grad_norm = flat_grad.abs().max(dim=1).values  # (M,)
        converged_grad = grad_norm <= tolerance_grad
        active &= ~converged_grad

        converged_eval = func_evals >= max_eval
        active &= ~converged_eval

        # Vectorized convergence checks
        param_change = (d * t.unsqueeze(1)).abs().max(dim=1).values  # (M,)
        converged_param = (param_change <= tolerance_change) & active
        active &= ~converged_param

        loss_change = (loss - prev_loss_iter).abs()  # (M,)
        converged_loss = (loss_change < tolerance_change) & active
        active &= ~converged_loss

    # Update x0 with final result
    x0.copy_(x)

    return x0