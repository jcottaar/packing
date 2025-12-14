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


def _cubic_interpolate_batch(x1, f1, g1, x2, f2, g2, xmin_bound, xmax_bound):
    """Batched cubic interpolation.

    Args:
        x1, f1, g1, x2, f2, g2: Tensors of shape (M,)
        xmin_bound, xmax_bound: Tensors of shape (M,)

    Returns:
        Tensor of shape (M,) with interpolated values
    """
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2

    # Where d2_square >= 0, use cubic interpolation
    valid = d2_square >= 0
    d2 = torch.sqrt(torch.clamp(d2_square, min=0))

    # Compute min_pos based on x1 <= x2
    x1_le_x2 = x1 <= x2
    min_pos_1 = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    min_pos_2 = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
    min_pos = torch.where(x1_le_x2, min_pos_1, min_pos_2)

    # Clamp to bounds
    result_valid = torch.clamp(min_pos, min=xmin_bound, max=xmax_bound)
    result_invalid = (xmin_bound + xmax_bound) / 2.0

    return torch.where(valid, result_valid, result_invalid)

@kgs.profile_each_line
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
        if active.any():
            xmin = bracket_t.min(dim=1).values
            xmax = bracket_t.max(dim=1).values
            t_new[active] = _cubic_interpolate_batch(
                bracket_t[active, 0], bracket_f[active, 0], bracket_gtd[active, 0],
                bracket_t[active, 1], bracket_f[active, 1], bracket_gtd[active, 1],
                xmin[active], xmax[active]
            )

        # Check progress
        eps = 0.1 * bracket_width
        dist_to_bounds = torch.stack([
            bracket_t.max(dim=1).values - t_new,
            t_new - bracket_t.min(dim=1).values
        ], dim=1).min(dim=1).values

        close_to_boundary = (dist_to_bounds < eps) & active

        # Vectorized insufficient progress adjustment
        bracket_min = bracket_t.min(dim=1).values
        bracket_max = bracket_t.max(dim=1).values

        # Check if we need to adjust (insufficient progress or out of bounds)
        need_adjust = close_to_boundary & (insuf_progress | (t_new >= bracket_max) | (t_new <= bracket_min))

        if need_adjust.any():
            # Determine which boundary is closer
            dist_to_max = (t_new - bracket_max).abs()
            dist_to_min = (t_new - bracket_min).abs()
            closer_to_max = dist_to_max < dist_to_min

            # Adjust t_new: move 0.1*eps away from nearest boundary
            t_new_adjusted = torch.where(closer_to_max, bracket_max - eps, bracket_min + eps)
            t_new[need_adjust] = t_new_adjusted[need_adjust]
            insuf_progress[need_adjust] = False

        # Mark insufficient progress for close systems that didn't need adjustment
        mark_insuf = close_to_boundary & ~need_adjust
        insuf_progress[mark_insuf] = True

        # Clear insufficient progress for systems not close to boundary
        insuf_progress[~close_to_boundary] = False

        # Evaluate new points for active systems
        x_new[active] = x[active] + t_new[active].unsqueeze(1) * d[active]
        f_new_batch, g_new_batch = obj_func(x_new[active])
        f_new[active] = f_new_batch
        g_new[active] = g_new_batch
        ls_func_evals[active] += 1
        gtd_new[active] = (g_new[active] * d[active]).sum(dim=1)

        ls_iter += 1

        # Update brackets based on conditions
        todo1 = active & (   (f_new > (f + c1 * t_new * gtd)) | (f_new >= bracket_f[torch.arange(M, device=device), low_pos])   )

        # Vectorized: Update high bracket for todo1 systems
        if todo1.any():
            hp_idx = high_pos[todo1].unsqueeze(1)  # (num_todo1, 1)
            idx_expanded = torch.where(todo1)[0].unsqueeze(1)  # (num_todo1, 1)

            # Create full indices for scatter
            scatter_idx = torch.cat([idx_expanded, hp_idx], dim=1)  # (num_todo1, 2)

            # Update bracket_t, bracket_f, bracket_gtd using advanced indexing
            bracket_t[scatter_idx[:, 0], scatter_idx[:, 1]] = t_new[todo1]
            bracket_f[scatter_idx[:, 0], scatter_idx[:, 1]] = f_new[todo1]
            bracket_g[scatter_idx[:, 0], scatter_idx[:, 1]] = g_new[todo1]
            bracket_gtd[scatter_idx[:, 0], scatter_idx[:, 1]] = gtd_new[todo1]
            # Vectorized: Update low/high positions for non-todo1 systems

            # mask where we actually update
            idx = todo1

            # among those, True means (0 <= 1) so low=0/high=1 else low=1/high=0
            m01 = bracket_f[:, 0] <= bracket_f[:, 1]

            low_pos[idx]  = torch.where(m01[idx], 0, 1)
            high_pos[idx] = 1 - low_pos[idx]

        todo2 = active & ~todo1

        # Vectorized: Check Wolfe conditions for todo2 systems
        wolfe_satisfied = todo2 & (gtd_new.abs() <= -c2 * gtd)
        if wolfe_satisfied.any():
            done[wolfe_satisfied] = True
            active[wolfe_satisfied] = False

        # Systems that need to potentially swap and update low bracket
        update_low = todo2 & ~wolfe_satisfied

        if update_low.any():
            # Vectorized: Check if old high should become new low
            lp_update = low_pos[update_low]
            hp_update = high_pos[update_low]

            # Compute bracket_t differences
            bracket_t_diff = bracket_t[update_low].gather(1, hp_update.unsqueeze(1)).squeeze(1) - \
                             bracket_t[update_low].gather(1, lp_update.unsqueeze(1)).squeeze(1)

            # Determine which systems need swap
            swap_needed = gtd_new[update_low] * bracket_t_diff >= 0

            if swap_needed.any():
                # Create mask for all M systems
                swap_mask = torch.zeros(M, dtype=torch.bool, device=device)
                swap_mask[update_low] = swap_needed

                # Copy low to high for swap_mask systems
                lp_swap = low_pos[swap_mask]
                hp_swap = high_pos[swap_mask]

                for m_idx, m in enumerate(torch.where(swap_mask)[0]):
                    lp = lp_swap[m_idx].item()
                    hp = hp_swap[m_idx].item()
                    bracket_t[m, hp] = bracket_t[m, lp]
                    bracket_f[m, hp] = bracket_f[m, lp]
                    bracket_g[m, hp] = bracket_g[m, lp]
                    bracket_gtd[m, hp] = bracket_gtd[m, lp]

        # Vectorized: Update low bracket with new point for all todo2 systems
        idx = todo2
        if idx.any():
            lp = low_pos[idx]
            bracket_t[idx, lp]   = t_new[idx]
            bracket_f[idx, lp]   = f_new[idx]
            bracket_g[idx, lp]   = g_new[idx]
            bracket_gtd[idx, lp] = gtd_new[idx]

    # Extract final results from brackets (vectorized)
    size_1_mask = bracket_size == 1
    size_2_mask = ~size_1_mask

    # For size == 1, use position 0
    if size_1_mask.any():
        t[size_1_mask] = bracket_t[size_1_mask, 0]
        f_new[size_1_mask] = bracket_f[size_1_mask, 0]
        g_new[size_1_mask] = bracket_g[size_1_mask, 0]

    # For size == 2, use low_pos position
    if size_2_mask.any():
        # Use gather to select from bracket_t, bracket_f based on low_pos
        t[size_2_mask] = bracket_t[size_2_mask].gather(1, low_pos[size_2_mask].unsqueeze(1)).squeeze(1)
        f_new[size_2_mask] = bracket_f[size_2_mask].gather(1, low_pos[size_2_mask].unsqueeze(1)).squeeze(1)
        g_new[size_2_mask] = bracket_g[size_2_mask].gather(1, low_pos[size_2_mask].unsqueeze(1).unsqueeze(2).expand(-1, -1, g_new.shape[1])).squeeze(1)

    return f_new, g_new, t, ls_func_evals


#@kgs.profile_each_line
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

    # Initialize history for each system using preallocated tensors
    # old_dirs: (M, history_size, N) - stores y vectors
    # old_stps: (M, history_size, N) - stores s vectors
    # ro: (M, history_size) - stores 1/ys values
    # hist_len: (M,) - tracks actual history length for each system (0 to history_size)
    old_dirs = torch.zeros(M, history_size, N, dtype=dtype, device=device)
    old_stps = torch.zeros(M, history_size, N, dtype=dtype, device=device)
    ro = torch.zeros(M, history_size, dtype=dtype, device=device)
    hist_len = torch.zeros(M, dtype=torch.long, device=device)
    H_diag = torch.ones(M, dtype=dtype, device=device)

    n_iter = 0
    prev_loss = loss.clone()

    while n_iter < max_iter and active.any():
        n_iter += 1

        # Compute search direction for all systems (vectorized two-loop recursion)
        if n_iter == 1:
            # Steepest descent direction for all systems
            d = -flat_grad.clone()
            H_diag.fill_(1.0)
        else:
            # L-BFGS two-loop recursion (vectorized)
            # Create mask for valid history entries: (M, history_size)
            hist_mask = torch.arange(history_size, device=device).unsqueeze(0) < hist_len.unsqueeze(1)

            # Initialize q for all systems: (M, N)
            q = -flat_grad.clone()

            # Storage for alpha values: (M, history_size)
            al = torch.zeros(M, history_size, dtype=dtype, device=device)

            # First loop (backward through history)
            for i in range(history_size - 1, -1, -1):
                # Mask for systems that have history at position i
                mask_i = hist_mask[:, i]  # (M,)

                if mask_i.any():
                    # Compute alpha[i] = s[i]^T * q * ro[i] for all systems
                    # old_stps[:, i]: (M, N), q: (M, N)
                    al[:, i] = (old_stps[:, i] * q).sum(dim=1) * ro[:, i]  # (M,)

                    # Update q = q - alpha[i] * y[i], but only for systems with history at i
                    # old_dirs[:, i]: (M, N), al[:, i]: (M,)
                    q = q - old_dirs[:, i] * al[:, i].unsqueeze(1) * mask_i.unsqueeze(1)

            # Multiply by initial Hessian approximation: r = H_diag * q
            r = q * H_diag.unsqueeze(1)  # (M, N)

            # Second loop (forward through history)
            for i in range(history_size):
                # Mask for systems that have history at position i
                mask_i = hist_mask[:, i]  # (M,)

                if mask_i.any():
                    # Compute beta[i] = y[i]^T * r * ro[i] for all systems
                    be_i = (old_dirs[:, i] * r).sum(dim=1) * ro[:, i]  # (M,)

                    # Update r = r + (alpha[i] - beta[i]) * s[i], but only for systems with history at i
                    r = r + old_stps[:, i] * (al[:, i] - be_i).unsqueeze(1) * mask_i.unsqueeze(1)

            d = r

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

        # Update history for active systems (vectorized)
        if n_iter > 1:
            # Compute y and s for all systems: (M, N)
            y = flat_grad - prev_flat_grad
            s = d * t.unsqueeze(1)

            # Compute ys for all systems: (M,)
            ys = (y * s).sum(dim=1)

            # Mask for systems that should update history (active and ys > threshold)
            update_mask = active & (ys > 1e-10)  # (M,)

            if update_mask.any():
                # Update H_diag for systems that pass the threshold
                H_diag[update_mask] = ys[update_mask] / (y[update_mask] * y[update_mask]).sum(dim=1)

                # For systems needing update, determine if we need to shift or append
                curr_len = hist_len[update_mask]  # lengths of systems being updated

                # Systems with room to append (curr_len < history_size)
                append_mask = update_mask.clone()
                append_mask[update_mask] = curr_len < history_size

                # Systems that need to shift (curr_len == history_size)
                shift_mask = update_mask & ~append_mask

                # Handle append case (vectorized)
                if append_mask.any():
                    # Get positions to write for each system: (num_append,)
                    append_positions = hist_len[append_mask]  # positions to write at
                    append_indices = torch.where(append_mask)[0]  # system indices

                    # Create scatter indices: (num_append, 2) with [system_idx, position]
                    scatter_idx = torch.stack([append_indices, append_positions], dim=1)

                    # Write to position hist_len[m] for each system using advanced indexing
                    old_dirs[scatter_idx[:, 0], scatter_idx[:, 1]] = y[append_mask]
                    old_stps[scatter_idx[:, 0], scatter_idx[:, 1]] = s[append_mask]
                    ro[scatter_idx[:, 0], scatter_idx[:, 1]] = 1.0 / ys[append_mask]

                    hist_len[append_mask] += 1

                # Handle shift case - need to shift left and add at end
                if shift_mask.any():
                    # Shift all entries left by 1
                    old_dirs[shift_mask, :-1] = old_dirs[shift_mask, 1:].clone()
                    old_stps[shift_mask, :-1] = old_stps[shift_mask, 1:].clone()
                    ro[shift_mask, :-1] = ro[shift_mask, 1:].clone()

                    # Add new entry at end
                    old_dirs[shift_mask, history_size - 1] = y[shift_mask]
                    old_stps[shift_mask, history_size - 1] = s[shift_mask]
                    ro[shift_mask, history_size - 1] = 1.0 / ys[shift_mask]

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