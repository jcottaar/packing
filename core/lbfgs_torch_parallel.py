# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import Tensor


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
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals


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
            t = torch.zeros(M, dtype=dtype, device=device)
            for m in range(M):
                if active[m]:
                    t[m] = min(1.0, 1.0 / flat_grad[m].abs().sum()) * lr
        else:
            t = torch.full((M,), lr, dtype=dtype, device=device)

        # Check directional derivative and mark converged
        gtd = (flat_grad * d).sum(dim=1)  # (M,)
        early_stop = gtd > -tolerance_change
        active &= ~early_stop

        if not active.any():
            break

        # Line search (per system, unbatched)
        ls_evals = torch.zeros(M, dtype=torch.long, device=device)

        for m in range(M):
            if not active[m]:
                continue

            if line_search_fn == 'strong_wolfe':
                # Create wrapper that adds batch dimension for this system
                def directional_evaluate(x_init, t_step, d_step):
                    x_new = x_init + t_step * d_step
                    x_batch = x_new.unsqueeze(0)  # (1, N)
                    loss_new, grad_new = func(x_batch)
                    return float(loss_new[0]), grad_new[0]

                loss_m, flat_grad_m, t_m, ls_evals_m = _strong_wolfe(
                    directional_evaluate, x[m], t[m].item(), d[m],
                    loss[m].item(), flat_grad[m], gtd[m].item()
                )
                loss[m] = loss_m
                flat_grad[m] = flat_grad_m
                t[m] = t_m
                ls_evals[m] = ls_evals_m
                x[m] = x[m] + t[m] * d[m]
            else:
                # Simple step
                x[m] = x[m] + t[m] * d[m]
                if n_iter != max_iter:
                    x_batch = x[m].unsqueeze(0)  # (1, N)
                    loss_batch, grad_batch = func(x_batch)
                    loss[m] = loss_batch[0]
                    flat_grad[m] = grad_batch[0]
                ls_evals[m] = 1

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

        for m in range(M):
            if not active[m]:
                continue

            param_change = (d[m] * t[m]).abs().max()
            if param_change <= tolerance_change:
                active[m] = False
                continue

            loss_change = abs(loss[m] - prev_loss_iter[m])
            if loss_change < tolerance_change:
                active[m] = False

    # Update x0 with final result
    x0.copy_(x)

    return x0