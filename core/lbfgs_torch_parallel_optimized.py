# mypy: allow-untyped-defs
"""
Optimized version of lbfgs_torch_parallel with aggressive compilation settings.

This module provides the same interface as lbfgs_torch_parallel but with:
- More aggressive torch.compile settings (mode="max-autotune")
- Dynamic shape optimization disabled for better performance with fixed batch sizes
- Optional TF32 precision for better performance on Ampere+ GPUs

Usage:
    from lbfgs_torch_parallel_optimized import lbfgs
    # Use exactly as you would use lbfgs_torch_parallel.lbfgs

Benchmarking recommendations:
    1. First run will be VERY slow (compilation)
    2. Subsequent runs should be significantly faster
    3. For best results, use consistent batch sizes
    4. Consider warming up with a small problem first
"""
from typing import Optional
import torch
from torch import Tensor
import kaggle_support as kgs

# Import base functions and recompile with aggressive settings
from lbfgs_torch_parallel import _cubic_interpolate, _strong_wolfe_batched

__all__ = ["lbfgs", "enable_performance_mode"]


def enable_performance_mode(use_tf32=True, matmul_precision='high'):
    """
    Enable PyTorch performance optimizations.

    Args:
        use_tf32: Enable TF32 on Ampere+ GPUs (faster but slightly less precise)
        matmul_precision: 'high', 'medium', or 'highest' (only for float32)

    Returns:
        dict with previous settings (for restoration)
    """
    prev_settings = {
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if matmul_precision in ['high', 'medium', 'highest']:
        torch.set_float32_matmul_precision(matmul_precision)

    return prev_settings


@torch.compile(mode="max-autotune", fullgraph=False, dynamic=False)
def _cubic_interpolate_batch_optimized(x1, f1, g1, x2, f2, g2, xmin_bound, xmax_bound):
    """Aggressively compiled cubic interpolation."""
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2

    valid = d2_square >= 0
    d2 = torch.sqrt(torch.clamp(d2_square, min=0))

    x1_le_x2 = x1 <= x2
    min_pos_1 = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    min_pos_2 = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
    min_pos = torch.where(x1_le_x2, min_pos_1, min_pos_2)

    result_valid = torch.clamp(min_pos, min=xmin_bound, max=xmax_bound)
    result_invalid = (xmin_bound + xmax_bound) / 2.0

    return torch.where(valid, result_valid, result_invalid)


@torch.compile(mode="max-autotune", fullgraph=False, dynamic=False)
def _lbfgs_two_loop_recursion_optimized(flat_grad, old_dirs, old_stps, ro, hist_len, H_diag, history_size):
    """
    Aggressively compiled L-BFGS two-loop recursion.

    This version uses mode="max-autotune" which tries many compilation strategies
    and selects the fastest. First compilation is very slow but runtime is optimal.
    """
    M, N = flat_grad.shape
    device = flat_grad.device
    dtype = flat_grad.dtype

    hist_mask = torch.arange(history_size, device=device).unsqueeze(0) < hist_len.unsqueeze(1)
    q = -flat_grad.clone()
    al = torch.zeros(M, history_size, dtype=dtype, device=device)

    # First loop (backward through history)
    for i in range(history_size - 1, -1, -1):
        mask_i = hist_mask[:, i]
        if mask_i.any():
            al[:, i] = (old_stps[:, i] * q).sum(dim=1) * ro[:, i]
            q = q - old_dirs[:, i] * al[:, i].unsqueeze(1) * mask_i.unsqueeze(1)

    r = q * H_diag.unsqueeze(1)

    # Second loop (forward through history)
    for i in range(history_size):
        mask_i = hist_mask[:, i]
        if mask_i.any():
            be_i = (old_dirs[:, i] * r).sum(dim=1) * ro[:, i]
            r = r + old_stps[:, i] * (al[:, i] - be_i).unsqueeze(1) * mask_i.unsqueeze(1)

    return r


@torch.compile(mode="reduce-overhead", fullgraph=False)
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
    """
    Optimized L-BFGS implementation with aggressive compilation.

    Interface identical to lbfgs_torch_parallel.lbfgs but with:
    - Compiled main loop (mode="reduce-overhead")
    - Compiled helper functions (mode="max-autotune")
    - Better performance after initial compilation

    WARNING: First call will be VERY slow (30s-2min depending on problem size)
    due to aggressive optimization search. Subsequent calls are much faster.

    Args:
        func: Callable that takes x (Tensor MxN) and returns (cost, gradient) tuple.
        x0: Initial parameters (M, N) - modified in-place
        lr: Learning rate (default: 1.0)
        max_iter: Maximum iterations (default: 20)
        max_eval: Maximum function evaluations (default: max_iter * 1.25)
        tolerance_grad: Gradient norm tolerance (default: 1e-7)
        tolerance_change: Change tolerance (default: 1e-9)
        history_size: Number of previous gradients to store (default: 100)
        line_search_fn: 'strong_wolfe' or None (default: None)

    Returns:
        Optimized parameters (same tensor as x0)
    """
    if max_eval is None:
        max_eval = max_iter * 5 // 4

    M, N = x0.shape
    device = x0.device
    dtype = x0.dtype

    x = x0.detach().clone()
    loss, flat_grad = func(x)
    func_evals = torch.ones(M, dtype=torch.long, device=device)

    active = torch.ones(M, dtype=torch.bool, device=device)
    grad_norm = flat_grad.abs().max(dim=1).values
    converged = grad_norm <= tolerance_grad
    active &= ~converged

    if not active.any():
        x0.copy_(x)
        return x0

    old_dirs = torch.zeros(M, history_size, N, dtype=dtype, device=device)
    old_stps = torch.zeros(M, history_size, N, dtype=dtype, device=device)
    ro = torch.zeros(M, history_size, dtype=dtype, device=device)
    hist_len = torch.zeros(M, dtype=torch.long, device=device)
    H_diag = torch.ones(M, dtype=dtype, device=device)

    n_iter = 0
    prev_loss = loss.clone()

    while n_iter < max_iter and active.any():
        n_iter += 1

        if n_iter == 1:
            d = -flat_grad.clone()
            H_diag.fill_(1.0)
        else:
            d = _lbfgs_two_loop_recursion_optimized(
                flat_grad, old_dirs, old_stps, ro, hist_len, H_diag, history_size
            )

        prev_flat_grad = flat_grad.clone()
        prev_loss_iter = loss.clone()

        if n_iter == 1:
            grad_sum = flat_grad.abs().sum(dim=1)
            t = torch.minimum(torch.ones_like(grad_sum), 1.0 / grad_sum) * lr
            t[~active] = 0
        else:
            t = torch.full((M,), lr, dtype=dtype, device=device)

        gtd = (flat_grad * d).sum(dim=1)
        early_stop = gtd > -tolerance_change
        active &= ~early_stop

        if not active.any():
            break

        if line_search_fn == 'strong_wolfe':
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
            x = x + t.unsqueeze(1) * d
            if n_iter != max_iter:
                loss, flat_grad = func(x)
            ls_evals = torch.ones(M, dtype=torch.long, device=device)

        func_evals += ls_evals

        if n_iter > 1:
            y = flat_grad - prev_flat_grad
            s = d * t.unsqueeze(1)
            ys = (y * s).sum(dim=1)
            update_mask = active & (ys > 1e-10)

            if update_mask.any():
                H_diag[update_mask] = ys[update_mask] / (y[update_mask] * y[update_mask]).sum(dim=1)

                curr_len = hist_len[update_mask]
                append_mask = update_mask.clone()
                append_mask[update_mask] = curr_len < history_size
                shift_mask = update_mask & ~append_mask

                if append_mask.any():
                    append_positions = hist_len[append_mask]
                    append_indices = torch.where(append_mask)[0]
                    scatter_idx = torch.stack([append_indices, append_positions], dim=1)

                    old_dirs[scatter_idx[:, 0], scatter_idx[:, 1]] = y[append_mask]
                    old_stps[scatter_idx[:, 0], scatter_idx[:, 1]] = s[append_mask]
                    ro[scatter_idx[:, 0], scatter_idx[:, 1]] = 1.0 / ys[append_mask]
                    hist_len[append_mask] += 1

                if shift_mask.any():
                    old_dirs[shift_mask, :-1] = old_dirs[shift_mask, 1:].clone()
                    old_stps[shift_mask, :-1] = old_stps[shift_mask, 1:].clone()
                    ro[shift_mask, :-1] = ro[shift_mask, 1:].clone()

                    old_dirs[shift_mask, history_size - 1] = y[shift_mask]
                    old_stps[shift_mask, history_size - 1] = s[shift_mask]
                    ro[shift_mask, history_size - 1] = 1.0 / ys[shift_mask]

        grad_norm = flat_grad.abs().max(dim=1).values
        converged_grad = grad_norm <= tolerance_grad
        active &= ~converged_grad

        converged_eval = func_evals >= max_eval
        active &= ~converged_eval

        param_change = (d * t.unsqueeze(1)).abs().max(dim=1).values
        converged_param = (param_change <= tolerance_change) & active
        active &= ~converged_param

        loss_change = (loss - prev_loss_iter).abs()
        converged_loss = (loss_change < tolerance_change) & active
        active &= ~converged_loss

    x0.copy_(x)
    return x0
