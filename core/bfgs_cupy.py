import cupy as cp

def lbfgs_minimize_batched_cupy(
    f,                       # callable: x[M,N] -> (cost[M], grad[M,N]) (CuPy)
    x0,                      # CuPy array [M,N]
    max_iter=5000,
    gtol=1e-2,               # infinity-norm stopping criterion
    m_hist=10,               # history size
    line_search_max=25,
    c1=1e-4,                 # Armijo constant
    tau=0.5,                 # backtracking factor
    alpha0=1.0,              # initial step
    min_alpha=1e-16,
    use_powell_damping=True,
    reset_on_failure=True,
):
    """
    Batched L-BFGS on GPU with CuPy.

    Returns dict with:
      x: [M,N] final positions
      cost: [M]
      grad: [M,N]
      converged: [M] bool
      n_iter: int
      n_eval: int objective/grad evaluations
    """

    x = cp.asarray(x0)
    if x.ndim != 2:
        raise ValueError("x0 must be 2D [M,N]")
    M, N = x.shape

    # Evaluate initial
    cost, grad = f(x)
    n_eval = 1

    # Convergence mask (kept on GPU)
    converged = cp.zeros((M,), dtype=cp.bool_)
    # Optional: store iteration of convergence
    # conv_iter = cp.full((M,), -1, dtype=cp.int32)

    # L-BFGS history (circular buffer)
    S = cp.zeros((M, m_hist, N), dtype=x.dtype)
    Y = cp.zeros((M, m_hist, N), dtype=x.dtype)
    Rho = cp.zeros((M, m_hist), dtype=x.dtype)  # 1/(y·s)
    hist_len = cp.zeros((M,), dtype=cp.int32)
    hist_pos = cp.zeros((M,), dtype=cp.int32)   # next insert position

    def inf_norm(g):
        return cp.max(cp.abs(g), axis=1)

    def two_loop_direction(g, active_mask):
        """
        Compute p = -H g using two-loop recursion, batched.
        g: [M,N]
        active_mask: [M] bool
        """
        M_, N_ = g.shape
        rows = cp.arange(M_, dtype=cp.int32)

        # Default: steepest descent
        p = -g.copy()

        has_hist = (hist_len > 0) & active_mask
        if not bool(cp.any(has_hist)):
            return p

        q = g.copy()
        alpha = cp.zeros((M_, m_hist), dtype=x.dtype)

        # Reverse loop: newest -> oldest
        for t in range(m_hist):
            idx = (hist_pos - 1 - t) % m_hist  # [M]
            valid = has_hist & (t < hist_len)
            if not bool(cp.any(valid)):
                continue

            # ✅ gather per-row
            s = S[rows, idx, :]          # [M,N]
            y = Y[rows, idx, :]          # [M,N]
            rho = Rho[rows, idx]         # [M]

            sq = cp.sum(s * q, axis=1)   # [M]
            a = rho * sq                 # [M]
            alpha[:, t] = a

            q = cp.where(valid[:, None], q - a[:, None] * y, q)

        # Initial scaling gamma using most recent pair
        idx0 = (hist_pos - 1) % m_hist   # [M]
        s0 = S[rows, idx0, :]            # [M,N]
        y0 = Y[rows, idx0, :]            # [M,N]
        yy = cp.sum(y0 * y0, axis=1)     # [M]
        sy = cp.sum(s0 * y0, axis=1)     # [M]

        gamma = cp.where(has_hist & (yy > 0), sy / yy, cp.ones((M_,), dtype=x.dtype))
        r = gamma[:, None] * q

        # Forward loop: oldest -> newest
        for t in range(m_hist - 1, -1, -1):
            idx = (hist_pos - 1 - t) % m_hist
            valid = has_hist & (t < hist_len)
            if not bool(cp.any(valid)):
                continue

            s = S[rows, idx, :]
            y = Y[rows, idx, :]
            rho = Rho[rows, idx]

            yr = cp.sum(y * r, axis=1)
            b = rho * yr
            a = alpha[:, t]
            r = cp.where(valid[:, None], r + s * (a - b)[:, None], r)

        p = cp.where(has_hist[:, None], -r, p)
        return p

    def store_history(s, y, active_mask):
        """
        Store (s,y) with safeguards + optional Powell damping.
        s,y: [M,N]
        active_mask: [M] bool (only update for active runs)
        """
        nonlocal S, Y, Rho, hist_len, hist_pos

        # Compute curvature
        ys = cp.sum(y * s, axis=1)  # [M]
        yy = cp.sum(y * y, axis=1)
        ss = cp.sum(s * s, axis=1)

        # Basic validity: positive curvature and not tiny
        eps = cp.finfo(x.dtype).eps
        min_curv = eps * cp.sqrt(yy * ss + eps)

        valid = active_mask & (ys > min_curv)

        if use_powell_damping:
            # Powell damping to enforce y^T s >= 0.2 * s^T B s
            # We approximate s^T B s using (y·s)/(s·s) * (s·s) = y·s, which is circular.
            # Better: use y as proxy can fail. A common cheap damping uses:
            # if y·s < 0.2 * s·s (in scaled space), modify y.
            # We'll use a heuristic: enforce y·s >= 0.2 * s·s * (||g|| scale)
            # but keep it simple & safe: if y·s < 0.2 * s·s, damp y towards s.
            # This is heuristic but works well in practice for noisy gradients.
            threshold = 0.2 * ss
            need = active_mask & (ys < threshold) & (ss > 0)
            if bool(cp.any(need)):
                # y_hat = theta*y + (1-theta)*s, choose theta to satisfy y_hat·s = threshold
                # => (theta*ys + (1-theta)*ss) = threshold
                # => theta = (threshold - ss) / (ys - ss)
                denom = (ys - ss)
                theta = cp.where(cp.abs(denom) > 0, (threshold - ss) / denom, cp.ones_like(ys))
                theta = cp.clip(theta, 0.0, 1.0)
                y = cp.where(need[:, None], theta[:, None] * y + (1.0 - theta)[:, None] * s, y)
                # recompute ys
                ys = cp.sum(y * s, axis=1)
                valid = active_mask & (ys > min_curv)

        # Insert at per-run hist_pos; we do it batched using advanced indexing
        pos = hist_pos  # [M]
        rows = cp.arange(M)

        # Write only where valid; keep old where not
        S[rows, pos, :] = cp.where(valid[:, None], s, S[rows, pos, :])
        Y[rows, pos, :] = cp.where(valid[:, None], y, Y[rows, pos, :])

        rho = cp.where(valid, 1.0 / ys, 0.0)
        Rho[rows, pos] = cp.where(valid, rho, Rho[rows, pos])

        # Update hist_len and hist_pos only where valid
        hist_len = cp.where(valid, cp.minimum(hist_len + 1, m_hist), hist_len)
        hist_pos = cp.where(valid, (hist_pos + 1) % m_hist, hist_pos)

        if reset_on_failure:
            # If not valid but active, you may want to reset history for that run to avoid bad curvature.
            bad = active_mask & (~valid)
            hist_len = cp.where(bad, 0, hist_len)
            hist_pos = cp.where(bad, 0, hist_pos)
            # Not strictly necessary to zero S/Y/Rho; len=0 ignores them.

    # Initial convergence
    converged = inf_norm(grad) < gtol

    for it in range(max_iter):
        active = ~converged  # [M]
        # Still must call f on full batch later per your constraint; we *mask updates only*.

        # Direction p (compute for all, but it will be ignored for converged runs)
        p = two_loop_direction(grad, active)

        # Ensure descent direction for active runs; if not, fall back to steepest descent.
        gp = cp.sum(grad * p, axis=1)  # [M]
        descent = active & (gp < 0)
        p = cp.where(descent[:, None], p, -grad)

        # Batched Armijo backtracking line search (always evaluate full batch)
        alpha = cp.full((M,), alpha0, dtype=x.dtype)
        x_base = x
        f_base = cost
        g_base = grad
        gp_base = cp.sum(g_base * p, axis=1)

        accepted = cp.zeros((M,), dtype=cp.bool_)

        # For converged runs: force alpha=0 so x doesn't change (but we still eval full batch anyway)
        alpha = cp.where(active, alpha, cp.zeros_like(alpha))

        for _ls in range(line_search_max):
            x_trial = x_base + alpha[:, None] * p
            f_trial, g_trial = f(x_trial)
            n_eval += 1

            # Armijo condition: f(x+a p) <= f(x) + c1*a*(g·p)
            rhs = f_base + c1 * alpha * gp_base
            ok = active & (f_trial <= rhs)

            # Accept where ok
            accepted = accepted | ok

            # Prepare for next backtrack: shrink alpha where not ok and still active
            still = active & (~ok)
            alpha = cp.where(still, alpha * tau, alpha)

            # If everyone active has accepted, stop
            if not bool(cp.any(still)):
                # keep the last computed f_trial/g_trial for the final update
                break

            # Also stop if step too small for remaining
            too_small = still & (alpha < min_alpha)
            if bool(cp.any(too_small)):
                # mark those as "failed" (won't update this iter)
                active = active & (~too_small)
                alpha = cp.where(~active, cp.zeros_like(alpha), alpha)
                # continue line search for remaining actives

        # Final trial evaluation at chosen alpha (we already have last f_trial/g_trial)
        # But if line search exited early due to accepted, we have corresponding f_trial/g_trial.
        # If it exited due to alpha < min_alpha, accepted may be false for those.

        # Compute updates (masked)
        x_new = x_base + alpha[:, None] * p

        # Re-evaluate at x_new to align with masked alpha changes (optional but safer).
        # This enforces "always full batch call"; costs one eval/iter.
        f_new, g_new = f(x_new)
        n_eval += 1

        # Only update for runs that are active and made a non-trivial step and improved Armijo
        # (You can relax this if you want to accept non-decreasing steps sometimes.)
        improved = (f_new <= (f_base + c1 * alpha * gp_base)) & (~converged)
        stepped = cp.abs(alpha) > 0
        do_update = improved & stepped

        s = x_new - x
        y = g_new - grad

        # Store history and commit state for do_update
        store_history(s, y, do_update)

        x = cp.where(do_update[:, None], x_new, x)
        cost = cp.where(do_update, f_new, cost)
        grad = cp.where(do_update[:, None], g_new, grad)

        # Update convergence (based on new grad)
        converged = converged | (inf_norm(grad) < gtol)

        # Optional: early exit if all converged (even though you asked to always pass full batch to f,
        # exiting means you won't call f anymore. If you truly want to keep calling f after convergence,
        # remove this break.)
        if bool(cp.all(converged)):
            return {
                "x": x,
                "cost": cost,
                "grad": grad,
                "converged": converged,
                "n_iter": it + 1,
                "n_eval": n_eval,
            }

    return {
        "x": x,
        "cost": cost,
        "grad": grad,
        "converged": converged,
        "n_iter": max_iter,
        "n_eval": n_eval,
    }
