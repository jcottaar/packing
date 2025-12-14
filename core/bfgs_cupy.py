# Batched L-BFGS in CuPy, closely mirroring SciPy's unconstrained L-BFGS-B behavior
# (Strong Wolfe line search, no bounds, float32/float64 supported)

import cupy as cp


def lbfgs_minimize_batched_cupy_scipy(
    f,
    x0,
    # SciPy L-BFGS-B defaults:
    max_iter=15000,
    maxcor=10,
    ftol=2.220446049250313e-09,
    gtol=1e-5,
    maxls=20,
    # Strong Wolfe parameters (SciPy defaults):
    c1=1e-4,
    c2=0.9,
):
    """
    Batched (unconstrained) L-BFGS in CuPy, aiming to mirror SciPy's L-BFGS-B behavior
    (without bounds).

    f: callable(x[M,N]) -> (cost[M], grad[M,N]) as CuPy arrays
    x0: CuPy array [M,N] (float32 or float64)

    Returns dict:
      x, cost, grad
      converged_f, converged_g, converged
      failed_line_search
      n_iter, n_eval
    """

    x = cp.asarray(x0)
    if x.ndim != 2:
        raise ValueError("x0 must be [M,N]")
    M, N = x.shape
    dtype = x.dtype
    rows = cp.arange(M, dtype=cp.int32)

    # mutable evaluation counter (safe for nested scopes)
    n_eval = [0]

    def assert_shapes(cost, grad):
        assert isinstance(cost, cp.ndarray) and isinstance(grad, cp.ndarray), \
            f"f must return CuPy arrays, got {type(cost)} and {type(grad)}"
        assert cost.shape == (M,), f"cost shape must be {(M,)}, got {cost.shape}"
        assert grad.shape == (M, N), f"grad shape must be {(M, N)}, got {grad.shape}"

    def inf_norm(g):
        return cp.max(cp.abs(g), axis=1)

    # Initial evaluation
    fval, grad = f(x)
    assert_shapes(fval, grad)
    n_eval[0] += 1

    # History buffers
    m_hist = int(maxcor)
    S = cp.zeros((M, m_hist, N), dtype=dtype)
    Y = cp.zeros((M, m_hist, N), dtype=dtype)
    Rho = cp.zeros((M, m_hist), dtype=dtype)
    hist_len = cp.zeros((M,), dtype=cp.int32)
    hist_pos = cp.zeros((M,), dtype=cp.int32)

    converged_g = cp.zeros((M,), dtype=cp.bool_)
    converged_f = cp.zeros((M,), dtype=cp.bool_)
    failed_ls = cp.zeros((M,), dtype=cp.bool_)

    # ---------------- L-BFGS direction ----------------
    def lbfgs_direction(g):
        q = g.copy()
        alpha = cp.zeros((M, m_hist), dtype=dtype)

        has_hist = hist_len > 0
        if not bool(cp.any(has_hist)):
            return -q

        # backward (newest -> oldest)
        for t in range(m_hist):
            idx = (hist_pos - 1 - t) % m_hist
            valid = (t < hist_len)
            if not bool(cp.any(valid)):
                continue
            s = S[rows, idx, :]
            y = Y[rows, idx, :]
            rho = Rho[rows, idx]
            a = rho * cp.sum(s * q, axis=1)
            alpha[:, t] = a
            q = cp.where(valid[:, None], q - a[:, None] * y, q)

        # scaling H0 = (s'y)/(y'y)
        idx0 = (hist_pos - 1) % m_hist
        s0 = S[rows, idx0, :]
        y0 = Y[rows, idx0, :]
        yy = cp.sum(y0 * y0, axis=1)
        sy = cp.sum(s0 * y0, axis=1)
        gamma = cp.where((has_hist) & (yy > 0), sy / yy, cp.ones((M,), dtype=dtype))
        r = gamma[:, None] * q

        # forward (oldest -> newest)
        for t in range(m_hist - 1, -1, -1):
            idx = (hist_pos - 1 - t) % m_hist
            valid = (t < hist_len)
            if not bool(cp.any(valid)):
                continue
            s = S[rows, idx, :]
            y = Y[rows, idx, :]
            rho = Rho[rows, idx]
            beta = rho * cp.sum(y * r, axis=1)
            r = cp.where(valid[:, None], r + s * (alpha[:, t] - beta)[:, None], r)

        return -r

    # ---------------- Strong Wolfe line search ----------------
    tiny = cp.finfo(dtype).tiny
    eps = cp.finfo(dtype).eps

    def _cubic_minimizer(a, fa, fpa, b, fb, fpb):
        d1 = fpa + fpb - 3.0 * (fa - fb) / (a - b)
        rad = d1 * d1 - fpa * fpb
        rad = cp.maximum(rad, 0.0)
        d2 = cp.sqrt(rad)
        denom = fpb - fpa + 2.0 * d2
        return b - (b - a) * (fpb + d2 - d1) / denom

    def _safe_step(cand, lo, hi):
        lo2 = cp.minimum(lo, hi)
        hi2 = cp.maximum(lo, hi)
        mid = 0.5 * (lo + hi)
        ok = (cand > lo2) & (cand < hi2) & cp.isfinite(cand)
        return cp.where(ok, cand, mid)

    def strong_wolfe_search(x, f0, g0, p, active_mask):
        """Batched strong-Wolfe line search (stateful per-run).

        Critical behavior for batching: once a run satisfies strong Wolfe (or fails),
        its step is *frozen* while we continue evaluating the full batch to satisfy
        the requirement that f is always called on the full [M,N] array.

        Returns (alpha, phi, g, ok, failed).
        """
        dphi0 = cp.sum(g0 * p, axis=1)

        # Only runs with a descent direction can succeed
        descent = active_mask & (dphi0 < 0)
        bad_dir = active_mask & (~descent)

        alpha_min = cp.asarray(tiny, dtype=dtype)
        alpha_max = cp.asarray(1e20, dtype=dtype)

        # Per-run line-search state
        ok = cp.zeros((M,), dtype=cp.bool_)
        failed = cp.zeros((M,), dtype=cp.bool_)
        ls_active = descent.copy()  # only these are searched

        # Start alpha ~1 (SciPy-ish). Keep as vector for general M.
        alpha = cp.ones((M,), dtype=dtype)
        alpha_prev = cp.zeros((M,), dtype=dtype)
        phi_prev = f0
        dphi_prev = dphi0

        # Bracket endpoints with values
        a_lo = cp.zeros((M,), dtype=dtype)
        phi_lo = f0
        dphi_lo = dphi0

        a_hi = cp.zeros((M,), dtype=dtype)
        phi_hi = cp.zeros((M,), dtype=dtype)
        dphi_hi = cp.zeros((M,), dtype=dtype)

        # Track best Armijo-satisfying point as a practical fallback (matches SciPy behavior better
        # on tricky objectives where strict strong-Wolfe may fail within maxls).
        have_armijo = cp.zeros((M,), dtype=cp.bool_)
        best_alpha = cp.zeros((M,), dtype=dtype)
        best_phi = cp.full((M,), cp.inf, dtype=dtype)
        best_g = g0.copy()

        # Last evaluated
        phi = f0
        g = g0
        dphi = dphi0

        def zoom(zoom_mask):
            """Zoom phase for runs in zoom_mask. Uses cubic interpolation with bisection fallback.

            Updates outer-scope variables and freezes finished runs.
            """
            nonlocal alpha, phi, g, dphi
            nonlocal a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi
            nonlocal ok, failed, ls_active
            nonlocal have_armijo, best_alpha, best_phi, best_g

            for _ in range(maxls):
                zm = zoom_mask & ls_active & (~ok) & (~failed)
                if not bool(cp.any(zm)):
                    return

                cand = _cubic_minimizer(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi)
                a_trial = _safe_step(cand, a_lo, a_hi)
                a_trial = cp.clip(a_trial, alpha_min, alpha_max)

                # Freeze alpha for non-zm runs
                alpha = cp.where(zm, a_trial, alpha)

                x_trial = x + alpha[:, None] * p
                phi, g = f(x_trial)
                assert_shapes(phi, g)
                n_eval[0] += 1
                dphi = cp.sum(g * p, axis=1)

                # Armijo check
                armijo_ok = phi <= (f0 + c1 * alpha * dphi0)

                # Track best Armijo point
                better = zm & armijo_ok & (phi < best_phi)
                best_phi = cp.where(better, phi, best_phi)
                best_alpha = cp.where(better, alpha, best_alpha)
                best_g = cp.where(better[:, None], g, best_g)
                have_armijo = have_armijo | (zm & armijo_ok)

                # Interval update logic
                high = (~armijo_ok) | (phi >= phi_lo)
                upd_hi = zm & high
                a_hi = cp.where(upd_hi, alpha, a_hi)
                phi_hi = cp.where(upd_hi, phi, phi_hi)
                dphi_hi = cp.where(upd_hi, dphi, dphi_hi)

                curv_ok = cp.abs(dphi) <= (-c2 * dphi0)
                accept = zm & (~high) & armijo_ok & curv_ok
                ok = ok | accept
                ls_active = ls_active & (~accept)

                upd_lo = zm & (~high) & armijo_ok & (~curv_ok) & (~accept)
                wrong_sign = upd_lo & (dphi * (a_hi - a_lo) >= 0)

                # Swap hi <- lo when needed
                a_hi = cp.where(wrong_sign, a_lo, a_hi)
                phi_hi = cp.where(wrong_sign, phi_lo, phi_hi)
                dphi_hi = cp.where(wrong_sign, dphi_lo, dphi_hi)

                a_lo = cp.where(upd_lo, alpha, a_lo)
                phi_lo = cp.where(upd_lo, phi, phi_lo)
                dphi_lo = cp.where(upd_lo, dphi, dphi_lo)

                # Collapse => mark failure for those runs (do not claim convergence)
                interval = cp.abs(a_hi - a_lo)
                collapsed = zm & (interval <= alpha_min)
                failed = failed | collapsed
                ls_active = ls_active & (~collapsed)

            # If zoom exceeded maxls, remaining zoomed runs are marked failed (unless Armijo fallback later)
            leftover = zoom_mask & ls_active & (~ok) & (~failed)
            failed = failed | leftover
            ls_active = ls_active & (~leftover)

        # Bracketing loop
        for i in range(maxls):
            bm = ls_active & (~ok) & (~failed)
            if not bool(cp.any(bm)):
                break

            alpha = cp.where(bm, cp.clip(alpha, alpha_min, alpha_max), alpha)

            x_trial = x + alpha[:, None] * p
            phi, g = f(x_trial)
            assert_shapes(phi, g)
            n_eval[0] += 1
            dphi = cp.sum(g * p, axis=1)

            armijo_ok = phi <= (f0 + c1 * alpha * dphi0)
            curv_ok = cp.abs(dphi) <= (-c2 * dphi0)

            # Track best Armijo for fallback
            better = bm & armijo_ok & (phi < best_phi)
            best_phi = cp.where(better, phi, best_phi)
            best_alpha = cp.where(better, alpha, best_alpha)
            best_g = cp.where(better[:, None], g, best_g)
            have_armijo = have_armijo | (bm & armijo_ok)

            accept = bm & armijo_ok & curv_ok
            ok = ok | accept
            ls_active = ls_active & (~accept)

            # Bracket condition 1: Armijo fails OR f increases vs previous
            bracket1 = bm & (~accept) & ((~armijo_ok) | ((i > 0) & (phi >= phi_prev)))
            if bool(cp.any(bracket1)):
                # Set bracket endpoints for these runs
                a_lo = cp.where(bracket1, alpha_prev, a_lo)
                phi_lo = cp.where(bracket1, phi_prev, phi_lo)
                dphi_lo = cp.where(bracket1, dphi_prev, dphi_lo)

                a_hi = cp.where(bracket1, alpha, a_hi)
                phi_hi = cp.where(bracket1, phi, phi_hi)
                dphi_hi = cp.where(bracket1, dphi, dphi_hi)

                zoom(bracket1)

            # Bracket condition 2: derivative becomes nonnegative
            bm2 = ls_active & (~ok) & (~failed)
            bracket2 = bm2 & (dphi >= 0)
            if bool(cp.any(bracket2)):
                a_lo = cp.where(bracket2, alpha, a_lo)
                phi_lo = cp.where(bracket2, phi, phi_lo)
                dphi_lo = cp.where(bracket2, dphi, dphi_lo)

                a_hi = cp.where(bracket2, alpha_prev, a_hi)
                phi_hi = cp.where(bracket2, phi_prev, phi_hi)
                dphi_hi = cp.where(bracket2, dphi_prev, dphi_hi)

                zoom(bracket2)

            # Prepare next step for remaining active runs
            bm3 = ls_active & (~ok) & (~failed)
            alpha_prev = cp.where(bm3, alpha, alpha_prev)
            phi_prev = cp.where(bm3, phi, phi_prev)
            dphi_prev = cp.where(bm3, dphi, dphi_prev)

            alpha = cp.where(bm3, alpha * 2.0, alpha)

        # Practical fallback: if strong Wolfe wasn't found but Armijo was, accept best Armijo.
        # This prevents early "failed" exits in cases where SciPy still proceeds.
        fallback = active_mask & descent & (~ok) & have_armijo
        alpha = cp.where(fallback, best_alpha, alpha)
        phi = cp.where(fallback, best_phi, phi)
        g = cp.where(fallback[:, None], best_g, g)
        ok = ok | fallback

        # Mark failures for those that still didn't find any acceptable step
        failed = failed | (active_mask & descent & (~ok)) | bad_dir
        return alpha, phi, g, ok, failed

    # ---------------- Main loop ----------------
    for it in range(int(max_iter)):
        gnorm = inf_norm(grad)
        converged_g |= (gnorm <= gtol)

        active = ~(converged_g | converged_f | failed_ls)
        if not bool(cp.any(active)):
            break

        p = lbfgs_direction(grad)
        gp = cp.sum(grad * p, axis=1)
        need_sd = active & (gp >= 0)
        p = cp.where(need_sd[:, None], -grad, p)

        alpha, f_new, g_new, ok, failed = strong_wolfe_search(x, fval, grad, p, active_mask=active)
        failed_ls |= failed
        accept = ok & active & (~failed)

        x_new = x + alpha[:, None] * p
        s = x_new - x
        y = g_new - grad
        ys = cp.sum(y * s, axis=1)
        good = accept & (ys > 0)

        pos = hist_pos
        S[rows, pos, :] = cp.where(good[:, None], s, S[rows, pos, :])
        Y[rows, pos, :] = cp.where(good[:, None], y, Y[rows, pos, :])
        Rho[rows, pos] = cp.where(good, 1.0 / ys, Rho[rows, pos])
        hist_len = cp.where(good, cp.minimum(hist_len + 1, m_hist), hist_len)
        hist_pos = cp.where(good, (hist_pos + 1) % m_hist, hist_pos)

        f_prev = fval
        x = cp.where(accept[:, None], x_new, x)
        fval = cp.where(accept, f_new, fval)
        grad = cp.where(accept[:, None], g_new, grad)

        denom = cp.maximum(cp.maximum(cp.abs(f_prev), cp.abs(fval)), cp.ones((M,), dtype=dtype))
        rel_red = (f_prev - fval) / denom
        converged_f |= (accept & (rel_red <= ftol))

        gnorm = inf_norm(grad)
        converged_g |= (gnorm <= gtol)

    converged = converged_f | converged_g
    return {
        "x": x,
        "cost": fval,
        "grad": grad,
        "converged_f": converged_f,
        "converged_g": converged_g,
        "converged": converged,
        "failed_line_search": failed_ls,
        "n_iter": it + 1,
        "n_eval": int(n_eval[0]),
    }
