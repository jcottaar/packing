import copy
import kaggle_support as kgs

def boolean_line_search(f, lo, hi, max_iter=60, tol=1e-20):
    """
    Generic binary search for smallest value in [lo, hi] where f(x) is False,
    assuming f(lo) == True and f(hi) == False, and f is monotonic.
    Returns the value in [lo, hi] where f(x) transitions from True to False.
    """
    assert f(lo)
    assert not f(hi)
    for _ in range(max_iter):
        mid = (hi + lo) / 2.0
        if f(mid):
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < tol:
            break
    return hi

def boolean_line_search_vectorized(f, lo, hi, n, max_iter=60, tol=1e-20):
    """
    Vectorized binary search for smallest values in [lo, hi] where f(x) is False.

    Args:
        f: Function that takes an array and returns a boolean array
        lo: Scalar lower bound (where f(lo) == True for all elements)
        hi: Scalar upper bound (where f(hi) == False for all elements)
        n: Number of parallel searches (length of arrays)
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence

    Returns:
        Array of dtype kgs.dtype_cp with the transition points for each search
    """
    cp = kgs.cp

    # Initialize lo and hi arrays with the scalar bounds
    lo_work = cp.full(n, lo, dtype=kgs.dtype_cp)
    hi_work = cp.full(n, hi, dtype=kgs.dtype_cp)

    # Validate initial conditions
    f_lo = f(lo_work)
    f_hi = f(hi_work)
    assert cp.all(f_lo)
    assert cp.all(~f_hi)

    for _ in range(max_iter):
        mid = (hi_work + lo_work) / 2.0

        # Get vectorized boolean results
        f_results = f(mid)

        # Update lo where f(mid) is True, hi where f(mid) is False
        lo_work = cp.where(f_results, mid, lo_work)
        hi_work = cp.where(f_results, hi_work, mid)

        # Check convergence (all differences below tolerance)
        if cp.all(cp.abs(hi_work - lo_work) < tol):
            break

    return hi_work
