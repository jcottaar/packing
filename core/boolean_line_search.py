"""
Boolean Line Search Module

This code is released under CC BY-SA 4.0, meaning you can freely use and adapt it 
(including commercially), but must give credit to the original author (Jeroen 
Cottaar) and keep it under this license.

Implements binary search algorithms for finding transition points where boolean
functions change from True to False. Used primarily for solution compaction in
pack_io.py to find maximum valid scale factors.

Key Functions:
- boolean_line_search: Scalar binary search for single transition point
- boolean_line_search_vectorized: Batched binary search for multiple searches

Used by:
- pack_io.solution_list_to_dataframe: Finding compaction factors
"""

import kaggle_support as kgs


def boolean_line_search(f, lo, hi, max_iter=60, tol=1e-20):
    """
    Binary search for transition point where boolean function changes value.
    
    Finds the smallest value x in [lo, hi] where f(x) is False, assuming f is
    monotonic (True below threshold, False above). Useful for finding maximum
    valid parameters in optimization contexts.
    
    Args:
        f: Callable[[float], bool] - Boolean function to search
        lo: Lower bound where f(lo) must be True
        hi: Upper bound where f(hi) must be False
        max_iter: Maximum number of bisection iterations (default 60)
        tol: Convergence tolerance for interval width (default 1e-20)
    
    Returns:
        float: Transition point x where f(x) changes from True to False
    
    Raises:
        AssertionError: If initial conditions f(lo)==True or f(hi)==False fail
    """
    # Validate search bounds
    assert f(lo)
    assert not f(hi)
    
    # Binary search iteration
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
    Vectorized binary search for multiple parallel transition points.
    
    Performs n independent binary searches simultaneously, finding transition
    points for each element in batched arrays. More efficient than calling
    boolean_line_search n times when function evaluation can be vectorized.
    
    Args:
        f: Callable[[cp.ndarray], cp.ndarray] - Vectorized boolean function
           Takes array of shape (n,), returns boolean array of shape (n,)
        lo: Scalar lower bound (f(lo) == True for all elements)
        hi: Scalar upper bound (f(hi) == False for all elements)
        n: Number of parallel searches to perform
        max_iter: Maximum number of bisection iterations (default 60)
        tol: Convergence tolerance for interval width (default 1e-20)
    
    Returns:
        cp.ndarray: Shape (n,), dtype kgs.dtype_cp - Transition points
    
    Raises:
        AssertionError: If initial conditions fail for any element
    """
    cp = kgs.cp

    # Initialize working arrays with scalar bounds
    lo_work = cp.full(n, lo, dtype=kgs.dtype_cp)  # Shape: (n,)
    hi_work = cp.full(n, hi, dtype=kgs.dtype_cp)  # Shape: (n,)

    # Validate initial conditions across all elements
    f_lo = f(lo_work)  # Shape: (n,)
    f_hi = f(hi_work)  # Shape: (n,)
    assert cp.all(f_lo)
    assert cp.all(~f_hi)

    # Vectorized binary search
    for _ in range(max_iter):
        mid = (hi_work + lo_work) / 2.0  # Shape: (n,)

        # Evaluate function at midpoints
        f_results = f(mid)  # Shape: (n,)

        # Update bounds based on function results
        lo_work = cp.where(f_results, mid, lo_work)
        hi_work = cp.where(f_results, hi_work, mid)

        # Check for convergence
        if cp.all(cp.abs(hi_work - lo_work) < tol):
            break

    return hi_work
