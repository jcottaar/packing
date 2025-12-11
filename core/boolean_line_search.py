import copy

def boolean_line_search(f, lo, hi, max_iter=60, tol=1e-12):
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
