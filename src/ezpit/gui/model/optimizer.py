import numpy as np
from numpy.linalg import lstsq

def optimize_poly_order(q_max, x, y):
    max_poly_order = int(1.5 * q_max / np.pi)
    min_rmse = float('inf')
    best_degree = 3

    #print('what is the problem here')
    for deg in range(3, max_poly_order + 1):
        X = np.vander(x, deg + 1, increasing=False)  # Vandermonde matrix
        coeffs, *_ = lstsq(X, y, rcond=None)         # Least squares solution
        y_fit = X @ coeffs                           # Predicted values
        rmse = np.sqrt(np.mean((y - y_fit)**2))      # Root Mean Squared Error
        if rmse < min_rmse:
            min_rmse = rmse
            best_degree = deg

    return best_degree


def optimize_qmax(x, y, qmin=22.0, qmax=25.0):
    """
    Returns the q value within a given range [qmin, qmax] where F(q) is closest to zero.

    Parameters:
        x (list or np.array): q values.
        y (list or np.array): corresponding F(q) values.
        qmin (float): lower bound of q range to search.
        qmax (float): upper bound of q range to search.

    Returns:
        float: q value within [qmin, qmax] where |F(q)| is minimized.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Create mask for range [qmin, qmax]
    mask = (x >= qmin) & (x <= qmax)
    x_range = x[mask]
    y_range = y[mask]

    if len(x_range) == 0:
        raise ValueError("No q values found in the specified range.")

    # Find the index where |F(q)| is minimized in the range
    min_idx = np.argmin(np.abs(y_range))
    return float(x_range[min_idx])

def optimize_background_scale(y, bkg_y):
    """
    Finds the maximum background scale such that (y - scale * bkg_y) ≥ 0 for all points.
    
    Assumes bkg_x is aligned with x (same length and ordering).
    
    Returns:
        float: optimal background scale.
    """
    if y.shape != bkg_y.shape:
        raise ValueError("y and bkg_y must have the same shape.")
    
    # Only use valid points: bkg_y > 0 and both values finite
    valid = (bkg_y > 0) & np.isfinite(y) & np.isfinite(bkg_y)
    if not np.any(valid):
        return 0.0

    safe_ratios = y[valid] / bkg_y[valid]
    optimal_scale = np.min(safe_ratios)

    print(optimal_scale)

    return max(optimal_scale * 98, 0.0)  # Slight margin to avoid floating point edge cases