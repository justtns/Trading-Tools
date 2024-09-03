import numpy as np

def hull_white_convexity_adjustment(a, sigma, t1, t2):
    """Compute the convexity adjustment for a Hull-White model.

    Parameters
    ----------
    a : float
        The mean reversion level.
    sigma : float
        The volatility of the short rate.
    t1 : float
        The current time.
    t2 : float
        The maturity time.
    Returns
    -------
    float
        The convexity adjustment.
    """
    term1 = (1 - np.exp(-a * (t2 - t1))) / (a * (t2 - t1))
    term2 = (1 - np.exp(-a * (t2 - t1))) / (a * (t2 - t1))
    term3 = 1 - np.exp(-2 * a * t1)
    term4 = 2 * a * (1 - np.exp(-a * t1)) / (a * t1)

    # Combine the terms to compute the final expression
    return ((1 - np.exp(-a * (t2 - t1))) / (a * (t2 - t1)) * (
        (1 - np.exp(-a * (t2 - t1))) / (a * (t2 - t1)) * (1 - np.exp(-2 * a * t1)) +
        2 * a * (1 - np.exp(-a * t1)) / (a * t1))
        * (sigma**2 / (4 * a)))

def find_sigma():
    return
def find_a():
    return