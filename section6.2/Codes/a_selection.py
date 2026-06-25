"""a_selection.py
Heuristic for Temporal coupling parameter a """

import numpy as np

# computing a_min
def compute_a_min(Total_days, rad0, dirichlet=True):
    """ Computes the minimum temporal coupling parameter """
    if dirichlet:
        a_min = Total_days / (np.pi * rad0) * 2.4048
    else:
        a_min = Total_days / (np.pi * rad0) * 1.8412
    return a_min


# computing a
def compute_a(Total_days, rad0, dirichlet=True, a_factor=1.0, verbose=True):
    """ Computes the temporal coupling parameter  a = a_factor * a_min """
    a_min = compute_a_min(Total_days, rad0, dirichlet=dirichlet)
    a = a_factor * a_min
    t_factor = a ** 2
    if verbose:
        print(f"a_min = {a_min}")
        print(" Computed a")
    return a, t_factor, a_min
