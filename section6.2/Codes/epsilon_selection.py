""" epsilon_selection.py  
Optimal Bandwidth Epsilon Via the Log-Log Slope Heuristic """

import numpy as np
from scipy.spatial import KDTree

# Nearest Neighbour Distance
def nndist_correct(A):
    """ Nearest Neighbour Distances """
    A = np.asarray(A, dtype=float)
    tree = KDTree(A, leafsize=16)
    distances, _ = tree.query(A, k=2, workers=-1)
    nn = distances[:, 1]
    return nn, float(np.mean(nn))


# Auto Epsilon
def auto_epsilon(SpacePointsarray, epsscalefactor=2, use_nndist=True, verbose=True):
    """ Compute the diffusion-maps bandwidth epsilon automatically """
    _, N, T = SpacePointsarray.shape
    epsx_vs_time = np.zeros(T)
    nn_func = nndist_correct
    for k in range(T):
        pts_k = SpacePointsarray[:, :, k].T   # (N, 2)
        _, mean_nn_k = nndist_correct(pts_k)
        epsx_vs_time[k] = mean_nn_k ** 2
    epsilonx = epsscalefactor * float(np.mean(epsx_vs_time))
    return epsilonx, epsx_vs_time