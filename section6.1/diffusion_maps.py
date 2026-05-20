""" diffusion_maps.py """

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree


def nndist(A: np.ndarray) -> np.ndarray:
    """ Nearest Neighbour distances """
    tree = KDTree(A)
    dists, _ = tree.query(A, k=2)                  
    return dists[:, 1]


def mean_nndist_sq(pts_slice: np.ndarray) -> float:
    nn = nndist(pts_slice.T)                       # pts_slice.T is (N, 2)
    return float(np.mean(nn) ** 2)


def diffusion_maps_matrix(pts: np.ndarray, epsilon: float):
    """ Builds the diffusion maps """
    m = pts.shape[1]
    data_T = pts.T                        # (N, d)
    r = np.sqrt(5.0 * epsilon)             # Cutoff  r
    tree = KDTree(data_T)
    idx_list = tree.query_ball_point(data_T, r)   
    lv = sum(len(idx_list[i]) for i in range(m))
    rows = np.empty(lv, dtype=np.int32)
    cols = np.empty(lv, dtype=np.int32)
    vals = np.empty(lv, dtype=np.float64)
    icurr = 0
    for i in range(m):
        nbrs = idx_list[i]
        li = len(nbrs)
        for jj, j in enumerate(nbrs):
            diff = data_T[i] - data_T[j]
            d2 = float(np.dot(diff, diff))
            rows[icurr + jj] = i
            cols[icurr + jj] = j
            vals[icurr + jj] = np.exp(-d2 / (4.0 * epsilon))
        icurr += li

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(m, m), dtype=float)
    diag_A = np.array(A.diagonal())
    A = A - sparse.diags(diag_A, format="csr") + sparse.eye(m, format="csr")
    row_means = np.array(A.mean(axis=1)).ravel()   
    q = 1.0 / row_means
    kalpha = q
    Adensnorm = sparse.diags(kalpha) @ A @ sparse.diags(kalpha)
    row_sums = np.array(Adensnorm.sum(axis=1)).ravel()
    DMM = sparse.diags(1.0 / row_sums) @ Adensnorm

    return DMM.tocsr(), A.tocsr()


def temp_laplace(Tspan: np.ndarray) -> np.ndarray:
    """ Laplacian """
    ts = np.asarray(Tspan).ravel()
    n  = len(ts)
    hs = np.diff(ts)          
    L = np.zeros((n, n))
    L[0, 0] = -1.0 / hs[0]**2                   # First row
    L[0, 1] =  1.0 / hs[0]**2
    L[n-1, n-2] =  1.0 / hs[-1]**2              # Last row
    L[n-1, n-1] = -1.0 / hs[-1]**2
    for i in range(1, n - 1):                    # Interior rows
        hm = hs[i - 1]
        hp = hs[i]
        L[i, i-1] =  2.0 / (hm * (hm + hp))
        L[i, i]   = -2.0 / (hm * hp)
        L[i, i+1] =  2.0 / (hp * (hm + hp))
    return L
