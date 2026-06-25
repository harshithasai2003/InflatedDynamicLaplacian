"""diffusion_maps.py
Kernel Computation and Temporal Laplacian """

import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree

# Helpers
def _rangesearch_kdtree(data, r):
    """ Range search with Euclidean distances fully vectorised """
    tree = KDTree(data, leafsize=40)
    idx_list = tree.query_ball_point(data, r, workers=-1)   # C level parallel
    D_list = []
    for i, nbrs in enumerate(idx_list):
        nbrs = np.asarray(nbrs, dtype=np.intp)
        diff = data[i] - data[nbrs]                         # (k, d)
        sq   = np.einsum('ij,ij->i', diff, diff)            
        D_list.append(np.sqrt(sq))
    return idx_list, D_list

# Building Kernel which is Sparse for Computations
def _build_sparse_kernel(idx_list, D_list, epsilon, n):
    """ Builds Sparse Kernel """
    lengths = np.fromiter((len(idx) for idx in idx_list), dtype=np.intp, count=n)
    rows    = np.repeat(np.arange(n, dtype=np.int32), lengths)
    cols    = np.concatenate([np.asarray(idx, dtype=np.int32) for idx in idx_list])
    dists   = np.concatenate(D_list)
    vals    = np.exp(-(dists * dists) / epsilon)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

# Performing Normalizations
def _density_normalise_and_markov(A, alpha=1):
    """ Normalizations """
    row_means = np.asarray(A.sum(axis=1)).ravel()
    np.maximum(row_means, 1e-14, out=row_means)
    q = 1.0 / row_means
    Anorm = sp.diags(q, format='csr') @ A @ sp.diags(q, format='csr')
    row_sums = np.asarray(Anorm.sum(axis=1)).ravel()
    np.maximum(row_sums, 1e-14, out=row_sums)
    DMM = sp.diags(1.0 / row_sums, format='csr') @ Anorm
    return DMM

# Diffusion Maps Matrix Creation
def diffusion_maps_matrix(pts, epsilon):
    """ Builds the diffusion maps Matrix """
    pts  = np.asarray(pts, dtype=np.float64)
    data = pts.T
    n    = data.shape[0]
    r    = np.sqrt(5.0 * epsilon)
    idx_list, D_list = _rangesearch_kdtree(data, r)
    A_raw = _build_sparse_kernel(idx_list, D_list, epsilon, n)
    A   = A_raw - sp.diags(A_raw.diagonal(), format='csr') + sp.eye(n, format='csr')
    DMM = _density_normalise_and_markov(A, alpha=1)
    return DMM, A

# Diffusion Maps Matrix For Custom Distance
def diffusion_maps_matrix_customdist(pts, distfun, epsilon):
    """Custom distance """
    pts  = np.asarray(pts, dtype=np.float64)
    data = pts.T
    n    = data.shape[0]
    r    = np.sqrt(np.log(100.0)) * np.sqrt(epsilon)
    idx_list, D_list = [], []
    for i in range(n):
        row_dist = distfun(data[i:i+1], data).ravel()
        mask = row_dist <= r
        idx_list.append(np.where(mask)[0])
        D_list.append(row_dist[mask])
    A_raw = _build_sparse_kernel(idx_list, D_list, epsilon, n)
    A     = A_raw - sp.diags(A_raw.diagonal(), format='csr') + sp.eye(n, format='csr')
    DMM   = _density_normalise_and_markov(A, alpha=1)
    return DMM, A

# Diffusion Maps Matrix With Epsilon Heuristic
def diffusion_maps_matrix_epsauto(pts, epsrange, numeps, plot=False):
    """ Diffusion maps with automatic epsilon selection """
    pts  = np.asarray(pts, dtype=np.float64)
    data = pts.T
    n    = data.shape[0]
    arr_eps = np.exp(np.linspace(np.log(epsrange[0]), np.log(epsrange[1]), numeps))
    r       = np.linalg.norm(pts.max(axis=1) - pts.min(axis=1)) / 2.0
    idx_list, D_list = _rangesearch_kdtree(data, r)
    # Flatten COO once
    lengths   = np.fromiter((len(idx) for idx in idx_list), dtype=np.intp, count=n)
    rows_arr  = np.repeat(np.arange(n, dtype=np.int32), lengths)
    cols_arr  = np.concatenate([np.asarray(idx, dtype=np.int32) for idx in idx_list])
    dist2_arr = np.concatenate(D_list) ** 2                     # (lv,)
    # Vectorised sweep
    log_kvals = -dist2_arr[np.newaxis, :] / arr_eps[:, np.newaxis]
    kvals_all = np.exp(log_kvals)                               # (numeps, lv)
    S         = kvals_all.sum(axis=1) / (n * n)                 # (numeps,)

    log_slope = (np.log(S[1:] / S[:-1]) / np.log(arr_eps[1:] / arr_eps[:-1]))
    # Plotting if required
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(arr_eps, S, '.-', markersize=10)
        plt.loglog(arr_eps[:-1], log_slope, 'r.-')
        plt.show()
    imdS    = int(np.argmax(log_slope))
    epsilon = arr_eps[imdS]
    dim_est = 2 * log_slope[imdS]
    print(f'Epsilon: {epsilon}')
    print(f'Dimension estimate: {dim_est}')
    kvals = kvals_all[imdS]
    A = sp.csr_matrix((kvals, (rows_arr, cols_arr)), shape=(n, n))
    A = A - sp.diags(A.diagonal(), format='csr') + sp.eye(n, format='csr')
    DMM = _density_normalise_and_markov(A, alpha=1)
    return DMM, epsilon

# Temporal Diffusion Maps Matrix Creation
def temp_diffusion_maps_matrix(Tspan, epsilon):
    """ Diffusion maps matrix for the 1D temporal domain."""
    Tspan = np.asarray(Tspan, dtype=np.float64).ravel()
    data  = Tspan[:, np.newaxis]
    m     = len(Tspan)
    r     = np.sqrt(np.log(100.0) * epsilon ** 5)
    idx_list, D_list = _rangesearch_kdtree(data, r)
    A_raw = _build_sparse_kernel(idx_list, D_list, epsilon, m)
    A = A_raw - sp.diags(A_raw.diagonal(), format='csr') + sp.eye(m, format='csr')
    DMM = _density_normalise_and_markov(A, alpha=1)
    return DMM, A

# Temporal Laplacian
def temp_laplace(Tspan):
    """ Finite Difference Second Order Laplacian  """
    ts = np.asarray(Tspan, dtype=np.float64).ravel()
    n  = len(ts)
    hs = np.diff(ts)
    L  = np.zeros((n, n), dtype=np.float64)
    L[0,   0],   L[0,   1]   = -1.0 / hs[0]**2,  1.0 / hs[0]**2
    L[n-1, n-2], L[n-1, n-1] =  1.0 / hs[-1]**2, -1.0 / hs[-1]**2
    if n > 2:
        hm = hs[:-1]; hp = hs[1:]
        ii = np.arange(1, n - 1)
        L[ii, ii - 1] =  2.0 / (hm * (hm + hp))
        L[ii, ii]     = -2.0 / (hm * hp)
        L[ii, ii + 1] =  2.0 / (hp * (hm + hp))
    return L
