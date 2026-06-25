"""spacetime.py  
Strang Splitting Spacetime Diffusion Operator """

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from diffusion_maps import diffusion_maps_matrix, diffusion_maps_matrix_customdist, temp_laplace

# name Alias for Imports
def tempLaplace(Tspan):
    """Name Alias of vectorised temp_laplace in diffusion_maps """
    return temp_laplace(Tspan)

# Fast Strang Splitting
def _matvec_strang(v, Pthalf_interval, Px_op, N, T):
    """ Applies  P = Pthalf @ Px @ Pthalf  """
    V = v.reshape(T, N)
    V = Pthalf_interval @ V                           # (T, T) @ (T, N)  →  (T, N) 
    y = V.ravel()
    y = Px_op @ y                                     # sparse (NT, NT) @ (NT,)
    V = y.reshape(T, N)
    V = Pthalf_interval @ V
    return np.real(V.ravel())

# Spacetime DiffusionMatrix Product form
def make_spacetime_DiffusionMat_productform(SpacePointsarray, TimePoints, epsilonx, epsilont, 
                                            t_factor, distfun=None):
    """ Builds the Strang Splitting factors of Spacetime Diffusion Matrix """
    _, N, T = SpacePointsarray.shape
    # Spatial blocks
    def _build_block(k):
        if distfun is None:
            P_slice, _ = diffusion_maps_matrix(SpacePointsarray[:, :, k], epsilonx)
        else:
            P_slice, _ = diffusion_maps_matrix_customdist(
                SpacePointsarray[:, :, k], distfun, epsilonx)
        return P_slice
    try:
        from joblib import Parallel, delayed
        blocks = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_build_block)(k) for k in range(T))
    except ImportError:
        blocks = []
        for k in range(T):
            if k % 10 == 0:
                print(f'    spatial block {k+1}/{T} ...', flush=True)
            blocks.append(_build_block(k))
    Px = sp.block_diag(blocks, format='csr')
    # Temporal Laplacian 
    Lt_interval      = temp_laplace(np.asarray(TimePoints, dtype=float))
    Pthalf_interval  = la.expm(Lt_interval * t_factor * (epsilonx / 4.0) / 2.0)
    # Sparse Kronecker
    Pthalf = sp.kron(Pthalf_interval, sp.eye(N), format='csr')
    return Pthalf, Px, Pthalf_interval, Lt_interval

# Non Product form 
def make_spacetime_DiffusionMat(SpacePointsarray, TimePoints, epsilonx, epsilont, t_factor):
    """ Full spacetime diffusion matrix """
    _, nx, nt = SpacePointsarray.shape
    K = sp.lil_matrix((nx * nt, nx * nt), dtype=float)
    for k in range(nt):
        _, Ax_slice = diffusion_maps_matrix(SpacePointsarray[:, :, k], epsilonx)
        K[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = Ax_slice
    K = K.tocsr()
    _, At = diffusion_maps_matrix(TimePoints[np.newaxis, :], t_factor * epsilont)
    K = K + sp.kron(At, sp.eye(nx), format='csr')
    K_dense = K.toarray()
    q       = 1.0 / np.mean(K_dense, axis=1)
    Kdensnorm = np.outer(q, q) * K_dense
    row_sums  = Kdensnorm.sum(axis=1)
    P = sp.diags(1.0 / row_sums) @ sp.csr_matrix(Kdensnorm)
    return P

# Surrogate Operator
def surrogate_operator(ct, a):
    """ Surrogate operator """
    ct = np.asarray(ct, dtype=float).ravel()
    n  = len(ct)
    L  = temp_laplace(np.linspace(0.0, 1.0, n))
    return a ** 2 * L + np.diag(ct)
