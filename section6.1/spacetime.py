""" spacetime.py
Implementation the Strang-splitting approximation """

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from diffusion_maps import diffusion_maps_matrix, temp_laplace

def make_spacetime_diffusion_mat_productform(SpacePointsarray: np.ndarray,
    TimePoints: np.ndarray, epsilonx: float, epsilont: float, t_factor: float,):
    _, N, T = SpacePointsarray.shape 
    Px_blocks = []
    for k in range(T):
        Px_slice, _ = diffusion_maps_matrix(SpacePointsarray[:, :, k], epsilonx)
        Px_blocks.append(Px_slice)
    Px = sparse.block_diag(Px_blocks, format="csr")
    Lt_interval = temp_laplace(TimePoints)                              # (T, T)
    Pthalf_interval = expm(Lt_interval * t_factor * epsilonx / 2)         # (T, T)
    Pthalf = sparse.kron(sparse.csr_matrix(Pthalf_interval), sparse.eye(N, format="csr"),
                                              format="csr",)

    return Pthalf, Px, Pthalf_interval, Lt_interval


def matvec_Pa(x: np.ndarray,Pthalf_interval: np.ndarray, Px_dir: sparse.csr_matrix, 
               N: int, T: int,) -> np.ndarray:
    def apply_Pthalf(v):
        V = v.reshape(T, N)                      
        return (Pthalf_interval @ V).ravel()
    y = apply_Pthalf(x)
    y = Px_dir @ y
    y = apply_Pthalf(y)
    return np.real(y)
