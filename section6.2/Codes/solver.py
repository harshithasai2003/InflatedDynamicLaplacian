""" solver.py 
Inflated Dynamic Laplacian for Polar Vortex """

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull, Delaunay
from spacetime import _matvec_strang


# Alpha-Shape Boundary
def _alpha_shape_boundary_indices(pts, shrink=0.5):
    """ Self-contained 2D alpha-shape boundary """
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    if n < 4:
        return np.arange(n)
    try:
        tri = Delaunay(pts)
    except Exception:
        hull = ConvexHull(pts)
        return np.unique(hull.vertices)
    simplices = tri.simplices
    # circumradius of each triangle
    a = pts[simplices[:, 0]]; b = pts[simplices[:, 1]]; c = pts[simplices[:, 2]]
    ab = np.linalg.norm(a - b, axis=1)
    bc = np.linalg.norm(b - c, axis=1)
    ca = np.linalg.norm(c - a, axis=1)
    s = (ab + bc + ca) / 2.0
    area = np.sqrt(np.maximum(s * (s - ab) * (s - bc) * (s - ca), 1e-30))
    circumradius = (ab * bc * ca) / (4.0 * area)
    # Cutoff radius scaled
    shrink = float(np.clip(shrink, 0.0, 1.0))
    edge_lengths = np.concatenate([ab, bc, ca])
    typical_spacing = np.median(edge_lengths)
    factor = 4.5 - 3.0 * shrink                      # shrink = 0.5 to factor=3.0
    cutoff = factor * typical_spacing
    keep = circumradius <= max(cutoff, 1e-12)
    if not np.any(keep):
        hull = ConvexHull(pts)
        return np.unique(hull.vertices)
    kept_simplices = simplices[keep]
    edges = {}
    for tr in kept_simplices:
        for e in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])):
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edges[key] = edges.get(key, 0) + 1
    boundary_pts = set()
    for (i, j), count in edges.items():
        if count == 1:
            boundary_pts.add(i); boundary_pts.add(j)
    if not boundary_pts:
        hull = ConvexHull(pts)
        return np.unique(hull.vertices)
    return np.array(sorted(boundary_pts))

# Dirichlet Boundary Indices
def compute_boundary_indices(SpacePointsarray, method=2, boundary_method='convexhull', shrink=0.5):
    """ Computes global Dirichlet Boundary Indices """
    _, N, T = SpacePointsarray.shape
    b_locind = []
    for k in range(T):
        slice_pts = SpacePointsarray[:, :, k].T
        if boundary_method == 'alphashape':
            b_locind.append(_alpha_shape_boundary_indices(slice_pts, shrink=shrink))
        else:
            hull = ConvexHull(slice_pts)
            b_locind.append(np.unique(hull.vertices))
    if method == 1:
        b_glob = np.unique(np.concatenate([N * k + b_locind[k] for k in range(T)]))
    else:
        pieces = []
        for k in range(T):
            loc = b_locind[k]                        # shape (m_k,)
            # add offset for every time slice
            pieces.append((np.arange(T, dtype=np.int64)[:, None] * N + loc[None, :]).ravel())
        b_glob = np.unique(np.concatenate(pieces))
    b_glob_bool = np.zeros(N * T, dtype=bool)
    b_glob_bool[b_glob] = True
    return b_glob_bool, b_locind
# Zero Rows and Columns
def _zero_rowcol_csr(A, idx):
    A = A.copy()                                   
    idx_arr = np.asarray(idx, dtype=np.int32)
    # zero rows, For each boundary row i, null out the stored values in-place.
    for i in idx_arr:
        A.data[A.indptr[i]:A.indptr[i + 1]] = 0.0
    # zero columns 
    col_mask = np.zeros(A.shape[1], dtype=bool)
    col_mask[idx_arr] = True
    A.data[col_mask[A.indices]] = 0.0
    A.eliminate_zeros()
    return A

# Building LinearOperator for Eigensolver
def build_Pfun(Pthalf, Px, b_globind=None, dirichlet=True, Pthalf_interval=None, N=None, T=None):
    """ Construct the LinearOperator for the Spacetime Eigenproblem """
    NT       = Px.shape[0]
    use_fast = (Pthalf_interval is not None) and (N is not None) and (T is not None)
    if dirichlet and b_globind is not None:
        b_idx = np.where(b_globind)[0]
        diag_vals = np.ones(NT, dtype=float)
        diag_vals[b_idx] = 0.0
        D = sp.diags(diag_vals, format='csr')
        if use_fast:
            Px_dir = _zero_rowcol_csr(Px, b_idx)
            def _matvec(x):
                x = np.asarray(x, dtype=float)
                xc = x.copy(); xc[b_idx] = 0.0
                y  = _matvec_strang(xc, Pthalf_interval, Px_dir, N, T)
                y[b_idx] = 0.0
                return y
        else:
            def _matvec(x):
                x = np.asarray(x, dtype=float)
                return D @ (Pthalf @ (D @ (Px @ (D @ (Pthalf @ (D @ x))))))
    else:
        D = None
        if use_fast:
            def _matvec(x):
                return _matvec_strang(np.asarray(x, dtype=float), Pthalf_interval, Px, N, T)
        else:
            def _matvec(x):
                return Pthalf @ (Px @ (Pthalf @ np.asarray(x, dtype=float)))
    Pfun = spla.LinearOperator((NT, NT), matvec=_matvec, dtype=float)
    return Pfun, D

# Eigen Decomposition
def compute_eigenmodes(Pfun, NT, num_evals=10, tol=1e-6,
                       maxiter=None, ncv=None, v0=None, random_seed=42):
    """ Top Eigenmodes of the Spacetime Diffusion Operator """
    if ncv     is None: ncv     = min(NT, max(3 * num_evals + 1, 60))
    if maxiter is None: maxiter = min(NT, 3000)
    if v0      is None:
        rng = np.random.default_rng(random_seed)
        v0  = rng.standard_normal(NT)
        v0 /= np.linalg.norm(v0)
    evals_raw, evecs_raw = spla.eigs(Pfun, k=num_evals, which='LM',tol=tol, maxiter=maxiter, 
                                     ncv=ncv, v0=v0, return_eigenvectors=True)
    MaxImEval = float(np.max(np.abs(np.imag(evals_raw))))
    B2      = np.argsort(-np.real(evals_raw))
    evals_s = np.real(evals_raw[B2])
    evecs   = np.real(evecs_raw[:, B2]) * np.sqrt(NT)
    return evals_s, evecs, MaxImEval, B2

# Decay Rate 
def compute_decay_rates(evecs, Pthalf, Px, epsilonx, k=0):
    phi   = evecs[:, k]
    norm2 = np.dot(phi, phi)
    inner_t = np.dot(phi, Pthalf @ (Pthalf @ phi))
    inner_x = np.dot(phi, Px @ phi)
    rate_t = 4.0 * np.log(inner_t / norm2) / epsilonx
    rate_x = 4.0 * np.log(inner_x / norm2) / epsilonx
    return rate_t, rate_x

# Time Averaged Spatial Operator
def compute_Px_avg(Px, Num_Space_Points, Num_Time_Points, b_globind=None, dirichlet=True):
    """ Time Averaged Spatial Diffusion Operator """
    N = Num_Space_Points
    T = Num_Time_Points
    if dirichlet and b_globind is not None:
        b_idx  = np.where(b_globind)[0]
        Px_dir = _zero_rowcol_csr(Px, b_idx)
    else:
        Px_dir = Px.tocsr()
    # Accumulate T sparse (N,N) blocks
    acc = sp.csr_matrix((N, N), dtype=float)
    for k in range(T):
        sl  = slice(k * N, (k + 1) * N)
        acc = acc + Px_dir[sl, sl]
    Px_avg = (acc / T).toarray()
    return Px_avg

def compute_Px_avg_eigenmodes(Px_avg, num_evals=10, maxiter=None, ncv=None, v0=None, random_seed=42):
    """Eigen Decomposition of Px_avg and then Sorted in Descending Order """
    N = Px_avg.shape[0]
    A = sp.csr_matrix(Px_avg)
    if ncv     is None: ncv     = min(N, max(3 * num_evals + 1, 60))
    if maxiter is None: maxiter = min(N, 3000)
    if v0      is None:
        rng = np.random.default_rng(random_seed)
        v0  = rng.standard_normal(N)
        v0 /= np.linalg.norm(v0)
    avg_evals_raw, avg_evecs_raw = spla.eigs(A, k=num_evals, which='LM', maxiter=maxiter, 
                                             ncv=ncv, v0=v0)
    B1          = np.argsort(-np.real(avg_evals_raw))
    avg_evals_s = np.real(avg_evals_raw[B1])
    avg_evecs   = np.real(avg_evecs_raw[:, B1])
    return avg_evals_s, avg_evecs, B1
