""" solver.py """

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, LinearOperator
from diffusion_maps import mean_nndist_sq, temp_laplace
from spacetime import make_spacetime_diffusion_mat_productform, matvec_Pa


class InflatedDynamicLaplacian:
    def __init__(self,epsilonx  : float  = None, a_factor  : float = 3.0, dirichlet : bool = False,   
        num_evals : int = 40, eps_scale : float  = 2.0, verbose   : bool   = True,):
        self.epsilonx   = epsilonx
        self.a_factor   = a_factor
        self.dirichlet  = dirichlet
        self.num_evals  = num_evals
        self.eps_scale  = eps_scale
        self.verbose    = verbose
        self.SpacePointsarray = None
        self.TimePoints       = None
        self.Pthalf_interval  = None   # (T, T)  dense
        self.Px_dir           = None   # (N·T, N·T)  sparse
        self.Px               = None   # (N·T, N·T)  sparse
        self.Lt_interval      = None   # (T, T) 
        self.evals_s   = None          # (K,) real, sorted descending
        self.evecs     = None          # (N·T, K)
        self.evecs_3d  = None          # (N, T, K)  
        self.dynmodes  = None          # spatial mode indices
        self.tempmodes = None          # temporal mode indices
        self.slicemeans  = None        # (T, K)
        self.var_slicemeans = None     # (K,)
        self.L2norms     = None        # (T, K)
        self.a          = None
        self.a_min      = None
        self.t_factor   = None
        self.epsilonx_  = None         
        self.epsilont_  = None


    def fit(self, pts : np.ndarray, TimePoints : np.ndarray, Ix: tuple  = (0, 3),
        Iy: tuple  = (0, 2), Time_sub_sample_rate : int = 1, Space_sub_sample_rate: int = 1, ):
        TimePoints = np.asarray(TimePoints).ravel()[::Time_sub_sample_rate]
        pts = pts[:, ::Space_sub_sample_rate, ::Time_sub_sample_rate]
        _, N, T = pts.shape
        SpacePointsarray = pts.copy()
        self.SpacePointsarray = SpacePointsarray
        self.TimePoints = TimePoints
        if self.verbose:
            print(f"SpacePointsarray: {SpacePointsarray.shape}  (2 * {N} * {T})")
        if self.epsilonx is None:
            epsx_vs_time = np.array([mean_nndist_sq(SpacePointsarray[:, :, k]) for k in range(T)])
            self.epsilonx_ = self.eps_scale * float(np.mean(epsx_vs_time))
            if self.verbose:
                print(f"Auto epsilon_x = {self.epsilonx_:.6f}")
        else:
            self.epsilonx_ = float(self.epsilonx)
        epsilonx = self.epsilonx_
        dt = float(np.max(np.diff(TimePoints)))
        self.epsilont_ = self.eps_scale * dt**2
        tau       = float(TimePoints[-1] - TimePoints[0])
        xlength   = abs(Ix[1] - Ix[0])
        ylength   = abs(Iy[1] - Iy[0])
        if self.dirichlet:
            a_min = tau * np.sqrt(1.0 / xlength**2 + 1.0 / ylength**2)
        else:
            a_min = tau / max(xlength, ylength)     
        self.a_min  = a_min
        self.a      = self.a_factor * a_min
        self.t_factor = self.a**2
        if self.verbose:
            print(f"a_min={a_min:.4f}  a={self.a:.4f}  t_factor={self.t_factor:.4f}")
        if self.verbose:
            print("Building P_x ...")
        Pthalf, Px, Pthalf_interval, Lt_interval = \
            make_spacetime_diffusion_mat_productform(SpacePointsarray, TimePoints,
                                   epsilonx, self.epsilont_, self.t_factor,)
        self.Pthalf_interval = Pthalf_interval   # (T, T)
        self.Px              = Px
        self.Lt_interval     = Lt_interval
        Px_dir = Px.copy()
        if self.dirichlet:
            if self.verbose:
                print("Applying Dirichlet BCs ...")
            b_globind = self._compute_dirichlet_indices(SpacePointsarray, N, T)
            Px_dir = Px_dir.tolil()
            Px_dir[b_globind, :] = 0
            Px_dir[:, b_globind] = 0
            Px_dir = Px_dir.tocsr()
        self.Px_dir = Px_dir
        if self.verbose:
            print(f"Computing top {self.num_evals} eigenpairs of Pa ...")
        def _matvec(x):
            return matvec_Pa(x, Pthalf_interval, Px_dir, N, T)
        linop = LinearOperator(shape=(N * T, N * T), matvec=_matvec, dtype=float)
        raw_evals, raw_evecs = eigs(linop, k=self.num_evals, which="LM", tol=1e-10, maxiter=10000,)
        order = np.argsort(np.real(raw_evals))[::-1]
        raw_evals = raw_evals[order]
        raw_evecs = raw_evecs[:, order]
        for k in range(self.num_evals):
            if not np.isreal(raw_evals[k]):
                u = np.real(raw_evecs[:, k])
                v = np.imag(raw_evecs[:, k])
                for q in range(k, self.num_evals):
                    if (np.abs(raw_evals[k] - np.conj(raw_evals[q])) < 1e-14
                            and not np.isreal(raw_evals[q])):
                        raw_evecs[:, k] = u
                        raw_evecs[:, q] = v
                        break
        self.evals_s = np.real(raw_evals)
        evecs_real   = np.real(raw_evecs)
        evecs_real *= np.sqrt(N * T)
        self.evecs = evecs_real
        self.evecs_3d = evecs_real.reshape(N, T, self.num_evals, order='F')
        self._classify_modes(T)
        if self.verbose:
            print(f"Done.  spatial={len(self.dynmodes)}  temporal={len(self.tempmodes)}")
            print(f"Top-10 nu_k (log(nu)/eps): {self.laplacian_eigenvalues()[:10]}")
        return self


    def _classify_modes(self, T: int):
        """ Classify modes as spatial or temporal """
        if self.dirichlet:
            self.dynmodes       = np.arange(self.num_evals)
            self.tempmodes      = np.array([], dtype=int)
            self.slicemeans     = np.mean(self.evecs_3d, axis=0)       # (T, K)
            self.var_slicemeans = np.var(self.slicemeans, axis=0)
            self.L2norms        = np.linalg.norm(self.evecs_3d, axis=0)
            return
        slicemeans     = np.mean(self.evecs_3d, axis=0)   # (N, T, K).mean(axis=0) to (T, K)
        var_slicemeans = np.var(slicemeans, axis=0)        # (K,)
        self.slicemeans     = slicemeans
        self.var_slicemeans = var_slicemeans
        self.tempmodes = np.where(var_slicemeans >= 0.1)[0]
        self.dynmodes  = np.where(var_slicemeans <  0.1)[0]
        self.L2norms   = np.linalg.norm(self.evecs_3d, axis=0)   # (T, K)

    def _compute_dirichlet_indices(self, SpacePointsarray, N, T):
        from scipy.spatial import ConvexHull
        boundary_spatial = set()
        for k in range(T):
            pts_k = SpacePointsarray[:, :, k].T
            try:
                hull = ConvexHull(pts_k)
                boundary_spatial.update(hull.vertices.tolist())
            except Exception:
                pass
        b_globind = []
        for k in range(T):
            for j in boundary_spatial:
                b_globind.append(k * N + j)
        return np.array(b_globind, dtype=int)

    def laplacian_eigenvalues(self) -> np.ndarray:
     nu = np.real(self.evals_s)
     nu = np.clip(nu, 1e-14, 1.0)
     return np.log(nu) / self.epsilonx_

    def px_average(self):
        _, N, T = self.SpacePointsarray.shape
        acc = sparse.csr_matrix((N, N), dtype=float)
        for k in range(T):
            sl = slice(k * N, (k + 1) * N)
            acc = acc + self.Px_dir[sl, sl]
        return acc / T

    def px_avg_eigs(self, num_evals=20):
        """Eigenpairs """
        Px_avg = self.px_average()
        ev, evec = eigs(Px_avg, k=num_evals, which="LM")
        order = np.argsort(np.real(ev))[::-1]
        return np.real(ev[order]), np.real(evec[:, order])
