""" seba.py 
Sparse eigenbasis approximation """

import numpy as np
from scipy.linalg import svd, qr


def seba(V: np.ndarray, Rinit: np.ndarray = None, tol: float = 1e-12):
    V, _ = qr(V, mode="economic")
    p, r = V.shape
    mu = 0.99 / np.sqrt(p)
    S  = np.zeros((p, r))
    # Initialise rotation
    if Rinit is None:
        Rnew = np.eye(r)
    else:
        U2, _, Vt2 = svd(Rinit, full_matrices=False)
        Rnew = U2 @ Vt2
        if np.linalg.det(Rnew) < 0:
            Rnew[:, 0] *= -1
    R = np.zeros_like(Rnew)
    iters = 0
    while np.linalg.norm(Rnew - R) > tol:
        R = Rnew.copy()
        Z = V @ R.T                              # (p, r)
        # Soft-thresholding
        for i in range(r):
            s = np.sign(Z[:, i]) * np.maximum(np.abs(Z[:, i]) - mu, 0.0)
            nrm = np.linalg.norm(s)
            S[:, i] = s / nrm if nrm > 0 else s
        # Polar decomposition 
        M = S.T @ V                              # (r, r)
        U2, _, Vt2 = svd(M, full_matrices=False)
        Rnew = U2 @ Vt2
        if np.linalg.det(Rnew) < 0:
            Rnew[:, 0] *= -1
        iters += 1
        if iters > 50_000:
            import warnings
            warnings.warn("SEBA: reached 50000 iterations without convergence.")
            break
    for i in range(r):
        S[:, i] *= np.sign(np.sum(S[:, i]))
        mx = np.max(S[:, i])
        if mx > 0:
            S[:, i] /= mx
    I = np.argsort(np.min(S, axis=0))[::-1]                  # Sorting
    S = S[:, I]
    return S, R, I
