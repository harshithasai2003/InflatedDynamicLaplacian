""" seba.py
Sparse Eigenbasis Approximation (SEBA) algorithm """

import numpy as np
import scipy.linalg as la

def seba(V, Rinit=None):
    """ Sparse Eigenbasis Approximation SEBA """
    V = np.asarray(V, dtype=float)
    # Enforce orthonormality 
    V, _ = np.linalg.qr(V)
    p, r = V.shape
    mu = 0.99 / np.sqrt(p)
    S = np.zeros_like(V)
    if Rinit is None:
        Rnew = np.eye(r)
    else:
        Rinit = np.asarray(Rinit, dtype=float)
        # Ensure orthonormality of Rinit via SVD polar factor
        P, _, Q = la.svd(Rinit, full_matrices=False)
        Rinit = P @ Q
        Rnew = Rinit.copy()
        if np.linalg.det(Rnew) < 0:
            Rnew[:, 0] *= -1
    R = np.zeros((r, r))                                 # Differs from Rnew
    # Main iteration
    while np.linalg.norm(Rnew - R) > 1e-12:
        R = Rnew.copy()
        Z = V @ R.T
        # Thresholding 
        for i in range(r):
            S[:, i] = np.sign(Z[:, i]) * np.maximum(np.abs(Z[:, i]) - mu, 0.0)
            norm_si = np.linalg.norm(S[:, i])
            if norm_si > 0:
                S[:, i] /= norm_si
        # Polar decomposition 
        P, _, Q = la.svd(S.T @ V, full_matrices=False)
        Rnew = P @ Q
        if np.linalg.det(Rnew) < 0:
            Rnew[:, 0] *= -1
    # Choosing correct parity and scale so largest value is 1
    for i in range(r):
        S[:, i] *= np.sign(S[:, i].sum())
        max_si = S[:, i].max()
        if max_si > 0:
            S[:, i] /= max_si
    # Sorting
    I = np.argsort(-np.min(S, axis=0))   
    S = S[:, I]
    return S, R, I

SEBA = seba
