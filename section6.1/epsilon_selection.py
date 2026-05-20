""" epsilon_selection.py
Optimal bandwidth epsilon via log-log slope heuristic. """

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d


# Heuristic computation
def _compute_S(X: np.ndarray, arr_eps: np.ndarray, eps_max: float) -> np.ndarray:
    """ Computes S(eps) = (1/N^2) * sum_{i,j} K_ij(eps) """
    N = X.shape[0]
    r = np.sqrt(5.0 * eps_max)
    d_all = pdist(X)
    mask  = d_all <= r
    d_cut = d_all[mask]
    if len(d_cut) == 0:
        return np.full(len(arr_eps), 1.0 / N)
    # Kernel
    K_off = np.exp(-0.25 * d_cut[None, :]**2 / arr_eps[:, None]).sum(axis=1) 
    return (N + 2.0 * K_off) / N**2

def compute_epsilon(X: np.ndarray, eps_min: float = 1e-4, eps_max: float = 1.0, 
                    n_eps: int = 40, verbose: bool = True) -> tuple:
    """ Selects optimal epsilon """
    arr_eps = np.exp(np.linspace(np.log(eps_min), np.log(eps_max), n_eps))
    S = _compute_S(X, arr_eps, eps_max)
    # Compute log-log slope
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slope = np.gradient(np.log(S), np.log(arr_eps))

    # Smooth slope to reduce noise
    slope_smooth = gaussian_filter1d(slope, sigma=1)
    slope_max = slope_smooth.max()
    threshold = 0.95 * slope_max
    plateau_idx = np.where(slope_smooth >= threshold)[0]
    if len(plateau_idx) == 0:
        i_opt = int(np.argmax(slope_smooth))
    else:
        i_opt = plateau_idx[len(plateau_idx)//2]
    eps_opt = float(arr_eps[i_opt])
    dim_est = 2.0 * float(slope_smooth[i_opt])
    if verbose:
        print(f'  Optimal epsilon = {eps_opt:.6f}')
        print(f'  Estimated intrinsic dimension = {dim_est:.2f}')
    return eps_opt, dim_est, arr_eps, S, slope_smooth


def plot_epsilon_heuristic(arr_eps, S, slope, eps_opt, dim_est,
                           paper_val: float = None, save: str = None):
    """ Plots the log-log S(epsilon) heuristic """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(arr_eps, S, 'o-', ms=6, lw=1.5, color='steelblue', label=r'$S(\varepsilon)$')
    ax.axvline(eps_opt, color='crimson', ls='--', lw=1.8,
               label=r'$\varepsilon^ = %.4f$' % eps_opt)
    if paper_val is not None:
        ax.axvline(paper_val, color='green', ls=':', lw=1.5,
                   label='Paper: %.4f' % paper_val)

    ax.set_xlabel(r'$\varepsilon$', fontsize=13)
    ax.set_ylabel(r'$S(\varepsilon)$', fontsize=13)
    ax.set_title('Log-log epsilon heuristic', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('Optimal epsilon heuristic = %.6f, d_hat = %.2f' % (eps_opt, dim_est), fontsize=11)
    plt.tight_layout(pad=2.0)
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f'  Saved {save}')
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    t0 = time.time()
    from generate_pts_GrowGyre import pts  
    paper_val = 0.0044
    save_npy  = 'epsilon_opt.npy'
    save_png  = 'epsilon_opt.png'
    X = pts[:, :, 0].T.astype(np.float64)
    N = X.shape[0]
    print(f'  N = {N}')
    eps_opt, dim_est, arr_eps, S, slope = compute_epsilon(X, verbose=True)
    print(f'\n  Paper reference value : {paper_val}')
    print(f'  Computed epsilon      : {eps_opt:.6f}')
    np.save(save_npy, np.float64(eps_opt))
    print(f'  Saved {save_npy}')
    plot_epsilon_heuristic(arr_eps, S, slope, eps_opt, dim_est, paper_val=paper_val, save=save_png)