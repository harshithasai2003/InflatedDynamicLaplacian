"""Compute the temporal diffusion parameter a using the heuristic."""

import warnings
import numpy as np
import matplotlib.pyplot as plt


def compute_a(tau: float, s1: float, s2: float, multiplier: float = 1.0) -> dict:
    """ Computes the temporal diffusion parameter a"""

    s_max = max(s1, s2)
    lambda_spat = -(np.pi / s_max) ** 2              # Spatial eigenvalue
    a_min = tau / s_max                              # Minimum a
    a = multiplier * a_min                           # Final parameter
    return {'a_min': a_min, 'a': a, 'lambda_spat': lambda_spat,}


def plot_a_heuristic(tau: float, result: dict, paper_a: float = None, save: str = None):
    """ Plots the heuristic selection of a """

    a_vals = np.linspace(0.01 * result['a_min'], 3.0 * result['a'], 300)
    lam_temp = -(a_vals * np.pi / tau) ** 2
    lam_spat = result['lambda_spat']
    plt.figure(figsize=(7, 5))

    # Temporal eigenvalue 
    plt.plot(a_vals, lam_temp, lw=2, color='darkorange', label=r'$\lambda_1^{\rm temp}(a)$')

    # Spatial eigenvalue
    plt.axhline(lam_spat, color='steelblue', ls='--', lw=2, label=r'$\lambda_1^{\rm spat}$')

    # Computed a
    plt.axvline(result['a'], color='green', lw=2, label=fr'$a={result["a"]:.4f}$')

    # Paper Reference value plotting
    if paper_a is not None:
        plt.axvline(paper_a, color='purple', ls=':', lw=2, label=fr'paper $a={paper_a}$' )

    plt.xlabel('$a$', fontsize=12)
    plt.ylabel('eigenvalue', fontsize=12)
    plt.title('Temporal diffusion parameter heuristic')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    from generate_pts_GrowGyre import Tspan, Ix, Iy
    paper_a = 0.3340
    multiplier = 1.0
    tau = float(np.asarray(Tspan).ravel()[-1] - np.asarray(Tspan).ravel()[0])
    s1 = float(np.asarray(Ix).ravel()[-1] - np.asarray(Ix).ravel()[0])
    s2 = float(np.asarray(Iy).ravel()[-1] - np.asarray(Iy).ravel()[0])
    result = compute_a(tau, s1, s2, multiplier=multiplier)
    print(f'Paper Reference a    = {paper_a}')
    print(f'Computed a = {result["a"]:.6f}')
    np.save('a_value.npy', np.float64(result['a']))
    plot_a_heuristic(tau, result, paper_a=paper_a, save='a_heuristic.png')