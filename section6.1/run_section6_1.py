""" run_section6_1.py """

import argparse
import time
import sys
from pathlib import Path


if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver import InflatedDynamicLaplacian
from plotting import (fig_streamfunction, fig_eigenvalues, fig_3d_spacetime,
                      fig_snapshot_grid, fig_high_variance_trajectories, fig_px_avg_modes,)

OUT = Path("results_section6_1")
OUT.mkdir(parents=True, exist_ok=True)

# Parameters
EPSILON = 0.0044
A_FACTOR = 3.0
NUM_EVALS = 40
IX = (0, 3)
IY = (0, 2)
ABS_CUT = 0.25


# Utilities
def load_mat(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"\nMAT file not found:\n{path}")
    print("\nLoading MAT file:")
    print(path)
    data = sio.loadmat(str(path), squeeze_me=True)
    required_keys = ["pts", "Tspan"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"MAT file missing required variable '{key}'")

    return data


def build_spacetime_arrays(data):
    pts = np.asarray(data["pts"], dtype=np.float64)
    Tspan = np.asarray(data["Tspan"], dtype=np.float64)
    if pts.ndim != 3:
        raise ValueError(f"'pts' must have shape (2,N,T), got {pts.shape}")

    _, N, T = pts.shape

    print(f"Detected trajectory array shape: {pts.shape}")
    SpacePointsarray = np.zeros((2, N, T), dtype=np.float64)
    for k in range(T):
        SpacePointsarray[0, :, k] = pts[0, :, k]
        SpacePointsarray[1, :, k] = pts[1, :, k]

    M0_pts = np.zeros_like(SpacePointsarray)
    for k in range(T):
        M0_pts[:, :, k] = SpacePointsarray[:, :, 0]

    return SpacePointsarray, Tspan, M0_pts


# Main computation
def run(mat_path):
    print("Switching Double Gyre")
    data = load_mat(mat_path)
    SpacePointsarray, TimePoints, M0_pts = build_spacetime_arrays(data)
    _, N, T = SpacePointsarray.shape
    print(f"\nParticles : {N}")
    print(f"Time steps: {T}")
    t0 = time.time()
    idl = InflatedDynamicLaplacian(epsilonx=EPSILON, a_factor=A_FACTOR, dirichlet=False,
        num_evals=NUM_EVALS, eps_scale=2.0, verbose=True, )

    idl.fit(SpacePointsarray, TimePoints, Ix=IX, Iy=IY,)
    elapsed = time.time() - t0
    lam = np.asarray(idl.laplacian_eigenvalues())
    print("\nTop eigenvalues:\n")
    for i, val in enumerate(lam[:10]):
        mode_type = ("spatial"
            if i in idl.dynmodes
            else "temporal")

        print(f"k={i+1:2d}   " f"L={val:+.6f}   " f"({mode_type})" )

    print("\nGenerating figures...\n")

    try:
        # Stream Function
        fig_streamfunction(outpath=OUT / "figA_streamfunction.png")
        print("Generated streamfunction.png")

        # Eigen Values
        fig_eigenvalues(idl, max_show=10, outpath=OUT / "figB_eigenvalues.png",)
        print("Generated eigenvalues.png")

        # 3D Plot
        dyn = np.asarray(idl.dynmodes)
        m2 = int(dyn[1])
        fig_3d_spacetime(idl, mode_idx=m2, domain="M0", abs_cutoff=ABS_CUT,M0_pts=M0_pts, 
                         time_step=1, outpath=OUT / "figC_3d_mode2_M0.png",)

        fig_3d_spacetime(idl, mode_idx=m2, domain="M1", abs_cutoff=ABS_CUT, M0_pts=M0_pts,
                          time_step=1, outpath=OUT / "figC_3d_mode2_M1.png",)

        print("Generated 3D View")

        # Snapshots
        fig_snapshot_grid(idl, mode_idx=m2, domain="M1", abs_cutoff=ABS_CUT,
                          n_times=10, outpath=OUT / "figD_snapshots_M1.png", )

        fig_snapshot_grid(idl, mode_idx=m2, domain="M0", abs_cutoff=ABS_CUT,
                          n_times=10, M0_pts=M0_pts, outpath=OUT / "figD_snapshots_M0.png",)

        print("Generated Snapshots")

        # High Variance

        fig_high_variance_trajectories(idl, mode_idx=m2, domain="M1", top_pct=0.05,
                                       abs_cutoff=ABS_CUT,add_streamfunction=True, 
                                       M0_pts=M0_pts, outpath=OUT / "figG_highvar_M1.png",)

        fig_high_variance_trajectories(idl, mode_idx=m2, domain="M0", top_pct=0.05,
                                       abs_cutoff=ABS_CUT,add_streamfunction=False, 
                                       M0_pts=M0_pts, outpath=OUT / "figG_highvar_M0.png",)

        print("Generated High Variance Trajectories")

        # Modes
        fig_px_avg_modes(idl, num_modes=8, t_idx=0, outpath=OUT / "figI_px_avg_modes.png",)
        print("Generated figI")

    except Exception as e:
        print("\nFigure generation failed:")
        print(e)

    print("\nAll outputs saved to:")
    print(OUT.resolve())
    return idl


# Main
if __name__ == "__main__":

    DEFAULT_MAT = (Path(__file__).resolve().parent
                   / "Pts_GrowGyre_regime_very-abrupt_strength20_45x30x101.mat")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default=str(DEFAULT_MAT),
                        help="Path to MATLAB trajectory file",)

    args = parser.parse_args()
    print("\nUsing MAT file:")
    print(args.mat)
    try:
        run(args.mat)
    except Exception as e:
        print("ERROR")
        print(e)
        sys.exit(1)