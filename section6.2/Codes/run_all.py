""" run_all.py 
Polar vortex Example with Inflated Dynamic Laplacian """

import numpy as np
import scipy.io as sio
import os
import sys
import time
# Windows Encoding Error Fix
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
# Imports 
from trajectories import tracer_isentropic_2d
from spacetime import make_spacetime_DiffusionMat_productform
from solver import (compute_boundary_indices, build_Pfun, compute_eigenmodes, compute_decay_rates,
                    compute_Px_avg, compute_Px_avg_eigenmodes)
from a_selection import compute_a
from plotting import (plot_eigenvalues, save_spatial_eigenmode_movies,
                      plot_px_avg_eigenvalues, save_Px_avg_movies,
                      plot_red_region_snapshots, plot_vortex_shape_deformation)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
# Saving Integrated Trajectories
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
TRAJ_CACHE_FILE = os.path.join(CACHE_DIR, 'trajectories_cache.npz')
tstart = 1
tend   = 240
# Load trajectory data 
if os.path.exists(TRAJ_CACHE_FILE):
    print(f' Loading Trajectories {TRAJ_CACHE_FILE}  ')
    t0_wall = time.time()
    _cache = np.load(TRAJ_CACHE_FILE)
    xdata1 = _cache['xdata1']
    ydata1 = _cache['ydata1']
    print(f' Loaded cached trajectories in {time.time() - t0_wall:.2f}s ')
else:
    print(' Loading and Integrating Trajectories from Dataset')
    [X, Y] = np.meshgrid(np.arange(-7000, 7000 + 187.5, 187.5), np.arange(-7000, 7000 + 187.5, 187.5))
    x0 = X.ravel(); y0 = Y.ravel()
    ind = np.linalg.norm(np.column_stack([x0, y0]), axis=1) < 7000
    x0 = x0[ind]; y0 = y0[ind]
    t0_wall = time.time()
    xdata, ydata = tracer_isentropic_2d(x0, y0, tend, tstart, data_dir='data')
    xdata1 = xdata[:, ::8]
    ydata1 = ydata[:, ::8]
    print(f' Integrated trajectories in {time.time() - t0_wall:.2f}s ')
    # Save to cache for future runs
    t0_wall = time.time()
    np.savez(TRAJ_CACHE_FILE, xdata1=xdata1, ydata1=ydata1)
    print(f' Saved trajectories to {TRAJ_CACHE_FILE} in {time.time() - t0_wall:.2f}s ')

# Initial Setup
Total_days       = xdata1.shape[1]
Time_sample_rate = 2
TimePoints       = np.arange(1, Total_days + 1, Time_sample_rate)
Num_Time_Points  = len(TimePoints)
timeunits_days = Time_sample_rate / 4
time_units     = f'{timeunits_days} days'
Num_Space_Points = int(np.ceil(xdata1.shape[0] / 1))
SpacePointsarray = np.zeros((2, Num_Space_Points, Num_Time_Points))
for k in range(Num_Time_Points):
    SpacePointsarray[0, :, k] = xdata1[:, TimePoints[k] - 1]
    SpacePointsarray[1, :, k] = ydata1[:, TimePoints[k] - 1]
dirichlet = True
BC        = 'dir'
M1_pts = SpacePointsarray.copy()
M0_pts = np.zeros_like(SpacePointsarray)
M0idx = np.argsort(SpacePointsarray[0, :, 0])
for k in range(Num_Time_Points):
    M0_pts[:, :, k] = SpacePointsarray[:, :, 0]

# Domain Parameters
num_evals = 10
Domain    = 'M1'
xmax0 = SpacePointsarray[0, :, 0].max();  xmin0 = SpacePointsarray[0, :, 0].min()
ymax0 = SpacePointsarray[1, :, 0].max();  ymin0 = SpacePointsarray[1, :, 0].min()
rad0  = (xmax0 - xmin0) / 2.0
xmax = SpacePointsarray[0].max();  xmin = SpacePointsarray[0].min()
ymax = SpacePointsarray[1].max();  ymin = SpacePointsarray[1].min()
# Epsilon Value 
epsilonx = 6835*4
epsilont = 1
# Temporal Coupling Parameter a 
a_factor = 1.0
a, t_factor, a_min = compute_a(Total_days, rad0, dirichlet=dirichlet,a_factor=a_factor,verbose=True)
# Boundary indices
t0_wall = time.time()
b_globind, b_locind = compute_boundary_indices(SpacePointsarray, method=2)
# Spacetime diffusion matrix
print(' Building Spacetime Diffusion Matrix ')
t0_wall = time.time()
Pthalf, Px, Pthalf_interval, Lt_interval = \
    make_spacetime_DiffusionMat_productform(SpacePointsarray, TimePoints, epsilonx, 
                                            epsilont, t_factor)


# Eigenmodes
print(' Computing Eigenmodes ')
t0_wall = time.time()
Pfun, D = build_Pfun(Pthalf, Px, b_globind=b_globind, dirichlet=dirichlet,
    Pthalf_interval=Pthalf_interval,  N=Num_Space_Points, T=Num_Time_Points)
NT = Num_Space_Points * Num_Time_Points
evals_s, evecs, MaxImEval, B2 = compute_eigenmodes(Pfun, NT, num_evals=num_evals)
print(" Computation Done ")
evecs_3d = evecs.reshape(Num_Time_Points, Num_Space_Points, num_evals).transpose(1, 0, 2)
dynmodes  = list(range(num_evals))
tempmodes = []
# Decay rates
rate_t, rate_x = compute_decay_rates(evecs, Pthalf, Px, epsilonx, k=0)
MM_pts = M1_pts                                          # plot on M1 
# Eigenvalue Spectrum
print(' Plotting Eigenvalues ')
plot_eigenvalues(evals_s, epsilonx, dynmodes=dynmodes, tempmodes=tempmodes,
                  results_dir=RESULTS_DIR, save=True)

# Title Heads
mov_fn_head = (f'mov_PV_{tend}_{Domain}_{BC}_a_{a}_eps_{epsilonx}'
              f'_TimeSampleRate_{Time_sample_rate}_')

# Spatial eigenmode movies
print(' Animating Spatial Eigenmodes ')
save_spatial_eigenmode_movies(evecs_3d, MM_pts, b_globind, Num_Space_Points, TimePoints,
    dynmodes, num_spat_mode_movies=2, MSR=1, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
    mov_fn_head=mov_fn_head, results_dir=RESULTS_DIR)

# Time Averaged Spatial Operator and Eigenmodes
Px_avg = compute_Px_avg(Px, Num_Space_Points, Num_Time_Points, b_globind=b_globind, 
                        dirichlet=dirichlet)
avg_evals_s, avg_evecs, B1 = compute_Px_avg_eigenmodes(Px_avg, num_evals=num_evals)
# Eigenvalue Spectrum 
plot_px_avg_eigenvalues(avg_evals_s, epsilonx, num_evals, results_dir=RESULTS_DIR)
mov_avg_fn_head = (f'mov_PV_{tend}_{Domain}_{BC}_eps_{epsilonx}'
                   f'_TimeSampleRate_{Time_sample_rate}_')
save_Px_avg_movies(MM_pts, avg_evecs, b_globind, Num_Space_Points, TimePoints=TimePoints,
    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mov_avg_fn_head=mov_avg_fn_head,
    num_avg_movies=1, Num_Time_Points=Num_Time_Points, results_dir=RESULTS_DIR)

# Polar Vortex Core Snapshots
print(' Plotting Snapshots ')
plot_red_region_snapshots(MM_pts, avg_evecs, TimePoints, b_globind=b_globind,
    Num_Space_Points=Num_Space_Points, mode_num=1, red_threshold=0.5, n_panels=12,
    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, results_dir=RESULTS_DIR)

# Vortex Shape Deformation
print(' Plotting Deformation ')
plot_vortex_shape_deformation(MM_pts, avg_evecs, TimePoints, mode_num=1, red_threshold=0.5,
                              results_dir=RESULTS_DIR)
print('Done! ')