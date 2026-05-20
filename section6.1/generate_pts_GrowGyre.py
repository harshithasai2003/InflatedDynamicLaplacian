# generate_pts_GrowGyre.py
""" Generate particle trajectories
Domain: [0,3] x [0,2], Time: [0,1] """


import numpy as np
import os
import time
from scipy.integrate import solve_ivp
from scipy.io import savemat, loadmat
from GrowGyre import grow_gyre
import matplotlib.pyplot as plt

# Parameters
mx, my = 45, 30
Ix, Iy = [0, 3], [0, 2]
T0, TF, dt = 0.0, 1.0, 0.01
Tspan = np.arange(T0, TF + dt, dt)
T = len(Tspan)
strength = 20.0
regime = 'very-abrupt'
N = mx * my
# File name
CACHE = (f"Pts_GrowGyre_regime_{regime}" f"_strength{int(strength)}" f"_{mx}x{my}x{T}.mat")

if os.path.exists(CACHE):
    data = loadmat(CACHE)
    pts = data["pts"]
    assert pts.shape == (2, N, T), (f"Cached pts shape {pts.shape} mismatch.")

else:
    print(f"Generating {N} trajectories x {T} time steps...")
    t_start = time.time()
    # Grid
    xg = np.linspace(Ix[0] + 1e-3 * (Ix[1] - Ix[0]), Ix[1] - 1e-3 * (Ix[1] - Ix[0]), mx)
    yg = np.linspace(Iy[0] + 1e-3 * (Iy[1] - Iy[0]), Iy[1] - 1e-3 * (Iy[1] - Iy[0]), my)
    Xg, Yg = np.meshgrid(xg, yg)
    x0 = Xg.ravel(order='F')
    y0 = Yg.ravel(order='F')
    pts = np.empty((2, N, T), dtype=np.float64)
    for i in range(N):
        sol = solve_ivp(lambda t, s: strength * grow_gyre(t, s, regime=regime),
            [T0, TF], [x0[i], y0[i]], t_eval=Tspan, method='RK45', rtol=1e-9, atol=1e-12)
        pts[:, i, :] = sol.y

    savemat(CACHE, {"pts": pts, "Tspan": Tspan, "mx": mx, "my": my, "strength": strength,
                    "regime": regime})

    print(f"Saved {CACHE} in " f"{time.time() - t_start:.1f}s, " f"shape={pts.shape}")

# Plotting
'''fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(Ix)
ax.set_ylim(Iy)
ax.set_aspect('equal')
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.tick_params(labelsize=12)
ax.set_title('Growing Gyre Flow', fontsize=16)
colors = pts[0,:,0]  # color by initial x-coordinate
sc = ax.scatter(pts[0,:,0], pts[1,:,0], c=colors, s=20, cmap='jet', marker='o', edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Initial x-coordinate', fontsize=12)
cbar.ax.tick_params(labelsize=10)
def update(k):
    sc.set_offsets(np.c_[pts[0,:,k], pts[1,:,k]])
    sc.set_array(colors)
    ax.set_title(f'Time: {Tspan[k]:.2f}', fontsize=16)
    return sc,
for k in range(T):
    update(k)
    plt.pause(0.15)
plt.show()'''