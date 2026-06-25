""" trajectories.py
Integrating trajectories """

import numpy as np
import scipy.io as sio
import scipy.interpolate as interp
import os

# km2degSH
def km2degSH(xx, yy):
    """ South Pole Centred Cartesian (km) to (longitude , latitude ) """
    Rad = 6378.135
    xx  = np.asarray(xx, dtype=float)
    yy  = np.asarray(yy, dtype=float)
    r   = np.sqrt(xx**2 + yy**2)
    lat = -90.0 + (180.0 / np.pi) * r / Rad
    lon = np.degrees(np.arctan2(xx, yy)) % 360.0           # vectorised
    return lon, lat

# RK4 integrator 
def myrk4_2dcartisotemp(FunFcn, tspan, y0, ssize):
    """Classical 4th-order RK """
    t0, tfinal = float(tspan[0]), float(tspan[1])
    pm = np.sign(tfinal - t0)
    if ssize < 0: ssize = -ssize
    h  = pm * ssize
    t  = t0
    y  = y0.ravel().copy()
    dt = abs(tfinal - t0)
    N  = int(np.floor(dt / ssize)) + 1
    if (N - 1) * ssize < dt: N += 1
    tout = np.array([t0, t0])
    yout = np.zeros((1, len(y)))
    yout[0] = y
    for k in range(1, N):
        h_eff = min(abs(h), abs(tfinal - t)) * pm
        s1 = FunFcn(t,             y).ravel()
        s2 = FunFcn(t + h_eff/2,   y + h_eff * s1 / 2).ravel()
        s3 = FunFcn(t + h_eff/2,   y + h_eff * s2 / 2).ravel()
        s4 = FunFcn(t + h_eff,     y + h_eff * s3).ravel()
        y  = y + h_eff * (s1 + 2*s2 + 2*s3 + s4) / 6.0
        t  = t + h_eff
        if abs(t - tfinal) < 1e-12: break
    tout[1]  = t
    yout[0]  = y
    return tout, yout

# tracer_isentropic_2d  
def tracer_isentropic_2d(xini, yini, tfinal, tini, data_dir='data'):
    """ Integrates Lagrangian particle trajectories on the 600 K isentropic surface """
    xini    = np.asarray(xini, dtype=float).ravel()
    yini    = np.asarray(yini, dtype=float).ravel()
    num_pts = len(xini)
    # load data 
    vvx = sio.loadmat(os.path.join(data_dir, 'u1km_600Kseptoct2002'))['uc1km'].astype(float)
    vvy = sio.loadmat(os.path.join(data_dir, 'v1km_600Kseptoct2002'))['vc1km'].astype(float)
    # periodic longitude wrap
    vvx = np.concatenate([vvx, vvx[:, 0:1, :]], axis=1)
    vvy = np.concatenate([vvy, vvy[:, 0:1, :]], axis=1)
    xx_lon_raw = sio.loadmat(os.path.join(data_dir, 'lon_isotemp'))['lon'].astype(float)
    yy_lat_raw = sio.loadmat(os.path.join(data_dir, 'lat_isotemp'))['lat'].astype(float)
    xx_lon     = np.concatenate([xx_lon_raw, 360.0 * np.ones((xx_lon_raw.shape[0], 1))], axis=1)
    yy_lat     = np.concatenate([yy_lat_raw, yy_lat_raw[:, 0:1]], axis=1)
    lon_vals = xx_lon[0, :]         # 1-D longitude axis
    lat_vals = yy_lat[:, 0]         # 1-D latitude axis
    nt_data  = vvx.shape[2]
    # Prebuild one interpolator per time level 
    interp_vx = [interp.RegularGridInterpolator((lat_vals, lon_vals), vvx[:, :, k],
            method='linear', bounds_error=False, fill_value=np.nan) for k in range(nt_data)]
    interp_vy = [interp.RegularGridInterpolator((lat_vals, lon_vals), vvy[:, :, k],
            method='linear', bounds_error=False, fill_value=np.nan)for k in range(nt_data)]
    print('  Done.', flush=True)
    # vectorised ODE RHS 
    def odes(t, state):
        x = state[:num_pts]
        y = state[num_pts:]
        x_lon, y_lat = km2degSH(x, y)
        t0_idx = max(0, min(int(np.floor(t)) - 1, nt_data - 1))
        t1_idx = max(0, min(int(np.ceil(t))  - 1, nt_data - 1))
        alpha  = t - np.floor(t)
        query = np.column_stack([y_lat, x_lon])           # (num_pts, 2)
        vx0 = interp_vx[t0_idx](query)
        vx1 = interp_vx[t1_idx](query)
        vy0 = interp_vy[t0_idx](query)
        vy1 = interp_vy[t1_idx](query)
        dx = vx0 + alpha * (vx1 - vx0)
        dy = vy0 + alpha * (vy1 - vy0)
        return np.concatenate([dx, dy])
    # time array and output storage 
    tt        = np.arange(tini, tfinal + 0.125 / 2, 0.125)
    num_steps = len(tt)
    ssize     = tt[1] - tt[0]                      # stepsize
    trajecx = np.empty((num_pts, num_steps), dtype=float)
    trajecy = np.empty((num_pts, num_steps), dtype=float)
    trajecx[:, 0] = xini
    trajecy[:, 0] = yini
    # Integrate step by step using fixed-step classical RK4
    state = np.concatenate([xini, yini])
    for i in range(1, num_steps):
        _, yout = myrk4_2dcartisotemp(odes, (tt[i - 1], tt[i]), state, ssize)
        state = yout[0]
        trajecx[:, i] = state[:num_pts]
        trajecy[:, i] = state[num_pts:]
    return trajecx, trajecy

# perdist
def perdist(ZI, ZJ, perdims, periods):
    """Euclidean distance with periodicity in selected dimensions """
    ZI   = np.asarray(ZI, dtype=float).ravel()
    ZJ   = np.asarray(ZJ, dtype=float)
    if ZJ.ndim == 1: ZJ = ZJ[np.newaxis, :]
    diffs = np.abs(ZI[np.newaxis, :] - ZJ)          # (m, d) broadcast
    for k_idx, dim in enumerate(perdims):
        per = periods[k_idx]
        d   = diffs[:, dim]
        diffs[:, dim] = np.minimum(d, np.minimum(
            np.abs(ZI[dim] + per - ZJ[:, dim]),
            np.abs(ZI[dim] - per - ZJ[:, dim])))
    return np.sqrt((diffs ** 2).sum(axis=1))
