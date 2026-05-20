""" plotting.py """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import (TwoSlopeNorm, Normalize, LinearSegmentedColormap,)
from mpl_toolkits.mplot3d import Axes3D  

# Blue-White-Red colormap
def _bluewhitered():
    colors = [ (0.00, 0.00, 0.50), (0.00, 0.50, 1.00), (1.00, 1.00, 1.00),
        (1.00, 0.00, 0.00), (0.50, 0.00, 0.00),]
    return LinearSegmentedColormap.from_list("bluewhitered", colors, N=512, )

BWR = _bluewhitered()

# Stream-function helper
def _p_very_abrupt(t):
    return (1.0 + np.tanh(40.0 * (t - 0.5))) / 2.0


def stream_function(t, x, y):
    p = _p_very_abrupt(t)
    a = np.pi * (1 - 2 * p) / (3 * (p - 2) * (p + 1))
    b = (2 * np.pi - 9 * a) / 3
    return (-np.sin(np.pi / 2 * y) * np.sin(a * x**2 + b * x) )


# Stream Function
def fig_streamfunction(outpath=None):
    x = np.linspace(0, 3, 400)
    y = np.linspace(0, 2, 300)
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(-0.9, 0.9, 10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, t_val, lbl in zip(axes, [0.0, 1.0], [r"$t=0$", r"$t=1$"],):
        Z = stream_function(t_val, X, Y)
        ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=1.2,)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title(f"Stream function {lbl}", fontsize=13,)
        ax.tick_params(labelsize=12)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',)
    return fig

# Eigen Values
def fig_eigenvalues(idl, max_show=10, outpath=None):
    epsx = idl.epsilonx_
    lam = idl.laplacian_eigenvalues()
    dyn = np.asarray(idl.dynmodes, dtype=int)
    temp = np.asarray(idl.tempmodes, dtype=int)
    dyn_show = dyn[dyn < max_show]
    temp_show = (temp[temp < max_show]
                 if len(temp)
        else np.array([], dtype=int))
    dyn_k = dyn_show + 1
    temp_k = (temp_show + 1
        if len(temp_show)
        else np.array([], dtype=int))
    avg_evals, _ = idl.px_avg_eigs(num_evals=max_show)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_lam = np.where(avg_evals > 0, np.log(avg_evals) / epsx, np.nan,)
    k_avg = np.arange(1, len(avg_lam) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), )

    # Left panel
    ax = axes[0]
    ax.plot(k_avg, avg_lam, 'bo', markersize=8, linewidth=1,)
    ax.set_xlabel(r'$k$', fontsize=16)
    ax.set_ylabel(r'$\lambda_k^D$', fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_xticks(k_avg)

    # Right panel
    ax = axes[1]
    ax.plot(dyn_k, lam[dyn_show], 'bo', markersize=8, linewidth=1, label='spatial',)
    if len(temp_show):
        ax.plot(temp_k, lam[temp_show], 'rx', markersize=10, markeredgewidth=2, label='temporal',)
        ax.legend(fontsize=12)
    ax.set_xlabel(r'$k$', fontsize=16)
    ax.set_ylabel(r'$\Lambda_k$', fontsize=16)
    ax.tick_params(labelsize=13)
    all_k = (np.concatenate([dyn_k, temp_k])
        if len(temp_show)
        else dyn_k)
    ax.set_xticks(sorted(all_k))
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',)
    return fig


def _scatter_mode(ax, xs, ys, vals, abs_cutoff=0.25, cmap=BWR, vmin=-1, vmax=1, s=8,):

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax,)
    mask = np.abs(vals) > abs_cutoff
    sc = ax.scatter(xs[mask], ys[mask], c=vals[mask], s=s,
        cmap=cmap, norm=norm, rasterized=True, linewidths=0, )
    return sc, mask


# Figure 3D
def fig_3d_spacetime(idl, mode_idx, domain='M1', abs_cutoff=0.25, time_step=2,
                     elev=55, azim=55, M0_pts=None, outpath=None,):

    SPA = idl.SpacePointsarray
    TP = idl.TimePoints
    _, N, T = SPA.shape
    V = idl.evecs_3d[:, :, mode_idx]
    V_norm = V / (np.max(np.abs(V)) + 1e-15)
    norm_c = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1,)
    sm = plt.cm.ScalarMappable(cmap=BWR, norm=norm_c,)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d',)

    for t in range(0, T, time_step):
        vec_t = V_norm[:, t]
        mask = np.abs(vec_t) > abs_cutoff
        if not mask.any():
            continue
        if domain == 'M1':
            xs = SPA[0, :, t][mask]
            ys = SPA[1, :, t][mask]
        else:
            xs = (M0_pts[0, :, t][mask]
                if M0_pts is not None
                else SPA[0, :, 0][mask])
            ys = (M0_pts[1, :, t][mask]
                if M0_pts is not None
                else SPA[1, :, 0][mask])
        ts = TP[t] * np.ones(mask.sum())
        rgba = sm.to_rgba(vec_t[mask])
        ax.scatter(ts, xs, ys, c=rgba, s=8, depthshade=False, rasterized=True,)
    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)
    ax.set_zlabel(r'$x_2$', fontsize=14)
    ax.set_xlim(TP[0], TP[-1])
    ax.set_ylim(SPA[0].min(),SPA[0].max(),)
    ax.set_zlim(SPA[1].min(),SPA[1].max(),)
    ax.view_init(elev=elev, azim=azim,)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.08,)
    dom_label = (r'$\mathbb{M}_0$'
        if domain == 'M0'
        else r'$\mathbb{M}_1$')
    ax.set_title(f'Eigenvector {mode_idx+1} in {dom_label}', fontsize=12,)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',)
    return fig, ax


# Figure Snapshot
def fig_snapshot_grid(idl, mode_idx, domain='M1', abs_cutoff=0.25, n_times=10, ncols=5,
                      M0_pts=None,outpath=None,):

    SPA = idl.SpacePointsarray
    TP = idl.TimePoints
    T = len(TP)
    V = idl.evecs_3d[:, :, mode_idx]
    V_n = V / (np.max(np.abs(V)) + 1e-15)
    t_idx = np.linspace(0, T - 1, n_times, dtype=int,)
    nrows = int(np.ceil(n_times / ncols))
    fig, axes = plt.subplots(nrows, ncols,figsize=(ncols * 3.2, nrows * 2.8),
                             constrained_layout=True,)

    axes = np.array(axes).ravel()
    for ii, t in enumerate(t_idx):
        ax = axes[ii]
        vec = V_n[:, t]
        if domain == 'M1':
            xs = SPA[0, :, t]
            ys = SPA[1, :, t]
        else:
            xs = (M0_pts[0, :, t]
                if M0_pts is not None
                else SPA[0, :, 0])
            ys = (M0_pts[1, :, t]
                if M0_pts is not None
                else SPA[1, :, 0])
        _scatter_mode(ax, xs, ys, vec, abs_cutoff=abs_cutoff,)
        ax.set_aspect('equal')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)
        ax.set_title(f't = {TP[t]:.2f}',fontsize=10,)
        ax.tick_params(labelsize=8)
    for j in range(ii + 1, len(axes)):
        axes[j].set_visible(False)

    dom_label = (r'$\mathbb{M}_0$'
        if domain == 'M0'
        else r'$\mathbb{M}_1$')

    fig.suptitle(f'Mode {mode_idx+1} in {dom_label}', fontsize=13,)
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',)
    return fig


# Figure High Variance
def fig_high_variance_trajectories(idl, mode_idx, domain='M1', top_pct=0.05,abs_cutoff=0.25, 
                                   time_step=1, M0_pts=None, add_streamfunction=True, outpath=None,):

    SPA = idl.SpacePointsarray
    TP = idl.TimePoints
    _, N, T = SPA.shape
    V = idl.evecs_3d[:, :, mode_idx]
    V_n = V / (np.max(np.abs(V)) + 1e-15)
    V2var = np.var(V_n, axis=1)
    n_top = max(1,int(round(N * top_pct)),)
    top_idx = np.argsort(V2var)[-n_top:]
    norm_c = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1,)
    sm = plt.cm.ScalarMappable(cmap=BWR, norm=norm_c,)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d',)
    for t in range(0, T, time_step):
        vec_t = V_n[top_idx, t]
        if domain == 'M1':
            xs = SPA[0, top_idx, t]
            ys = SPA[1, top_idx, t]
        else:
            xs = (M0_pts[0, top_idx, t]
                if M0_pts is not None
                else SPA[0, top_idx, 0])
            ys = (M0_pts[1, top_idx, t]
                if M0_pts is not None
                else SPA[1, top_idx, 0])
        ts = TP[t] * np.ones(n_top)
        rgba = sm.to_rgba(vec_t)
        ax.scatter(ts, xs, ys, c=rgba, s=12, depthshade=False, rasterized=True,)
    if add_streamfunction:
        xg = np.linspace(0, 3, 200)
        yg = np.linspace(0, 2, 150)
        XG, YG = np.meshgrid(xg, yg)
        levels = [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9,]
        _fig_tmp, _ax_tmp = plt.subplots()
        for t_val in [TP[0], TP[-1]]:
            ZG = stream_function(t_val, XG, YG,)
            cs = _ax_tmp.contour(XG, YG, ZG, levels=levels,)
            for lvl_segs in cs.allsegs:
                for seg in lvl_segs:
                    if len(seg) < 2:
                        continue
                    ax.plot(t_val * np.ones(len(seg)), seg[:, 0], seg[:, 1], 'k-',
                            linewidth=0.6, alpha=0.7,)
        plt.close(_fig_tmp)

    dom_label = (r'$\mathbb{M}_0$'
        if domain == 'M0'
        else r'$\mathbb{M}_1$')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$x_1$', fontsize=13)
    ax.set_zlabel(r'$x_2$', fontsize=13)
    ax.set_xlim(TP[0], TP[-1])
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 2)
    ax.view_init(elev=55, azim=55,)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.08,)
    ax.set_title(f'Top variance trajectories in {dom_label}', fontsize=12,)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',)
    return fig, ax

def fig_px_avg_modes(idl, num_modes=8, t_idx=0,outpath=None,):
    SPA = idl.SpacePointsarray
    avg_evals, avg_evecs = idl.px_avg_eigs(num_evals=num_modes + 1)
    start = 0
    ncols = 4
    nrows = int(np.ceil(num_modes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), 
                             constrained_layout=True,)
    axes = np.array(axes).ravel()
    for ii in range(num_modes):
        ax = axes[ii]
        v = np.real(avg_evecs[:, start + ii])
        v /= np.max(np.abs(v)) + 1e-15
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1,)
        ax.scatter(SPA[0, :, t_idx],  SPA[1, :, t_idx], c=v, cmap=BWR, norm=norm, s=8,
                   rasterized=True, linewidths=0,)
        ax.set_aspect('equal')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)
        ax.set_title(f'mode {start+ii+1}',fontsize=10,)
        ax.tick_params(labelsize=8)
    for j in range(ii + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(r'Top eigenvectors of $P_x^{\mathrm{avg}}$', fontsize=13,)
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',)
    return fig