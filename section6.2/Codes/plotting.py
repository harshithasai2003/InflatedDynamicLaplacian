""" plotting.py  
Polar vortex analysis figures and animations """

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import warnings
matplotlib.use('Agg')                   # non-interactive backend


# Colormap utilities
def bluegrayred(n=256):
    return mcolors.LinearSegmentedColormap.from_list('bluegrayred',
                                                      [[0,0,1],[0.5,0.5,0.5],[1,0,0]], N=n)
def bluewhitered(n=256):
    return mcolors.LinearSegmentedColormap.from_list('bluewhitered', 
                                                     [[0,0,1],[1,1,1],[1,0,0]], N=n)
def whitered(n=256):
    return mcolors.LinearSegmentedColormap.from_list('whitered', 
                                                     [[0.80,0.79,0.79],[1,0,0],[0.50,0,0]], N=n)
def register_colormaps():
    for name, func in [('bluegrayred', bluegrayred), ('bluewhitered', bluewhitered),
                       ('whitered', whitered)]:
        try:
            matplotlib.colormaps.register(func(), name=name)
        except ValueError:
            pass
register_colormaps()


# Animation helper
def _save_animation(ani, fn, fps):
    """Saves animations """
    saved = False
    try:
        ani.save(fn, writer=animation.FFMpegWriter(fps=fps))  
        print(f'Saved {fn}')
        saved = True
    except Exception:
        pass
    if not saved:
        gif_fn = fn.replace('.mp4', '.gif')                   # If FFMpeg isnt working
        try:
            ani.save(gif_fn, writer='pillow', fps=max(1, int(fps)))
            print(f'Saved {gif_fn}')
        except Exception as e2:
            warnings.warn(f'Could not save animation {fn} or {gif_fn}: {e2}')


# Eigenvalue spectrum plot
def plot_eigenvalues(evals_s, epsilonx, dynmodes, tempmodes=None,
                     fig_num=10, results_dir='results', save=True):
    """ Eigenvalue spectrum of the Laplacian """
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    dyn_idx = np.asarray(dynmodes) + 1
    if tempmodes is not None and len(tempmodes) > 0:
        ax.plot(np.asarray(tempmodes) + 1, np.log(evals_s[tempmodes]) / (epsilonx / 4.0),
                'rx', markersize=10, linewidth=1, label='temporal')
    ax.plot(dyn_idx, np.log(evals_s[dynmodes]) / (epsilonx / 4.0),
            'bo', markersize=8, linewidth=1, label='spatial')
    ax.set_xlabel('$k$', fontsize=16)
    ax.set_ylabel(r'$\Lambda_k$', fontsize=16)
    ax.tick_params(labelsize=16)
    if tempmodes is not None and len(tempmodes) > 0:
        ax.legend()
    plt.tight_layout()
    if save:
        fp = os.path.join(results_dir, 'IDL_eigenvalues.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'Saved {fp}')
    return fig, ax

# Scatter snapshot helper
def _scatter_snapshot(ax, MM_pts, evec_slice, k, bp_inds, xmin, xmax, ymin, ymax, 
                      cmap, caxbound=None):
    ax.cla()
    sc = ax.scatter(MM_pts[0, :, k], MM_pts[1, :, k], s=20, c=evec_slice, cmap=cmap)
    ax.scatter(MM_pts[0, bp_inds, k], MM_pts[1, bp_inds, k],
               s=20, facecolors='none', edgecolors=[0, 0.8, 0])
    ax.set_xlabel('km')
    ax.set_ylabel('km')
    ax.tick_params(labelsize=16)
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if caxbound is not None:
        sc.set_clim(-caxbound, caxbound)
    return sc


# Px_avg coherent set animation
def save_Px_avg_movies(MM_pts, avg_evecs, b_globind, Num_Space_Points, TimePoints, xmin, xmax,
                       ymin, ymax, mov_avg_fn_head='', num_avg_movies=1, fps=None, 
                       Num_Time_Points=None, results_dir='results'):
    """ Animation of the time-averaged spatial operator eigenvectors """
    os.makedirs(results_dir, exist_ok=True)
    # Time Setup 
    if Num_Time_Points is None:
        Num_Time_Points = MM_pts.shape[2]
    TimePoints = np.asarray(TimePoints)
    # Boundary Indices 
    bp_inds = np.where(b_globind[:Num_Space_Points])[0]
    if fps is None:
        fps = max(5, min(15, int(Num_Time_Points / 10)))
    for mode_num in range(1, num_avg_movies + 1):
        # Eigenvector
        evec = avg_evecs[:, mode_num - 1].copy()
        # Sign Consistency
        if np.sum(evec) < 0:
            evec *= -1
        evec = evec - np.mean(evec)
        evec[evec < 0] = 0
        vmax = np.max(evec)
        if vmax == 0:
            vmax = 1.0
        # Output Saving
        fn = os.path.join(results_dir, f'coherent_set_animation.mp4')
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#eeeeee')
        cmap = matplotlib.colormaps['whitered']
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        def _make_frame(frame_idx):
            k = frame_idx
            ax.cla()
            # Main Scatter - Raw Eigenvector Values
            ax.scatter(MM_pts[0, :, k], MM_pts[1, :, k], s=20, c=evec, cmap=cmap, norm=norm)
            # Boundary Points
            ax.scatter(MM_pts[0, bp_inds, k], MM_pts[1, bp_inds, k], s=20, facecolors='none', 
                       edgecolors=[0, 0.8, 0])
            # Formatting 
            ax.set_xlabel('km')
            ax.set_ylabel('km')
            ax.set_aspect('equal')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f'$t={TimePoints[k]/4.0:.2f}$ days')
            return ax.collections
        ani = animation.FuncAnimation(fig, _make_frame, frames=Num_Time_Points, blit=False)
        _save_animation(ani, fn, fps)
        plt.close(fig)


# Red Region Helper
def _compute_red_region_stats(MM_pts, avg_evecs, TimePoints, mode_num=1, red_threshold=0.5):
    evec = avg_evecs[:, mode_num - 1].copy()
    if np.sum(evec) < 0:
        evec = -evec
    vmax = evec.max()
    if vmax <= 0:
        vmax = 1.0
    evec_n = evec / vmax
    red_mask = evec_n > red_threshold
    days   = np.asarray(TimePoints) / 4.0
    T      = len(days)
    cx       = np.zeros(T)
    cy       = np.zeros(T)
    area     = np.zeros(T)
    mean_val = np.zeros(T)
    spread_x = np.zeros(T)
    spread_y = np.zeros(T)
    # spread_r = np.zeros(T)
    for k in range(T):
        xs = MM_pts[0, red_mask, k]
        ys = MM_pts[1, red_mask, k]
        vals = evec_n[red_mask]
        area[k] = xs.size
        if xs.size == 0:
            continue
        w      = vals / vals.sum()
        cx[k]  = np.dot(w, xs)
        cy[k]  = np.dot(w, ys)
        mean_val[k] = vals.mean()
        spread_x[k] = xs.std()
        spread_y[k] = ys.std()
        # spread_total = spread_x + spread_y            # For sum of SD
        '''xc = xs.mean()                            # For Radial Spread
        yc = ys.mean()
        r2 = (xs - xc)**2 + (ys - yc)**2
        spread_r[k] = np.sqrt(np.mean(r2))'''
    return days, cx, cy, area, mean_val, spread_x, spread_y


# Snapshots Plot
def plot_red_region_snapshots(MM_pts, avg_evecs, TimePoints, b_globind, Num_Space_Points,
                               mode_num=1, red_threshold=0.5, n_panels=12,xmin=None, xmax=None,
                                ymin=None, ymax=None, results_dir='results',
                               filename='polar_vortex_core_snapshots.png'):
    """ Multi-panel Plot of the Polar Vortex Core at evenly-spaced Time Slices """
    os.makedirs(results_dir, exist_ok=True)
    evec   = avg_evecs[:, mode_num - 1].copy()
    if np.sum(evec) < 0:
        evec = -evec
    vmax   = evec.max() if evec.max() > 0 else 1.0
    evec_n = evec / vmax
    red_mask = evec_n > red_threshold
    days      = np.asarray(TimePoints) / 4.0
    T         = len(days)
    bp_inds   = np.where(b_globind[:Num_Space_Points])[0]
    # Choose evenly-spaced time indices across the full period
    panel_idx = np.round(np.linspace(0, T - 1, n_panels)).astype(int)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    # Layout and Width Ratios
    panel_w   = 5.0          # width of each scatter panel in inches
    panel_h   = 5.0          # height of each scatter panel in inches
    cbar_w    = 0.55          # colorbar column width in inches
    cbar_pad  = 0.25          # gap between last panel col and cbar col
    fig_w     = ncols * panel_w + cbar_pad + cbar_w + 0.5
    fig_h     = nrows * panel_h + 1.0    # +1 for suptitle headroom
    fig = plt.figure(figsize=(fig_w, fig_h))
    # GridSpec with an extra column for the colorbar
    gs = gridspec.GridSpec(nrows, ncols + 1, figure=fig,
                           width_ratios=[1] * ncols + [cbar_w / panel_w], wspace=0.08, 
                           hspace=0.35, left=0.06, right=0.98, top=0.92, bottom=0.04)

    fig.suptitle(f'Polar Vortex Core Evolution Snapshots\n', fontsize=14, fontweight='bold')
    cmap_bg = bluegrayred()
    norm    = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    axes = []
    for panel, k in enumerate(panel_idx):
        row = panel // ncols
        col = panel  % ncols
        ax  = fig.add_subplot(gs[row, col])
        axes.append(ax)
        # All particles coloured by avg eigenvector
        ax.scatter(MM_pts[0, :, k], MM_pts[1, :, k], c=evec_n, cmap=cmap_bg, norm=norm,
                   s=5, alpha=0.6, linewidths=0, zorder=1)
        # Red particles: bold red outline to highlight vortex core
        ax.scatter(MM_pts[0, red_mask, k], MM_pts[1, red_mask, k], s=14, facecolors='none', 
                   edgecolors='red', linewidths=0.6, alpha=0.8, zorder=3, label='Vortex core')
        # Domain boundary 
        ax.scatter(MM_pts[0, bp_inds, k], MM_pts[1, bp_inds, k], s=8, facecolors='none', 
                   edgecolors=[0, 0.75, 0], linewidths=0.5, alpha=0.6, zorder=2)
        ax.set_title(f't = {days[k]:.1f} days', fontsize=10)
        ax.set_xlabel('km', fontsize=8)
        ax.set_ylabel('km', fontsize=8)
        ax.set_aspect('equal')
        if xmin is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None:
            ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=7)
        if panel == 0:
            ax.legend(fontsize=7, loc='lower right', markerscale=1.5)
    # Hide unused panels in the data columns
    for panel in range(n_panels, nrows * ncols):
        row = panel // ncols
        col = panel  % ncols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_visible(False)
    # Colorbar in its own column 
    cbar_ax = fig.add_subplot(gs[:, ncols])   
    sm = plt.cm.ScalarMappable(cmap=cmap_bg, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(' Eigenvector', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fp = os.path.join(results_dir, filename)
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fp}')
    return fp

# Spatial Eigenmodes Animations
def save_spatial_eigenmode_movies(evecs_3d, MM_pts, b_globind, Num_Space_Points, TimePoints,
                                  dynmodes, num_spat_mode_movies=2, MSR=1,
                                  xmin=None, xmax=None, ymin=None, ymax=None,
                                  mov_fn_head='', fps=None, results_dir='results'):
    """ Animates each Spatial Eigenmodes """
    os.makedirs(results_dir, exist_ok=True)
    Num_Time_Points = MM_pts.shape[2]
    TimePoints = np.asarray(TimePoints)
    bp_inds = np.where(b_globind[:Num_Space_Points])[0]
    if fps is None:
        fps = max(1, 4 * Num_Time_Points / 45)               
    saved_files = []
    for jj in range(num_spat_mode_movies):
        mode = dynmodes[jj]
        mode_vals = evecs_3d[:, :, mode].copy()
        if np.sum(mode_vals) < 0:
            mode_vals *= -1
        caxbound = max(abs(mode_vals.min()), abs(mode_vals.max()))
        if caxbound == 0:
            caxbound = 1.0
        frame_ks = list(range(MSR - 1, Num_Time_Points, MSR))
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#eeeeee')
        cmap = matplotlib.colormaps['bluegrayred']
        norm = mcolors.Normalize(vmin=-caxbound, vmax=caxbound)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        def _make_frame(k):
            ax.cla()
            ax.scatter(MM_pts[0, :, k], MM_pts[1, :, k], s=20,
                       c=mode_vals[:, k], cmap=cmap, norm=norm)
            ax.scatter(MM_pts[0, bp_inds, k], MM_pts[1, bp_inds, k],
                      s=20, facecolors='none', edgecolors=[0, 0.8, 0])
            ax.set_xlabel('km'); ax.set_ylabel('km')
            ax.tick_params(labelsize=16)
            ax.set_aspect('equal')
            if xmin is not None:
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_title(f'$t={TimePoints[k] / 4.0:.2f}$ days')
            return ax.collections
        ani = animation.FuncAnimation(fig, _make_frame, frames=frame_ks, blit=False)
        fn = os.path.join(results_dir, f'spatial_mode_{mode + 1}.mp4')
        _save_animation(ani, fn, fps)
        plt.close(fig)
        saved_files.append(fn)
    return saved_files


# Eigenvalue Spectrum
def plot_px_avg_eigenvalues(avg_evals_s, epsilonx, num_evals, results_dir='results',
                            filename='Eigenvalues_dyn.png'):
    """ Eigenvalue Spectrum """
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, num_evals + 1), np.log(avg_evals_s) / (epsilonx / 4.0),
            'bo', markersize=8, linewidth=1)
    ax.set_xlabel('$k$', fontsize=16)
    ax.set_ylabel(r'$\lambda_k^D$', fontsize=16)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    fp = os.path.join(results_dir, filename)
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fp}')
    return fp


# Vortex Shape Deformation
def plot_vortex_shape_deformation(MM_pts, avg_evecs, TimePoints,
                                   mode_num=1, red_threshold=0.5, results_dir='results',
                                   filename='vortex_shape_deformation.png'):
    """ Vortex shape deformation - spatial standard deviation of the red
    vortex core particles in x and y over the full time window """
    os.makedirs(results_dir, exist_ok=True)
    days, cx, cy, area, mean_val, spread_x, spread_y = \
        _compute_red_region_stats(MM_pts, avg_evecs, TimePoints, mode_num=mode_num,
                                  red_threshold=red_threshold)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, spread_x / 1e3, color='red',  linewidth=1.8,
            label='Zonal spread (x)')
    ax.plot(days, spread_y / 1e3, color='blue', linewidth=1.8,
            label='Meridional spread (y)')
    # ax.plot(days, spread_total/1e3, linewidth=2.0, label='stdx + stdy')      # for sum of SD
    # For Radial Spread
    # ax.plot(days, spread_r / 1e3, color='black', linewidth=2.2, label='Radial spread')   
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Spread (km)',  fontsize=12)
    ax.set_title(' Vortex shape deformation', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(days[0], days[-1])
    ax.tick_params(labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fp = os.path.join(results_dir, filename)
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fp}')
    return fp