"""__init__.py 
Full pipeline and Modules for Polar Vortex Example. """

from trajectories import (tracer_isentropic_2d, myrk4_2dcartisotemp, km2degSH, perdist)
from diffusion_maps import (diffusion_maps_matrix, diffusion_maps_matrix_customdist,
                            diffusion_maps_matrix_epsauto, temp_diffusion_maps_matrix)
from spacetime import (tempLaplace, make_spacetime_DiffusionMat_productform,
                       make_spacetime_DiffusionMat, surrogate_operator)
from solver import (compute_boundary_indices, build_Pfun,compute_eigenmodes, 
                    compute_decay_rates, compute_Px_avg, compute_Px_avg_eigenmodes)
from seba import seba, SEBA
from epsilon_selection import nndist_correct, auto_epsilon
from a_selection import compute_a_min, compute_a
from plotting import (bluegrayred, bluewhitered, plot_eigenvalues, save_Px_avg_movies, 
                      plot_red_region_snapshots)

__all__ = [
    'tracer_isentropic_2d', 'myrk4_2dcartisotemp', 'km2degSH', 'perdist','diffusion_maps_matrix', 
    'diffusion_maps_matrix_customdist','diffusion_maps_matrix_epsauto', 'temp_diffusion_maps_matrix',
    'tempLaplace', 'make_spacetime_DiffusionMat_productform','make_spacetime_DiffusionMat', 
    'surrogate_operator','compute_boundary_indices', 'build_Pfun', 'compute_eigenmodes',
    'compute_decay_rates', 'compute_Px_avg', 'compute_Px_avg_eigenmodes','seba', 'SEBA','nndist',
    'nndist_correct', 'auto_epsilon', 'compute_a_min', 'compute_a','bluegrayred', 'bluewhitered',
    'plot_eigenvalues', 'save_Px_avg_movies', 'plot_red_region_snapshots', ]
