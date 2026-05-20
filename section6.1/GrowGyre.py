""" GrowGyre.py
Switching double-gyre flow on Domain : [0, 3] x [0, 2],  time : [0, 1]"""

import numpy as np

def grow_gyre(t: float, x: np.ndarray, regime: str = 'very-abrupt') -> np.ndarray:
    if regime == 'uniform':
        r = t
    elif regime == 'abrupt':
        r = 0.5 * (1.0 + np.tanh(10.0 * (t - 0.5)))
    elif regime == 'very-abrupt':
        r = 0.5 * (1.0 + np.tanh(40.0 * (t - 0.5)))
    else:
        raise ValueError(f"Unknown regime '{regime}', choose from 'uniform', 'abrupt', 'very-abrupt'")
    
    # Separatrix parameters 
    a = np.pi * (1.0 - 2.0 * r) / (3.0 * (r - 2.0) * (r + 1.0))
    b = (2.0 * np.pi - 9.0 * a) / 3.0
    x1, x2 = float(x[0]), float(x[1])
    sub1 = a * x1**2 + b * x1
    sub2 = (np.pi / 2.0) * x2
    u = -(np.pi / 2.0) * np.cos(sub2) * np.sin(sub1)
    v = np.sin(sub2) * np.cos(sub1) * (2.0 * a * x1 + b)
    
    '''alpha = (1.0 - 2.0 * r) / (3.0 * (r - 2.0) * (r + 1.0))
    beta  = (2.0 - 9.0 * alpha) / 3.0

    # Particle coordinates
    x1, x2 = float(x[0]), float(x[1])
    f = alpha * (x1**2) + beta * x1

    # Velocity from stream function psi = -sin(pi*x2/2) * sin(pi*f) 
    u = -(np.pi / 2.0) * np.sin(np.pi * f) * np.cos(np.pi * x2 / 2.0)
    v = ((2.0 * alpha * x1) + beta) * np.cos(np.pi * f) * np.sin(np.pi * x2 / 2.0)'''

    return np.array([u, v], dtype=np.float64)