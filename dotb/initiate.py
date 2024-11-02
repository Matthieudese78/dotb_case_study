#!/usr/bin//python3
from __future__ import annotations

import numpy as np

# Generic 2nd member treatment :


def intiate_y(t, **kw):
    """
    Initiate the tensor field y(t=0)
    from input.yaml file keywords arguments.

    Parameters: depend on the use case
    (input.yaml keywords arguments)

    Returns:

    - ballistic : 4D-array : (x0,y0,dxdt0,dydt0')
    - rabbit : 1D-array : N0
    - diffusion (1D) : 1D-array : T(x,t=0)
    - diffusion (2D) : 2D-array : T(x,y,t=0)
    """

    if kw['case'] == 'ballistic':
        # Reading args :
        x0 = kw['x_0']
        y0 = kw['y_0']
        dxdt0 = kw['v_0']*np.cos(kw['theta_0'])
        dydt0 = kw['v_0']*np.cos(kw['theta_0'])
        y_0 = np.array([x0, y0, dxdt0, dydt0])

    # y - initialization : adding a dimension for time
    n_steps = len(t)
    y_shape_with_time = y_0.shape + (n_steps,)
    y = np.zeros(y_shape_with_time, dtype=y_0.dtype)
    y[..., 0] = y_0

    return y, y_0
