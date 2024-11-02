#!/usr/bin//python3
from __future__ import annotations

import numpy as np

from dotb.input_yaml import mydict_ballistic
# import numpy.typing as npt


def intiate_y(**kw: mydict_ballistic) -> tuple:
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
        dxdt0 = kw['v_0']*np.cos(np.pi*kw['theta_0']/180.)
        dydt0 = kw['v_0']*np.cos(np.pi*kw['theta_0']/180.)
        y_0 = np.array([x0, y0, dxdt0, dydt0])

    return y_0
