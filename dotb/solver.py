#!/usr/bin/python3
from __future__ import annotations

import numpy as np


# %%
def euler_explicit(F, y0, x, t, dx=0.01, dt=0.001):
    """
    Solve ∂y/∂t = F(x,y,y') using Euler explicit method

    Parameters:
    F (function): Function defining the PDE
    y0 (array): Initial condition
    x (array): Spatial grid
    t (array): Time array
    dx (float): Spatial step size
    dt (float): Time step size

    Returns:
    y (ndarray): Solution tensor field
    """
    n_steps = len(t)
    n_points = len(x)

    y = np.zeros((n_steps, n_points))
    y[0] = y0

    for i in range(n_steps - 1):
        # dydx = np.gradient(y[i], axis=1) / dx
        dydx = np.gradient(y[i]) / dx

        # Apply Euler explicit formula
        # y[i + 1] = y[i] + dt * F(x[:, np.newaxis], y[i], dydx)
        y[i + 1] = y[i] + dt * F(x, y[i], dydx)

    return y
