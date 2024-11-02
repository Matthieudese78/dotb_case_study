#!/usr/bin/python3
from __future__ import annotations

import numpy as np

import dotb.second_member as second


# %%
def euler_explicit(y, F, t, **kw):
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
    dt = t[1] - t[0]
    sol = []
    for i, ti in enumerate(t):
        # Apply Euler explicit formula
        y_new = y + dt * second.F(y, **kw)
        if ti % kw['nsave']:
            sol.append(y_new)
    return sol


def euler_explicit_ballistic(F, y0, c, A, g, t):
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
    dt = t[1] - t[0]

    n = len(y0)

    y = np.zeros((n, n_steps))
    y[:, 0] = y0
    print(y[:, 0])
    # print(f'len(y) = {len(y[0,:])}')
    # print(f'len(t) = {len(t)}')

    for i, t in enumerate(t[:-1]):
        y[:, i + 1] = y[:, i] + dt * F(y[:, i], t, c, A, g)
    print(f'sol size = {np.shape(y)}')
    # print(f'sol = {y}')
    return y


def euler_explicit_rabbit(F, y0, k, b, t):
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
    dt = t[1] - t[0]

    # initialization :
    y_shape_with_time = y0.shape + (n_steps,)
    y = np.zeros(y_shape_with_time, dtype=y0.dtype)
    y[..., 0] = y0

    for i, t in enumerate(t[:-1]):
        y[..., i + 1] = y[..., i] + dt * F(y[..., i], k, b)
    return y


def scalar_laplacian_edge_order(arr, dx, dy, edge_order=2):
    grad = np.gradient(arr, dx, dy, edge_order=edge_order)
    grad2 = grad * grad
    print(grad2)
    return np.sum(grad2, axis=tuple(range(len(arr.shape))))


def euler_explicit_diffusion(F, y0, D, dx, dy, t, edge_order=2):
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
    dt = t[1] - t[0]

    # initialization :
    y_shape_with_time = y0.shape + (n_steps,)
    y = np.zeros(y_shape_with_time, dtype=y0.dtype)
    y[..., 0] = y0

    for i, t in enumerate(t[:-1]):
        y[..., i + 1] = y[..., i] + dt * \
            F(y[..., i], D, dx, dy, edge_order=edge_order)
    return y
