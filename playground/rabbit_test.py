#!/usr/bin/python3
# %%
from __future__ import annotations

import numpy as np


def rabbit(y, k, b):
    """
    Ballistic function for F(x,y,dydx')

    Parameters:
    x (array): Spatial coordinates
    y (array): Current state of y
    dxdt (array): Time derivative of x
    dydt (array): Time derivative of y

    Returns:
    array: F(x,y,dxdt,dydt')
    """
    return np.array([k * y * (1 - (y / b))])


# %% Set up the problem
t = np.linspace(0, 1, 1000)
# Physics :
# rabibts : 5 litters annualy of 10 kits each :
k = 5. * 10.
# but only half are females :
k = k / 2.0
# discretization : dx, dy in km
dx = 1
dy = 1
# For 1 km^2 : 100 rabbits max
b = 500 / dx * dy
# Initial conditions : 10 rabbits
# y0 = np.array([2]).astype(float)
y0 = np.random.randint(2, 10, size=(5, 5)).astype(float)
# y0 = np.ones((5 , 5)).astype(float)

print(f'y0 = {y0}')
# solver :
n_steps = len(t)
dt = t[1] - t[0]

# %% initialization :
# 1D :
# n = len(y0)
# y = np.zeros((n, n_steps))
# y[:, 0] = y0

# 2D :
y_shape_with_time = y0.shape + (n_steps,)
y = np.zeros(y_shape_with_time, dtype=y0.dtype)
y[..., 0] = y0

# %% solve :
for i, t in enumerate(t[:-1]):
    y[..., i + 1] = y[..., i] + dt * rabbit(y[..., i], k, b)

print(f'sol size = {np.shape(y)}')
print(f'sol final = {y[...,5]}')

# postt :
# postt.postt_ballistic(t, sol)

# %%
