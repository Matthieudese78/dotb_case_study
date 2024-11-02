#!/usr/bin//python3
from __future__ import annotations

import numpy as np


def F(y, **kw):
    if kw['case'] == 'ballistic':
        return ballistic(y, kw['rho'], kw['c'], kw['A'], kw['g'])
    if kw['case'] == 'rabbit':
        return rabbit(y, kw['k'], kw['b'])


def ballistic(y, rho: float, c: float, A: float, g: float):
    """
    Ballistic function for F(x,y,dydx')

    y (array): Current state of y
    dxdt (array): Time derivative of x
    dydt (array): Time derivative of y

    Returns:
    2nd member = 4D-array : F(x,y,dxdt,dydt')
    """
    # Takes the x and y direction velocities
    dxdt = y[2]
    dydt = y[3]
    # Computes the mu coeff.
    mu = 0.5*rho*c*A
    # Computes velocity magnitude
    magv = np.sqrt(dxdt**2 + dydt**2)
    return np.array([dxdt, dydt, -mu*dxdt*magv, - g - mu*dydt*magv])


def rabbit(N: np.ndarray, k: float, b: float):
    """
    Rabbit puplation dynamics function for F(x,y,dydx')

    N (array): Current state of N (population)
    k (float): reproduction rate
    b (float): carrying capacity (max population / area)

    Returns:
    2nd member = 1D-array : F(N,k,b)
    """
    return k*N*(1. - (N/b))
