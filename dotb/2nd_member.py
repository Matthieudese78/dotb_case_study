#!/usr/bin//python3
from __future__ import annotations

import numpy as np

# Ballistic :


def ballistic(y, rho, c, A, g):
    """
    Ballistic function for F(x,y,dydx')

    Parameters:
    x (array): Spatial coordinates
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
    return np.array([dxdt, dydt, -mu*dxdt*magv, -g-mu*dydt*magv])
