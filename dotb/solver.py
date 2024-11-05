#!/usr/bin/python3
from __future__ import annotations

import numpy as np
import numpy.linalg as LA

import dotb.boundary_conditions as BC
import dotb.second_member as second


# %%
def euler_explicit(y: np.ndarray, t: np.ndarray, **kw) -> np.ndarray:
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
    print(f'solver : dt = {dt}')
    sol = []
    sol.append(y)
    for i, ti in enumerate(t[:-1]):
        # Apply the boundary conditions :
        y = BC.boundary_conds(y, **kw)[0]
        # Apply Euler explicit formula
        y_new = y + dt * second.F(y, **kw)
        if (i + 1) % kw['n_save'] == 0:
            print(f'saving {i}^th time step')
            sol.append(y_new)
            # print(f'solver : sol = {sol}')
        y = y_new
    return np.array(sol)


def adams_bashforth(y: np.ndarray, t: np.ndarray, **kw) -> np.ndarray:
    """
    Solve ∂y/∂t = F(x,y,y') using Euler explicit method

    Parameters:
    y (tensor): integrated tensor field
    F (function): Function defining the PDE
    x (array): Spatial grid
    t (array): Time array
    dx (float): Spatial step size
    dt (float): Time step size

    Returns:
    y (ndarray) : tensor field value at the saved time steps
    y_{n+2} = y_{n+1} + (3/2)*dt*F_{n+1} - (1/2)*dt*F_n
    """
    dt = t[1] - t[0]
    print(f'solver : dt = {dt}')
    sol = []
    sol.append(y)
    y = BC.boundary_conds(y, **kw)[0]
    # Initialization of y_{n+1} (n=0) via Euler method :
    y1 = y + dt * second.F(y, **kw)
    if 1 % kw['n_save'] == 0:
        # .i.e if nsave = 1 :
        sol.append(y1)
    #
    for i, ti in enumerate(t[:-2]):
        # Apply the boundary conditions :
        y = BC.boundary_conds(y, **kw)[0]
        y1 = BC.boundary_conds(y1, **kw)[0]
        # Apply adams-bashforth formula :
        y_new = y1 + dt * (
            (3.0 / 2.0) * second.F(y1, **kw) - (1.0 / 2.0) * second.F(y, **kw)
        )
        if (i + 2) % kw['n_save'] == 0:
            print(f'saving {i}^th time step')
            sol.append(y_new)

        # y, y1, y2 values update for next time step :
        #   y_{n} = y_{n+1} :
        y = y1
        #   y_{n+1} = y_{n+2} :
        y1 = y_new
    return np.array(sol)


def crank_nicolson(y: np.ndarray, t: np.ndarray, **kw) -> np.ndarray:
    """
    Solve ∂y/∂t = F(x,y,y') using Euler explicit method
    using the Crank-Nicolson method (implicit) :

    y_{n+1} = y_{n} + (dt/2)*(F_{n+1} + F_n)

    Parameters:
    y (tensor): integrated tensor field
    F (function): Function defining the PDE
    t (array): Time array

    Returns:
    sol (ndarray) : stacked tensor field values at the saved time steps
    """
    dt = t[1] - t[0]
    sol = []
    sol.append(y)
    for i, ti in enumerate(t[:-1]):
        # Applying boundary conditions for y :
        y = BC.boundary_conds(y, **kw)[0]
        f = second.F(y, **kw)
        # Fist estimation of the residu :
        y1 = y + dt * f
        # Applying boundary conditions for y1 :
        y1 = BC.boundary_conds(y1, **kw)[0]
        f1 = second.F(y1, **kw)
        # Matching residu :
        res = y1 - y - (dt/2.)*(f + f1)
        # Computing the RMS (absolute value in 1D) of the residu :
        res_norm = np.sqrt(np.sum(res**2))
        # Convergence criterium :
        crit = 1.e-2
        # Already converged ?
        if res_norm < crit:
            y = y1
        #   save
            if (i + 1) % kw['n_save'] == 0:
                sol.append(y1)
        #   move to the next time step :
            continue
        # Else : Newton Raphson :
        y1 = newton_raphson_diffusion(y1, dt, res, crit, nmax=100, **kw)
        if (i + 1) % kw['n_save'] == 0:
            sol.append(y1)
            # print(f'solver : sol = {sol}')
        y = y1
    return np.array(sol)


def newton_raphson_diffusion(y, dt, res, crit, nmax=100, **kw):
    print('Newton Raphson')
    res_norm = np.sqrt(np.sum(res**2))
    y_norm = np.sqrt(np.sum(y**2))
    f = second.F(y, **kw)
    i = 0
    # First estimation of the residu value :
    dres = res / res_norm
    while res_norm > crit:
        i += 1
        y1 = y - np.matmul(LA.inv(dres), res)
        y1 = BC.boundary_conds(y1, **kw)[0]
        f1 = second.F(y1, **kw)
        # new residu computation :
        res1 = y1 - y - (dt/2.)*(f1 + f)
        res_norm = np.sqrt(np.sum(res**2))
        print(f'res norm = {res_norm}')
        # converged ?
        if res_norm < crit:
            return y1
        # time derivative of the residu function :
        dres = (res1 - res) / res_norm
        # max number of iteration reached ?
        if i >= nmax:
            print('Error : Newton Raphson divergence!')
            print('        Max nb of iterations reached')
            return y1
        # other wise next Newton Raphson iteration :
        y = y1
        f = f1
        res = res1
