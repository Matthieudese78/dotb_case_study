#!/usr/bin//python3
from __future__ import annotations

import numpy as np

import dotb.differential_operators as diffops
from dotb.refined_mesh_class import Mesh


def F(y, **kw):
    if kw['case'] == 'ballistic':
        return ballistic(y, kw['rho'], kw['c'], kw['A'], kw['g'])
    if kw['case'] == 'rabbit':
        return rabbit(y, kw['k'], kw['b'])
    if kw['case'] == 'diffusion_2D':
        return diffusion(y, edge_order=1, **kw)


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


def F_diffusion(mesh: Mesh, data: np.ndarray, D: np.ndarray) -> np.ndarray:
    """"
    Computes the scalar laplacian of D * y where D is the diffusion field.
    """

    grads = diffops.gradient_mesh(mesh, data)
    grads[0] = D * grads[0]
    grads[1] = D * grads[1]
    grad2 = []
    [
        grad2.append(diffops.gradient_mesh(mesh, gradi)[i])
        for i, gradi in enumerate(grads)
    ]
    return np.sum(np.array(grad2), axis=0)


def diffusion(data, edge_order=1, **kw):
    # print(f'second.diffusion : np.shape(data) = {np.shape(data)}')
    # print(f'second.diffusion : len(np.shape(data)) = {len(np.shape(data))}')
    """
    Difffusion for F(x,y,dydx')

    data(x,y,t) (array): Current temperature field
    D(x,y,t) (array): diffusion coefficient
    edge_order = 1

    Returns : F(x,y,t) : div[ D(x,y) * grad(T(x,y,t)) ]
    2nd member of the diffusion equation
    """
    # Spatial discretization :
    x = np.linspace(-kw['l_x']/2., kw['l_x']/2., kw['n_x'])
    y = np.linspace(-kw['l_x']/2., kw['l_y']/2., kw['n_y'])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    discr = [dx, dy]

    # Gradient of the temperature field :
    grads = gradient(data, discr, edge_order)

    # Multiplying the temperature gradient by the conduction coeff. :

    # 1kw['D'] :
    if len(np.shape(data)) == 1:
        grads = kw['D'] * grads
        # order two derivative = (in 1kw['D']) scalar laplacian
        sl = np.gradient(grads, dx)

    # 2kw['D'] :
    if len(np.shape(data)) == 2:
        grads[0] = kw['D'] * grads[0]
        grads[1] = kw['D'] * grads[1]
        # order two derivative :
        grad2s = []
        [
            grad2s.append(
                np.gradient(
                    gradi, discr[i], axis=i, edge_order=edge_order,
                ),
            )
            for i, gradi in enumerate(grads)
        ]
        # scalar laplacian :
        sl = np.sum(grad2s, axis=0)

    return sl
