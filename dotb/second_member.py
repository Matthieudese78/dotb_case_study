#!/usr/bin//python3
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.sparse import identity

import dotb.differential_operators as diffops
from dotb.boundary_conditions import apply_boundaries
from dotb.refined_mesh_class import Mesh


class Ballistic:
    def __init__(self, **kw):
        self.g = kw['g']
        # Compute drag coeff.
        self.mu = 0.5 * kw['rho'] * kw['c'] * kw['A']
        self.rho = kw['rho']
        self.c = kw['c']
        self.A = kw['A']

    # Right hand side of the ballistic equation :
    def dydt(self, y, **kw):  # unused kw args for consistency
        dxdt = y[2]
        dydt = y[3]
        # Computes velocity magnitude
        magv = np.sqrt(dxdt**2 + dydt**2)
        return np.array(
            [dxdt, dydt, -self.mu * dxdt * magv, -self.g - self.mu * dydt * magv],
        )

    # Neutral boundary condition function :

    def bcond(self, y):
        return y

    def newton_raphson(self, y, y1, f, dt, niter_max, tol):
        for iter in range(niter_max):
            # y1 riht hand side :
            f1 = F_ballistic(y, self.rho, self.c, self.A, self.g)
            # residual computation :
            residual = y1 - y - (dt / 2.0) * (f + f1)
            residual_norm = np.linalg.norm(residual)
            # Convergence ?
            if residual_norm < tol:
                print(f'convergence after niter = {iter}')
                break
            # Tangent matrix computation :
            # Crank-Nicolson : I - dt/2 * jacobian(F)
            jacobian = (
                identity(len(y1)) - (dt / 2.0) *
                jacobian_ballistic(y1, self.mu)
            )
            # y-increment computation :
            delta_y = solve(jacobian, -residual, check_finite=True)
            y1 += delta_y
            # yielding a new residual value :
            residual = y1 - y - (dt / 2.0) * (f + f1)
            residual_norm = np.linalg.norm(residual)
            # Convergence ?
            if residual_norm < tol:
                print(f'convergence after niter = {iter}')
                break
            if iter == (niter_max - 1):
                print(f'divergence, ||res|| = {residual_norm}')


class Rabbit:
    def __init__(self, **kw):
        self.k = kw['k']
        self.b = kw['b']

    # Right hand side of the rabbit equation :

    def dydt(self, y, **kw):  # unused kw args for consistency
        return self.k * y * (1.0 - (y / self.b))

    # Neutral boundary condition function :

    def bcond(self, y):
        return y

    def newton_raphson(self, y, y1, f, dt, niter_max, tol):
        for iter in range(niter_max):
            # y1 riht hand side :
            f1 = F_rabbit(y, self.k, self.b)
            # residual computation :
            residual = y1 - y - (dt / 2.0) * (f + f1)
            residual_norm = np.abs(residual)
            # Convergence ?
            if residual_norm < tol:
                print(f'convergence after niter = {iter}')
                break
            # Tangent matrix computation :
            # Crank-Nicolson : I - dt/2 * jacobian(F)
            jacobian = (
                1. - (dt / 2.0) * jacobian_rabbit(y1, self.k, self.b)
            )
            # y-increment computation :
            delta_y = -residual/jacobian
            y1 += delta_y
            # yielding a new residual value :
            residual = y1 - y - (dt / 2.0) * (f + f1)
            residual_norm = np.linalg.norm(residual)
            # Convergence ?
            if residual_norm < tol:
                print(f'convergence after niter = {iter}')
                break
            if iter == (niter_max - 1):
                print(f'divergence, ||res|| = {residual_norm}')


class Diffusion:
    def __init__(self, **kw):
        self.D = kw['D']
        self.mesh = kw['mesh']
        self.dict_bc = {
            k: kw.get(k)
            for k in [
                'dirichlet',
                'neumann',
                'left_boundary',
                'bottom_boundary',
                'right_boundary',
                'top_boundary',
                'interpolation_coeff',
            ]
        }
        # Warning : quick fix : apply_boundaries does not appreciate having mesh as positional argument and in the dict with the same name!
        # To do : create a config class

    # Right hand side of the diffusion equation :
    def dydt(self, y, **kw):
        # k1 = {k: kw.get(k) for k in ['dirichlet','neumann','left_boundary','bottom_boundary','right_boundary','top_boundary','interpolation_coeff'] }
        # print(k1)
        # Applying the boundary conditions :
        y = apply_boundaries(self.mesh, y, **self.dict_bc)
        grads = diffops.gradient_mesh(self.mesh, y)
        grads[0] = self.D * grads[0]
        grads[1] = self.D * grads[1]
        grad2 = []
        [
            grad2.append(diffops.gradient_mesh(self.mesh, gradi)[i])
            for i, gradi in enumerate(grads)
        ]
        return np.sum(np.array(grad2), axis=0)
        # test
        # sl = diffops.scalar_laplacian_tensor(self.mesh) @ (self.D.flatten() * y.flatten())
        # return sl.reshape(y.shape)

    # Method to apply the boundary conds :

    def bcond(self, y):
        return apply_boundaries(self.mesh, y, **self.dict_bc)

    # Newton-Raphson :
    #   remark : the BCs are not applied within the N-R loop
    def newton_raphson(self, y, y1, f, dt, niter_max, tol):
        for iter in range(niter_max):
            # y1 riht hand side :
            f1 = F_diffusion(self.mesh, y1, self.D)
            # flattening arrays :
            y1 = y1.flatten()
            f1 = f1.flatten()
            # residual computation :
            residual = y1 - y.flatten() - (dt / 2.0) * (f.flatten() + f1)
            residual_norm = np.linalg.norm(residual)
            # Convergence ?
            if residual_norm < tol:
                y1 = y1.reshape(y.shape)
                print(f'convergence after niter = {iter}')
                break
            # Tangent matrix computation :
            # Crank-Nicolson : I - dt/2 * nabla^2
            jacobian = (
                identity(self.mesh.nx * self.mesh.ny)
                - (dt / 2.0) * diffops.scalar_laplacian_tensor(self.mesh)
            ).toarray()
            # y-increment computation :
            delta_y = solve(jacobian, -residual, check_finite=True)
            y1 += delta_y
            # yielding a new residual value :
            residual = y1 - y.flatten() - (dt / 2.0) * (f.flatten() + f1)
            residual_norm = np.linalg.norm(residual)
            # Convergence ?
            if residual_norm < tol:
                y1 = y1.reshape(y.shape)
                print(f'convergence after niter = {iter}')
                break
            if iter == (niter_max - 1):
                print(f'divergence, ||res|| = {residual_norm}')
            # reshaping y1 :
            y1 = y1.reshape(y.shape)


def F(y, **kw):
    if kw['case'] == 'ballistic':
        return F_ballistic(y, kw['rho'], kw['c'], kw['A'], kw['g'])
    if kw['case'] == 'rabbit':
        return F_rabbit(y, kw['k'], kw['b'])
    if kw['case'] == 'diffusion_2D':
        return F_diffusion(y, edge_order=1, **kw)


def F_ballistic(y, rho: float, c: float, A: float, g: float):
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
    mu = 0.5 * rho * c * A
    # Computes velocity magnitude
    magv = np.sqrt(dxdt**2 + dydt**2)
    return np.array([dxdt, dydt, -mu * dxdt * magv, -g - mu * dydt * magv])


def F_rabbit(N: np.ndarray, k: float, b: float):
    """
    Rabbit puplation dynamics function for F(x,y,dydx')

    N (array): Current state of N (population)
    k (float): reproduction rate
    b (float): carrying capacity (max population / area)

    Returns:
    2nd member = 1D-array : F(N,k,b)
    """
    return k * N * (1.0 - (N / b))


def F_diffusion(mesh: Mesh, data: np.ndarray, D: np.ndarray) -> np.ndarray:
    """ "
    Computes the scalar laplacian of D * y where D is the diffusion field.

    Parameters :

    data(x,y,t) (array): Current temperature field
    D(x,y,t) (array): diffusion coefficient

    Returns :

    F(x,y,t) : div[ D(x,y) * grad(T(x,y,t)) ]
    2nd member of the diffusion equation
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


def jacobian_rabbit(y, k, b):
    return 1. - k - (2.*y/b)


def jacobian_ballistic(y, mu):
    z = y[2]
    w = y[3]
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [
                0,
                0,
                -1.0 * mu * z**2 / (w**2 + z**2) ** 0.5 -
                mu * (w**2 + z**2) ** 0.5,
                -1.0 * mu * w * z / (w**2 + z**2) ** 0.5,
            ],
            [
                0,
                0,
                -1.0 * mu * w * z / (w**2 + z**2) ** 0.5,
                -1.0 * mu * w**2 / (w**2 + z**2) ** 0.5 -
                mu * (w**2 + z**2) ** 0.5,
            ],
        ],
    )

# sympy output : Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1.0*mu*z**2/(w**2 + z**2)**0.5 - mu*(w**2 + z**2)**0.5, -1.0*mu*w*z/(w**2 + z**2)**0.5], [0, 0, -1.0*mu*w*z/(w**2 + z**2)**0.5, -1.0*mu*w**2/(w**2 + z**2)**0.5 - mu*(w**2 + z**2)**0.5]])
