#!/usr/bin/python3
from __future__ import annotations

import numpy as np
# import numpy.linalg as LA
# import dotb.boundary_conditions as BC
# import dotb.second_member as second


# %%
class EulerExplicit:
    def __init__(self, y0, t, rhs, **kw):
        self.t = t
        self.dt = t[1] - t[0]
        self.y0 = y0
        self.rhs = rhs
        self.n_save = kw['n_save']

    def solve(self, **kw):
        sol = []
        sol.append(self.y0)
        # Aply possible boundary conditions :
        y = self.rhs.bcond(self.y0)
        # print(f'solve : kw = {kw}')
        # print(f'dydyt 0 : {self.rhs.dydt(y,**kw)}')
        for i, ti in enumerate(self.t[:-1]):
            # Apply Euler explicit formula
            y_new = y + self.dt * self.rhs.dydt(y, **kw)
            # Aply possible boundary conditions :
            y_new = self.rhs.bcond(y_new)
            # print(f'dydyt : {self.rhs.dydt(y,**kw)}')
            # print(self.rhs.dydt(y, **kw))
            if (i + 1) % self.n_save == 0:
                # print(f'saving {i}^th time_step')
                sol.append(y_new)
            y = y_new
        return np.array(sol)


class AdamsBashforth:
    def __init__(self, y0, t, rhs, **kw):
        self.t = t
        self.dt = t[1] - t[0]
        self.y0 = y0
        self.rhs = rhs
        self.n_save = kw['n_save']

    def solve(self, **kw):
        sol = []
        sol.append(self.y0)
        # Apply possible BC to y0
        y = self.rhs.bcond(self.y0)
        # Initialization with euler explicit :
        y1 = y + self.dt * self.rhs.dydt(y, **kw)
        # Apply possible BC to y0
        y1 = self.rhs.bcond(y1)
        if 1 % self.n_save == 0:
            sol.append(y1)
        for i, ti in enumerate(self.t[:-2]):
            y_new = y1 + self.dt * (
                (3.0 / 2.0) * self.rhs.dydt(y1, **kw) -
                (1.0 / 2.0) * self.rhs.dydt(y, **kw)
            )
            # Aply possible boundary conditions :
            y_new = self.rhs.bcond(y_new)
            if (i + 2) % self.n_save == 0:
                # print(f'saving {i}^th time step')
                sol.append(y_new)

            # y, y1 update for next time step :
            #   y_{n} = y_{n+1} :
            y = y1
            #   y_{n+1} = y_{n+2} :
            y1 = y_new
        return np.array(sol)


class CrankNicolson:
    def __init__(self, y0, t, rhs, **kw):
        self.t = t
        self.dt = t[1] - t[0]
        self.y0 = y0
        self.rhs = rhs
        self.n_save = kw['n_save']
        self.n_iter_max = kw['n_iter_max']
        self.tol = kw['tol']

    def solve(self, **kw):
        sol = []
        sol.append(self.y0)
        y = self.rhs.bcond(self.y0)
        for i, ti in enumerate(self.t[:-2]):
            # Apply possible BC to y0
            y = self.rhs.bcond(y)
            # Matching right hand side :
            f = self.rhs.dydt(y)
            # Initialization with euler explicit :
            y1 = y + self.dt * f
            # Apply possible BC to y0
            y1 = self.rhs.bcond(y1)
            # # Matching right hand side :
            # f1 = self.rhs.dydt(y1)
            # Iterative Newton-Raphson :
            self.rhs.newton_raphson(
                y, y1, f, self.dt, self.n_iter_max, self.tol,
            )
            # iteration :
            if (i + 1) % self.n_save == 0:
                # print(f'saving {i}^th time step')
                sol.append(y)
            y = y1
        return np.array(sol)


# def euler_explicit(y: np.ndarray, t: np.ndarray, **kw) -> np.ndarray:
#     """
#     Solve ∂y/∂t = F(x,y,y') using Euler explicit method

#     Parameters:
#     F (function): Function defining the PDE
#     y0 (array): Initial condition
#     x (array): Spatial grid
#     t (array): Time array
#     dx (float): Spatial step size
#     dt (float): Time step size

#     Returns:
#     y (ndarray): Solution tensor field
#     """
#     dt = t[1] - t[0]
#     print(f'solver : dt = {dt}')
#     sol = []
#     sol.append(y)
#     for i, ti in enumerate(t[:-1]):
#         # Apply the boundary conditions :
#         # y = BC.apply_boundaries(kw['mesh'], y, **kw)
#         # Apply Euler explicit formula
#         y_new = y + dt * second.F(y, **kw)
#         if (i + 1) % kw['n_save'] == 0:
#             print(f'saving {i}^th time step')
#             sol.append(y_new)
#             # print(f'solver : sol = {sol}')
#         y = y_new
#     return np.array(sol)


# def adams_bashforth(y: np.ndarray, t: np.ndarray, **kw) -> np.ndarray:
#     """
#     Solve ∂y/∂t = F(x,y,y') using Euler explicit method

#     Parameters:
#     y (tensor): integrated tensor field
#     F (function): Function defining the PDE
#     x (array): Spatial grid
#     t (array): Time array
#     dx (float): Spatial step size
#     dt (float): Time step size

#     Returns:
#     y (ndarray) : tensor field value at the saved time steps
#     y_{n+2} = y_{n+1} + (3/2)*dt*F_{n+1} - (1/2)*dt*F_n
#     """
#     dt = t[1] - t[0]
#     print(f'solver : dt = {dt}')
#     sol = []
#     sol.append(y)
#     # y = BC.apply_boundaries(mesh, y, **kw)
#     # Initialization of y_{n+1} (n=0) via Euler method :
#     y1 = y + dt * second.F(y, **kw)
#     if 1 % kw['n_save'] == 0:
#         # .i.e if nsave = 1 :
#         sol.append(y1)
#     #
#     for i, ti in enumerate(t[:-2]):
#         # Apply the boundary conditions :
#         # y = BC.apply_boundaries(mesh, y, **kw)
#         # y1 = BC.apply_boundaries(mesh, y1, **kw)
#         # Apply adams-bashforth formula :
#         y_new = y1 + dt * (
#             (3.0 / 2.0) * second.F(y1, **kw) - (1.0 / 2.0) * second.F(y, **kw)
#         )
#         if (i + 2) % kw['n_save'] == 0:
#             print(f'saving {i}^th time step')
#             sol.append(y_new)

#         # y, y1, y2 values update for next time step :
#         #   y_{n} = y_{n+1} :
#         y = y1
#         #   y_{n+1} = y_{n+2} :
#         y1 = y_new
#     return np.array(sol)
