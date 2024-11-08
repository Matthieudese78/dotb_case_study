#!/usr/bin/python3
# %%
from __future__ import annotations

import numpy as np

from dotb.refined_mesh_class import Mesh
# %%


def boundary_conditions_mesh(mesh: Mesh, **kw) -> list:
    """
    Creates the arrays corresponding to the Dirichlet or Neumann
    boundary conditions on the boundaries of the domain.
    If non zero interpolation coeff, the values of two adjacent
    edges are linked through a C^1 interolation function.
    The merge at the corner value : (example of the lower left corner)
    - Dirichlet : (Tleft + Tbottom)/2
    - Neumann : 0. (insulated corners)

    Parameters :
    - only a 2D mesh and the config dictionnary kw
    Returns :
    A list of the [left - bottom - right - top] boundary conditions array
    (interpolated or not)
    """
    x = mesh.X[:, 0]
    y = mesh.Y[0, :]
    arr_left = kw['left_boundary'] * np.ones_like(mesh.Y[0, :])
    arr_right = kw['right_boundary'] * np.ones_like(mesh.Y[-1, :])
    arr_bottom = kw['bottom_boundary'] * np.ones_like(mesh.X[:, 0])
    arr_top = kw['top_boundary'] * np.ones_like(mesh.X[:, -1])

    if kw['interpolation_coeff'] > 0.0:
        if kw['dirichlet']:
            val_lower_left = np.mean(
                [kw['left_boundary'], kw['bottom_boundary']],
            )
            val_upper_left = np.mean(
                [kw['left_boundary'], kw['top_boundary']],
            )
            val_lower_right = np.mean(
                [kw['right_boundary'], kw['bottom_boundary']],
            )
            val_upper_right = np.mean(
                [kw['right_boundary'], kw['top_boundary']],
            )

        if kw['neumann']:  # all corners are insulated by default :
            val_lower_left = 0.0
            val_upper_left = 0.0
            val_lower_right = 0.0
            val_upper_right = 0.0

        alpha = kw['interpolation_coeff']

        ix_sup = np.where(x >= x[0] + mesh.lx * (1.0 - alpha))
        ix_mid = np.where(
            (x > x[0] + mesh.lx * alpha) & (
                x <
                x[0] + mesh.lx * (1.0 - alpha)
            ),
        )
        iy_sup = np.where(y >= y[0] + mesh.ly * (1.0 - alpha))
        iy_mid = np.where(
            (y > y[0] + mesh.ly * alpha) & (
                y <
                y[0] + mesh.ly * (1.0 - alpha)
            ),
        )

        vlim_x = mesh.lx * alpha
        vlim_y = mesh.ly * alpha

        x_interp = np.linspace(0.0, vlim_x, len(ix_sup[0]))
        y_interp = np.linspace(0.0, vlim_y, len(iy_sup[0]))

        beta_x = 0.5 * (
            1.0
            + (2.0 - np.abs(x_interp - (vlim_x / 2.0)) / (vlim_x / 2.0))
            * ((x_interp - (vlim_x / 2.0)) / (vlim_x / 2.0))
        )

        beta_y = 0.5 * (
            1.0
            + (2.0 - np.abs(y_interp - (vlim_y / 2.0)) / (vlim_y / 2.0))
            * ((y_interp - (vlim_y / 2.0)) / (vlim_y / 2.0))
        )

        # bottom x array :
        arr_x_interp_lower_right = (
            kw['bottom_boundary'] * (1.0 - beta_x) + val_lower_right * beta_x
        )

        arr_x_interp_lower_left = kw['bottom_boundary'] * beta_x + val_lower_left * (
            1.0 - beta_x
        )

        arr_bottom = np.concatenate(
            (
                arr_x_interp_lower_left,
                arr_bottom[ix_mid],
                arr_x_interp_lower_right,
            ),
        )

        # top x array :
        arr_x_interp_upper_right = (
            kw['top_boundary'] * (1.0 - beta_x) + val_upper_right * beta_x
        )

        arr_x_interp_upper_left = kw['top_boundary'] * beta_x + val_upper_left * (
            1.0 - beta_x
        )

        arr_top = np.concatenate(
            (
                arr_x_interp_upper_left,
                arr_top[ix_mid],
                arr_x_interp_upper_right,
            ),
        )

        # left y array :
        arr_y_interp_lower_left = kw['left_boundary'] * beta_y + val_lower_left * (
            1.0 - beta_y
        )

        arr_y_interp_upper_left = (
            kw['left_boundary'] * (1.0 - beta_y) + val_upper_left * beta_y
        )

        arr_left = np.concatenate(
            (
                arr_y_interp_lower_left,
                arr_left[iy_mid],
                arr_y_interp_upper_left,
            ),
        )

        # right y array :
        arr_y_interp_lower_right = kw['right_boundary'] * beta_y + val_lower_right * (
            1.0 - beta_y
        )

        arr_y_interp_upper_right = (
            kw['right_boundary'] * (1.0 - beta_y) + val_upper_right * beta_y
        )

        arr_right = np.concatenate(
            (
                arr_y_interp_lower_right,
                arr_right[iy_mid],
                arr_y_interp_upper_right,
            ),
        )

    return [arr_left, arr_bottom, arr_right, arr_top]


def apply_boundaries(mesh: Mesh, data: np.ndarray, **kw) -> np.ndarray:
    """
    Applies the boundary conditions computed with boundary_conditions_mesh to the edges of the 2D domain.

    Parameters:
    - 2D mesh
    - data field
    - the config input dictionnary

    Returns :
    The modified dataset with new values at its edges.
    """
    # Computations of boundary conditions :
    boundaries = boundary_conditions_mesh(mesh, **kw)
    if kw['dirichlet']:
        # returns a list left-bottom-right-top of interpolated thus C^1 boundary conditions
        # setting y boundaries :
        # left :
        data[0, :] = boundaries[0]
        # right :
        data[-1, :] = boundaries[2]
        # setting x boundaries :
        # bottom :
        data[:, 0] = boundaries[1]
        # top :
        data[:, -1] = boundaries[3]

    if kw['neumann']:
        # Rreproduction at the 1st order of the gradient values at the boundaries by modifying the border temperature values.
        # Remark :
        # - left border : (forward difference) grad = (T(x+dx) - T(x)) / dx
        # - right border : (backward difference) grad = (T(x) - T(x-dx)) / dx
        # - bottom border : (forward difference) grad = (T(y+dy) - T(y)) / dy
        # - top border : (backward difference) grad = (T(y) - T(y-dy)) / dy
        # x-gradient left = boundaries[0]
        #          BC_0 = grad_x_0 :
        data[0, :] = data[1, :] - boundaries[0] * mesh.delta_x_minus[0, :]
        # x-gradient right = boundaries[2]
        #          BC_n = grad_x_n :
        data[-1, :] = mesh.delta_x_plus[-1, :] * \
            boundaries[2] + data[-2, :]
        # y-gradient bottom = boundaries[1]
        #          BC_0 = grad_y_0 :
        data[:, 0] = data[:, 1] - mesh.delta_y_minus[:, 0] * boundaries[1]
        # y-gradient top = boundaries[3]
        #          BC_n = grad_y_n :
        data[:, -1] = mesh.delta_y_plus[:, -1] * \
            boundaries[3] + data[:, -2]

    return data
