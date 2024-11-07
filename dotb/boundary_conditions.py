#!/usr/bin/python3
# %%
from __future__ import annotations

import numpy as np
# %%


def boundary_conds(data: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    """
    # Creation of contour_test, and arr_test :
    # (filled only for 2D cases)
    # --> for ploting of the boundary conditions in initiate_y
    #     - arr_test : concatenated values of the boundary conditions
    #     - contour_test : contour of the square domain
    contour_test, arr_test = (np.empty(0), np.empty(0))

    # BC only for the diffusion case :
    if kw['case'] == 'ballistic':
        return data, contour_test, arr_test
    if kw['case'] == 'rabbit':
        return data, contour_test, arr_test

    # 1D :
    if len(np.shape(data)) == 1:
        if kw['dirichlet']:
            data[0] = kw['left_boundary']
            data[-1] = kw['right_boundary']
        if kw['neumann']:
            data[0] = data[1] - kw['n_x'] * kw['left_boundary']
            data[-1] = data[-2] + kw['n_x'] * kw['right_boundary']

    # 2D :
    if len(np.shape(data)) == 2:
        # Domain coordinates :
        x = np.linspace(-kw['l_x']/2., kw['l_x']/2., kw['n_x'])
        y = np.linspace(-kw['l_x']/2., kw['l_y']/2., kw['n_y'])
        lx = x[-1] - x[0]
        ly = y[-1] - y[0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        arr_left = kw['left_boundary'] * np.ones_like(data[:, 0])
        arr_right = kw['right_boundary'] * np.ones_like(arr_left)
        arr_bottom = kw['bottom_boundary'] * np.ones_like(data[0, :])
        arr_top = kw['top_boundary'] * np.ones_like(arr_bottom)

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

            ix_sup = np.where(x >= x[0] + lx * (1.0 - alpha))
            ix_mid = np.where((x > x[0] + lx * alpha)
                              & (x < x[0] + lx * (1.0 - alpha)))
            iy_sup = np.where(y >= y[0] + ly * (1.0 - alpha))
            iy_mid = np.where((y > y[0] + ly * alpha)
                              & (y < y[0] + ly * (1.0 - alpha)))

            vlim_x = lx * alpha
            vlim_y = ly * alpha

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
                kw['bottom_boundary'] *
                (1.0 - beta_x) + val_lower_right * beta_x
            )

            arr_x_interp_lower_left = kw[
                'bottom_boundary'
            ] * beta_x + val_lower_left * (1.0 - beta_x)

            arr_bottom = np.concatenate(
                (
                    arr_x_interp_lower_left,
                    arr_bottom[ix_mid], arr_x_interp_lower_right,
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
                    arr_top[ix_mid], arr_x_interp_upper_right,
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
                    arr_left[iy_mid], arr_y_interp_upper_left,
                ),
            )

            # right y array :
            arr_y_interp_lower_right = kw[
                'right_boundary'
            ] * beta_y + val_lower_right * (1.0 - beta_y)

            arr_y_interp_upper_right = (
                kw['right_boundary'] * (1.0 - beta_y) +
                val_upper_right * beta_y
            )

            arr_right = np.concatenate(
                (
                    arr_y_interp_lower_right,
                    arr_right[iy_mid], arr_y_interp_upper_right,
                ),
            )

        if kw['dirichlet']:
            # returns a list left-bottom-right-top of interpolated thus C^1 boundary conditions
            # setting y boundaries :
            # left :
            data[0, :] = arr_left
            # right :
            data[-1, :] = arr_right
            # setting x boundaries :
            # bottom :
            data[:, 0] = arr_bottom
            # top :
            data[:, -1] = arr_top

        if kw['neumann']:
            # Rreproduction at the 1st order of the gradient values at the boundaries by modifying the border temperature values.
            # Remark :
            # - left border : (forward difference) grad = (T(x+dx) - T(x)) / dx
            # - right border : (backward difference) grad = (T(x) - T(x-dx)) / dx
            # - bottom border : (forward difference) grad = (T(y+dy) - T(y)) / dy
            # - top border : (backward difference) grad = (T(y) - T(y-dy)) / dy

            # setting x-gradient values :
            # left :
            data[0, :] = data[1, :] - dx * arr_left
            # right :
            data[-1, :] = data[-2, :] + dx * arr_right

            # setting y-gradient values :
            # bottom :
            data[:, 0] = data[:, 1] - dy * arr_bottom
            # top :
            data[:, -1] = data[:, -2] + dy * arr_top

        arr_test = np.concatenate(
            (np.flip(arr_left), arr_bottom, arr_right, np.flip(arr_top)),
        )

        contour_test = np.linspace(
            0.0, 2.*(x[-1]-x[0]) + 2.*(y[-1]-y[0]), len(arr_test),
        )

    return data, contour_test, arr_test
