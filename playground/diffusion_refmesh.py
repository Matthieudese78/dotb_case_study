#!/usr/bin/python3
# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dotb.refined_mesh_class import Mesh


# from scipy.sparse.linalg import spsolve
# from scipy.sparse import diags, identity
# %%
def postt_2Dmap(mesh: Mesh, data, title, labelx, labely, labelbar, s=5):
    # Create meshgrid for plotting
    X = mesh.X
    Y = mesh.Y

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter plot with colors determined by data values
    scatter = ax.scatter(
        X.ravel(),
        Y.ravel(),
        c=data,
        cmap='inferno',
        s=s,
    )

    # Set limits and aspect ratio
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_xlim(np.min(Y), np.max(Y))
    # ax.set_aspect("equal")

    ax.set_xlabel(f'{labelx}')
    ax.set_ylabel(f'{labely}')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel(f'{labelbar}')

    plt.title(f'{title}')
    plt.show()
    plt.close('all')


def postt_2Dmap_interior(mesh: Mesh, data, title, labelx, labely, labelbar, s=5):
    # Create meshgrid for plotting
    X = mesh.X[1:-1, 1:-1]
    Y = mesh.Y[1:-1, 1:-1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter plot with colors determined by data values
    scatter = ax.scatter(
        X.ravel(),
        Y.ravel(),
        c=data,
        cmap='inferno',
        s=s,
    )

    # Set limits and aspect ratio
    ax.set_xlim(-1.05 * mesh.lx / 2.0, 1.05 * mesh.lx / 2.0)
    ax.set_ylim(-1.05 * mesh.ly / 2.0, 1.05 * mesh.ly / 2.0)
    # ax.set_aspect("equal")

    ax.set_xlabel(f'{labelx}')
    ax.set_ylabel(f'{labely}')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel(f'{labelbar}')

    plt.title(f'{title}')
    plt.show()
    plt.close('all')


def gradient(data, discr, edge_order=1):
    grads = []
    [
        grads.append(
            np.gradient(
                data,
                discr[i],
                axis=i,
                edge_order=edge_order,
            ),
        )
        for i, si in enumerate(np.shape(data))
    ]
    return np.array(grads)


# def gradient(data, discr, edge_order):
def gradient_mesh(mesh: Mesh, data: np.ndarray) -> np.ndarray:
    # Interior nodes : for i = 1 --> n-1
    #       (cf numpy.gradient notice --> bibliographic sources)
    # For structured non uniformely sized meshes :
    # \pratial y / \partial x =
    # [ (hs**2)*f(xi + hd) + (hd**2 - hs**2)*f(xi)   - hd**2 * f(xi - hs) ] / hs*hd*(hd + hs)
    # Where
    #       hs is the node distance to its left neighbor
    #       hd is the node distance to its right neighbor
    # Example node 1 :
    # [ (hs**2)*data[2]    + (hd**2 - hs**2)*data[1] - hd**2 * data[0] ] / hs*hd*(hd + hs)
    # [ (hs**2)*f(xi + hd) + (hd**2 - hs**2)*f(xi) - hd**2 * f(xi - hs) ] / hs*hd*(hd + hs)
    hs = (mesh.delta_x_minus)
    hd = (mesh.delta_x_plus)
    grad_x = (
        (hs**2) * data[2:, :] + (hd**2 - hs**2) *
        data[1:-1, :] - (hd**2) * data[:-2, :]
    ) / (hs * hd * (hd + hs))
    print(f'shape grad_x = {grad_x.shape}')
    # grad_x = (
    #     (hs**2) * data[:, 2:] + (hd**2 - hs**2) * data[:, 1:-1] - (hd**2) * data[:, :-2]
    # ) / (hs * hd * (hd + hs))

    hs = (mesh.delta_y_minus)
    hd = (mesh.delta_y_plus)
    grad_y = (
        (hs**2) * data[:, 2:] + (hd**2 - hs**2) *
        data[:, 1:-1] - hd**2 * data[:, :-2]
    ) / (hs * hd * (hd + hs))
    print(f'shape grad_y = {grad_y.shape}')

    # Boundary nodes : for i = 0, i = -1
    # i = 0  : forward difference
    # i = -1 : backward difference
    #
    grad_x_0 = (data[1, :] - data[0, :]) / (mesh.delta_x_minus)[0, :]
    print(f'shape grad_x_0 = {grad_x_0.shape}')
    grad_x_n = (data[-1, :] - data[-2, :]) / (mesh.delta_x_plus)[-1, :]
    print(f'shape grad_x_n = {grad_x_n.shape}')
    print(f'len border x : {len(grad_x_n)}')
    # stacking above : lower x values
    grad_x = np.vstack((grad_x_0, grad_x))
    # stacking under : higher x values
    grad_x = np.vstack((grad_x, grad_x_n))
    #
    grad_y_0 = (data[:, 1] - data[:, 0]) / (mesh.delta_y_minus)[:, 0]
    grad_y_n = (data[:, -1] - data[:, -2]) / (mesh.delta_y_plus)[:, -1]
    # print(f'right stacking : {np.hstack((M, right_column.reshape(-1,1)))}')
    # print(f"left stacking : {np.hstack((left_column.reshape(-1,1),M))}")
    # stacking left : lower y values
    grad_y = np.hstack((grad_y_0.reshape(-1, 1), grad_y))
    # stacking right : higher y values
    grad_y = np.hstack((grad_y, grad_y_n.reshape(-1, 1)))

    return np.array([grad_x, grad_y])


def divergence_mesh(mesh: Mesh, data: np.ndarray) -> np.ndarray:
    grads = gradient_mesh(mesh, data)
    return np.sum(grads, axis=0)


def gradient_square_mesh(mesh: Mesh, data: np.ndarray) -> np.ndarray:
    grads = gradient_mesh(mesh, data)
    grad_square = []
    [
        grad_square.append(gradient_mesh(mesh, gradi)[i])
        for i, gradi in enumerate(grads)
    ]
    return np.array(grad_square)


def F_diffusion(mesh: Mesh, data: np.ndarray, D: np.ndarray) -> np.ndarray:
    grads = gradient_mesh(mesh, data)
    grads[0] = D * grads[0]
    grads[1] = D * grads[1]
    grad2 = []
    [
        grad2.append(gradient_mesh(mesh, gradi)[i])
        for i, gradi in enumerate(grads)
    ]
    return np.sum(np.array(grad2), axis=0)


def diffusion(x, y, mesh, data, D, discr, edge_order=1, **kw) -> np.ndarray:
    # 1D :
    if len(np.shape(data)) == 1:
        if kw['dirichlet']:
            data[0] = kw['left_boundary']
            data[-1] = kw['right_boundary']
        if kw['neumann']:
            data[0] = data[1] - discr[0] * kw['left_boundary']
            data[-1] = data[-2] + discr[0] * kw['right_boundary']

    # 2D :
    if len(np.shape(data)) == 2:
        # Computations of boundary conditions :
        # boundaries = boundary_conditions_2D(x, y, data, **kw)
        boundaries = boundary_conditions_2D_mesh(mesh, data, **kw)
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

            # setting x-gradient values :
            # left :
            data[0, :] = data[1, :] - discr[0] * boundaries[0]
            # right :
            data[-1, :] = data[-2, :] + discr[0] * boundaries[2]

            # setting y-gradient values :
            # bottom :
            data[:, 0] = data[:, 1] - discr[1] * boundaries[1]
            # top :
            data[:, -1] = data[:, -2] + discr[1] * boundaries[3]

    grads = gradient_mesh(mesh, data)
    # grads = gradient(data, discr, edge_order)

    # multiplying the temperature gradient by the conduction coeff. :

    # 1D :
    if len(np.shape(data)) == 1:
        grads = D * grads
        # order two derivative = (in 1D) scalar laplacian
        sl = np.gradient(grads)

    # 2D :
    if len(np.shape(data)) == 2:
        grads[0] = D * grads[0]
        grads[1] = D * grads[1]
        # order two derivative :
        grad2s = []
        [
            grad2s.append(
                np.gradient(
                    gradi,
                    discr[i],
                    axis=i,
                    edge_order=edge_order,
                ),
            )
            for i, gradi in enumerate(grads)
        ]
        grad2s = np.array(grad2s)
        # scalar laplacian :
        sl = np.sum(grad2s, axis=0)

    return sl


def scalar_laplacian_mesh(mesh: Mesh, data: np.ndarray):
    return np.sum(gradient_square_mesh(mesh, data), axis=0)


def boundary_conditions_2D_mesh(mesh: Mesh, data: np.ndarray, **kw) -> list:
    print('BOUNDARY CONDITIONS')
    x = mesh.X[:, 0]
    y = mesh.Y[0, :]
    arr_left = kw['left_boundary'] * np.ones_like(mesh.Y[0, :])
    arr_right = kw['right_boundary'] * np.ones_like(mesh.Y[-1, :])
    arr_bottom = kw['bottom_boundary'] * np.ones_like(mesh.X[:, 0])
    arr_top = kw['top_boundary'] * np.ones_like(mesh.X[:, -1])
    print(f'size : BC_left {len(arr_left)}')
    print(f'size : BC_bottom {len(arr_bottom)}')
    print(f'size : BC_right {len(arr_right)}')
    print(f'size : BC_top {len(arr_top)}')

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
            kw['bottom_boundary'] *
            (1.0 - beta_x) + val_lower_right * beta_x
        )

        arr_x_interp_lower_left = kw[
            'bottom_boundary'
        ] * beta_x + val_lower_left * (1.0 - beta_x)

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
                arr_right[iy_mid],
                arr_y_interp_upper_right,
            ),
        )

    return [arr_left, arr_bottom, arr_right, arr_top]


def boundary_conditions_2D(x, y, data, **kw):
    if len(np.shape(data)) == 2:
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

            lx = x[-1] - x[0]
            ly = y[-1] - y[0]
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
                    arr_right[iy_mid],
                    arr_y_interp_upper_right,
                ),
            )

        return [arr_left, arr_bottom, arr_right, arr_top]


def apply_bounds(mesh: Mesh, data: np.ndarray, **kw) -> np.ndarray:
    # 1D :

    # 2D :
    if len(np.shape(data)) == 2:
        # Computations of boundary conditions :
        boundaries = boundary_conditions_2D_mesh(mesh, data, **kw)
        if kw['dirichlet']:
            # returns a list left-bottom-right-top of interpolated thus C^1 boundary conditions
            # setting y boundaries :
            # left :
            data[0, :] = boundaries[0]
            print('left boundary OK')
            # right :
            data[-1, :] = boundaries[2]
            print('right boundary OK')
            # setting x boundaries :
            # bottom :
            data[:, 0] = boundaries[1]
            print('bottom boundary OK')
            # top :
            data[:, -1] = boundaries[3]
            print('top boundary OK')

        if kw['neumann']:
            # Rreproduction at the 1st order of the gradient values at the boundaries by modifying the border temperature values.
            # Remark :
            # - left border : (forward difference) grad = (T(x+dx) - T(x)) / dx
            # - right border : (backward difference) grad = (T(x) - T(x-dx)) / dx
            # - bottom border : (forward difference) grad = (T(y+dy) - T(y)) / dy
            # - top border : (backward difference) grad = (T(y) - T(y-dy)) / dy
            #   todo : multiply with D once the config class created :
            # x-gradient left = boundaries[0]
            #          BC_0 = grad_x_0 = (data[:, 1] - data[:, 0]) / mesh.delta_x_minus[:, 0]
            data[:, 0] = data[:, 1] - boundaries[0] * mesh.delta_x_minus[:, 0]
            # x-gradient right = boundaries[2]
            #          BC_n = grad_x_n = (data[:, -1] - data[:, -2]) / mesh.delta_x_plus[:, -1]
            data[:, -1] = mesh.delta_x_plus[:, -1] * \
                boundaries[2] + data[:, -2]
            # y-gradient bottom = boundaries[1]
            #          BC_0 = grad_y_0 = (data[1, :] - data[0, :]) / mesh.delta_y_minus[0, :]
            data[0, :] = data[1, :] - mesh.delta_y_minus[0, :] * boundaries[1]
            # y-gradient top = boundaries[3]
            #          BC_n = grad_y_n = (data[-1, :] - data[-2, :]) / mesh.delta_y_minus[-1, :]
            data[-1, :] = mesh.delta_y_plus[-1, :] * \
                boundaries[3] + data[-2, :]

    return data


# %% Set up the problem
# time :
t_end = 1.0
# mesh :
lx = 1.0
ly = 2.0
# ly = 1.
# dx_fine = 0.01
# dy_fine = 0.01
dx_fine = 0.01
dy_fine = 0.01
dx_coarse = 3.0 * dx_fine
dy_coarse = 3.0 * dy_fine
# dx_coarse = 2.0 * dx_fine
# dy_coarse = 2.0 * dy_fine
x_refine_percent = 20.0
y_refine_percent = 20.0
# x_refine_percent = 100.
# y_refine_percent = 100.

mesh = Mesh(
    lx=lx,
    ly=ly,
    dx_fine=dx_fine,
    dy_fine=dy_fine,
    dx_coarse=dx_coarse,
    dy_coarse=dy_coarse,
    x_refine_percent=x_refine_percent,
    y_refine_percent=y_refine_percent,
)

mesh.visualize()

print(mesh.Y.shape)
print(mesh.delta_x_plus.shape)
print(mesh.delta_y_plus.shape)

# mesh.X_pos = positions[0].reshape(22,22)
# mesh.Y_pos = positions[1].reshape(21, 41)

# list_of_points = np.c_[mesh.X.ravel(), mesh.Y.ravel()].shape
# index_positions = np.c_[mesh.X.ravel(), mesh.Y.ravel()].shape
# %% checke interior points delta_x + & delta_x minus values :

postt_2Dmap_interior(
    mesh,
    mesh.delta_x_minus[:, 1:-1],
    'delta_x_minus \n',
    'X',
    'Y',
    'delta_x_minus',
    s=10,
)

postt_2Dmap_interior(
    mesh,
    mesh.delta_x_plus[:, 1:-1],
    'delta_x_plus \n',
    'X',
    'Y',
    'delta_x_plus',
    s=10,
)

postt_2Dmap_interior(
    mesh,
    mesh.delta_y_minus[1:-1, :],
    'delta_y_minus \n',
    'y',
    'Y',
    'delta_y_minus',
    s=10,
)

postt_2Dmap_interior(
    mesh,
    mesh.delta_y_plus[1:-1, :],
    'delta_y_plus \n',
    'y',
    'Y',
    'delta_y_plus',
    s=10,
)
# %%
delta_y_minus = np.abs(mesh.Y[:, 1:] - mesh.Y[:, :-1])
# delta_y_minus = np.abs(mesh.Y[:, ::] - mesh.Y[::, ::])
delta_y_plus = delta_y_minus
# retrieving the values at the boundaries :
# --> creating the shift
#   retrieve last value of minus distance array :
#   retrieve last value of plus distance array :
delta_y_minus = delta_y_minus[:, :-1]
delta_y_plus = delta_y_plus[:, 1:]

postt_2Dmap_interior(
    mesh,
    delta_y_minus[1:-1, :],
    'delta_y_minus \n',
    'y',
    'Y',
    'delta_y_minus',
    s=10,
)

postt_2Dmap_interior(
    mesh,
    delta_y_plus[1:-1, :],
    'delta_y_plus \n',
    'y',
    'Y',
    'delta_y_plus',
    s=10,
)
# %% intial solution :
T0 = 20.0
# y0 = np.random.randint(2.0, 40.0, size=(nx, ny)).astype(float)
wx = 2.0 * np.pi / lx
wy = 2.0 * np.pi / ly
# y0 = T0 * (
#     1.0 + np.array([
#         [
#             np.sin(wx * xi) * np.sin(wy * yi)
#             for xi in mesh.X
#         ] for yi in mesh.Y
#     ])
# )
y0 = T0 * (1.0 + np.sin(wx * mesh.X) * np.sin(wx * mesh.Y))

spoints = 23
postt_2Dmap(
    mesh, y0, 'T init \n', 'X', 'Y', 'Initial Temperature' + r'$(\degree)$', s=spoints,
)
# %% gradient of y0 :
grad_y0 = gradient_mesh(mesh, y0)
postt_2Dmap(
    mesh,
    grad_y0[0],
    'x-Gradient of T(t=0) \n',
    'X',
    'Y',
    'Initial Temperature x-gradient',
    s=spoints,
)
postt_2Dmap(
    mesh,
    grad_y0[1],
    'y-Gradient of T(t=0) \n',
    'X',
    'Y',
    'Initial Temperature y-gradient',
    s=spoints,
)

postt_2Dmap(
    mesh,
    divergence_mesh(mesh, y0),
    'Divergence of T(t=0) \n',
    'X',
    'Y',
    'Initial Temperature divergence',
    s=spoints,
)

postt_2Dmap(
    mesh,
    scalar_laplacian_mesh(mesh, y0),
    'Scalar Laplacian of T(t=0) \n',
    'X',
    'Y',
    'Initial Temperature scalar laplacian',
    s=spoints,
)

# %% test gradient_mesh fucntion :
# hs = mesh.delta_x_minus
# hd = mesh.delta_x_plus
# grad_x = (hs**2)*y0[:,2:] + (hd**2 - hs**2)*y0[:,1:-1] - hd**2 * y0[:,:-2]  / hs*hd*(hd + hs)
# print(f'gradx shape : {grad_x.shape}')

# hs = mesh.delta_y_minus
# hd = mesh.delta_y_plus
# grad_y = (hs**2)*y0[2:,:] + (hd**2 - hs**2)*y0[1:-1,:] - hd**2 * y0[:-2,:]  / hs*hd*(hd + hs)
# print(f'grady shape : {grad_y.shape}')

# ### Boundary nodes : for i = 0, i = -1
# # i = 0  : forward difference
# # i = -1 : backward difference
# grad_x_0 = (y0[:,1] - y0[:,0]) / mesh.delta_x_minus[:,0]
# grad_x_n = (y0[:,-1] - y0[:,-2]) / mesh.delta_x_plus[:,-1]
# print(f'len border x : {len(grad_x_n)}')
# # stacking left :
# grad_x = np.hstack((grad_x_0.reshape(-1,1),grad_x))
# # stacking right :
# grad_x = np.hstack((grad_x,grad_x_n.reshape(-1,1)))
# #
# grad_y_0 = (y0[1,:] - y0[0,:]) / mesh.delta_y_minus[0,:]
# grad_y_n = (y0[-1,:] - y0[-2,:]) / mesh.delta_y_plus[-1,:]
# # stacking under :
# grad_y = np.vstack(grad_y,grad_y_0)
# # stacking above :
# grad_y = np.vstack(grad_y_n,grad_y)
# #
# print(f'len border y : {len(grad_y_n)}')
# #
# %%
M = np.random.randint(10, size=(5, 3))
print(M)
right_column = np.random.randint(10, size=(5))
print(f'right col : {right_column}')
print(f'right stacking : {np.hstack((M, right_column.reshape(-1,1)))}')

M = np.random.randint(10, size=(5, 3))
print(M)
left_column = np.random.randint(10, size=(5))
print(f'left col : {left_column}')
print(f'left stacking : {np.hstack((left_column.reshape(-1,1),M))}')

# M = np.random.randint(10, size=(5, 3))
# print(M)
# upper_line = np.random.randint(10, size=(3))
# print(f"upper line : {upper_line}")
# print(f"upper stacking : {np.vstack((upper_line,M))}")
# lower_line = np.random.randint(10, size=(3))
# print(f"lower line : {lower_line}")
# print(f"lower stacking : {np.vstack((M,lower_line))}")
# %% Limit conditions :
kw_dirichlet = {
    'neumann': False,
    'dirichlet': True,
    'interpolation_coeff': 0.2,
    'left_boundary': 25.0,
    'bottom_boundary': 30.0,
    'right_boundary': 35.0,
    'top_boundary': 40.0,
}
kw_neumann = {
    'neumann': True,
    'dirichlet': False,
    'interpolation_coeff': 0.2,
    'right_boundary': -1.0,
    'left_boundary': 1.0,
    'bottom_boundary': 0.0,
    'top_boundary': 0.0,
}
dirichlet = True
# neumann =  True

if dirichlet:
    neumann = False
    kw = kw_dirichlet
    print(f"T_right = {kw_dirichlet['right_boundary']  }")
    print(f"T_left = {kw_dirichlet['left_boundary']   }")
    print(f"T_bottom = {kw_dirichlet['bottom_boundary'] }")
    print(f"T_top = {kw_dirichlet['top_boundary']    }")

if not dirichlet:
    neumann = True
    kw = kw_neumann
    print(f"Flow_right = {kw_neumann['right_boundary']  }")
    print(f"Flow_left = {kw_neumann['left_boundary']   }")
    print(f"Flow_bottom = {kw_neumann['bottom_boundary'] }")
    print(f"Flow_top = {kw_neumann['top_boundary']    }")

# boundaries = boundary_conditions_2D(mesh.X[0, :], mesh.Y[:, 0], y0, **kw)

boundaries = boundary_conditions_2D_mesh(mesh, y0, **kw)

# test with concatenated arrays : left-bottom-right-top :
arr_test = np.concatenate(
    (
        np.flip(boundaries[0]),
        boundaries[1],
        boundaries[2],
        np.flip(boundaries[3]),
    ),
)
xleft = np.linspace(0.0, mesh.ly, len(mesh.Y[0, :]))
xbottom = np.linspace(mesh.ly, mesh.ly + mesh.lx, len(mesh.X[:, 0]))
xright = np.linspace(
    mesh.ly + mesh.lx, 2.0 * mesh.ly +
    mesh.lx, len(mesh.Y[-1, :]),
)
xtop = np.linspace(
    2.0 * mesh.ly + mesh.lx, 2.0 * mesh.ly + 2.0 * mesh.lx, len(mesh.X[:, -1]),
)

contour_test = np.concatenate((xleft, xbottom, xright, xtop))
# contour_test = np.concatenate(mesh.Y[:, 0], mesh.X[0, :], mesh.Y[:, -1], mesh.X[-1, :])


plt.scatter(mesh.Y[0, :], boundaries[0], s=4)
plt.title('left interpolated array')
plt.show()
plt.close('all')
print(f'len(boundaries[1]) = {len(boundaries[1])}')
# %%
plt.scatter(mesh.Y[-1, :], boundaries[2], s=4)
plt.title('right interpolated array')
plt.show()
plt.close('all')
print(f'len(boundaries[3]) = {len(boundaries[3])}')

plt.scatter(mesh.X[:, 0], boundaries[1], s=4)
plt.title('bottom interpolated array')
plt.show()
plt.close('all')

plt.scatter(mesh.X[:, -1], boundaries[3], s=4)
plt.title('top interpolated array')
plt.show()
plt.close('all')

print(f'len(contour) = {len(contour_test)}')
print(f'len(interpolated array) = {len(arr_test)}')
plt.scatter(contour_test, arr_test, s=4)
plt.title('Concatenated interpolated array')
plt.show()
plt.close('all')


# %% diffusion coefficient map :
D0 = 1.0e-2
# uniform
#  = D0 * np.ones((nx, ny))
D = D0 * np.ones_like(mesh.X)
# D = np.random.uniform(D0, 10.*D0, size=(nx, ny))
# D = D0 * np.array([[1. + (xi / lx) for xi in x] for yi in y])
# D = D0 * (np.array([[1.0 + (xi / lx) * (yi / ly) for xi in x] for yi in y]))
# D = (D0/T0) * y0

print(type(D))
print(type(y0))
print(np.shape(D))
print(np.shape(y0))
print(type(D * y0))
print(np.shape(D * y0))

print(f'diffusion max value = {np.max(D)}')

# %% CFL criteria :
# np.max(D) * dt / (np.min(dx,dy))**2 <= 1./2.
val_D_max = D[np.unravel_index(np.argmax(D, axis=None), D.shape)]
min_spatial_discr = np.min([mesh.dx_fine, mesh.dy_fine])
dt = 0.1 * (0.5 * (min_spatial_discr**2) / val_D_max)
print(f'min spatial discr = {min_spatial_discr}')

n_t = int(t_end / dt) + 1

t = np.linspace(0.0, t_end, n_t)

print(f'len(t) = {len(t)}')
# %% Plot D : diffusion coeff
postt_2Dmap(
    mesh,
    D,
    'Diffusion coefficient value \n',
    'X',
    'Y',
    'Diffusion coefficient',
    s=spoints,
)
# %% Second member : initial values
Fini = F_diffusion(mesh, y0, D)

postt_2Dmap(mesh, Fini, 'F ini \n', 'X', 'Y', 'Second member ini', s=spoints)

# applying boundary conditions

# BCs = boundary_conditions_2D(mesh.X[0, :], mesh.Y[:, 0], y0, **kw)
BCs = boundary_conditions_2D_mesh(mesh, y0, **kw)
print(len(BCs[0]))
print(len(BCs[1]))
print(len(BCs[2]))
print(len(BCs[3]))
print(np.shape(y0))

y0 = apply_bounds(mesh, y0, **kw)
# %%
Fini = F_diffusion(mesh, y0, D)

postt_2Dmap(
    mesh,
    Fini,
    'F ini with coundary coditions \n',
    'X',
    'Y',
    'Second member ini',
    s=spoints,
)
# %%
# Fini = diffusion(
#     mesh.X[0, :], mesh.Y[:, 0], mesh, y0, D, [dx_fine, dy_fine], edgeorder=1, **kw
# )

# postt_2Dmap(
#     mesh,
#     Fini,
#     "F ini with coundary coditions \n",
#     "X",
#     "Y",
#     "Second member ini",
#     s=spoints,
# )

# %% solver :
n_steps = len(t)
dt = t[1] - t[0]
print(f'dt = {dt}')

# %% initialization :
y_shape_with_time = y0.shape + (n_steps,)
sol = np.zeros(y_shape_with_time, dtype=y0.dtype)
sol[..., 0] = y0

print(type(sol[..., 0]))
print(np.shape(sol[..., 0]))
euler_explicit = True
crank_nicolson = False

if euler_explicit:
    for i, ti in enumerate(t[:-1]):
        # Boundary conditions :
        sol[..., i] = apply_bounds(mesh, sol[..., i], **kw)
        # Right had side i.e. 2nd member :
        F = F_diffusion(mesh, sol[..., i], D)
        # F = diffusion(
        #     mesh.X[0, :],
        #     mesh.Y[:, 0],
        #     mesh,
        #     sol[..., i],
        #     D,
        #     [dx_fine, dy_fine],
        #     edge_order=1,
        #     **kw,
        # )
        # F[:,0] = -F[:,1]
        # F[0,:] = -F[1,:]
        # F[-1,:] = -F[-2,:]
        # F[:,-1] = -F[:,-2]
        print(f'type(F) = {F}')
        print(f'shape(F) = {np.shape(F)}')
        print(f'type(yi) = {type(sol[...,i])}')
        print(f'shape(yi) = {np.shape(sol[...,i])}')
        sol[..., i + 1] = sol[..., i] + dt * F

if crank_nicolson:
    y0 = apply_bounds(y0, discr, **kw)
    y_0 = y0
    yini = y0
    niter_max = 20
    tol = 5.0e-2 * T0
    print(f'Crank-Nicolson y : {np.shape(y0)}')
    # Fist estimation of the y1 : euler explicit
    f = diffusion(yini, D, discr, edge_order, **kw)
    print(f'Crank-Nicolson f : {np.shape(f)}')
    y1 = y_0 + dt * f

    for i, ti in enumerate(t[:-1]):
        # y1 = apply_bounds(y1, discr, **kw)
        for iter in range(niter_max):
            # new y1 right hand side :
            y1 = apply_bounds(y1, discr, **kw)
            f1 = scalar_laplacian(D * y1, discr, edge_order)
            # Residual's Jacobian :
            jacobian = np.eye(nx, M=ny, dtype=yini.dtype) - (dt / 2.0) * f1
            # print(f'shape(jacobian) {np.shape(jacobian)}')
            # Flattening arrays :
            y_0 = y_0.flatten()
            f = f.flatten()
            y1 = y1.flatten()
            f1 = f1.flatten()
            # print(f'y0 = {y0}')
            # print(f'y1 = {y1}')
            # print(f'f = {f}')
            # print(f'f1 = {f1}')
            # Matching residu :
            residual = y1 - y_0 - (dt / 2.0) * (f + f1)
            # print(f'residual shape = {residual.shape}')
            # solving :
            # delta_y = spsolve(jacobian, -residual)
            delta_y = np.matmul(
                np.linalg.inv(jacobian),
                -residual.reshape(yini.shape),
            )
            delta_y = delta_y.flatten()
            # Incrementing y1 :
            y_0 = y1
            f = f1
            y1 += delta_y
            # Reshape :
            y_0 = y_0.reshape(yini.shape)
            f = f.reshape(yini.shape)
            y1 = y1.reshape(yini.shape)
            # Check for convergence :
            residual_norm = np.linalg.norm(delta_y)
            if residual_norm < tol:
                print('convergence')
                break
            if iter == max(range(niter_max)):
                print(f'divergence, ||res|| = {residual_norm}')

        sol[..., i + 1] = y1

# %%
postt_2Dmap(
    mesh,
    sol[..., 20],
    'solution 5^th rime step \n',
    'X',
    'Y',
    'Temperature',
    s=spoints,
)
# %%
postt_2Dmap(
    mesh,
    sol[..., -1],
    'final solution \n',
    'X',
    'Y',
    'Temperature',
    s=spoints,
)

Ffinal = F_diffusion(mesh, sol[..., -1], D)
postt_2Dmap(
    mesh,
    Ffinal,
    'final F \n',
    'X',
    'Y',
    'F',
    s=spoints,
)

sl = scalar_laplacian_mesh(mesh, sol[..., -1])
postt_2Dmap(
    mesh,
    sl,
    'final scalar laplacian \n',
    'X',
    'Y',
    'scalar laplacian',
    s=spoints,
)

# %%
