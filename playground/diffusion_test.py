#!/usr/bin/python3
# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
# from scipy.sparse.linalg import spsolve
# from scipy.sparse import diags, identity


# %%
def postt_2Dmap(x, y, data, title, labelx, labely, labelbar):
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter plot with colors determined by data values
    scatter = ax.scatter(
        X.ravel(),
        Y.ravel(),
        c=data.ravel(),
        cmap='inferno',
        s=200,
    )

    # Set limits and aspect ratio
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect('equal')

    ax.set_xlabel(f'{labelx}')
    ax.set_ylabel(f'{labely}')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel(f'{labelbar}')

    plt.title(f'{title}')
    plt.show()
    plt.close('all')


def gradient(data, discr, edge_order):
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


def divergence(data, discr, edge_order):
    grads = gradient(data, discr, edge_order)
    return np.sum(grads, axis=0)


def gradient_square(data, discr, edge_order):
    grads = gradient(data, discr, edge_order)
    grad2s = []
    [
        grad2s.append(np.gradient(gradi, discr[i], axis=i, edge_order=1))
        for i, gradi in enumerate(grads)
    ]
    return np.array(grad2s)


def scalar_laplacian(data, discr, edge_order):
    grad2s = gradient_square(data, discr, edge_order)
    return np.sum(grad2s, axis=0)


def scalar_laplacian_neumann_boundaries(data, discr, edge_order, D, **kw):
    grads = gradient(data, discr, edge_order)
    # 1D :
    if len(np.shape(data)) == 1:
        grads[0] = kw['left_boundary']
        grads[-1] = kw['right_boundary']
    # 2D :
    if len(np.shape(data)) == 2:
        # setting x-gradient values :
        grads[0][0, :] = kw['left_boundary']
        grads[0][-1, :] = kw['right_boundary']
        # setting y-gradient values :
        grads[1][:, 0] = kw['bottom_boundary']
        grads[1][:, -1] = kw['top_boundary']
    grad2s = gradient(grads, discr, edge_order)
    return np.sum(grad2s, axis=0)


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


def diffusion(x, y, data, D, discr, edge_order=1, **kw) -> np.ndarray:
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
        boundaries = boundary_conditions_2D(x, y, data, **kw)
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

    grads = gradient(data, discr, edge_order)

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


def apply_bounds(data, discr, **kw) -> np.ndarray:
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
        boundaries = boundary_conditions_2D(x, y, data, **kw)
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

    return data


# %% Set up the problem
# t = np.linspace(0., 1.0, 1000)
t_end = 1.
# Physics :
# rabibts : 5 litters annualy of 10 kits each :
# discretization : dx, dy in km
# nx = 100
# ny = 100
nx = 100
ny = 100
lx = 1.0
ly = 1.0
# x = np.linspace(0.0, lx, nx)
# y = np.linspace(0.0, ly, ny)
x = np.linspace(-lx / 2.0, lx / 2.0, nx)
y = np.linspace(-ly / 2.0, ly / 2.0, ny)
# mesh = np.array([(xi,yj) for ])
dx = x[1] - x[0]
dy = y[1] - y[0]
discr = [dx, dy]

n_t = int((10.*t_end)/np.min(discr))
t = np.linspace(0., t_end, n_t)
# intial solution :
T0 = 20.0
# y0 = np.random.randint(2.0, 40.0, size=(nx, ny)).astype(float)
wx = 2.0 * np.pi / lx
wy = 2.0 * np.pi / ly
y0 = T0 * (
    1.0 + np.array([
        [
            np.sin(wx * xi) * np.sin(wy * yi)
            for xi in x
        ] for yi in y
    ])
)

postt_2Dmap(
    x,
    y,
    y0,
    'T init \n',
    'X',
    'Y',
    'Initial Temperature' + r'$(\degree)$',
)
# %% Limit conditions :
kw_dirichlet = {
    'neumann': False,
    'dirichlet': True,
    'interpolation_coeff': 0.,
    'right_boundary': 1.5 * T0,
    'left_boundary': 0.5 * T0,
    'bottom_boundary': 0.3 * T0,
    'top_boundary': 1.7 * T0,
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

boundaries = boundary_conditions_2D(x, y, y0, **kw)

# test with concatenated arrays : left-bottom-right-top :
arr_test = np.concatenate(
    (
        np.flip(boundaries[0]),
        boundaries[1],
        boundaries[2],
        np.flip(boundaries[3]),
    ),
)
contour_test = np.linspace(0.0, 2.0 * lx + 2.0 * ly, len(arr_test))

plt.scatter(x, boundaries[1], s=4)
plt.title('Bottom interpolated array')
plt.show()
plt.close('all')
print(f'len(x) = {len(x)}')
print(f'len(boundaries[1]) = {len(boundaries[1])}')

plt.scatter(x, boundaries[3], s=4)
plt.title('top interpolated array')
plt.show()
plt.close('all')
print(f'len(x) = {len(x)}')
print(f'len(boundaries[3]) = {len(boundaries[3])}')

plt.scatter(y, boundaries[0], s=4)
plt.title('left interpolated array')
plt.show()
plt.close('all')
print(f'len(y) = {len(y)}')
print(f'len(boundaries[0]) = {len(boundaries[0])}')

plt.scatter(y, boundaries[2], s=4)
plt.title('right interpolated array')
plt.show()
plt.close('all')
print(f'len(y) = {len(y)}')
print(f'len(boundaries[2]) = {len(boundaries[2])}')

plt.scatter(contour_test, arr_test, s=4)
plt.title('Concatenated interpolated array')
plt.show()
plt.close('all')
print(f'len(contour) = {len(contour_test)}')
print(f'len(interpolated array) = {len(arr_test)}')

# x_interp * kw['left']
# %% Gradient :
edge_order = 1

# grads = []
# [ grads.append(np.gradient(y0, discr[i], axis=i, edge_order=2)) for i,si in enumerate(np.shape(y0)) ]
# grads = np.array(grads)

grads = gradient(y0, discr, edge_order)

gradx = grads[0]
grady = grads[1]

# div = np.sum(grads, axis=0)
div = divergence(y0, discr, edge_order)

postt_2Dmap(x, y, gradx, 'x=gradient \n', 'X', 'Y', 'x-gradient')
postt_2Dmap(x, y, grady, 'y=gradient \n', 'X', 'Y', 'y-gradient')
postt_2Dmap(x, y, div, 'Divergence \n', 'X', 'Y', 'Divergence')

# %% Gradient ^ 2 :
# grad2s = []
# [ grad2s.append(np.gradient(gradi, discr[i], axis=i, edge_order=2)) for i,gradi in enumerate(grads) ]

grad2s = gradient_square(y0, discr, edge_order)

postt_2Dmap(x, y, grad2s[0], 'x-gradient^2 \n', 'X', 'Y', 'x-gradient^2')
postt_2Dmap(x, y, grad2s[1], 'y=gradient^2 \n', 'X', 'Y', 'y-gradient^2')
# %% Scalar laplacian :
# SL_y0 = np.sum(grad2s, axis=0)

SL_y0 = scalar_laplacian(y0, discr, edge_order)

postt_2Dmap(x, y, SL_y0, 'Scalar Laplacian \n', 'X', 'Y', 'Scalar Laplacian')

# %% diffusion coefficient map :
D0 = 0.01
D = D0 * np.ones((nx, ny))
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

#%% CFL criteria :
# np.max(D) * dt / (np.min(dx,dy))**2 <= 1./2.
val_D_max = D[np.unravel_index(np.argmax(D, axis=None), D.shape)]
dt =  0.1 * 0.5 * ((np.min([dx,dy]))**2) / val_D_max
n_t = int(t_end / dt) + 1
t = np.linspace(0.,t_end,n_t)
# %% Plot D : diffusion coeff
postt_2Dmap(
    x,
    y,
    D,
    'Diffusion coefficient value \n',
    'X',
    'Y',
    'Diffusion coefficient',
)
# %%
F = scalar_laplacian(D * y0, discr, edge_order)
print(type(F))
print(np.shape(F))

# %%
grads = gradient(D, discr, edge_order)
gradx = grads[0]
grady = grads[1]

div = divergence(D, discr, edge_order)

postt_2Dmap(x, y, gradx, 'x=gradient \n', 'X', 'Y', 'x-gradient')
postt_2Dmap(x, y, grady, 'y=gradient \n', 'X', 'Y', 'y-gradient')
postt_2Dmap(x, y, div, 'Divergence \n', 'X', 'Y', 'Divergence')

# %% Second member : initial values
Fini = diffusion(x, y, y0, D0, discr, edge_order=1, **kw)
postt_2Dmap(x, y, Fini, 'F ini \n', 'X', 'Y', 'Second member ini')

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
        F = diffusion(x, y, sol[..., i], D, discr, edge_order, **kw)
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
            # f1 = scalar_laplacian(D * y1, discr, edge_order)
            f1 = diffusion(y1, D, discr, edge_order, **kw)
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
                np.linalg.inv(jacobian), -
                residual.reshape(yini.shape),
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
    x,
    y,
    sol[..., 20],
    'solution 5^th rime step \n',
    'X',
    'Y',
    'Temperature',
)
# %%
postt_2Dmap(x, y, sol[..., -1], 'final solution \n', 'X', 'Y', 'Temperature')

# %%
