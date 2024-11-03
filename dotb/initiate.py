#!/usr/bin//python3
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import dotb.boundary_conditions as BC
import dotb.postt as postt
import dotb.second_member as second

# For input testing plots :

# from typing import TypedDict
# from typing import Union
# from dotb.input_yaml import my_dict_ballistic
# from dotb.input_yaml import my_dict_diffusion_2D
# from dotb.input_yaml import my_dict_rabbit


def intiate_y(kw: dict) -> tuple:
    """
    Initiate the tensor field y(t=0)
    from input.yaml file keywords arguments.

    Parameters: depend on the use case
    (input.yaml keywords arguments)

    Returns:

    - ballistic : 4D-array : (x0,y0,dxdt0,dydt0')
    - rabbit : 1D-array : N0
    - diffusion (1D) : 1D-array : T(x,t=0)
    - diffusion (2D) : 2D-array : T(x,y,t=0)
    """

    if kw['case'] == 'ballistic':
        # Reading args :
        x0 = kw['x_0']
        y0 = kw['y_0']
        dxdt0 = kw['v_0'] * np.cos(np.pi * kw['theta_0'] / 180.0)
        dydt0 = kw['v_0'] * np.cos(np.pi * kw['theta_0'] / 180.0)
        y_0 = np.array([x0, y0, dxdt0, dydt0])

    if kw['case'] == 'rabbit':
        # Reading args :
        y_0 = np.array([kw['N_0']])

    if kw['case'] == 'diffusion_2D':
        # Computation of the initial temperature field :
        x = np.linspace(-kw['l_x'] / 2.0, kw['l_x'] / 2.0, kw['n_x'])
        y = np.linspace(-kw['l_x'] / 2.0, kw['l_y'] / 2.0, kw['n_y'])
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        # mesh = np.array([(xi,yj) for ])
        kw['dx'] = x[1] - x[0]
        kw['dy'] = y[1] - y[0]
        # y0 = np.random.randint(2.0, 40.0, size=(kw['n_x'], kw['n_y'])).astype(float)
        wx = kw['p_x'] * 2.0 * np.pi / kw['l_x']
        wy = kw['p_y'] * 2.0 * np.pi / kw['l_y']
        y_0 = kw['T_0'] * (
            1.0
            + np.array([
                [
                    np.sin(wx * xi) * np.sin(wy * yi)
                    for xi in x
                ] for yi in y
            ])
        )
        # Computation of the diffusion coefficient field :
        if kw['D_uni']:
            kw['D'] = kw['D_0'] * np.ones((kw['n_x'], kw['n_y']))
        if kw['D_lin']:
            kw['D'] = kw['D_0'] * np.array(
                [
                    [
                        1.0 + (xi / kw['l_x']) * (yi / kw['l_y'])
                        for xi in x
                    ] for yi in y
                ],
            )

        # Boundary conditions : Neumann or Dirichlet ?
        if kw['dirichlet']:
            kw['right_boundary'] = kw['dirichlet_right_boundary']
            kw['left_boundary'] = kw['dirichlet_left_boundary']
            kw['bottom_boundary'] = kw['dirichlet_bottom_boundary']
            kw['top_boundary'] = kw['dirichlet_top_boundary']

        if kw['neumann']:
            kw['right_boundary'] = kw['neumann_right_boundary']
            kw['left_boundary'] = kw['neumann_left_boundary']
            kw['bottom_boundary'] = kw['neumann_bottom_boundary']
            kw['top_boundary'] = kw['neumann_top_boundary']

        # Debug : postt of input data :
        print(f'intiate_y np.shape(y_0) = {np.shape(y_0)}')
        input_dir = kw['save_dir'] + 'input/'
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        bc_condtions = BC.boundary_conds(y_0, **kw)
        y_0 = bc_condtions[0]
        print(f'intiate_y after BC : np.shape(y_0) = {np.shape(y_0)}')

        # plotting y_0 :
        save_name = 'T_field_ini'
        title = 'Temperature at t = 0 s \n'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = 'Temperature ' + r'$(\degree C)$'
        postt.postt_2Dmap(
            x,
            y,
            y_0,
            title,
            save_name,
            input_dir,
            labelx,
            labely,
            labelbar,
            **kw,
        )

        # plotting y_0 gradient :
        grads_ini = second.gradient(y_0, [dx, dy], edge_order=1)
        # plotting y_0 x-gradient :
        save_name = 'dTdx_ini'
        title = r'$\frac{\partial T}{\partial x}(t=0)$'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = r'$\frac{\partial T}{\partial x}(t=0)$'
        postt.postt_2Dmap(
            x,
            y,
            grads_ini[0],
            title,
            save_name,
            input_dir,
            labelx,
            labely,
            labelbar,
            **kw,
        )
        # plotting y_0 y-gradient :
        save_name = 'dTdy_ini'
        title = r'$\frac{\partial T}{\partial y}(t=0)$'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = r'$\frac{\partial T}{\partial y}(t=0)$'
        postt.postt_2Dmap(
            x,
            y,
            grads_ini[1],
            title,
            save_name,
            input_dir,
            labelx,
            labely,
            labelbar,
            **kw,
        )
        # plotting y_0 divergence :
        div_y_0 = second.divergence(y_0, [dx, dy], edge_order=1)
        # plotting y_0 divergence :
        save_name = 'div_T_ini'
        title = r'$div(T)(t=0)$'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = r'$\frac{\partial T}{\partial x} + \frac{\partial T}{\partial y} \quad (t=0)$'
        postt.postt_2Dmap(
            x,
            y,
            div_y_0,
            title,
            save_name,
            input_dir,
            labelx,
            labely,
            labelbar,
            **kw,
        )

        # plotting second member F initial value :
        sec_0 = second.F(y_0, **kw)
        print(f'type sec_0 : {type(sec_0)}')
        save_name = 'F_ini'
        title = 'F at t = 0 s \n'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = 'F'
        postt.postt_2Dmap(
            x,
            y,
            sec_0,
            title,
            save_name,
            input_dir,
            labelx,
            labely,
            labelbar,
            **kw,
        )

        # plotting diffusion D(x,y) :
        save_name = 'diffusion_coeff'
        title = 'Diffusion coefficient map \n'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = 'D'
        postt.postt_2Dmap(
            x,
            y,
            kw['D'],
            title,
            save_name,
            input_dir,
            labelx,
            labely,
            labelbar,
            **kw,
        )

        # plotting boundary conditions :
        contour_test = bc_condtions[1]
        bc_test = bc_condtions[2]
        plt.scatter(contour_test, bc_test, s=4)
        plt.title('Concatenated interpolated array')
        plt.savefig(input_dir + 'BC' + '.png')
        plt.close('all')

    return y_0, kw
