#!/usr/bin//python3
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import dotb.differential_operators as diffops
import dotb.postt as postt
import dotb.second_member as second
from dotb.refined_mesh_class import Mesh
# import dotb.boundary_conditions as BC
# from dotb.second_member import Diffusion

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
        # time array :
        print(f"ballistic, t_end = {kw['t_end']}")
        print(f"type = {type(kw['t_end'])}")
        print(f"ballistic, dt = {kw['dt']}")
        print(f"type = {type(kw['dt'])}")
        n_t = int(kw['t_end'] / kw['dt'] + 1.)
        t = np.linspace(0.0, kw['t_end'], n_t)
        # Reading args :
        x0 = kw['x_0']
        y0 = kw['y_0']
        dxdt0 = kw['v_0'] * np.cos(np.pi * kw['theta_0'] / 180.0)
        dydt0 = kw['v_0'] * np.cos(np.pi * kw['theta_0'] / 180.0)
        y_0 = np.array([x0, y0, dxdt0, dydt0])

    if kw['case'] == 'rabbit':
        # time array :
        t = np.linspace(0, kw['t_end'], kw['n_t'])
        # Reading args :
        y_0 = np.array([kw['N_0']])

    if kw['case'] == 'diffusion_2D':
        # time array :
        n_t = int(kw['t_end'] / kw['dt']) + 1
        t = np.linspace(0.0, kw['t_end'], n_t)
        # nsave : laisse vide dans le input.yaml
        kw['n_save'] = int(n_t/25) + 1

        # For now the Crank Nicolson solver only runs with a square domain :
        if kw['solver'] == 'crank_nicolson':
            if kw['l_x'] != kw['l_y']:
                raise ValueError(
                    'Crank Nicolson method only runs with square domain : please consider using equal x and y dimensions :  lx = ly in your input file.',
                )
        # Mesh creation :
        mesh = Mesh(
            lx=kw['l_x'],
            ly=kw['l_y'],
            dx_fine=kw['dx_fine'],
            dy_fine=kw['dy_fine'],
            dx_coarse=kw['dx_coarse'],
            dy_coarse=kw['dy_coarse'],
            x_refine_percent=kw['x_refine_percent'],
            y_refine_percent=kw['y_refine_percent'],
        )
        # Socked in the config dict :
        kw['mesh'] = mesh
        # y0 = np.random.randint(2.0, 40.0, size=(kw['n_x'], kw['n_y'])).astype(float)
        wx = kw['p_x'] * 2.0 * np.pi / kw['l_x']
        wy = kw['p_y'] * 2.0 * np.pi / kw['l_y']
        y_0 = kw['T_0'] * (1.0 + np.sin(wx * mesh.X) * np.sin(wy * mesh.Y))
        # Computation of the diffusion coefficient field :
        if kw['D_uni']:
            kw['D'] = kw['D_0'] * np.ones_like(mesh.X)
        if kw['D_lin']:
            kw['D'] = kw['D_0'] * (1. + (mesh.X/mesh.lx) * (mesh.Y/mesh.ly))

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

        k1 = {
            k: kw.get(k) for k in [
                'dirichlet', 'neumann', 'left_boundary',
                'bottom_boundary', 'right_boundary', 'top_boundary', 'interpolation_coeff',
            ]
        }
        print(k1)
        # Applying the boundary conditions to y_0 :
        # y_0 = BC.apply_boundaries(mesh, y_0, **kw)

        ###### Post treatment of input data #####
        input_dir = kw['save_dir'] + 'input/'
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        plt.figure(figsize=(10, 10))
        plt.scatter(mesh.X.ravel(), mesh.Y.ravel(), s=4)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(
            f'Refined Mesh Lattice (X-refine: {mesh.x_refine_percent}%, Y-refine: {mesh.y_refine_percent}%)',
        )
        plt.savefig(input_dir + 'mesh_2D_diffusion.png')

        # plotting y_0 :
        save_name = 'T_field_ini'
        title = 'Temperature at t = 0 s \n'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = 'Temperature ' + r'$(\degree C)$'
        postt.postt_2Dmap(
            mesh,
            y_0,
            title,
            labelx,
            labely,
            labelbar,
            save_dir=input_dir,
            save_name=save_name,
        )

        # plotting y_0 gradient :
        grads_ini = diffops.gradient_mesh(mesh, y_0)
        # plotting y_0 x-gradient :
        save_name = 'dTdx_ini'
        title = r'$\frac{\partial T}{\partial x}(t=0)$'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = r'$\frac{\partial T}{\partial x}(t=0)$'
        postt.postt_2Dmap(
            mesh,
            grads_ini[0],
            title,
            labelx,
            labely,
            labelbar,
            save_dir=input_dir,
            save_name=save_name,
        )
        # plotting y_0 y-gradient :
        save_name = 'dTdy_ini'
        title = r'$\frac{\partial T}{\partial y}(t=0)$'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = r'$\frac{\partial T}{\partial y}(t=0)$'
        postt.postt_2Dmap(
            mesh,
            grads_ini[1],
            title,
            labelx,
            labely,
            labelbar,
            save_dir=input_dir,
            save_name=save_name,
        )
        # plotting y_0 divergence :
        div_y_0 = diffops.divergence_mesh(mesh, y_0)
        # plotting y_0 divergence :
        save_name = 'div_T_ini'
        title = r'$div(T)(t=0)$'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = r'$\frac{\partial T}{\partial x} + \frac{\partial T}{\partial y} \quad (t=0)$'
        postt.postt_2Dmap(
            mesh,
            div_y_0,
            title,
            labelx,
            labely,
            labelbar,
            save_dir=input_dir,
            save_name=save_name,
        )

        # plotting second member F initial value :
        sec_0 = second.F_diffusion(mesh, y_0, kw['D'])
        save_name = 'F_ini'
        title = 'F at t = 0 s \n'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = 'F'
        postt.postt_2Dmap(
            mesh,
            sec_0,
            title,
            labelx,
            labely,
            labelbar,
            save_dir=input_dir,
            save_name=save_name,
        )

        # plotting diffusion D(x,y) :
        save_name = 'diffusion_coeff'
        title = 'Diffusion coefficient map \n'
        labelx = 'X (m)'
        labely = 'Y (m)'
        labelbar = 'D'
        postt.postt_2Dmap(
            mesh,
            kw['D'],
            title,
            labelx,
            labely,
            labelbar,
            save_dir=input_dir,
            save_name=save_name,
        )

        # plotting boundary conditions :
        print(f'input_dir {input_dir}')
        print(f'save_name {save_name}')

        savename = 'BCs'
        postt.plot_boundary_conditions(mesh, input_dir, savename, **k1)

        # print(f'Visualize your initial configuration in {input_dir}!')

    return y_0, t, kw
