#!/usr/bin/python3
# %%
from __future__ import annotations

import os

import numpy as np

import dotb.boundary_conditions as BC
import dotb.initiate as init
import dotb.postt as postt
import dotb.second_member as second
from dotb.build_model import create_solver
from dotb.refined_mesh_class import Mesh

# %%
rep_input = './results/demo_diffusion/input/'
rep_output = './results/demo_diffusion/'

for i, repi in enumerate([rep_input, rep_output]):
    if not os.path.exists(repi):
        os.makedirs(repi)

# %% Set up the problem
# time :
t_end = 3.0
dt = 0.001
# mesh :
lx = 1.0
ly = 1.2
dx_fine = 0.005
dy_fine = 0.005
dx_coarse = 4.0 * dx_fine
dy_coarse = 4.0 * dy_fine
x_refine_percent = 2.5
y_refine_percent = 2.5

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

# %% Limit conditions :
dirichlet = True
neumann = False
config = {
    # save directory :
    'save_dir': rep_output,
    # Time :
    't_end': t_end,
    'dt': dt,
    'n_iter_max': 60,
    'tol': 0.2,
    # mesh definition
    'l_x': lx,
    'l_y': ly,
    'dx_fine': dx_fine,
    'dy_fine': dy_fine,
    'dx_coarse': dx_coarse,
    'dy_coarse': dy_coarse,
    'x_refine_percent': x_refine_percent,
    'y_refine_percent': y_refine_percent,
    # boundary conditons definition
    'neumann': neumann,
    'dirichlet': dirichlet,
    'interpolation_coeff': 0.2,
    'dirichlet_left_boundary': 10.0,
    'dirichlet_bottom_boundary': 30.0,
    'dirichlet_right_boundary': 20.0,
    'dirichlet_top_boundary': 5.0,
    'neumann_left_boundary': 1.0,
    'neumann_bottom_boundary': 0.0,
    'neumann_right_boundary': -1.0,
    'neumann_top_boundary': 0.0,
    'left_boundary': None,
    'bottom_boundary': None,
    'right_boundary': None,
    'top_boundary': None,
}
if config['dirichlet']:
    config['right_boundary'] = config['dirichlet_right_boundary']
    config['left_boundary'] = config['dirichlet_left_boundary']
    config['bottom_boundary'] = config['dirichlet_bottom_boundary']
    config['top_boundary'] = config['dirichlet_top_boundary']

if config['neumann']:
    config['right_boundary'] = config['neumann_right_boundary']
    config['left_boundary'] = config['neumann_left_boundary']
    config['bottom_boundary'] = config['neumann_bottom_boundary']
    config['top_boundary'] = config['neumann_top_boundary']

# Plotting BCs :
postt.plot_boundary_conditions(mesh, rep_input, 'BCs', **config)

# %% intial solution :
T0 = 20.0
p_x = 1.0
p_y = 1.0
wx = p_x * 2.0 * np.pi / lx
wy = p_y * 2.0 * np.pi / ly
y0 = T0 * (1.0 + np.sin(wx * mesh.X) * np.sin(wy * mesh.Y))

spoints = 23
postt.postt_2Dmap(
    mesh,
    y0,
    'T init \n',
    'X',
    'Y',
    'Initial Temperature' + r'$(\degree)$',
    save_dir=rep_input,
    save_name='Tini',
    s=45,
)
# Adding to input config dictionnary :
config['T_0'] = T0
config['p_x'] = p_x
config['p_y'] = p_y

# %% diffusion coefficient map :
D0 = 1.0e-2
# uniform
D = D0 * np.ones_like(mesh.X)
# D = D0 * (1.0 + (mesh.X / mesh.lx) * (mesh.Y / mesh.ly))
# D = (D0/T0) * y0

postt.postt_2Dmap(
    mesh,
    D,
    'Diffusion coefficient \n',
    'X',
    'Y',
    'Diffusion coefficient',
    save_dir=rep_input,
    save_name='D',
    s=45,
)

# Adding to input config dictionnary :
config['D_lin'] = False
config['D_uni'] = False
config['D'] = D

# %% Second member : initial values
Fini = second.F_diffusion(mesh, y0, D)


postt.postt_2Dmap(
    mesh, Fini, 'F ini \n', 'X', 'Y',
    'Second member ini', save_dir=rep_input, save_name='Fini', s=45,
)

# %% Second member : with boundary conditions
config2 = {k: v for k, v in config.items() if k != 'mesh'}
y0 = BC.apply_boundaries(mesh, y0, **config2)

Fini_BC = second.F_diffusion(mesh, y0, D)

postt.postt_2Dmap(
    mesh, Fini_BC, 'F ini with boundary conditions \n', 'X', 'Y',
    'Second member ini with boundary conditions', save_dir=rep_input, save_name='Fini_BCs', s=35,
)

# %% Simulation :
# config['solver'] = 'crank_nicolson'
config['solver'] = 'euler_explicit'
# config['solver'] = 'adams_bashforth'
# Assembling a solver to a case study :
# config['case'] = 'rabbit'
# config['case'] = 'ballistic'
config['case'] = 'diffusion_2D'

# %% Initializing :
y, t, config = init.intiate_y(config)

# %% Creating simulation :
simu = create_solver(y, t, config)

# %% Solving :
sol = simu.solve(**config)
print(f'Number of saved time steps = {np.shape(sol)[0]}')
print(f'sol.shape = {sol.shape}')

# %% Post treatent :
t_save = np.linspace(0., config['t_end'], len(sol))
postt.plot_postt(t_save, sol, **config)
# %%
