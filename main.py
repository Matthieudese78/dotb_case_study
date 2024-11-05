#!/usr/bin/python3
# %% Loading packages :
# Standard packages :
from __future__ import annotations

import os.path

import numpy as np

import dotb.initiate as init
import dotb.input_yaml as input
import dotb.postt as postt
import dotb.solver as solver
# import dotb.solver as solver

# Local packages :


# %% main function definition :
def main():
    print('This is the main program')
    # Default input :
    # main directory : contains the default .yaml input files
    maindir = os.path.dirname(__file__)
    # default .yaml file name :
    # default_input = 'ballistic_default_input.yaml'
    # default_input = 'rabbit_default_input.yaml'
    default_input = 'diffusion_2D_default_input.yaml'
    # Reading argument :
    # arg parsing for inputs :
    config_file_path = input.get_config_file(maindir, default_input)
    print(f'Using config file: {config_file_path}')
    # input.yaml --> config dict
    config = input.load_config(config_file_path)
    # Checking the types of the input variables :
    # (is the config dict conform to one of the tolerated TypedDict classes?)
    input.check_type(config)

    print(f"Solving the {config['case']} case")
    print(f"using the {config['solver']} solver")

    # time vector :
    t = np.linspace(0, config['t_end'], config['n_t'])

    # y-initialization :
    #   n-dimensional tensor field y(t=0) from config :
    #   some values in config dict are computed in initiate_y
    #   --> config dict is therefore reloaded.
    y, config = init.intiate_y(config)
    #       ... then re-checked :
    input.check_type(config)

    print(f'type y_0 = {type(y)}')
    print(f'outupt type {type(init.intiate_y(config))}')

    # Solver :
    if config['solver'] == 'euler_explicit':
        sol = solver.euler_explicit(y, t, **config)
    if config['solver'] == 'adams_bashforth':
        sol = solver.adams_bashforth(y, t, **config)
    if config['solver'] == 'crank_nicolson':
        sol = solver.crank_nicolson(y, t, **config)
    print(f'Number of saved time steps = {np.shape(sol)[0]}')
    print(f'sol.shape = {sol.shape}')
    # print(f'sol = {sol}')

    # Post-treatment :
    t_save = t[::config['n_save']]
    postt.plot_postt(t_save, sol, **config)


# %% direct execution :
if __name__ == '__main__':
    main()
else:
    print('This script was imported')
