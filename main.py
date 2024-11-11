#!/usr/bin/python3
# %% Loading packages :
# Standard packages :
from __future__ import annotations

import os.path

import numpy as np

import dotb.initiate as init
import dotb.input_yaml as input
import dotb.postt as postt
from dotb.build_model import create_solver

# %% main function definition :


def main():
    maindir = os.path.dirname(__file__)
    # default_input = 'ballistic_default_input.yaml'
    # default_input = 'rabbit_default_input.yaml'
    default_input = 'diffusion_2D_default_input.yaml'
    # Reading argument :
    config_file_path = input.get_config_file(maindir, default_input)
    print(f'Using config file: {config_file_path}')
    # input.yaml --> config dict
    config = input.load_config(config_file_path)
    # Checking the types of the input variables :
    # (conformity to TypedDict classes?)
    input.check_type(config)

    # y-initialization :
    print(f'Initializing problem ...')
    y, t, config = init.intiate_y(config)
    #       ... then re-checked :
    input.check_type(config)

    # Check time step value (diffusion only : CFL) to avoid divergence :
    print(f'Checking input data types...')
    input.check_time_step(config)

    # Solver creation :
    print(
        f"Solving the {config['case']} case \n using the {config['solver']} solver",
    )
    simu = create_solver(y, t, config)

    # Run simulation :
    sol = simu.solve(**config)
    print(f'Number of saved time steps = {np.shape(sol)[0]}')
    print(f'sol.shape = {sol.shape}')

    # Post-treatment :
    print(f'Postreatment...')
    t_save = np.linspace(0., config['t_end'], len(sol))
    postt.plot_postt(t_save, sol, **config)


# %% direct execution :
if __name__ == '__main__':
    main()

else:
    print('This script was imported')
