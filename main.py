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
from dotb.second_member import Ballistic
from dotb.second_member import Diffusion
from dotb.second_member import Rabbit
# from dotb.solver import EulerExplicit, AdamsBashforh

# Local packages :

solver_map = {
    'euler_explicit': solver.EulerExplicit,
    'adams_bashforth': solver.AdamsBashforth,
    # 'crank_nicolson': CrankNicolson
}

rhs_map = {
    'ballistic': Ballistic,
    'rabbit': Rabbit,
    'diffusion_2D': Diffusion,
}


def create_solver(y0, t, case_name, solver_name, config):
    if case_name not in rhs_map:
        raise ValueError(
            f"Invalid study case '{case_name}'. Choose from {list(rhs_map.keys())}",
        )
    if solver_name not in solver_map:
        raise ValueError(
            f"Invalid method '{solver_name}'. Choose from {list(solver_map.keys())}",
        )

    rhs_instance = rhs_map[case_name](**config)
    return solver_map[solver_name](y0, t, rhs=rhs_instance, **config)
    # return rhs_map[case_name](method=solver_instance, **kwargs)

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

    # y-initialization :
    #   n-dimensional tensor field y(t=0) from config :
    #   some values in config dict are computed in initiate_y
    #   --> config dict is therefore reloaded.
    y, t, config = init.intiate_y(config)
    #       ... then re-checked :
    input.check_type(config)

    # k1 = {k: config.get(k) for k in [
    #     'dirichlet','neumann','left_boundary','bottom_boundary','right_boundary','top_boundary','interpolation_coeff',
    #     'n_save','D','mesh'
    #     ] }
    # print(f'type y_0 = {type(y)}')
    # print(f'outupt type {type(init.intiate_y(config))}')

    # Solver creation :
    simu = create_solver(y, t, config['case'], config['solver'], config)

    # Run simulation :
    sol = simu.solve(**config)
    # print(f'solution shape : {sol.shape}')
    # Solver : on laisse en fonctionnel :
    # if config['solver'] == 'euler_explicit':
    #     sol = solver.euler_explicit(y, t, **config)
    # if config['solver'] == 'adams_bashforth':
    #     sol = solver.adams_bashforth(y, t, **config)
    # if config['solver'] == 'crank_nicolson':
    #     sol = solver.crank_nicolson(y, t, **config)
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
