#!/usr/bin/python3
from __future__ import annotations

import argparse
import os
from typing import TypedDict

import numpy as np
import yaml

from dotb.refined_mesh_class import Mesh


class MyDictBallistic(TypedDict):
    case: str
    solver: str
    save_dir: str
    t_end: float
    dt: float
    n_save: int
    x_0: float
    y_0: float
    v_0: float
    theta_0: float
    g: float
    rho: float
    A: float
    c: float


class MyDictRabbit(TypedDict):
    case: str
    solver: str
    save_dir: str
    t_end: float
    n_t: int
    n_save: int
    N_0: float
    k: float
    b: float


class MyDictDiffusion2D(TypedDict):
    case: str
    solver: str
    save_dir: str
    t_end: float
    dt: float
    n_save: int

    l_x: float
    l_y: float

    dx_fine: float
    dy_fine: float

    dx_coarse: float
    dy_coarse: float

    x_refine_percent: float
    y_refine_percent: float

    mesh: Mesh

    T_0: float

    p_x: float
    p_y: float

    D_0: float
    D_uni: bool
    D_lin: bool
    D: None | np.ndarray

    dirichlet: bool
    neumann: bool

    interpolation_coeff: float

    dirichlet_right_boundary: float
    dirichlet_left_boundary: float
    dirichlet_bottom_boundary: float
    dirichlet_top_boundary: float

    neumann_right_boundary: float
    neumann_left_boundary: float
    neumann_bottom_boundary: float
    neumann_top_boundary: float

    right_boundary: float
    left_boundary: float
    bottom_boundary: float
    top_boundary: float

# Function "check_type" :
# checks if the input dict belongs to one ofe the three classes :
#  - my_dict_ballistic
#  - my_dict_rabbit
#  - my_dict_diffusion_2D


def is_ballistic(d: dict) -> bool:
    return all(key in d for key in list(MyDictBallistic.__annotations__.keys()))


def is_rabbit(d: dict) -> bool:
    return all(key in d for key in list(MyDictRabbit.__annotations__.keys()))


def is_diffusion_2D(d: dict) -> bool:
    return all(key in d for key in list(MyDictDiffusion2D.__annotations__.keys()))


def check_type(d: dict) -> str:
    if is_ballistic(d):
        return 'Ballistic'
    if is_rabbit(d):
        return 'Rabbit'
    if is_diffusion_2D(d):
        return 'Diffusion2D'
    raise KeyError('Some inputs are missing in your input_file.yaml')


def check_time_step(d: dict) -> str:
    if d['case'] == 'diffusion_2D':
        discr_min = np.min([
            np.min(d['mesh'].delta_x_minus),
            np.min(d['mesh'].delta_y_minus),
        ])
        D_max = np.max(d['D'])
        if d['dt'] > (discr_min/(2.*D_max)):
            raise ValueError(
                'Time step is to big. \n Please respect the CFL criterium : \n dt <= min(delta_x,delta_y)**2 / (2 * max(D) ) ',
            )

    return 'time step size seems OK'


def get_config_file(default_config_path, default_input) -> str:
    """
    Parses the argument for a .yaml input file

    Parameters : default_config_path
    For now = the root directory
    - appends the root dir name to the argument
    - if the .yaml exists, returns the input.file path
    - if not : uses ballistic_default_input.yaml file as input
    """
    # Define the default config file path
    default_config_path = f'{default_config_path}/{default_input}'

    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument(
        'config',
        nargs='?',
        default=default_config_path,
        help='Path to the config file (optional)',
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the config file path
    config_path = args.config

    # Check if the provided config file exists
    if not os.path.exists(config_path):
        print(
            f"Warning: Config file '{config_path}' does not exist. Using default config.",
        )
        config_path = default_config_path

    return config_path


# my_dict : type being either one of the three possible TypedDict :
my_dict = MyDictBallistic | MyDictRabbit | MyDictDiffusion2D
# turns the entries of a config.yaml file into a dict :


def load_config(file_path) -> my_dict:
    """
    Turns the entries of a config.yaml file into a dict.
    """
    with open(file_path) as config_file:
        return yaml.safe_load(config_file)
