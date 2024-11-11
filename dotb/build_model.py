#!/usr/bin/pyhon3
from __future__ import annotations

import numpy as np

import dotb.solver as solver
from dotb.second_member import Ballistic
from dotb.second_member import Diffusion
from dotb.second_member import Rabbit
# solver & right hand side maps :
# --> enables simulation construction without case disjunction
solver_map = {
    'euler_explicit': solver.EulerExplicit,
    'adams_bashforth': solver.AdamsBashforth,
    'crank_nicolson': solver.CrankNicolson,
}

rhs_map = {
    'ballistic': Ballistic,
    'rabbit': Rabbit,
    'diffusion_2D': Diffusion,
}

solver_type = solver.EulerExplicit | solver.AdamsBashforth | solver.CrankNicolson


def create_solver(y0: np.ndarray, t: np.ndarray, config: dict):
    """
    Reads
    Creates an instance of a solver among :
    - EulerExplicit
    - AdamsBashforth
    - CrankNicolson
    ... and an instance of a right hand side class among :
    - Ballistic
    - Rabbit
    - Diffusion
    which is an argument of the solver.
    """
    if config['case'] not in rhs_map:
        raise ValueError(
            f"Invalid study case '{config['case']}'. Choose from {list(rhs_map.keys())}",
        )
    if config['solver'] not in solver_map:
        raise ValueError(
            f"Invalid method '{config['solver']}'. Choose from {list(solver_map.keys())}",
        )
    rhs_instance = rhs_map[config['case']](**config)
    return solver_map[config['solver']](y0, t, rhs=rhs_instance, **config)
