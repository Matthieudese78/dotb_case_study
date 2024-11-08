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


def create_solver(y0: np.ndarray, t: np.ndarray, case_name: str, solver_name: str, config: dict) -> solver_type:
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
