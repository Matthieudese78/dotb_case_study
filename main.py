#!/usr/bin/python3
# %% Loading packages :
# Standard packages :
from __future__ import annotations

import numpy as np

import dotb.postt as postt
import dotb.solver as solver
# Local packages :


# %% main function definition :
def main():
    print('This is the main program')

    # Example usage
    def example_F(x, y, dydx):
        """
        Example function for F(x,y,y')

        Parameters:
        x (array): Spatial coordinates
        y (array): Current state of y
        dydx (array): Spatial derivative of y

        Returns:
        array: F(x,y,y')
        """
        return np.sin(x) * y + np.cos(dydx)

    # input :
    # Set up the problem
    x = np.linspace(-np.pi, np.pi, 100)
    t = np.linspace(0, 1, 100)

    # Initial condition
    # y0 = np.sin(x)[:, np.newaxis]
    y0 = np.sin(x)

    # Solve the PDE
    sol = solver.euler_explicit(example_F, y0, x, t)

    # Post Treatment :
    postt.postt_1D(t, x, sol)


# %% direct execution :
if __name__ == '__main__':
    main()
else:
    print('This script was imported')
