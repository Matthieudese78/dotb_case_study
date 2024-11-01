#!/usr/bin/python3
# %% Loading packages :
# Standard packages :
from __future__ import annotations

import os.path

import numpy as np

import dotb.input_yaml as input
# import dotb.postt as postt
# import dotb.solver as solver

# Local packages :


# %% main function definition :
def main():
    print('This is the main program')
    # main directory : contains the default .yaml input files
    maindir = os.path.dirname(__file__)
    # default .yaml file name :
    default_input = 'ballistic_default_input.yaml'
    config_file_path = input.get_config_file(maindir, default_input)
    print(f'Using config file: {config_file_path}')
    # Parsing the input.yaml --> config dict
    config = input.load_config(config_file_path)
    print(f"Solving the {config['case']} case")
    print(f"using the {config['solver']} solver")
    # time vector :
    t = np.linspace(0, config['tend'], config['nt'])

    # Example usage
    # def example_F(x, y, dydx):
    #     """
    #     Example function for F(x,y,dydx')

    #     Parameters:
    #     x (array): Spatial coordinates
    #     y (array): Current state of y
    #     dydx (array): Spatial derivative of y

    #     Returns:
    #     array: F(x,y,dydx')
    #     """
    #     return np.sin(x) * y + np.cos(dydx)

    # # input :
    # # Set up the problem
    # x = np.linspace(-np.pi, np.pi, 100)
    # t = np.linspace(0, 1, 100)

    # # Initial condition
    # # y0 = np.sin(x)[:, np.newaxis]
    # y0 = np.sin(x)

    # # Solve the PDE
    # sol = solver.euler_explicit(example_F, y0, x, t)

    # # Post Treatment :
    # postt.postt_1D(t, x, sol)

    # # Ballistic :
    # def ballistic(y, rho, c, A, g):
    #     """
    #     Ballistic function for F(x,y,dydx')

    #     Parameters:
    #     x (array): Spatial coordinates
    #     y (array): Current state of y
    #     dxdt (array): Time derivative of x
    #     dydt (array): Time derivative of y

    #     Returns:
    #     array: F(x,y,dxdt,dydt')
    #     """
    #     dxdt = y[2]
    #     dydt = y[3]
    #     mu = 0.5*rho*c*A
    #     magv = np.sqrt(dxdt**2 + dydt**2)
    #     return np.array([dxdt,dydt,-mu*dxdt*magv,-g-mu*dydt*magv])

    # # Set up the problem
    # t = np.linspace(0, 30, 100)
    # # Physics :
    # g = 9.81
    # rho = 1.225
    # A = np.pi*(2.5e-2)**2
    # c = 0.47
    # # Initial conditions :
    # y0 = np.zeros(4)
    # y0[0], y0[1] = 0., 0.
    # v0 = 50.
    # theta0 = 45.
    # y0[2] = v0*np.cos(theta0)
    # y0[3] = v0*np.sin(theta0)
    # # solver :
    # sol = solver.euler_explicit_ballistic(ballistic, y0, c, A, g, t)
    # # postt :
    # postt.postt_ballistic(t, sol)

    # # Rabbit :
    # def rabbit(y, k, b):
    #     """
    #     Ballistic function for F(x,y,dydx')

    #     Parameters:
    #     x (array): Spatial coordinates
    #     y (array): Current state of y
    #     dxdt (array): Time derivative of x
    #     dydt (array): Time derivative of y

    #     Returns:
    #     array: F(x,y,dxdt,dydt')
    #     """
    #     return np.array([k * y * (1 - (y / b))])

    # # Set up the problem
    # t = np.linspace(0, 1, 1000)
    # # Physics :
    # # rabibts : 5 litters annualy of 10 kits each :
    # k = 5 * 10
    # # but only half are females :
    # k = k / 2.0
    # # discretization : dx, dy in km
    # dx = 1
    # dy = 1
    # # For 1 km^2 : 100 rabbits max
    # # b = 10000 / dx * dy
    # b = 500 / dx * dy
    # # Initial conditions : 10 rabbits
    # y0 = np.array([2]).astype(float)
    # # y0 = np.random.randint(2, 10, size=(5 , 5)).astype(float)
    # print(f'y0 = {y0}')
    # # solver :
    # sol = solver.euler_explicit_rabbit(rabbit, y0, k, b, t)
    # print(f"sol size = {np.shape(sol)}")
    # print(f'sol at time step 5 = {sol[...,5]}')
    # # postt :
    # postt.postt_ballistic(t, sol)


# %% direct execution :
if __name__ == '__main__':
    main()
else:
    print('This script was imported')
