#!/usr/bin/python3
# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# %%
def euler_explicit(F, y0, x, t, dx=0.01, dt=0.001):
    """
    Solve ∂y/∂t = F(x,y,y') using Euler explicit method

    Parameters:
    F (function): Function defining the PDE
    y0 (array): Initial condition
    x (array): Spatial grid
    t (array): Time array
    dx (float): Spatial step size
    dt (float): Time step size

    Returns:
    y (ndarray): Solution tensor field
    """
    n_steps = len(t)
    n_points = len(x)

    y = np.zeros((n_steps, n_points))
    y[0] = y0

    for i in range(n_steps - 1):
        # dydx = np.gradient(y[i], axis=1) / dx
        dydx = np.gradient(y[i]) / dx

        # Apply Euler explicit formula
        # y[i + 1] = y[i] + dt * F(x[:, np.newaxis], y[i], dydx)
        y[i + 1] = y[i] + dt * F(x, y[i], dydx)

    return y


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


# %%

# Set up the problem
x = np.linspace(-np.pi, np.pi, 100)
t = np.linspace(0, 1, 100)

# Initial condition
# y0 = np.sin(x)[:, np.newaxis]
y0 = np.sin(x)

# %%
# Solve the PDE
sol = euler_explicit(example_F, y0, x, t)
# %%
# Plot the solution at different times
for i in range(len(t)):
    if i % 10 == 0:
        # plt.plot(x, sol[i, :, 0], label=f"t={t[i]:.2f}")
        plt.plot(x, sol[i, :], label=f't={t[i]:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution over time')
plt.legend()
plt.show()

# %%
