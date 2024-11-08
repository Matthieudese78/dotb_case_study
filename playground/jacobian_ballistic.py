#!/usr/bin/python3
# %%
from __future__ import annotations

from sympy import Matrix
from sympy import symbols

# %% Define variables
x, y, z, w, mu, g = symbols('x y z w mu g')

# Define a 4x1 vector field as a column matrix

f = Matrix([
    z,
    w,
    -mu*z*(((z**2)+(w**2))**(1/2)),
    -g-mu*w*(((z**2)+(w**2))**(1/2)),

])

# Define the variables as a vector
variables = Matrix([x, y, z, w])

# Compute the Jacobian
jacobian = f.jacobian(variables)

display(jacobian)
# %%
