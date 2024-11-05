#!/usr/bin/pyton3
# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# %%
class Mesh:
    def __init__(
        self,
        lx,
        ly,
        dx_fine,
        dy_fine,
        dx_coarse,
        dy_coarse,
        x_refine_percent=10,
        y_refine_percent=10,
    ):
        self.lx = lx
        self.ly = ly
        self.dx_coarse = dx_coarse
        self.dy_coarse = dy_coarse
        self.dx_fine = dx_fine
        self.dy_fine = dy_fine
        self.x_refine_percent = x_refine_percent
        self.y_refine_percent = y_refine_percent

        self.create_mesh()
        self.delta_x_right()

    def create_mesh(self):
        # Create a coarse grid
        x_coarse = np.linspace(
            -self.lx / 2.0, self.lx / 2.0, int(self.lx / self.dx_coarse) + 1,
        )
        y_coarse = np.linspace(
            -self.ly / 2.0, self.ly / 2.0, int(self.ly / self.dy_coarse) + 1,
        )

        # Create a finer grid
        x_fine = np.linspace(
            -self.lx / 2.0, self.lx / 2.0, int(self.lx / self.dx_fine) + 1,
        )
        y_fine = np.linspace(
            -self.ly / 2.0, self.ly / 2.0, int(self.ly / self.dy_fine) + 1,
        )

        ix_fine_right = np.where(
            x_fine >= x_fine[0] + self.lx *
            (1.0 - self.x_refine_percent / 100.0),
        )[0]

        ix_fine_left = np.where(
            x_fine <= x_fine[0] + self.lx * (self.x_refine_percent / 100.0),
        )[0]

        iy_fine_top = np.where(
            y_fine >= y_fine[0] + self.ly *
            (1.0 - self.y_refine_percent / 100.0),
        )[0]

        iy_fine_bottom = np.where(
            y_fine <= y_fine[0] + self.ly * (self.y_refine_percent / 100.0),
        )[0]

        ix_middle = np.where(
            ((x_coarse > x_coarse[0] + self.lx * (self.x_refine_percent / 100.0))
             & (x_coarse < x_coarse[0] + self.lx * (1.0 - self.x_refine_percent / 100.0))),
        )[0]
        # print(f'ix_middle = {ix_middle}')

        iy_middle = np.where(
            ((y_coarse > y_coarse[0] + self.ly * (self.y_refine_percent / 100.0))
             & (y_coarse < y_coarse[0] + self.ly * (1.0 - self.y_refine_percent / 100.0))),
        )[0]

        # Combine coarse and fine grids
        x = np.concatenate(
            [x_fine[ix_fine_left], x_coarse[ix_middle], x_fine[ix_fine_right]],
        )
        y = np.concatenate(
            [y_fine[iy_fine_bottom], y_coarse[iy_middle], y_fine[iy_fine_top]],
        )
        # Create meshgrid
        self.X, self.Y = np.meshgrid(x, y)
        # print(self.X.shape)

    # x delta to the right neghbor :
    def delta_x_right(self):
        # x-distance to the left and right nodes :
        delta_x_minus = self.X[:, 1:] - self.X[:, :-1]
        delta_x_plus = delta_x_minus
        # retrieving the values at the boundaries :
        # --> creating the shift
        #   retrieve last value of minus distance array :
        #   retrieve first value of plus distance array :
        self.delta_x_minus = delta_x_minus[:, :-1]
        self.delta_x_plus = delta_x_plus[:, 1:]
        # y-distance to the minus and plus nodes :
        delta_y_minus = self.Y[1:, :] - self.Y[:-1, :]
        delta_y_plus = delta_y_minus
        # retrieving the values at the boundaries :
        # --> creating the shift
        #   retrieve last value of minus distance array :
        #   retrieve last value of plus distance array :
        self.delta_y_minus = delta_y_minus[:-1, :]
        self.delta_y_plus = delta_y_plus[1:, :]

    def visualize(self):
        plt.figure(figsize=(10, 10))

        # plt.plot(self.X[:, ::-1], self.Y[:, ::-1], 'k-', linewidth=0.5)

        # Plot horizontal lines
        plt.scatter(self.X.ravel(), self.Y.ravel(), s=4)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(
            f'Refined Mesh Lattice (X-refine: {self.x_refine_percent}%, Y-refine: {self.y_refine_percent}%)',
        )
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # Example usage
    lx = 1.
    ly = 1.
    dx_fine = 0.001
    dy_fine = 0.001
    dx_coarse = 50. * dx_fine
    dy_coarse = 50. * dy_fine

    mesh = Mesh(
        lx=lx,
        ly=ly,
        dx_fine=dx_fine,
        dy_fine=dy_fine,
        dx_coarse=dx_coarse,
        dy_coarse=dy_coarse,
        x_refine_percent=2,
        y_refine_percent=2,
    )

    # mesh.visualize()


# %%
