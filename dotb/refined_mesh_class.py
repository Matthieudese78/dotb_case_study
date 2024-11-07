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
        x_refine_percent=10.,
        y_refine_percent=10.,
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
        self.delta_x_y()

    def create_mesh(self):
        # Create a coarse grid
        x_coarse = np.linspace(
            -self.lx / 2.0,
            self.lx / 2.0,
            int(self.lx / self.dx_coarse) + 1,
        )
        y_coarse = np.linspace(
            -self.ly / 2.0,
            self.ly / 2.0,
            int(self.ly / self.dy_coarse) + 1,
        )

        # Create a finer grid
        x_fine = np.linspace(
            -self.lx / 2.0,
            self.lx / 2.0,
            int(self.lx / self.dx_fine) + 1,
        )
        y_fine = np.linspace(
            -self.ly / 2.0,
            self.ly / 2.0,
            int(self.ly / self.dy_fine) + 1,
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
            (
                (
                    x_coarse > x_coarse[0] + self.lx *
                    (self.x_refine_percent / 100.0)
                )
                & (
                    x_coarse
                    < x_coarse[0] + self.lx * (1.0 - self.x_refine_percent / 100.0)
                )
            ),
        )[0]
        # print(f'ix_middle = {ix_middle}')

        iy_middle = np.where(
            (
                (
                    y_coarse > y_coarse[0] + self.ly *
                    (self.y_refine_percent / 100.0)
                )
                & (
                    y_coarse
                    < y_coarse[0] + self.ly * (1.0 - self.y_refine_percent / 100.0)
                )
            ),
        )[0]

        # Combine coarse and fine grids
        # refined :
        x = np.concatenate(
            (x_fine[ix_fine_left], x_coarse[ix_middle], x_fine[ix_fine_right]),
        )
        y = np.concatenate(
            (y_fine[iy_fine_bottom], y_coarse[iy_middle], y_fine[iy_fine_top]),
        )
        print(f'ix_fine_right : {ix_fine_right}')
        print(f'ix_fine_left : {ix_fine_left}')
        print(f'ix_middle : {ix_middle}')
        # non refined :
        # x = np.linspace(-self.lx/2.,self.lx/2.,int(self.lx/self.dx_fine)+1)
        # y = np.linspace(-self.ly/2.,self.ly/2.,int(self.ly/self.dy_fine)+1)

        #
        # x1 = np.geomspace(-self.lx/2.,-self.dx_coarse/2.)
        # x2 = np.geomspace(self.dx_fine/2.,self.lx/2.)
        # #
        # x1 = np.logspace(-self.lx/2.,-self.dx_coarse/2.,base=100.*self.lx)
        # n = 1.
        # x2 = np.logspace(self.dx_fine/2.,self.lx/2.,num=20, base= dx_fine/10.)
        # x2 = x2 * ((self.lx/2.) / np.max(np.abs(x2)) )
        # x2 = np.flip(x2)
        # x2 = np.insert(x2,0,self.dx_fine)
        # print(f'x2[0] = {x2[0]}')
        # print(f'self.dx_fine = {self.dx_fine}')
        # print(f'self.dx_coarse = {self.dx_coarse}')
        # print(f'x2[-1] = {x2[-1]}')
        #
        # plt.scatter(np.linspace(self.dx_fine,1.,len(x1)),x1)
        # plt.show()
        # plt.scatter(np.linspace(0.,1.,len(x2)),x2)
        # plt.show()

        # N = 20

        # # x1 = np.logspace(0.1, 1, N, endpoint=True)
        # x1 = np.logspace(0., 1, N, endpoint=True)

        # x2 = np.logspace(0.1, 1, N, endpoint=False)

        # x1 = x1 * ((self.lx/2.) / np.max(np.abs(x1)) )

        # y = np.zeros(N)

        # plt.plot(x1, y, 'o')

        # # plt.plot(x2, y + 0.5, 'o')

        # plt.ylim([-0.5, 1])
        # (-0.5, 1)

        # plt.show()

        # print(x1[0])
        # print(x1[-1])
        # print(f'y = {dy}')
        # Create meshgrid
        # self.X, self.Y = np.meshgrid(x, y)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.nx = len(x)
        self.ny = len(y)
        # get the points positions grid :
        # positions = np.vstack([self.X.ravel(), self.Y.ravel()])
        # self.X_pos = positions[0].reshape(len(x),len(y))
        # self.Y_pos = positions[1].reshape(len(x),len(y))
        # self.X_calc, self.Y_calc = np.mgrid
        # interior points :
        self.interior_X, self.interior_Y = np.meshgrid(x[1:-1], y[1:-1])
        # print(self.X.shape)

    # x delta to the right neghbor :
    def delta_x_y(self):
        # x-distance to the left and right nodes :
        # delta_x_minus = np.abs(self.X[:, 1:] - self.X[:, :-1])
        # delta_x_plus = delta_x_minus
        delta_x_minus = np.abs(self.X[1:, :] - self.X[:-1, :])
        delta_x_plus = delta_x_minus
        # retrieving the values at the boundaries :
        # --> creating the shift
        #   retrieve last value of minus distance array :
        #   retrieve first value of plus distance array :
        self.delta_x_minus = delta_x_minus[:-1, :]
        self.delta_x_plus = delta_x_plus[1:, :]
        # y-distance to the minus and plus nodes :
        # delta_y_minus = np.abs(self.Y[1:, :] - self.Y[:-1, :])
        delta_y_minus = np.abs(self.Y[:, 1:] - self.Y[:, :-1])
        delta_y_plus = delta_y_minus
        # retrieving the values at the boundaries :
        # --> creating the shift
        #   retrieve last value of minus distance array :
        #   retrieve last value of plus distance array :
        # self.delta_y_minus = delta_y_minus[:-1, :]
        # self.delta_y_plus = delta_y_plus[1:, :]
        self.delta_y_minus = delta_y_minus[:, :-1]
        self.delta_y_plus = delta_y_plus[:, 1:]

        # np.round(self.delta_x_minus, int(np.abs(np.log10(np.min(self.delta_x_minus)))) + 1)
        # np.round(self.delta_x_plus, int(np.abs(np.log10(np.min(self.delta_x_plus)))) + 1)
        # np.round(self.delta_y_minus, int(np.abs(np.log10(np.min(self.delta_y_minus)))) + 1)
        # np.round(self.delta_y_plus, int(np.abs(np.log10(np.min(self.delta_y_plus)))) + 1)
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
        # plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # Example usage
    lx = 1.0
    ly = 1.0
    dx_fine = 0.1
    dy_fine = 0.1
    dx_coarse = 2. * dx_fine
    dy_coarse = 2. * dy_fine
    x_refine_percent = 20.
    y_refine_percent = 20.

    mesh = Mesh(
        lx=lx,
        ly=ly,
        dx_fine=dx_fine,
        dy_fine=dy_fine,
        dx_coarse=dx_coarse,
        dy_coarse=dy_coarse,
        x_refine_percent=x_refine_percent,
        y_refine_percent=x_refine_percent,
    )

    mesh.visualize()


# %%
