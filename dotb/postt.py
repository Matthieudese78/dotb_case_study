from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def postt_1D(t, x, sol):
    for i in range(len(t)):
        if i % 10 == 0:
            # plt.plot(x, sol[i, :, 0], label=f"t={t[i]:.2f}")
            plt.plot(x, sol[i, :], label=f't={t[i]:.2f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution over time')
    plt.legend()
    plt.show()
    plt.close('all')


def postt_ballistic(t, sol):
    for i, slice in enumerate(sol):
        print(f'i = {i}')
        plt.plot(t, slice, label=f'slice={i}')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Solution {i}^th y component over time')
    plt.legend()
    plt.show()
    plt.close('all')


def postt_rabbit(t, sol):
    for i, slice in enumerate(sol):
        print(f'i = {i}')
        plt.plot(t, slice, label=f'slice={i}')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Solution {i}^th y component over time')
    plt.legend()
    plt.show()
    plt.close('all')


def postt_2Dmap(x, y, data, title, labelx, labely, labelbar):
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter plot with colors determined by data values
    scatter = ax.scatter(
        X.ravel(), Y.ravel(),
        c=data.ravel(), cmap='inferno', s=100,
    )

    # Set limits and aspect ratio
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect('equal')

    ax.set_xlabel(f'{labelx}')
    ax.set_ylabel(f'{labely}')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel(f'{labelbar}')

    plt.title(
        f'{title}',
    )
    plt.show()
    plt.close('all')
