from __future__ import annotations

import matplotlib.pyplot as plt


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
