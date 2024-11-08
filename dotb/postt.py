from __future__ import annotations

import os

import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from PIL import Image

import dotb.differential_operators as diffops
import dotb.second_member as second
from dotb.boundary_conditions import boundary_conditions_mesh
from dotb.refined_mesh_class import Mesh


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


def plot_postt(t, sol, **kw):
    if not os.path.exists(kw['save_dir']):
        os.makedirs(kw['save_dir'])
    if kw['case'] == 'ballistic':
        return postt_ballistic_2(t, sol, **kw)
    if kw['case'] == 'rabbit':
        return postt_rabbit(t, sol, **kw)
    if kw['case'] == 'diffusion_2D':
        return postt_diffusion_2D(t, sol, **kw)


def postt_rabbit(t: np.ndarray, sol: np.ndarray, **kw):
    print(f"{kw['save_dir']}")
    labs = [r'$N(t)$' + ' (year)']
    fig_names = ['N_ft']
    for i in range(np.shape(sol)[1]):
        # plt.plot(t, sol[:,i], label=f'slice={i}')
        plt.plot(t, sol[:, i])
        plt.xlabel('t (s)')
        plt.ylabel(labs[i])
        plt.title(f'Solution {i}^th y component over time \n')
        # plt.legend()
        plt.savefig(kw['save_dir'] + fig_names[i] + '.png')
        plt.close('all')


def postt_ballistic_2(t: np.ndarray, sol: np.ndarray, **kw):
    print(f"{kw['save_dir']}")
    labs = [
        r'$x(t)$' + ' (m)',
        r'$y(t)$' + ' (m)',
        r'$\frac{dx}{dt}$' + ' (m.' + r'$s^{-1}$' + ')',
        r'$\frac{dy}{dt}$' + ' (m.' + r'$s^{-1}$' + ')',
    ]
    fig_names = ['x', 'y', 'vx', 'vy']
    for i in range(np.shape(sol)[1]):
        # plt.plot(t, sol[:,i], label=f'slice={i}')
        plt.plot(t, sol[:, i])
        plt.xlabel('t (s)')
        plt.ylabel(labs[i])
        plt.title(f'Solution {i}^th y component over time \n')
        # plt.legend()
        plt.savefig(kw['save_dir'] + fig_names[i] + '.png')
        plt.close('all')

    plt.plot(sol[:, 0], sol[:, 1])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Trajectory \n')
    # plt.legend()
    plt.savefig(kw['save_dir'] + 'Trajectory' + '.png')
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


def postt_diffusion_2D(t, sol, **kw):
    output_dir_gifs = kw['save_dir']
    # x & y coordinates and discr :
    # x = np.linspace(-kw['l_x'] / 2.0, kw['l_x'] / 2.0, kw['n_x'])
    # y = np.linspace(-kw['l_x'] / 2.0, kw['l_y'] / 2.0, kw['n_y'])
    # dx = x[1] - x[0]
    # dy = y[1] - y[0]

    print(f"save_dir = {kw['save_dir']}")
    labelx = 'X (m)'
    labely = 'Y (m)'

    # gradient values :
    grads = [diffops.gradient_mesh(kw['mesh'], soli) for soli in sol]
    gradx = [gradi[0] for gradi in grads]
    grady = [gradi[1] for gradi in grads]
    # divergence values :
    divs = [diffops.divergence_mesh(kw['mesh'], soli) for soli in sol]
    # second member F values :
    Fs = [second.F_diffusion(kw['mesh'], soli, kw['D']) for soli in sol]
    # matching max & min values :
    print(
        f'max T(x,y,t=0) = {sol[0][np.unravel_index(np.argmax(sol[0],axis=None), sol[0].shape)]}',
    )
    print(
        f'min T(x,y,t=0) = {sol[0][np.unravel_index(np.argmin(sol[0],axis=None), sol[0].shape)]}',
    )

    # Minima :
    min_T = np.min(
        [
            vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
            for vali in sol
        ],
    )
    max_T = np.max(
        [
            vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
            for vali in sol
        ],
    )
    min_div = np.min(
        [
            vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
            for vali in divs
        ],
    )
    max_div = np.max(
        [
            vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
            for vali in divs
        ],
    )
    min_F = np.min(
        [
            vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
            for vali in Fs
        ],
    )
    max_F = np.max(
        [
            vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
            for vali in Fs
        ],
    )
    min_gradx = np.min(
        [
            vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
            for vali in gradx
        ],
    )
    max_gradx = np.max(
        [
            vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
            for vali in gradx
        ],
    )
    min_grady = np.min(
        [
            vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
            for vali in grady
        ],
    )
    max_grady = np.max(
        [
            vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
            for vali in grady
        ],
    )

    print(f'min Temperature reached at {min_T}')
    print(f'max Temperature reached at {max_T}')

    # Saving snapshots of Temperature field :
    dir_T = kw['save_dir'] + 'T/'
    labelbar = 'Temperature ' + r'$(\degree C)$'
    if not os.path.exists(dir_T):
        os.makedirs(dir_T)
    for i, soli in enumerate(sol):
        save_name = f'T_{i}'
        title = f'Temperature at {t[i]:.2f} s \n'
        postt_2Dmap_limits(
            kw['mesh'],
            soli,
            title,
            save_name,
            dir_T,
            labelx,
            labely,
            labelbar,
            limitbar=[min_T, max_T],
        )
    # Outputing gifs :
    input_dir = dir_T
    output_file = 'T.gif'
    create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

    # Saving snapshots of temperature divergence :
    dir_T = kw['save_dir'] + 'div_T/'
    labelbar = r'$div(T)$'
    if not os.path.exists(dir_T):
        os.makedirs(dir_T)
    for i, soli in enumerate(divs):
        save_name = f'div_T_{i}'
        title = f'Temperature divergence at {t[i]:.2f} s \n'
        postt_2Dmap_limits(
            kw['mesh'],
            soli,
            title,
            save_name,
            dir_T,
            labelx,
            labely,
            labelbar,
            limitbar=[min_div, max_div],
        )
    # Outputing gifs :
    input_dir = dir_T
    output_file = 'div_T.gif'
    create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

    # Saving snapshots of 2nd member F :
    dir_T = kw['save_dir'] + 'F/'
    labelbar = 'F'
    if not os.path.exists(dir_T):
        os.makedirs(dir_T)
    for i, soli in enumerate(Fs):
        save_name = f'F_{i}'
        title = f'2nd member F at {t[i]:.2f} s \n'
        postt_2Dmap_limits(
            kw['mesh'],
            soli,
            title,
            save_name,
            dir_T,
            labelx,
            labely,
            labelbar,
            limitbar=[min_F, max_F],
        )
    # Outputing gifs :
    input_dir = dir_T
    output_file = 'F.gif'
    create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

    # Saving snapshots of x-gradient :
    dir_T = kw['save_dir'] + 'gradx/'
    labelbar = r'$\frac{\partial T}{\partial x}$'
    if not os.path.exists(dir_T):
        os.makedirs(dir_T)
    for i, soli in enumerate(gradx):
        save_name = f'gradx_{i}'
        title = f'x-gradient at {t[i]:.2f} s \n'
        postt_2Dmap_limits(
            kw['mesh'],
            soli,
            title,
            save_name,
            dir_T,
            labelx,
            labely,
            labelbar,
            limitbar=[min_gradx, max_gradx],
        )
    # Outputing gifs :
    input_dir = dir_T
    output_file = 'gradx.gif'
    create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

    # Saving snapshots of y-gradient :
    dir_T = kw['save_dir'] + 'grady/'
    labelbar = r'$\frac{\partial T}{\partial y}$'
    if not os.path.exists(dir_T):
        os.makedirs(dir_T)
    for i, soli in enumerate(grady):
        save_name = f'grady_{i}'
        title = f'y-gradient at {t[i]:.2f} s \n'
        postt_2Dmap_limits(
            kw['mesh'],
            soli,
            title,
            save_name,
            dir_T,
            labelx,
            labely,
            labelbar,
            limitbar=[min_grady, max_grady],
        )
    # Outputing gifs :
    input_dir = dir_T
    output_file = 'grady.gif'
    create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

# def postt_diffusion_2D(t, sol, **kw):
#     output_dir_gifs = kw['save_dir']
#     # x & y coordinates and discr :
#     x = np.linspace(-kw['l_x'] / 2.0, kw['l_x'] / 2.0, kw['n_x'])
#     y = np.linspace(-kw['l_x'] / 2.0, kw['l_y'] / 2.0, kw['n_y'])
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]

#     print(f"save_dir = {kw['save_dir']}")
#     labelx = 'X (m)'
#     labely = 'Y (m)'

#     # gradient values :
#     grads = [second.gradient(soli, [dx, dy], edge_order=1) for soli in sol]
#     gradx = [gradi[0] for gradi in grads]
#     grady = [gradi[1] for gradi in grads]
#     # divergence values :
#     divs = [second.divergence(soli, [dx, dy], edge_order=1) for soli in sol]
#     # second member F values :
#     Fs = [second.F(soli, **kw) for soli in sol]
#     # matching max & min values :
#     print(
#         f'max T(x,y,t=0) = {sol[0][np.unravel_index(np.argmax(sol[0],axis=None), sol[0].shape)]}',
#     )
#     print(
#         f'min T(x,y,t=0) = {sol[0][np.unravel_index(np.argmin(sol[0],axis=None), sol[0].shape)]}',
#     )

#     # Minima :
#     min_T = np.min(
#         [
#             vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
#             for vali in sol
#         ],
#     )
#     max_T = np.max(
#         [
#             vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
#             for vali in sol
#         ],
#     )
#     min_div = np.min(
#         [
#             vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
#             for vali in divs
#         ],
#     )
#     max_div = np.max(
#         [
#             vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
#             for vali in divs
#         ],
#     )
#     min_F = np.min(
#         [
#             vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
#             for vali in Fs
#         ],
#     )
#     max_F = np.max(
#         [
#             vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
#             for vali in Fs
#         ],
#     )
#     min_gradx = np.min(
#         [
#             vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
#             for vali in gradx
#         ],
#     )
#     max_gradx = np.max(
#         [
#             vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
#             for vali in gradx
#         ],
#     )
#     min_grady = np.min(
#         [
#             vali[np.unravel_index(np.argmin(vali, axis=None), vali.shape)]
#             for vali in grady
#         ],
#     )
#     max_grady = np.max(
#         [
#             vali[np.unravel_index(np.argmax(vali, axis=None), vali.shape)]
#             for vali in grady
#         ],
#     )

#     print(f'min Temperature reached at {min_T}')
#     print(f'max Temperature reached at {max_T}')

#     # Saving snapshots of Temperature field :
#     dir_T = kw['save_dir'] + 'T/'
#     labelbar = 'Temperature ' + r'$(\degree C)$'
#     if not os.path.exists(dir_T):
#         os.makedirs(dir_T)
#     for i, soli in enumerate(sol):
#         save_name = f'T_{i}'
#         title = f'Temperature at {t[i]:.2f} s \n'
#         postt_2Dmap_limits(
#             x,
#             y,
#             soli,
#             title,
#             save_name,
#             dir_T,
#             labelx,
#             labely,
#             labelbar,
#             limitbar=[min_T, max_T],
#             **kw,
#         )
#     # Outputing gifs :
#     input_dir = dir_T
#     output_file = 'T.gif'
#     create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

#     # Saving snapshots of temperature divergence :
#     dir_T = kw['save_dir'] + 'div_T/'
#     labelbar = r'$div(T)$'
#     if not os.path.exists(dir_T):
#         os.makedirs(dir_T)
#     for i, soli in enumerate(divs):
#         save_name = f'div_T_{i}'
#         title = f'Temperature divergence at {t[i]:.2f} s \n'
#         postt_2Dmap_limits(
#             x,
#             y,
#             soli,
#             title,
#             save_name,
#             dir_T,
#             labelx,
#             labely,
#             labelbar,
#             limitbar=[min_div, max_div],
#             **kw,
#         )
#     # Outputing gifs :
#     input_dir = dir_T
#     output_file = 'div_T.gif'
#     create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

#     # Saving snapshots of 2nd member F :
#     dir_T = kw['save_dir'] + 'F/'
#     labelbar = 'F'
#     if not os.path.exists(dir_T):
#         os.makedirs(dir_T)
#     for i, soli in enumerate(Fs):
#         save_name = f'F_{i}'
#         title = f'2nd member F at {t[i]:.2f} s \n'
#         postt_2Dmap_limits(
#             x,
#             y,
#             soli,
#             title,
#             save_name,
#             dir_T,
#             labelx,
#             labely,
#             labelbar,
#             limitbar=[min_F, max_F],
#             **kw,
#         )
#     # Outputing gifs :
#     input_dir = dir_T
#     output_file = 'F.gif'
#     create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

#     # Saving snapshots of x-gradient :
#     dir_T = kw['save_dir'] + 'gradx/'
#     labelbar = r'$\frac{\partial T}{\partial x}$'
#     if not os.path.exists(dir_T):
#         os.makedirs(dir_T)
#     for i, soli in enumerate(gradx):
#         save_name = f'gradx_{i}'
#         title = f'x-gradient at {t[i]:.2f} s \n'
#         postt_2Dmap_limits(
#             x,
#             y,
#             soli,
#             title,
#             save_name,
#             dir_T,
#             labelx,
#             labely,
#             labelbar,
#             limitbar=[min_gradx, max_gradx],
#             **kw,
#         )
#     # Outputing gifs :
#     input_dir = dir_T
#     output_file = 'gradx.gif'
#     create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

#     # Saving snapshots of y-gradient :
#     dir_T = kw['save_dir'] + 'grady/'
#     labelbar = r'$\frac{\partial T}{\partial y}$'
#     if not os.path.exists(dir_T):
#         os.makedirs(dir_T)
#     for i, soli in enumerate(grady):
#         save_name = f'grady_{i}'
#         title = f'y-gradient at {t[i]:.2f} s \n'
#         postt_2Dmap_limits(
#             x,
#             y,
#             soli,
#             title,
#             save_name,
#             dir_T,
#             labelx,
#             labely,
#             labelbar,
#             limitbar=[min_grady, max_grady],
#             **kw,
#         )
#     # Outputing gifs :
#     input_dir = dir_T
#     output_file = 'grady.gif'
#     create_gif_from_png_files(input_dir, output_dir_gifs, output_file)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mplcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


# truncated inferno values :
cmapinf = truncate_colormap(mplcm.inferno, 0.0, 0.9, n=100)


def postt_2Dmap(mesh: Mesh, data, title: str, labelx: str, labely: str, labelbar: str, save_dir: str = '', save_name: str = '', s: int | float = 10):
    """
    Plots a 2D value array over a 2D grid.
    """
    # Create meshgrid for plotting
    X = mesh.X
    Y = mesh.Y

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter plot with colors determined by data values
    scatter = ax.scatter(
        X.ravel(),
        Y.ravel(),
        c=data,
        cmap=cmapinf,
        s=s,
    )

    # Set limits and aspect ratio
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_xlim(np.min(Y), np.max(Y))
    # ax.set_aspect("equal")

    ax.set_xlabel(f'{labelx}')
    ax.set_ylabel(f'{labely}')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel(f'{labelbar}')

    plt.title(f'{title}')
    # plt.show()
    plt.savefig(save_dir + save_name + '.png')
    plt.close('all')


def plot_boundary_conditions(mesh, savedir, savename,  **kw):

    boundaries = boundary_conditions_mesh(mesh, **kw)
    arr_test = np.concatenate(
        (
            np.flip(boundaries[0]),
            boundaries[1],
            boundaries[2],
            np.flip(boundaries[3]),
        ),
    )
    xleft = np.linspace(0.0, mesh.ly, len(mesh.Y[0, :]))
    xbottom = np.linspace(mesh.ly, mesh.ly + mesh.lx, len(mesh.X[:, 0]))
    xright = np.linspace(
        mesh.ly + mesh.lx,
        2.0 * mesh.ly + mesh.lx,
        len(mesh.Y[-1, :]),
    )
    xtop = np.linspace(
        2.0 * mesh.ly + mesh.lx,
        2.0 * mesh.ly + 2.0 * mesh.lx,
        len(mesh.X[:, -1]),
    )

    contour_test = np.concatenate((xleft, xbottom, xright, xtop))

    plt.scatter(contour_test, arr_test, s=4)
    plt.title('Concatenated interpolated array')
    plt.savefig(savedir+savename+'.png')
    plt.close('all')
    return


def postt_2Dmap_limits(
    mesh, data, title, save_name, dirsave, labelx, labely, labelbar, limitbar=[],
):
    f = plt.figure(figsize=(8, 6))
    # init subplots :
    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 0.5])
    # 1st subplot :
    plt.subplot(gs[0])
    # Create figure and axis
    ax = plt.gca()

    # Plot scatter plot with colors determined by data values
    ax.scatter(
        mesh.X.ravel(),
        mesh.Y.ravel(),
        c=data.ravel(),
        cmap='inferno',
        s=18,
    )

    # Set limits and aspect ratio
    x = mesh.X[:, 0]
    y = mesh.Y[0, :]
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect('equal')

    ax.set_xlabel(f'{labelx}')
    ax.set_ylabel(f'{labely}')

    ax.set_title(title)

    # 2nd subplot :
    plt.subplot(gs[1])
    plt.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax = f.gca()
    norm = mplcolors.Normalize(vmin=limitbar[0], vmax=limitbar[1])
    plt.colorbar(
        mplcm.ScalarMappable(norm=norm, cmap='inferno'),
        cax=ax,
        orientation='vertical',
        label=labelbar,
    ).formatter.set_useOffset(False)
    f.tight_layout(pad=0.5)
    # Save :
    f.savefig(dirsave + save_name + '.png')
    plt.close('all')


def create_gif_from_png_files(
    input_directory, output_directory, output_file, duration=5000,
):
    # Get all .png files in the input directory
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Sort the files numerically
    # png_files.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    sorted_files = sorted(
        png_files, key=lambda x: int(
            ''.join(filter(str.isdigit, x)),
        ),
    )

    # Open all images
    images = [
        Image.open(os.path.join(input_directory, f))
        for f in sorted_files
    ]

    # Save as GIF
    images[0].save(
        os.path.join(output_directory, output_file),
        save_all=True,
        append_images=images[1:],
        duration=duration // len(images),  # Calculate duration per frame
        loop=0,  # Loop indefinitely
    )


# Usage
