#!/usr/bin/python3
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse import kron

from dotb.refined_mesh_class import Mesh


def gradient_mesh(mesh: Mesh, data: np.ndarray) -> np.ndarray:
    """
    On any structured 2D mesh with uniform or non-uniform mesh size :
    Computes the gradient of a 2D array using the distance 2D-arrays
    to the right & left neighbors of Mesh class :
    - delta_x_minus : x-distance to the left neighbor
    - delta_x_plus : x-distance to the right neighbor
    - delta_y_minus : y-distance to the lower neighbor
    - delta_y_plus  : y-distance to the upper neighbor

    Boundaries : gradient estimation
        - x-component -> left & bottom boundaries
                      -> forward difference
        - y-component -> right & top boundaries
                      -> backward difference
    Parameters:
    - a 2D mesh
    - a dataset
    Returns:
    - the 2D gradient field value over the mesh
    Raises: none
    """
    # Interior nodes : for i = 1 --> n-1
    #       (cf numpy.gradient notice --> bibliographic sources)
    # For structured non uniformely sized meshes :
    # \pratial y / \partial x =
    # [ (hs**2)*f(xi + hd) + (hd**2 - hs**2)*f(xi)   - hd**2 * f(xi - hs) ] / hs*hd*(hd + hs)
    # Where
    #       hs is the node distance to its left neighbor
    #       hd is the node distance to its right neighbor
    # Example node 1 :
    # [ (hs**2)*data[2]    + (hd**2 - hs**2)*data[1] - hd**2 * data[0] ] / hs*hd*(hd + hs)
    hs = mesh.delta_x_minus
    hd = mesh.delta_x_plus
    grad_x = (
        (hs**2) * data[2:, :] + (hd**2 - hs**2) *
        data[1:-1, :] - (hd**2) * data[:-2, :]
    ) / (hs * hd * (hd + hs))

    hs = mesh.delta_y_minus
    hd = mesh.delta_y_plus
    grad_y = (
        (hs**2) * data[:, 2:] + (hd**2 - hs**2) *
        data[:, 1:-1] - hd**2 * data[:, :-2]
    ) / (hs * hd * (hd + hs))

    # Boundary nodes : for i = 0, i = -1
    # i = 0  : forward difference
    # i = -1 : backward difference
    #
    grad_x_0 = (data[1, :] - data[0, :]) / (mesh.delta_x_minus)[0, :]
    grad_x_n = (data[-1, :] - data[-2, :]) / (mesh.delta_x_plus)[-1, :]
    # stacking above : lower x values
    grad_x = np.vstack((grad_x_0, grad_x))
    # stacking under : higher x values
    grad_x = np.vstack((grad_x, grad_x_n))
    #
    grad_y_0 = (data[:, 1] - data[:, 0]) / (mesh.delta_y_minus)[:, 0]
    grad_y_n = (data[:, -1] - data[:, -2]) / (mesh.delta_y_plus)[:, -1]
    # stacking left : lower y values
    grad_y = np.hstack((grad_y_0.reshape(-1, 1), grad_y))
    # stacking right : higher y values
    grad_y = np.hstack((grad_y, grad_y_n.reshape(-1, 1)))

    return np.array([grad_x, grad_y])


def divergence_mesh(mesh: Mesh, data: np.ndarray) -> np.ndarray:
    """
    Computes the divergence field by summing up the components of the gradient_mesh output array.
    """
    grads = gradient_mesh(mesh, data)
    return np.sum(grads, axis=0)


def gradient_x_tensor(mesh: Mesh):
    """
    Computes the x-gradient operator according to varying mesh size.
    Inspired from the gradident computation formula form
    numpy.gradient notice.
    Works for non uniform structured meshes.

    Parameters : mesh (Mesh instance)

    Returns : x-gradient operator (ndarray of size nx x ny)
    """
    hd = mesh.delta_x_plus[:, 0]
    hs = mesh.delta_x_minus[:, 0]

    upper_diag = (hs**2 / (hs * hd * (hd + hs)))[:-1]
    main_diag = (hd**2 - hs**2) / (hs * hd * (hd + hs))
    lower_diag = (-(hd**2) / (hs * hd * (hd + hs)))[1:]

    upper_diag = np.pad(upper_diag, (1, 1), mode='constant')
    main_diag = np.pad(main_diag, (1, 1), mode='constant')
    lower_diag = np.pad(lower_diag, (1, 1), mode='constant')

    Lx = diags(
        [lower_diag, main_diag, upper_diag],
        [-1, 0, 1],
        shape=((len(main_diag)), (len(main_diag))),
    )

    # # Convert to LIL format to allow for easy modification at boundary points
    Lx = Lx.tolil()

    # Forward difference at the left boundary
    Lx[0, 0] = -1 / mesh.delta_x_minus[0, 0]
    Lx[0, 1] = 1 / mesh.delta_x_minus[0, 0]
    # Backward difference at the right boundary
    Lx[-1, -1] = 1 / mesh.delta_x_plus[-1, -1]
    Lx[-1, -2] = -1 / mesh.delta_x_plus[-1, -1]
    # print(f"Lx shape {Lx.shape}")
    # additional :
    Lx[1, 0] = -1 / (2.*mesh.delta_x_minus[0, 0])
    Lx[-2, -1] = 1 / (2.*mesh.delta_x_plus[-1, -1])
    # # Construct the 2D Laplacian operator using Kronecker products
    Iy = identity(mesh.ny)

    return kron(Lx, Iy)


def gradient_y_tensor(mesh: Mesh):
    """
    Computes the y-gradient operator according to varying mesh size.
    Inspired from the gradident computation formula form
    numpy.gradient notice.
    Works for non uniform structured meshes.

    Parameters : mesh (Mesh instance)

    Returns : y-gradient operator (ndarray of size nx x ny)
    """
    hd = mesh.delta_y_plus[0, :]
    hs = mesh.delta_y_minus[0, :]
    upper_diag = (hs**2 / (hs * hd * (hd + hs)))[:-1]
    main_diag = (hd**2 - hs**2) / (hs * hd * (hd + hs))
    lower_diag = (-(hd**2) / (hs * hd * (hd + hs)))[1:]

    upper_diag = np.pad(upper_diag, (1, 1), mode='constant')
    main_diag = np.pad(main_diag, (1, 1), mode='constant')
    lower_diag = np.pad(lower_diag, (1, 1), mode='constant')

    Ly = diags(
        [lower_diag, main_diag, upper_diag],
        [-1, 0, 1],
        shape=((len(main_diag)), (len(main_diag))),
    )

    # # Convert to LIL format to allow for easy modification at boundary points
    Ly = Ly.tolil()

    # Forward difference at the left boundary
    Ly[0, 0] = -1 / mesh.delta_y_minus[0, 0]
    Ly[0, 1] = 1 / mesh.delta_y_minus[0, 0]
    # Backward difference at the right boundary
    Ly[-1, -1] = 1 / mesh.delta_y_plus[-1, -1]
    Ly[-1, -2] = -1 / mesh.delta_y_plus[-1, -1]
    # print(f"Ly shape {Ly.shape}")
    # additional :
    Ly[1, 0] = -1 / (2.*mesh.delta_y_minus[0, 0])
    Ly[-2, -1] = 1 / (2.*mesh.delta_y_plus[-1, -1])

    # # Construct the 2D Laplacian operator using Kronecker products
    Ix = identity(mesh.nx)

    return kron(Ix, Ly)


def scalar_laplacian_tensor(mesh: Mesh):
    """
    Computes the scalar laplacian operator i.e. nabla**2.
    Uses gradient_x and y_tensor functions.

    Parameters : mesh (instance of Mesh)

    Returns : scalar laplacian operator (ndarray of size nx x ny)
    """
    return (gradient_x_tensor(mesh) @ gradient_x_tensor(mesh)) + (gradient_y_tensor(mesh) @ gradient_y_tensor(mesh))
