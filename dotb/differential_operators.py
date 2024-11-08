#!/usr/bin/python3
from __future__ import annotations

import numpy as np

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
