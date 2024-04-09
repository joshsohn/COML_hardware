"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import jax
import jax.numpy as jnp
from utils import hat, vee

# System constants
g_acc = 9.81    # gravitational acceleration
β = (0.1, 1.)   # drag coefficients


def prior(q, dq, g_acc=g_acc):
    """TODO: docstring."""
    nq = 3
    m = 1.3 # kg
    # sinϕ, cosϕ = jnp.sin(q[2]), jnp.cos(q[2])
    H = m*jnp.eye(nq)
    C = jnp.zeros((nq, nq))
    g = m*jnp.array([0., 0., g_acc])
    # R = jnp.array([
    #     [cosϕ, -sinϕ, 0],
    #     [sinϕ,  cosϕ, 0],
    #     [0,    0,     1],
    # ])
    # B = jnp.array([
    #     [-sinϕ, 0, cosϕ],
    #     [cosϕ,  0, sinϕ],
    #     [0,     1,    0],
    # ])
    B = jnp.eye(nq)
    return H, C, g, B


def plant(q, dq, u, f_ext, prior=prior):
    """TODO: docstring."""
    H, C, g, B = prior(q, dq)
    ddq = jax.scipy.linalg.solve(H, f_ext + B@u - C@dq - g, assume_a='pos')
    return ddq


k_R = jnp.array([1400.0, 1400.0, 1260.0])/1000.0
k_Omega = jnp.array([330.0, 330.0, 300.0])/1000.0
J = jnp.diag(jnp.array([0.03, 0.03, 0.09]))

def plant_attitude(R_flatten, Omega, u):
    R = R_flatten.reshape((3,3))

    # f_d = jnp.linalg.norm(u)
    b_3d = -u / jnp.linalg.norm(u)
    b_1d = jnp.array([1, 0, 0])
    cross = jnp.cross(b_3d, b_1d)
    b_2d = cross / jnp.linalg.norm(cross)

    R_d = jnp.column_stack((jnp.cross(b_2d, b_3d), b_2d, b_3d))

    Omega_d = jnp.array([0, 0, 0])
    dOmega_d = jnp.array([0, 0, 0])

    e_R = 0.5 * vee(R_d.T@R - R.T@R_d)
    e_Omega = Omega - R.T@R_d@Omega_d

    M = - k_R*e_R \
        - k_Omega*e_Omega \
        + jnp.cross(Omega, J@Omega) \
        - J@(hat(Omega)@R.T@R_d@Omega_d - R.T@R_d@dOmega_d)

    dOmega = jax.scipy.linalg.solve(J, M - jnp.cross(Omega, J@Omega), assume_a='pos')
    dR = R@hat(Omega)
    dR_flatten = dR.flatten()

    return (dR_flatten, dOmega)

def disturbance(q, dq, w, β=β):
    """TODO: docstring."""
    β = jnp.asarray(β)
    ϕ, dx, dy = q[2], dq[0], dq[1]
    sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
    R = jnp.array([
        [cosϕ, -sinϕ],
        [sinϕ,  cosϕ]
    ])
    v = R.T @ jnp.array([dx - w, dy])
    f_ext = - jnp.array([*(R @ (β * v * jnp.abs(v))), 0.])
    return f_ext

# def ensemble_disturbance(q, dq, R_flatten, Omega, W, b, A):
#     f_ext = jnp.concatenate((q, dq, R_flatten, Omega), axis=0)
#     print(f_ext.shape)
#     print(W.shape)
#     for W, b in zip(W, b):
#         f_ext = jnp.tanh(W@f_ext + b)
#     f_ext = A @ f_ext

#     return f_ext