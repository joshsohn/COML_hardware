import argparse
import os
import pickle
from itertools import product

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import jax.debug as jdb

import numpy as np

from tqdm.auto import tqdm

from dynamics import prior, disturbance, plant
from utils import params_to_posdef
from utils import random_ragged_spline, spline
from utils import (tree_normsq, rk38_step, epoch,   # noqa: E402
                   odeint_fixed_step, random_ragged_spline, spline,
            params_to_cholesky, params_to_posdef)

from functools import partial
import matplotlib.pyplot as plt
import csv



def convert_p_qbar(p):
    return np.sqrt(1/(1 - 1/p) - 1.1)

def convert_qbar_p(qbar):
    return 1/(1 - 1/(1.1 + qbar**2))

# Support functions for generating loop reference trajectory
def reference(t):
    T = 10.            # loop period
    d = 4.             # displacement along `x` from `t=0` to `t=T`
    w = 4.             # loop width
    h = 6.             # loop height
    ϕ_max = jnp.pi/3   # maximum roll angle (achieved at top of loop)

    x = (w/2)*jnp.sin(2*jnp.pi * t/T) + d*(t/T)
    y = (h/2)*(1 - jnp.cos(2*jnp.pi * t/T))
    ϕ = 4*ϕ_max*(t/T)*(1-t/T)
    r = jnp.array([x, y, ϕ])
    return r

def test_simulate(ts, w, params, reference,
                plant=plant, prior=prior, disturbance=disturbance, NGD_flag=False):
    """TODO: docstring."""
    # Required derivatives of the reference trajectory
    def ref_derivatives(t):
        ref_vel = jax.jacfwd(reference)
        ref_acc = jax.jacfwd(ref_vel)
        r = reference(t)
        dr = ref_vel(t)
        ddr = ref_acc(t)
        return r, dr, ddr

    # Adaptation law
    def adaptation_law(q, dq, r, dr, params=params, NGD_flag=NGD_flag):
        # Regressor features
        y = jnp.concatenate((q, dq))
        for W, b in zip(params['W'], params['b']):
            y = jnp.tanh(W@y + b)

        # Auxiliary signals
        Λ, P = params['Λ'], params['P']
        e, de = q - r, dq - dr
        s = de + Λ@e

        if NGD_flag:
            dA = jnp.outer(s, y) @ P
        else:
            dA = P @ jnp.outer(s, y)
        return dA, y

    # Controller
    def controller(q, dq, r, dr, ddr, f_hat, params=params):
        # Auxiliary signals
        Λ, K = params['Λ'], params['K']
        e, de = q - r, dq - dr
        s = de + Λ@e
        v, dv = dr - Λ@e, ddr - Λ@de

        # Control input and adaptation law
        H, C, g, B = prior(q, dq)
        τ = H@dv + C@v + g - f_hat - K@s
        u = jnp.linalg.solve(B, τ)
        return u, τ

    # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on
    # the controller
    def ode(x, t, u, w=w):
        q, dq = x
        f_ext = disturbance(q, dq, w)
        ddq = plant(q, dq, u, f_ext)
        dx = (dq, ddq)
        return dx

    # Simulation loop
    def loop(carry, input_slice, params=params):
        t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev, pA_prev = carry
        t = input_slice
        qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]),
                            u_prev)
        q, dq = qs[-1], dqs[-1]

        r, dr, ddr = ref_derivatives(t)

        if NGD_flag:
            qn = 1.1 + params['pnorm']**2

            # Integrate adaptation law via trapezoidal rule
            dA, y = adaptation_law(q, dq, r, dr)
            pA = pA_prev + (t - t_prev)*(dA_prev + dA)/2
            # A = (jnp.maximum(jnp.abs(pA), 1e-6 * jnp.ones_like(pA))**(qn-1) * jnp.sign(pA)* (jnp.ones_like(pA) - jnp.isclose(pA, 0, atol=1e-6)) ) @ params['P']
            A = jnp.abs(pA)**(qn-1) * jnp.sign(pA) @ params['P']
        else:
            # Integrate adaptation law via trapezoidal rule
            dA, y = adaptation_law(q, dq, r, dr)
            A = A_prev + (t - t_prev)*(dA_prev + dA)/2
            pA = pA0

        # Compute force estimate and control input
        f_hat = A @ y
        u, τ = controller(q, dq, r, dr, ddr, f_hat)

        f_ext = disturbance(q, dq, w)

        carry = (t, q, dq, u, A, dA, pA)
        flat_A = A.flatten()
        output_slice = (q, dq, u, τ, r, dr, f_hat, f_ext, y, flat_A)
        return carry, output_slice

    # Initial conditions
    t0 = ts[0]
    r0, dr0, ddr0 = ref_derivatives(t0)
    q0, dq0 = r0, dr0
    dA0, y0 = adaptation_law(q0, dq0, r0, dr0)
    A0 = jnp.zeros((q0.size, y0.size))
    pA0 = jnp.ones((q0.size, y0.size))
    f0 = A0 @ y0
    u0, τ0 = controller(q0, dq0, r0, dr0, ddr0, f0)
    f_ext0 = disturbance(q0, dq0, w)

    flat_A0 = A0.flatten()

    # Run simulation loop
    carry = (t0, q0, dq0, u0, A0, dA0, pA0)
    carry, output = jax.lax.scan(loop, carry, ts[1:])
    q, dq, u, τ, r, dr, f_hat, f_ext, y, flat_A = output

    # Prepend initial conditions
    q = jnp.vstack((q0, q))
    dq = jnp.vstack((dq0, dq))
    u = jnp.vstack((u0, u))
    τ = jnp.vstack((τ0, τ))
    r = jnp.vstack((r0, r))
    dr = jnp.vstack((dr0, dr))
    f_hat = jnp.vstack((f0, f_hat))
    f_ext = jnp.vstack((f_ext0, f_ext))
    flat_A = jnp.vstack((flat_A0, flat_A))
    y = jnp.vstack((y0, y))

    sim = {"q": q, "dq": dq, "u": u, "τ": τ, "r": r, "dr": dr, "f_hat": f_hat, "f_ext": f_ext, "y": y, "A": flat_A}

    return sim

def eval_single_model(model_dir, filename, T, dt, w, pnorm_flag=True):
    model_pkl_loc = os.path.join(model_dir, filename)
    with open(model_pkl_loc, 'rb') as f:
        train_results = pickle.load(f)

    test_results = {}
    test_params = {}  

    parts = filename.replace('.pkl', '').split('_')
        
    # Dictionary to hold the attributes for this file
    test_results['train_params'] = {}
    # Loop through each part of the filename
    for part in parts:
        # Split each part by '=' to separate the key and value
        key, value = part.split('=')
        # Convert value to float if it looks like a number, else keep as string
        try:
            test_results['train_params'][key] = float(value)
        except ValueError:
            test_results['train_params'][key] = value

    # Post-process training loss information
    # train_aux = train_results['train_lossaux_history']
    # train_loss_history = jnp.zeros(train_results['train_params']['E'])
    # for i in range(test_results['train_params']['E']):
    #     train_loss_history[i] = train_aux[i]['tracking_loss'] + 1e-3 * train_aux[i]['control_loss'] + 1e-4 * train_aux[i]['l2_penalty'] + test_results['train_params']['regP'] * train_aux[i]['reg_P_penalty']

    test_results['train_info'] = {
        'best_step_meta': train_results['best_step_meta'],
        'ensemble': train_results['ensemble'],
        'valid_loss_history': train_results['valid_loss_history'],
        # 'train_loss_history': train_results['train_loss_history'],
        'pnorm_history': train_results['pnorm_history']
    } 

    if pnorm_flag:
        test_results['final_p'] = train_results['pnorm']
        # Note that the pnorm stored in pickle is the actual p
        # To run evaluation script, we convert params['pnorm'] to qbar
        test_params['pnorm'] = convert_p_qbar(train_results['pnorm'])
    else:
        test_results['final_p'] = 2.0
    

    # Store the model parameters
    test_params['W'] = train_results['model']['W']
    test_params['b'] = train_results['model']['b']
    test_params['Λ'] = params_to_posdef(train_results['controller']['Λ'])
    test_params['K'] = params_to_posdef(train_results['controller']['K'])
    test_params['P'] = params_to_posdef(train_results['controller']['P'])

    # Test on new trajectories
    ts = jnp.arange(0, T, dt)
    sim = test_simulate(ts, w, test_params, reference, NGD_flag=pnorm_flag)

    sim_e = sim['q'] - sim['r']
    tracking_error = jnp.mean(jnp.linalg.norm(sim_e, axis=1))
    sim_ftilde = sim['f_hat'] - sim['f_ext']
    estimation_error = jnp.mean(jnp.linalg.norm(sim_ftilde, axis=1))

    test_results['tracking_err'] = tracking_error
    test_results['estimation_err'] = estimation_error

    return sim, test_params, test_results

def eval_dir_models(model_dir, T, dt, w, pnorm_flag=True):
    csv_file_path = os.path.join(model_dir, 'model_test_results.csv')
    Header = ['Seed', 'M', 'Epoch', 'init_p', 'p_freq', 'reg_P', 'best_step', 'final_p', 'Test Tracking Error', 'Test Estimation Error']
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(Header)

        for filename in tqdm(os.listdir(model_dir)):
            if filename.endswith(".pkl"):
                model_sim, model_test_params, model_test_results = eval_single_model(model_dir, filename, T, dt, w, pnorm_flag)
                # Write the results to the csv file
                row = [model_test_results['train_params']['seed'], model_test_results['train_params']['M'], model_test_results['train_params']['E'], model_test_results['train_params']['pinit'], model_test_results['train_params']['pfreq'], model_test_results['train_params']['regP'], model_test_results['train_info']['best_step_meta'], model_test_results['final_p'], model_test_results['tracking_err'], model_test_results['estimation_err']]
                writer.writerow(row)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='set model directory', type=str)
    parser.add_argument('--T', help='set simulation time', type=float)
    parser.add_argument('--dt', help='set simulation time step', type=float)
    parser.add_argument('--w', help='set disturbance wind speed', type=float)
    parser.add_argument('--pnorm', help='set pnorm flag', action='store_true')
    args = parser.parse_args()
    eval_dir_models(args.model_dir, args.T, args.dt, args.w, args.pnorm)