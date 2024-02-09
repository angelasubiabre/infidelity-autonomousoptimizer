# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:18:34 2023

@author: Angela
"""

import copy

import numpy as np
import scipy.linalg
import scipy.stats
import torch
from matplotlib import pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch import linalg
import autonomous_optimizer

import torch.nn as nn
import torch.optim as optim

def infidelity():

    #genera parámetros de un estado puro: Re(alpha_rho), Im(alpha_rho),
    # Re(beta_rho), Im(beta_rho)
    def random_state():
        dim=4
        #torch.manual_seed(123456)
        state = torch.rand(dim)
        state=state/((state[0]**2)+(state[1]**2)+(state[2]**2)+(state[3]**2)).pow(1/2)
        state = torch.tensor(state, dtype=torch.float,requires_grad=True)
        return state
    rho=random_state()
    x = torch.eye(4,dtype=torch.float)
    model0 =nn.Linear(4,1)
    def infidelity0(model):
        sigma = model(x).view(-1)
        sigma=sigma/((sigma[0]**2)+(sigma[1]**2)+(sigma[2]**2)+(sigma[3]**2)).pow(1/2)
        fidelities=((rho[0]*sigma[0]+sigma[1]*rho[1]+rho[2]*sigma[2]+
                     rho[3]*sigma[3])**2)+((rho[0]*sigma[1]-rho[1]*sigma[0]+
                                                rho[2]*sigma[3]-rho[3]*sigma[2])**2)                                  
        infidelities=(1-fidelities)
        return infidelities
    return {"model0": model0, "obj_function": infidelity0}



# códigos originales https://github.com/stewy33/Learning-to-Optimize   
def run_optimizer(make_optimizer, problem, iterations, hyperparams):
    # Initial solution
    model = copy.deepcopy(problem["model0"])
    obj_function = problem["obj_function"]

    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **hyperparams)

    # We will keep track of the objective values and weight trajectories
    # throughout the optimization process.
    values = []
    trajectory = []

    # Passed to optimizer. This setup is required to give the autonomous
    # optimizer access to the objective value and not just its gradients.
    def closure():
        trajectory.append(copy.deepcopy(model))
        optimizer.zero_grad()

        obj_value = obj_function(model)
        obj_value.backward()

        values.append(obj_value.item())
        return obj_value

    # Minimize
    for i in range(iterations):
        optimizer.step(closure)

        # Stop optimizing if we start getting nans as objective values
        if np.isnan(values[-1]) or np.isinf(values[-1]):
            break

    return np.nan_to_num(values, 1e6), trajectory


def run_all_optimizers(problem, iterations, tune_dict, policy):
    # SGD
    sgd_vals, sgd_traj = run_optimizer(
        torch.optim.SGD, problem, iterations, tune_dict["sgd"]["hyperparams"]
    )
    print(f"SGD best loss: {sgd_vals.min()}")

    # Momentum
    momentum_vals, momentum_traj = run_optimizer(
        torch.optim.SGD, problem, iterations, tune_dict["momentum"]["hyperparams"]
    )
    print(f"Momentum best loss: {momentum_vals.min()}")

    # Adam
    adam_vals, adam_traj = run_optimizer(
        torch.optim.Adam, problem, iterations, tune_dict["adam"]["hyperparams"]
    )
    print(f"Adam best loss: {adam_vals.min()}")

    # LBFGS
    lbfgs_vals, lbfgs_traj = run_optimizer(
        torch.optim.LBFGS, problem, iterations, tune_dict["lbfgs"]["hyperparams"]
    )
    print(f"LBFGS best loss: {lbfgs_vals.min()}")

    # Autonomous optimizer
    ao_vals, ao_traj = run_optimizer(
        autonomous_optimizer.AutonomousOptimizer,
        problem,
        iterations,
        {"policy": policy},
    )
    print(f"Autonomous Optimizer best loss: {ao_vals.min()}")

    return {
        "sgd": (sgd_vals, sgd_traj),
        "momentum": (momentum_vals, momentum_traj),
        "adam": (adam_vals, adam_traj),
        "lbfgs": (lbfgs_vals, lbfgs_traj),
        "ao": (ao_vals, ao_traj),
    }


def plot_trajectories(trajectories, problem, get_weights, set_weights):
    """Plot optimization trajectories on top of a contour plot.

    Parameters:
        trajectories (List(nn.Module))
        problem (dict)
        get_weights (Callable[[], Tuple[float, float]])
        set_weights (Callable[[float, float], None])

    """
    data = {}
    for name, traj in trajectories.items():
        data[name] = np.array([get_weights(model) for model in traj])

    xmin = min(np.array(d)[:, 0].min() for d in data.values())
    ymin = min(np.array(d)[:, 1].min() for d in data.values())
    xmax = max(np.array(d)[:, 0].max() for d in data.values())
    ymax = max(np.array(d)[:, 1].max() for d in data.values())

    X = np.linspace(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2)
    Y = np.linspace(ymin - (ymax - ymin) * 0.2, ymax + (ymax - ymin) * 0.2)

    model = copy.deepcopy(problem["model0"])
    Z = np.empty((len(Y), len(X)))
    for i in range(len(X)):
        for j in range(len(Y)):
            set_weights(model, X[i], Y[j])
            Z[j, i] = problem["obj_function"](model)

    plt.figure(figsize=(10, 6), dpi=500)
    plt.contourf(X, Y, Z, 30, cmap="RdGy")
    plt.colorbar()

    for name, traj in data.items():
        plt.plot(traj[:, 0], traj[:, 1], label=name)

    plt.title("Sigma Trajectory Plot")
    plt.plot(*get_weights(problem["model0"]), "bo")
    plt.legend()

    plt.plot()
    plt.show()