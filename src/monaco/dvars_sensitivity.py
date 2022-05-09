# dvars_sensitivity.py

'''
Implements the D-VARS algorithm as dscribed in:
Sheikholeslami, Razi, and Saman Razavi. "A fresh look at variography: measuring
dependence and possible sensitivities across geophysical systems from any given
data." Geophysical Research Letters 47.20 (2020): e2020GL089829.

Multi-SQP is replaced with scipy's minimize function implementing L-BFGS-B.
'''

from monaco.mc_sim import Sim
from monaco.helper_functions import vprint
import numpy as np
from scipy.optimize import minimize
from numba import jit


def full_states(sim : Sim,
                outvarname: str,
                ) -> tuple:
    X = np.zeros((sim.ncases, sim.ninvars))
    Y = np.zeros((sim.ncases, 1))
    for i, varname in enumerate(sim.invars):
        X[:, i] = sim.invars[varname].pcts
    Y[:, 0] = sim.outvars[outvarname].nums
    return X, Y


@jit(nopython=True)
def calc_rj(hj   : float,
            phij : float
            ) -> float:
    r_j = max(0, 1-phij*abs(hj))  # linear covariance function
    # r_j = np.exp(-(abs(hj)/phij))  # exponential covariance function
    # r_j = np.exp(-(hj/phij)**2)  # squared exponential covariance function
    return r_j


@jit(nopython=True)
def calc_Ruw(phi : np.ndarray,
             Xu  : np.ndarray,
             Xw  : np.ndarray
             ) -> float:
    h = Xu-Xw
    Ruw = 1
    for j, hj in enumerate(h):
        Ruw = Ruw*calc_rj(hj, phi[j])
    return Ruw


@jit(nopython=True)
def calc_R(phi : np.ndarray,
           X   : np.ndarray,
           ) -> np.ndarray:
    m = X.shape[0]
    R = np.ones((m, m))
    for u in range(1, m):
        # do lower triangle only and duplicate across diag
        # diag will be all 1s
        for w in range(1, u):
            Ruw = calc_Ruw(phi, X[u, :], X[w, :])
            R[u, w] = Ruw
            R[w, u] = Ruw
    return R


@jit(nopython=True)
def calc_L(phi : np.ndarray,
           X   : np.ndarray,
           Y   : np.ndarray
           ) -> float:
    m = len(Y)
    M = np.ones((m, 1))
    R = calc_R(phi, X)
    Rinv = np.linalg.inv(R)
    Rdet = max(np.linalg.det(R), 1e-12)  # Protect for poor conditioning

    mu = np.linalg.inv(M.T @ Rinv @ M) @ (M.T @ Rinv @ Y)

    L_inner = Y - M*mu
    L = np.log(Rdet)/m + m*np.log(L_inner.T @ Rinv @ L_inner)
    L = L[0][0]
    return L


def L_runner(phi     : np.ndarray,
             X       : np.ndarray,
             Y       : np.ndarray,
             verbose : bool = False
             ) -> float:
    L = calc_L(phi, X, Y)
    vprint(verbose, f'L = {L:0.4f}, Φ = {phi}')
    return L


def calc_phi_opt(sim        : Sim,
                 outvarname : str,
                 tol        : float = 1e-6,
                 ) -> np.ndarray:
    phi_max = 1e6
    phi_min = 0
    phi0 = 1

    phi0s = [phi0, phi0, phi0, phi0, phi0, phi0]
    bounds = []
    for _ in range(sim.ninvars):
        bounds.append((phi_min, phi_max))

    vprint(sim.verbose, 'Calculating optimal covariance hyperparameters Φ for ' +
                       f"'{outvarname}' covariances...")
    X, Y = full_states(sim, outvarname)
    method = 'L-BFGS-B'
    res = minimize(L_runner, phi0s, args=(X, Y, sim.verbose), bounds=bounds,
                   tol=tol, method=method)
    phi_opt = res.x
    vprint(sim.verbose, 'Done calculating optimal hyperparameters')

    return phi_opt


def calc_Gammaj(Hj       : float,
                phij     : np.ndarray,
                variance : float
                ) -> float:

    dh = 1e-3
    q = int(np.floor(Hj/dh))
    rjs = []
    for i in range(q+1):
        rjs.append(calc_rj(dh*i, phij))
    rjs = np.array(rjs)

    Gamma = np.trapz(1 - rjs) * dh * variance

    return Gamma


def calc_sensitivities(sim        : Sim,
                       outvarname : str,
                       Hj         : float = 1.0,
                       tol        : float = 1e-6,
                       ) -> np.ndarray:
    phi_opt = calc_phi_opt(sim, outvarname, tol)
    variance = np.var(np.array([sim.outvars[outvarname].nums]))
    sensitivities = []
    for j in range(sim.ninvars):
        sensitivities.append(calc_Gammaj(Hj, phi_opt[j], variance))
    sensitivities = np.array(sensitivities)
    ratios = sensitivities/sum(sensitivities)

    return sensitivities, ratios
