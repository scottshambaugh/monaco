# dvars_sensitivity.py

'''
Implements the D-VARS algorithm as dscribed in:
Sheikholeslami, Razi, and Saman Razavi. "A fresh look at variography: measuring
dependence and possible sensitivities across geophysical systems from any given
data." Geophysical Research Letters 47.20 (2020): e2020GL089829.

Multi-SQP is replaced with scipy's minimize function implementing L-BFGS-B.
'''

import monaco as mc
import numpy as np
from scipy.optimize import minimize
from numba import jit


def full_states(sim : mc.MCSim) -> tuple:
    X = np.zeros((sim.ncases, sim.ninvars))
    Y = np.zeros((sim.ncases, 1))
    for i, varname in enumerate(sim.mcinvars):
        X[:, i] = sim.mcinvars[varname].pcts
    for i, varname in enumerate(sim.mcoutvars):
        Y[:, i] = sim.mcoutvars[varname].nums
    return X, Y


@jit(nopython=True)
def calc_rj(hj   : float,
            phij : float
            ) -> float:
    r_j = max(0, 1-phij*abs(hj))
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
           Y   : np.ndarray
           ) -> np.ndarray:
    m = len(Y)
    R = np.ones((m, m))
    for u in range(m):
        # do lower triangle only and duplicate across diag
        # diag will be all 1s
        for w in range(u):
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
    R = calc_R(phi, X, Y)
    Rinv = np.linalg.inv(R)
    Rdet = max(np.linalg.det(R), 1e-9)  # Protect for poor conditioning

    mu = np.linalg.inv(M.T @ Rinv @ M) @ (M.T @ Rinv @ Y)

    L_inner = Y - M*mu
    L = np.log(Rdet) + m*np.log(L_inner.T @ Rinv @ L_inner)
    L = L[0][0]
    return L


def L_runner(phi     : np.ndarray,
             X       : np.ndarray,
             Y       : np.ndarray,
             verbose : bool = False
             ) -> float:
    L = mc.calc_L(phi, X, Y)
    mc.vprint(verbose, f'L = {L:0.4f}, Φ = {phi}')
    return L


def calc_Gammaj(Hj       : float,
                dh       : float,
                phij     : np.ndarray,
                variance : float
                ) -> float:
    Gamma = 0
    q = int(np.floor(Hj/dh))
    for t in range(1, q+1):
        Gamma = Gamma + ((1-calc_rj(dh*(t-1), phij)) + (1-calc_rj(dh*t, phij)))
    Gamma = Gamma * dh/2 * variance
    return Gamma


def calc_phi_opt(sim    : mc.MCSim,
                 Hj_min : float
                 ) -> np.ndarray:
    phi_max = 1/Hj_min
    phi_min = 1e-6

    phi0 = [1, 1, 1, 1, 1, 1]
    bounds = []
    X, Y = mc.full_states(sim)
    verbose = True
    for j in range(sim.ninvars):
        bounds.append((phi_min, phi_max))
    res = minimize(L_runner, phi0, args=(X, Y, verbose), bounds=bounds, tol=1e-6)
    phi_opt = res.x

    return phi_opt
