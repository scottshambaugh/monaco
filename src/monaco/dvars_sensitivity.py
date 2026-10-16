# dvars_sensitivity.py
from __future__ import annotations

# Somewhat hacky type checking to avoid circular imports:
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from monaco.mc_sim import Sim

import logging
import numpy as np
from scipy.optimize import minimize
from warnings import warn
from monaco.helper_functions import get_cases

# numba is recommended for speed, as this will be very slow otherwise
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # create a dummy decorator
    def jit(f=None, *args, **kwargs):
        def decorator(func):
            return func

        if callable(f):
            return f
        else:
            return decorator


def calc_sensitivities(
    sim: "Sim",
    outvarname: str,
    cases: None | int | Iterable[int] = None,
    Hj: float = 1.0,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the global sensitivity indices and ratios for a specific output
    variable to each of a simulation's input variables.

    This implements the D-VARS algorithm as dscribed in Reference [1]_ to
    calculate global sensitivity indices from a set of given data. The
    implementation largely follows the Reference's supplementary notes, though
    Multi-SQP is replaced with scipy's minimize function implementing L-BFGS-B.
    See also Reference [2]_ for some theoretical background on the VARS and
    IVARS methods of calculating sensitivity indices.

    Parameters
    ----------
    sim : monaco.mc_sim.Sim
        The input simulation.
    outvarname : str
        The name of the output variable to calculate sensitivities for. Note
        that the output variable must be scalar.
    cases : None | int | Iterable[int], default None
        The cases to use for the sensitivity analysis. If None, all cases will
        be used.
    Hj : float, default: 1.0
        The fraction of the total parameter space to integrate over. Note that
        the linear correlation function only has one hyperparameter, so as the
        Reference notes it is unable to distinguish variogram effects at varying
        length scales. So, this should not be set to anything other than 1 in
        practice.
    tol : float, default 1e-6
        The convergence tolerance for scipy's minimize function acting on the
        negative log likelihood function.
    verbose : bool, default False
        Whether to print diagnostic information.

    Returns
    -------
    sensitivities : numpy.ndarray
        The global sensitivity indices for the output variable for each of the
        sim's input variables.
    ratios : numpy.ndarray
        The global sensitivity ratios for the output variable for each of the
        sim's input variables, essentially the fraction of each input variable's
        ability to explain the output variance.

    References
    ----------
    .. [1] Sheikholeslami, Razi, and Saman Razavi. "A fresh look at variography:
           measuring dependence and possible sensitivities across geophysical
           systems from any given data." Geophysical Research Letters 47.20
           (2020): e2020GL089829.
    .. [2] Razavi, Saman, and Hoshin V. Gupta. "A new framework for
           comprehensive, robust, and efficient global sensitivity analysis:
           1. Theory." Water Resources Research 52.1 (2016): 423-439.
    """
    if not HAS_NUMBA:
        warn("Consider installing numba for better performance", UserWarning)

    cases = get_cases(ncases=sim.ncases, cases=cases)
    ncases = len(cases)
    if ncases != sim.ncases:
        warn(f"Only using {ncases} of {sim.ncases} cases for D-VARS sensitivity analysis")

    phi_opt = calc_phi_opt(sim, outvarname, cases, tol, verbose)

    variance = np.var(np.asarray(sim.outvars[outvarname].nums)[cases])
    sensitivities = np.zeros(sim.ninvars)
    for j in range(sim.ninvars):
        sensitivities[j] = calc_Gammaj(Hj, phi_opt[j], variance)
    sensitivities_sum = sum(sensitivities)
    if sensitivities_sum == 0:
        ratios = np.zeros(sim.ninvars)
    else:
        ratios = sensitivities / sensitivities_sum

    return sensitivities, ratios


def calc_phi_opt(
    sim: "Sim", outvarname: str, cases: list[int], tol: float = 1e-6, verbose: bool = False
) -> np.ndarray:
    """
    Calculate the optimal hyperparameters for the covariance functions between
    the output variable and each of the input variables via maximum likelihood
    estimation (MLE). MLE works by minimizing a negative log-likelihood
    function.

    Parameters
    ----------
    sim : monaco.mc_sim.Sim
        The input simulation.
    outvarname : str
        The name of the output variable to calculate sensitivities for. Note
        that the output variable must be scalar.
    cases : list[int]
        The cases to use for the sensitivity analysis.
    tol : float, default 1e-6
        The convergence tolerance for scipy's minimize function acting on the
        negative log likelihood function.
    verbose : bool, default False
        Whether to print diagnostic information.

    Returns
    -------
    phi_opt : numpy.ndarray
        The learned hyperparameters for the covariance functions.
    """
    phi_max = 1e6
    phi_min = 0
    phi0 = 1

    phi0s = []
    bounds = []
    for _ in range(sim.ninvars):
        phi0s.append(phi0)
        bounds.append((phi_min, phi_max))

    logger = logging.getLogger("monaco")
    logger.info("Calculating optimal hyperparameters Φ for " + f"'{outvarname}' covariances...")
    X, Y = full_states(sim, outvarname, cases)
    D = calc_dists(X)
    res = minimize(
        L_runner, phi0s, args=(X, Y, verbose, D), bounds=bounds, tol=tol, method="L-BFGS-B"
    )
    phi_opt = res.x
    logger.info("Done calculating optimal hyperparameters.")

    return phi_opt


def full_states(sim: "Sim", outvarname: str, cases: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the full input and output states in combined matrices.

    Parameters
    ----------
    sim : monaco.mc_sim.Sim
        The input simulation.
    outvarname : str
        The name of the output variable to calculate sensitivities for. Note
        that the output variable must be scalar.
    cases : list[int]
        The cases to use for the sensitivity analysis.

    Returns
    -------
    X : numpy.ndarray
        The input variables' percentiles.
    Y : numpy.ndarray
        The output variable's nums.
    """
    ncases = len(cases)
    X = np.zeros((ncases, sim.ninvars))
    Y = np.zeros((ncases, 1))
    for i, varname in enumerate(sim.invars):
        X[:, i] = np.asarray(sim.invars[varname].pcts)[cases]
    Y[:, 0] = np.asarray(sim.outvars[outvarname].nums)[cases]
    return X, Y


def calc_Gammaj(Hj: float, phij: float, variance: float) -> float:
    """
    Calculates the IVARS sensitivity index from a learned covariance function.

    This integrates the directional variogram for a specific input variable
    using trapezoidal integration.

    Parameters
    ----------
    Hj : float
        The fraction of the total parameter space to integrate over. Note that
        the linear correlation function only has one hyperparameter, so as the
        Reference notes it is unable to distinguish variogram effects at varying
        length scales. So, this should not be set to anything other than 1 in
        practice.
    phij : float
        The learned hyperparameter for the covariance function between the
        output and input variable.
    variance : float
        The variance of the output variable.

    Returns
    -------
    Gammaj : float
        The global sensitivity index for this output-input variable pair.
    """
    dh = 1e-3
    q = int(np.floor(Hj / dh))
    hjs = dh * np.arange(q + 1)
    rjs = np.maximum(0, 1 - phij * np.abs(hjs))

    Gammaj = np.trapezoid(1 - rjs) * dh * variance
    return Gammaj


def L_runner(
    phi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    verbose: bool = False,
    D: np.ndarray | None = None,
) -> float:
    """
    A wrapper function for calculating the negative log-likelihood cost.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.
    Y : numpy.ndarray
        The state matrix for the output variables nums.
    verbose : bool
        Whether to print the values at each step.
    D : numpy.ndarray | None, default None
        Optional precomputed pairwise distances from calc_dists.

    Returns
    -------
    L : float
        The negative log-likelihood cost.
    """
    L = calc_L(phi, X, Y, D)
    if verbose:
        logger = logging.getLogger("monaco")
        logger.info(f"L = {L:0.4f}, Φ = {phi}")
    return L


def calc_dists(X: np.ndarray) -> np.ndarray | None:
    """
    Precompute the per-dimension absolute pairwise distances between each of
    the input states, for reuse across likelihood evaluations.

    Parameters
    ----------
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.

    Returns
    -------
    D : numpy.ndarray | None
        Array of shape (ninvars, m, m) where D[j, u, w] = |X[u, j] - X[w, j]|,
        or None if it would be larger than 256 MiB.
    """
    m, d = X.shape
    if d * m * m * 8 > 256 * 1024**2:
        return None
    D = np.empty((d, m, m), dtype=np.float64)
    for j in range(d):
        D[j] = np.abs(X[:, None, j] - X[None, :, j])
    return D


def calc_L(phi: np.ndarray, X: np.ndarray, Y: np.ndarray, D: np.ndarray | None = None) -> float:
    """
    Calculate the negative log-likelihood cost.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.
    Y : numpy.ndarray
        The state matrix for the output variables nums.
    D : numpy.ndarray | None, default None
        Optional precomputed pairwise distances from calc_dists.

    Returns
    -------
    L : float
        The negative log-likelihood cost.
    """
    m = len(Y)
    M = np.ones((m, 1))
    R = calc_R(phi, X, D)
    Rinv = np.linalg.inv(R)
    Rdet = max(np.linalg.det(R), 1e-12)  # Protect for poor conditioning

    mu = np.linalg.inv(M.T @ Rinv @ M) @ (M.T @ Rinv @ Y)

    L_inner = Y - M * mu
    L = np.log(Rdet) + m * np.log(L_inner.T @ Rinv @ L_inner)
    L = L[0][0]
    return L


def calc_R(
    phi: np.ndarray,
    X: np.ndarray,
    D: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the correlation matrix between each of the input states. This is
    the elementwise product over each input variable of the covariance function
    calc_rj applied to the pairwise distances.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    X : numpy.ndarray
        The state matrix for all the input variables percentiles.
    D : numpy.ndarray | None, default None
        Optional precomputed pairwise distances from calc_dists. If None, the
        distances are computed on the fly one input variable at a time.

    Returns
    -------
    R : float
        The correlation matrix.
    """
    m, d = X.shape
    R = np.ones((m, m), dtype=np.float64)
    rj = np.empty((m, m), dtype=np.float64)
    for j in range(d):
        if D is None:
            np.subtract(X[:, None, j], X[None, :, j], out=rj)
            np.abs(rj, out=rj)
            np.multiply(rj, phi[j], out=rj)
        else:
            np.multiply(D[j], phi[j], out=rj)
        np.subtract(1, rj, out=rj)
        np.maximum(rj, 0, out=rj)
        R *= rj
    np.fill_diagonal(R, 1)
    return R


@jit(nopython=True, cache=True)
def calc_Ruw(phi: np.ndarray, Xu: np.ndarray, Xw: np.ndarray) -> float:
    """
    Calculate the correlation between two input states.
    Note that this is just-in-time compiled by numba for increased speed.

    Parameters
    ----------
    phi : numpy.ndarray
        The hyperparameters for the covariance function.
    Xu : numpy.ndarray
        The u'th input state.
    Xw : numpy.ndarray
        The w'th input state.

    Returns
    -------
    Ruw : float
        The correlation between the two states.
    """
    h = Xu - Xw
    Ruw = 1
    for j, hj in enumerate(h):
        Ruw = Ruw * calc_rj(hj, phi[j])
    return Ruw


@jit(nopython=True, cache=True)
def calc_rj(hj: float, phij: float) -> float:
    """
    The covariance function (also called a kernel). We currently use a linear
    kernel which has a single hyperparameter that must be learned.
    Note that this is just-in-time compiled by numba for increased speed.

    Parameters
    ----------
    hj : float
        The distance between two state elements.
    phij : float
        The hyperparameter for the function.

    Returns
    -------
    rj : float
        The covariance.
    """
    rj = max(0, 1 - phij * abs(hj))  # linear covariance function
    # rj = np.exp(-(abs(hj)/phij))  # exponential covariance function
    # rj = np.exp(-(hj/phij)**2)  # squared exponential covariance function
    return rj
