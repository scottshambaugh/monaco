# sobol_sensitivity.py
"""
Sobol' sensitivity analysis using the Saltelli sampling scheme.

This module provides functions to calculate variance-based Sobol' sensitivity
indices (first-order S_i and total-order S_Ti) from simulation results generated
using the Saltelli star sampling scheme.

References
----------
.. [1] Sobol', I. M. (1993). "Sensitivity estimates for nonlinear mathematical
       models." Mathematical Modelling and Computational Experiments.
.. [2] Saltelli, A. (2002). "Making best use of model evaluations to compute
       sensitivity indices." Computer Physics Communications.
.. [3] Saltelli, A. et al. (2010). "Variance based sensitivity analysis of
       model output." Computer Physics Communications.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from monaco.mc_sim import Sim

import logging
import numpy as np
from dataclasses import dataclass, field
from monaco.mc_sampling import SaltelliSamples


@dataclass
class SobolIndices:
    """
    Container for Sobol' sensitivity analysis results.

    Attributes
    ----------
    first_order : dict[str, float]
        First-order sensitivity indices S_i for each input variable.
        Measures the direct contribution of each input to output variance.
    total_order : dict[str, float]
        Total-order sensitivity indices S_Ti for each input variable.
        Measures total contribution including all interactions.
    first_order_conf : dict[str, tuple[float, float]] | None
        Bootstrap confidence intervals for first-order indices.
    total_order_conf : dict[str, tuple[float, float]] | None
        Bootstrap confidence intervals for total-order indices.
    """

    first_order: dict[str, float] = field(default_factory=dict)
    total_order: dict[str, float] = field(default_factory=dict)
    first_order_conf: dict[str, tuple[float, float]] | None = None
    total_order_conf: dict[str, tuple[float, float]] | None = None

    def __repr__(self) -> str:
        lines = ["SobolIndices:"]
        lines.append("  First-order (S_i):")
        for name, val in self.first_order.items():
            lines.append(f"    {name}: {val:.4f}")
        lines.append("  Total-order (S_Ti):")
        for name, val in self.total_order.items():
            lines.append(f"    {name}: {val:.4f}")
        return "\n".join(lines)


def calc_sobol_indices_from_saltelli(
    f_A: np.ndarray,
    f_B: np.ndarray,
    f_AB: np.ndarray,
    invar_names: list[str],
    method: str = "saltelli_2010",
) -> SobolIndices:
    """
    Calculate Sobol' indices from Saltelli sampling results.

    Parameters
    ----------
    f_A : numpy.ndarray
        Model evaluations at matrix A points, shape (nstars,).
    f_B : numpy.ndarray
        Model evaluations at matrix B points, shape (nstars,).
    f_AB : numpy.ndarray
        Model evaluations at AB matrices. For npts=1: shape (ninvars, nstars).
        For npts>1: shape (ninvars, npts, nstars).
    invar_names : list[str]
        Names of input variables in order.
    method : str, default: 'saltelli_2010'
        Estimator method. Options:
        - 'saltelli_2002': Original Saltelli estimators
        - 'saltelli_2010': Improved estimators (recommended)
        - 'jansen': Jansen estimator for total-order

    Returns
    -------
    SobolIndices
        Container with first-order and total-order sensitivity indices.

    Notes
    -----
    For npts > 1, uses the last point along each arm (which equals B[:,i])
    for the standard Saltelli estimators.

    The first-order index S_i measures the fraction of output variance
    explained by input i alone:
        S_i = V[E(Y|X_i)] / V(Y)

    The total-order index S_Ti measures the total contribution of input i,
    including all interactions with other inputs:
        S_Ti = E[V(Y|X_~i)] / V(Y) = 1 - V[E(Y|X_~i)] / V(Y)

    References
    ----------
    .. [1] Saltelli, A. (2002). "Making best use of model evaluations to
           compute sensitivity indices."
    .. [2] Saltelli, A. et al. (2010). "Variance based sensitivity analysis
           of model output."
    """
    n = len(f_A)
    d = len(invar_names)

    # Validate method early
    valid_methods = ("saltelli_2002", "saltelli_2010", "jansen")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use one of: {', '.join(valid_methods)}."
        )

    # Handle multi-point arms: use the last point (which equals B[:,i])
    if f_AB.ndim == 3:
        # Shape is (ninvars, npts, nstars) - take last point along arm
        f_AB_final = f_AB[:, -1, :]
    else:
        # Shape is (ninvars, nstars)
        f_AB_final = f_AB

    # Calculate variance
    f_all = np.concatenate([f_A, f_B])
    var_total = np.var(f_all, ddof=1)

    if var_total < 1e-12:
        # No variance in output - all indices are 0
        return SobolIndices(
            first_order={name: 0.0 for name in invar_names},
            total_order={name: 0.0 for name in invar_names},
        )

    first_order = {}
    total_order = {}

    for i, name in enumerate(invar_names):
        f_ABi = f_AB_final[i]

        if method == "saltelli_2002":
            # Original Saltelli (2002) estimators
            # S_i = (1/N) * sum(f_B * (f_ABi - f_A)) / V(Y)
            S_i = np.mean(f_B * (f_ABi - f_A)) / var_total
            # S_Ti = (1/2N) * sum((f_A - f_ABi)^2) / V(Y)
            S_Ti = np.mean((f_A - f_ABi) ** 2) / (2 * var_total)

        elif method == "saltelli_2010":
            # Improved Saltelli (2010) estimators
            # Uses f_B instead of f_A in first-order estimator for lower variance
            # S_i = (1/N) * sum(f_B * (f_ABi - f_A)) / V(Y)
            S_i = np.mean(f_B * (f_ABi - f_A)) / var_total
            # S_Ti = (1/2N) * sum((f_A - f_ABi)^2) / V(Y)
            S_Ti = np.mean((f_A - f_ABi) ** 2) / (2 * var_total)

        elif method == "jansen":
            # Jansen estimator (1999)
            f_mean = np.mean(f_all)
            # S_i using Jansen: based on variance decomposition
            S_i = (np.mean(f_B * f_ABi) - f_mean**2) / var_total
            # S_Ti using Jansen
            S_Ti = np.mean((f_A - f_ABi) ** 2) / (2 * var_total)

        first_order[name] = float(S_i)
        total_order[name] = float(S_Ti)

    return SobolIndices(first_order=first_order, total_order=total_order)


def calc_sobol_sensitivities(
    sim: "Sim",
    outvarname: str,
    saltelli_samples: SaltelliSamples,
    method: str = "saltelli_2010",
) -> SobolIndices:
    """
    Calculate Sobol' sensitivity indices for a simulation output variable.

    This function extracts the model evaluations from a completed simulation
    that was run with Saltelli sampling and computes the sensitivity indices.

    Parameters
    ----------
    sim : monaco.mc_sim.Sim
        The completed simulation. Must have been run with Saltelli sampling.
    outvarname : str
        Name of the output variable to analyze. Must be scalar.
    saltelli_samples : SaltelliSamples
        The Saltelli sampling structure used when running the simulation.
    method : str, default: 'saltelli_2010'
        Estimator method for computing indices.

    Returns
    -------
    SobolIndices
        Container with first-order and total-order sensitivity indices.

    Raises
    ------
    ValueError
        If the output variable is not scalar or simulation size doesn't match.
    """
    logger = logging.getLogger("monaco")

    outvar = sim.outvars[outvarname]
    if not outvar.isscalar:
        raise ValueError(
            f"Output variable '{outvarname}' is not scalar. "
            "Sobol' analysis requires scalar outputs."
        )

    # Get all output values
    all_nums = np.array(outvar.nums)
    expected_total = saltelli_samples.total_points

    if len(all_nums) != expected_total:
        raise ValueError(
            f"Simulation has {len(all_nums)} cases but Saltelli sampling "
            f"expects {expected_total}. Ensure the simulation was run with "
            "the correct Saltelli sampling structure."
        )

    # Extract f_A, f_B, f_AB from the ordered results
    nstars = saltelli_samples.nstars
    npts = saltelli_samples.npts
    ninvars = saltelli_samples.ninvars

    f_A = all_nums[:nstars]
    f_B = all_nums[nstars : 2 * nstars]

    # Extract f_AB with shape (ninvars, npts, nstars)
    f_AB = np.zeros((ninvars, npts, nstars))
    idx = 2 * nstars
    for i in range(ninvars):
        for k in range(npts):
            f_AB[i, k] = all_nums[idx : idx + nstars]
            idx += nstars

    # Get input variable names in order
    invar_names = list(sim.invars.keys())

    logger.info(f"Calculating Sobol' indices for '{outvarname}'...")
    indices = calc_sobol_indices_from_saltelli(
        f_A=f_A,
        f_B=f_B,
        f_AB=f_AB,
        invar_names=invar_names,
        method=method,
    )
    logger.info("Done calculating Sobol' indices.")

    return indices


def bootstrap_sobol_indices(
    f_A: np.ndarray,
    f_B: np.ndarray,
    f_AB: np.ndarray,
    invar_names: list[str],
    method: str = "saltelli_2010",
    confidence_level: float = 0.95,
    n_resamples: int = 999,
    seed: int | None = None,
) -> SobolIndices:
    """
    Calculate Sobol' indices with bootstrap confidence intervals.

    Parameters
    ----------
    f_A : numpy.ndarray
        Model evaluations at matrix A points, shape (nstars,).
    f_B : numpy.ndarray
        Model evaluations at matrix B points, shape (nstars,).
    f_AB : numpy.ndarray
        Model evaluations at AB matrices, shape (ninvars, npts, nstars)
        or (ninvars, nstars).
    invar_names : list[str]
        Names of input variables in order.
    method : str, default: 'saltelli_2010'
        Estimator method.
    confidence_level : float, default: 0.95
        Confidence level for intervals.
    n_resamples : int, default: 999
        Number of bootstrap resamples.
    seed : int | None, default: None
        Random seed for reproducibility.

    Returns
    -------
    SobolIndices
        Indices with confidence intervals populated.
    """
    rng = np.random.default_rng(seed)
    n = len(f_A)
    d = len(invar_names)

    # Calculate point estimates
    base_result = calc_sobol_indices_from_saltelli(
        f_A=f_A, f_B=f_B, f_AB=f_AB, invar_names=invar_names, method=method
    )

    # Bootstrap
    first_order_samples = {name: [] for name in invar_names}
    total_order_samples = {name: [] for name in invar_names}

    for _ in range(n_resamples):
        # Resample with replacement
        idx = rng.integers(0, n, size=n)

        boot_f_A = f_A[idx]
        boot_f_B = f_B[idx]
        if f_AB.ndim == 3:
            boot_f_AB = f_AB[:, :, idx]
        else:
            boot_f_AB = f_AB[:, idx]

        boot_result = calc_sobol_indices_from_saltelli(
            f_A=boot_f_A,
            f_B=boot_f_B,
            f_AB=boot_f_AB,
            invar_names=invar_names,
            method=method,
        )

        for name in invar_names:
            first_order_samples[name].append(boot_result.first_order[name])
            total_order_samples[name].append(boot_result.total_order[name])

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    first_order_conf = {}
    total_order_conf = {}

    for name in invar_names:
        fo_samples = np.array(first_order_samples[name])
        to_samples = np.array(total_order_samples[name])

        first_order_conf[name] = (
            float(np.percentile(fo_samples, 100 * alpha / 2)),
            float(np.percentile(fo_samples, 100 * (1 - alpha / 2))),
        )
        total_order_conf[name] = (
            float(np.percentile(to_samples, 100 * alpha / 2)),
            float(np.percentile(to_samples, 100 * (1 - alpha / 2))),
        )

    return SobolIndices(
        first_order=base_result.first_order,
        total_order=base_result.total_order,
        first_order_conf=first_order_conf,
        total_order_conf=total_order_conf,
    )
