# mc_sampling.py
from __future__ import annotations

import warnings
import sys
import scipy.stats
import numpy as np
from functools import lru_cache
from dataclasses import dataclass
from monaco.mc_enums import SampleMethod


@dataclass
class SaltelliSamples:
    """
    Container for Saltelli sampling matrices used in Sobol' sensitivity analysis.

    Attributes
    ----------
    A : numpy.ndarray
        Base matrix A, shape (nstars, ninvars). The "star centers".
    B : numpy.ndarray
        Base matrix B, shape (nstars, ninvars). Independent samples for
        total-order estimation.
    AB : numpy.ndarray
        Hybrid matrices, shape (ninvars, npts, nstars, ninvars).
        AB[i, k] is the matrix where column i is interpolated k/(npts) of the
        way from A[:,i] to B[:,i].
    nstars : int
        Number of star centers.
    npts : int
        Number of points per arm of the star.
    ninvars : int
        Number of input variables (dimensions).
    total_points : int
        Total number of sample points: nstars * (2 + ninvars * npts).
    """

    A: np.ndarray
    B: np.ndarray
    AB: np.ndarray
    nstars: int
    npts: int
    ninvars: int

    @property
    def total_points(self) -> int:
        """Total number of sample points."""
        return self.nstars * (2 + self.ninvars * self.npts)

    def get_all_points(self) -> np.ndarray:
        """
        Get all sample points as a single array.

        Returns
        -------
        points : numpy.ndarray
            Shape (total_points, ninvars). Points ordered as:
            A (nstars), B (nstars), AB_0 (npts*nstars), AB_1 (npts*nstars), ...
        """
        points = [self.A, self.B]
        for i in range(self.ninvars):
            for k in range(self.npts):
                points.append(self.AB[i, k])
        return np.vstack(points)

    def get_point_labels(self) -> list[str]:
        """
        Get labels identifying each point's role.

        Returns
        -------
        labels : list[str]
            Labels for each point: 'A', 'B', or 'AB_i_k' where i is the
            dimension and k is the arm point index.
        """
        labels = ["A"] * self.nstars + ["B"] * self.nstars
        for i in range(self.ninvars):
            for k in range(self.npts):
                labels.extend([f"AB_{i}_{k}"] * self.nstars)
        return labels


def saltelli_sampling(
    nstars: int,
    ninvars: int,
    npts: int = 1,
    scramble: bool = True,
    seed: int = 0,
) -> SaltelliSamples:
    """
    Generate Saltelli sampling matrices for Sobol' sensitivity analysis.

    Creates the "star" sampling pattern where each center point has arms
    extending along each dimension. This implements a generalized Saltelli
    scheme that supports multiple points per arm.

    Parameters
    ----------
    nstars : int
        Number of star centers. Should be a power of 2 for best Sobol'
        sequence properties.
    ninvars : int
        Number of input variables (dimensions).
    npts : int, default: 1
        Number of points along each arm of the star. With npts=1, this gives
        the standard Saltelli scheme. Higher values provide finer resolution
        along each dimension.
    scramble : bool, default: True
        Whether to apply Owen's scrambling to the Sobol' sequence.
    seed : int, default: 0
        Random seed for scrambling.

    Returns
    -------
    samples : SaltelliSamples
        Container with matrices A, B, and AB.

    Notes
    -----
    Total function evaluations required: nstars * (2 + ninvars * npts)

    For standard Saltelli (npts=1): nstars * (ninvars + 2)

    The star structure from each center in A:
    - The center point itself (from A)
    - For each dimension i, npts points where only dimension i varies
      (interpolating from A[:,i] toward B[:,i])

    Matrix B is needed for the total-order sensitivity index estimator.

    References
    ----------
    .. [1] Saltelli, A. (2002). "Making best use of model evaluations to
           compute sensitivity indices." Computer Physics Communications.
    .. [2] Sobol', I. M. (1993). "Sensitivity estimates for nonlinear
           mathematical models." Mathematical Modelling and Computational
           Experiments.
    """
    # Generate 2*ninvars dimensions to get independent A and B matrices
    sampler = scipy.stats.qmc.Sobol(d=2 * ninvars, scramble=scramble, seed=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        samples = sampler.random(n=nstars)

    # Split into A and B matrices
    A = samples[:, :ninvars]
    B = samples[:, ninvars:]

    # Create AB hybrid matrices with interpolation for multiple points per arm
    # AB[i, k] is matrix where column i is interpolated k+1 steps toward B
    AB = np.zeros((ninvars, npts, nstars, ninvars))
    for i in range(ninvars):
        for k in range(npts):
            # Interpolation factor: k=0 gives 1/npts, k=npts-1 gives 1.0
            # This ensures the last point equals B[:,i] (standard Saltelli)
            t = (k + 1) / npts
            AB[i, k] = A.copy()
            AB[i, k, :, i] = A[:, i] * (1 - t) + B[:, i] * t

    return SaltelliSamples(
        A=A,
        B=B,
        AB=AB,
        nstars=nstars,
        npts=npts,
        ninvars=ninvars,
    )


def get_saltelli_total_cases(nstars: int, ninvars: int, npts: int = 1) -> int:
    """
    Calculate total number of cases needed for Saltelli sampling.

    Parameters
    ----------
    nstars : int
        Number of star centers.
    ninvars : int
        Number of input variables.
    npts : int, default: 1
        Number of points per arm.

    Returns
    -------
    total : int
        Total cases needed: nstars * (2 + ninvars * npts)
    """
    return nstars * (2 + ninvars * npts)


def sampling(
    ndraws: int,
    method: SampleMethod = SampleMethod.SOBOL_RANDOM,
    ninvar: int | None = None,
    ninvar_max: int | None = None,
    seed: int = np.random.get_state(legacy=False)["state"]["key"][0],
) -> np.ndarray:
    """
    Draws random samples according to the specified method.

    Parameters
    ----------
    ndraws : int
        The number of samples to draw.
    method : monaco.mc_enums.SampleMethod
        The sample method to use.
    ninvar : int
        For all but the 'random' method, must define which number input
        variable is being sampled, ninvar >= 1. The 'sobol' and
        'sobol_random' methods must have ninvar <= 21201
    ninvar_max : int
        The total number of invars, ninvar_max >= ninvar. Used for caching.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random seed. Not used in 'sobol' or 'halton' methods.

    Returns
    -------
    pcts : numpy.ndarray
        The random samples. Each sample is 0 <= pct <= 1.
    """
    if ninvar_max is None:
        ninvar_max = ninvar

    if method == SampleMethod.RANDOM:
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed)

    elif method in (
        SampleMethod.SOBOL,
        SampleMethod.SOBOL_RANDOM,
        SampleMethod.HALTON,
        SampleMethod.HALTON_RANDOM,
        SampleMethod.LATIN_HYPERCUBE,
    ):
        if ninvar is None:
            raise ValueError(f"{ninvar=} must defined for the {method} method")
        elif (not 1 <= ninvar <= 21201) and method in (
            SampleMethod.SOBOL,
            SampleMethod.SOBOL_RANDOM,
        ):
            raise ValueError(f"{ninvar=} must be between 1 and 21201 for the {method} method")

        scramble = False
        if method in (SampleMethod.SOBOL_RANDOM, SampleMethod.HALTON_RANDOM):
            scramble = True
        elif method in (SampleMethod.SOBOL, SampleMethod.HALTON):
            seed = 0  # These do not use randomness, so keep seed constant for caching

        all_pcts = cached_pcts(
            ndraws=ndraws, method=method, ninvar_max=ninvar_max, scramble=scramble, seed=seed
        )
        pcts = all_pcts[:, ninvar - 1]  # ninvar will always be >= 1

    else:
        raise ValueError(
            "".join(
                [
                    f"{method=} must be one of the following: "
                    + f"{SampleMethod.RANDOM}, {SampleMethod.SOBOL}, "
                    + f"{SampleMethod.SOBOL_RANDOM}, {SampleMethod.HALTON}, "
                    + f"{SampleMethod.HALTON_RANDOM}, {SampleMethod.LATIN_HYPERCUBE}"
                ]
            )
        )

    return pcts


@lru_cache(maxsize=1)
def cached_pcts(
    ndraws: int,
    method: str,
    ninvar_max: int,
    scramble: bool,
    seed: int,
) -> np.ndarray:
    """
    Wrapper function to cache the qmc draws so that we don't repeat calculation
    of lower numbered invars for the higher numbered invars.

    Parameters
    ----------
    ndraws : int
        The number of samples to draw.
    method : monaco.mc_enums.SampleMethod
        The sample method to use.
    ninvar_max : int
        The total number of invars.
    scramble : bool
        Whether to scramble the sobol or halton points. Should only be True if
        method is in {'sobol_random', 'halton_random'}
    seed : int
        The random seed. Not used in 'sobol' or 'halton' methods.

    Returns
    -------
    all_pcts : numpy.ndarray
        The random samples. Each sample is 0 <= pct <= 1.
    """
    if method in (SampleMethod.SOBOL, SampleMethod.SOBOL_RANDOM):
        sampler = scipy.stats.qmc.Sobol(d=ninvar_max, scramble=scramble, seed=seed)
    elif method in (SampleMethod.HALTON, SampleMethod.HALTON_RANDOM):
        sampler = scipy.stats.qmc.Halton(d=ninvar_max, scramble=scramble, seed=seed)
    elif method == SampleMethod.LATIN_HYPERCUBE:
        sampler = scipy.stats.qmc.LatinHypercube(d=ninvar_max, seed=seed)

    if not sys.warnoptions:
        with warnings.catch_warnings():
            # Suppress the power of 2 warning for sobol / halton sequences
            warnings.simplefilter("ignore", category=UserWarning)
            points = sampler.random(n=ndraws)
    else:
        points = sampler.random(n=ndraws)

    all_pcts = np.asarray(points)
    return all_pcts
