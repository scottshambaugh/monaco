# mc_sampling.py
from __future__ import annotations

import scipy.stats
import numpy as np
from functools import lru_cache
import warnings
import sys
from monaco.mc_enums import SampleMethod


def sampling(ndraws     : int,
             method     : SampleMethod = SampleMethod.SOBOL_RANDOM,
             ninvar     : int | None   = None,
             ninvar_max : int | None   = None,
             seed       : int          = np.random.get_state(legacy=False)['state']['key'][0],
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

    elif method in (SampleMethod.SOBOL, SampleMethod.SOBOL_RANDOM,
                    SampleMethod.HALTON, SampleMethod.HALTON_RANDOM, SampleMethod.LATIN_HYPERCUBE):
        if ninvar is None:
            raise ValueError(f'{ninvar=} must defined for the {method} method')
        elif (not 1 <= ninvar <= 21201) and method in (SampleMethod.SOBOL,
                                                       SampleMethod.SOBOL_RANDOM):
            raise ValueError(f'{ninvar=} must be between 1 and 21201 for the {method} method')

        scramble = False
        if method in (SampleMethod.SOBOL_RANDOM, SampleMethod.HALTON_RANDOM):
            scramble = True
        elif method in (SampleMethod.SOBOL, SampleMethod.HALTON):
            seed = 0  # These do not use randomness, so keep seed constant for caching

        all_pcts = cached_pcts(ndraws=ndraws, method=method, ninvar_max=ninvar_max,
                               scramble=scramble, seed=seed)
        pcts = all_pcts[:, ninvar-1]  # ninvar will always be >= 1

    else:
        raise ValueError("".join([f'{method=} must be one of the following: ' +
                                  f'{SampleMethod.RANDOM}, {SampleMethod.SOBOL}, ' +
                                  f'{SampleMethod.SOBOL_RANDOM}, {SampleMethod.HALTON}, ' +
                                  f'{SampleMethod.HALTON_RANDOM}, {SampleMethod.LATIN_HYPERCUBE}']))

    return pcts


@lru_cache(maxsize=1)
def cached_pcts(ndraws     : int,
                method     : str,
                ninvar_max : int,
                scramble   : bool,
                seed       : int,
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

    all_pcts = np.array(points)
    return all_pcts
