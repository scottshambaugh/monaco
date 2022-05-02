# integration_statistics.py
from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar
from typing import Optional
from monaco.gaussian_statistics import pct2sig
from monaco.mc_enums import SampleMethod

def integration_error(nums         : np.ndarray,
                      dimension    : int,
                      volume       : float        = 1,
                      conf         : float        = 0.95,
                      samplemethod : SampleMethod = SampleMethod.RANDOM,
                      runningerror : bool         = False,
                      ) -> float | np.ndarray:
    """
    Returns the bounding integration error for an input array of numbers. This
    error can be a float point estimate if runningerror == False, or a numpy
    array showing running error over the samples, to demonstrate convergence.
    If volume == 1, the error returned is a percent error. Otherwise, the error
    is absolute over the integration volume.

    Parameters
    ----------
    nums : list[float]
        A list of the integration point estimates across the volume.
    dimension : int
        The number of dimensions in the integration volume, dimension > 0.
    volume : float
        The integration volume, > 0. If volume == 1 (default), then the error
        returned is a percentage of the true integration volume.
    conf : float
        Confidence level of the calculated error. This must be 0 < conf < 1,
        and should be 0.5 < conf < 1.
    samplemethod : monaco.mc_enums.SampleMethod
        Monte Carlo sample method. Either 'random' (default and bounding), or
        'sobol'. If using a different sample method, use 'random' here.
    runningerror : bool
        If False, returns a point estimate. If True, returns an array
        containing the running error over all of the integration estimates.

    Returns
    -------
    error : float | np.ndarray
        Either a point estimate of the error if runningerror == False, or an
        array of the running error if runningerror is True.
    """
    integration_args_check(error=None, dimension=dimension, volume=volume,
                           stdev=None, conf=conf, samplemethod=samplemethod)

    n = len(nums)
    if n == 1:
        error1sig = np.array(volume)

    elif not runningerror:
        stdev = np.std(nums, ddof=1)
        error1sig_random = volume*np.sqrt((2**(-1*dimension) - 3**(-1*dimension))/n)
        if samplemethod == SampleMethod.RANDOM:
            error1sig = error1sig_random
        elif samplemethod == SampleMethod.SOBOL:
            error1sig_sobol = volume*stdev*np.log(n)**dimension/n
            error1sig = np.minimum(error1sig_random, error1sig_sobol)

    else:
        # Use Welford's algorithm to calculate the running variance
        M = np.zeros(n)  # Running mean
        S = np.zeros(n)  # Sum of variances
        M[0] = nums[0]
        for i in range(1, n):
            M[i] = M[i-1] + (nums[i]-M[i-1])/(i+1)
            S[i] = S[i-1] + (nums[i]-M[i-1])*(nums[i]-M[i])
        variances = np.zeros(n)
        variances[1:] = S[1:]/np.arange(1, n)
        stdevs = np.sqrt(variances)

        error1sig_random = volume*np.sqrt((2**(-1*dimension) - 3**(-1*dimension))/np.arange(1, n+1))
        if samplemethod == SampleMethod.RANDOM:
            error1sig = error1sig_random
        elif samplemethod == SampleMethod.SOBOL:
            error1sig_sobol = volume*stdevs*np.log(np.arange(1, n+1))**dimension/np.arange(1, n+1)
            error1sig = np.minimum(error1sig_random, error1sig_sobol)

        # Leading zeros will throw off plots, fill with reasonable dummy data
        error1sig[error1sig == 0] = np.max(error1sig)

    error : float | np.ndarray = error1sig*pct2sig(conf)
    return error


def integration_n_from_err(error        : float,
                           dimension    : int,
                           volume       : float,
                           stdev        : float,
                           conf         : float        = 0.95,
                           samplemethod : SampleMethod = SampleMethod.RANDOM,
                           ) -> int:
    """
    Returns the bounding integration error for an input array of numbers. This
    error can be a float point estimate if runningerror == False, or a numpy
    array showing running error over the samples, to demonstrate convergence.
    If volume == 1, the error returned is a percent error. Otherwise, the error
    is absolute over the integration volume.
    We generally do not know a-priori what the standard deviation will be, so
    best practice is to set to the max range of values on the interval, and
    then calculate a better stdev on a lower number of cases, which can then
    be subsituted in here to bootleg a more efficient computation.
    For sobol sampling, remember to round n to the next power of 2 for balance.
    monaco.helper_functions.next_power_of_2(n) can help with this.

    Parameters
    ----------
    error : float
        The target error.
    dimension : int
        The number of dimensions in the integration volume, dimension > 0.
    volume : float
        The integration volume, > 0. If volume == 1 (default), then the target
        error is a percentage of the true integration volume.
    stdev : float
        The standard deviation of the integration estimates, stdev > 0. We
        generally do not know this a-priori, so use
        monaco.integration_statistics.max_stdev to calculate this in that
        instance. Or, do a limited number of cases to estimate this before
        performing the full run.
    conf : float
        Confidence level of the calculated error. This must be 0 < conf < 1,
        and should be 0.5 < conf < 1.
    samplemethod : monaco.mc_enums.SampleMethod
        Monte Carlo sample method. Either 'random' (default and bounding), or
        'sobol'. If using a different sample method, use 'random' here.

    Returns
    -------
    n : int
        The number of sample points required to meet the target integration
        error.
    """
    integration_args_check(error=error, dimension=dimension, volume=volume,
                           stdev=stdev, conf=conf, samplemethod=samplemethod)

    n_random = (volume*pct2sig(conf)*stdev/error)**2
    if samplemethod == SampleMethod.RANDOM:
        n = n_random
    elif samplemethod == SampleMethod.SOBOL:
        def f(n : float) -> float:
            return volume*stdev*pct2sig(conf)*np.log(n)**dimension/n - error

        try:
            rootResults = root_scalar(f, method='brentq', bracket=[2**8, 2**31-1],
                                         xtol=0.1, maxiter=int(1e3))
            n_sobol = rootResults.root
            n = np.min([n_random, n_sobol])
        except Exception:
            # For higher than 3 dimensions, reaching n may be difficult, and
            # will be much larger than n_random anyways
            # warn(f'Cannot reach error tolerance of ±{error}. ' +
            #      f'Falling back to samplemethod={SampleMethod.RANDOM}')
            n = n_random

    n = int(np.ceil(n))
    return n


def integration_args_check(error        : Optional[float],
                           dimension    : int,
                           volume       : float,
                           stdev        : Optional[float],
                           conf         : float,
                           samplemethod : SampleMethod,
                           ) -> None:
    """
    Raises a ValueError if any of the inputs for the integration functions are
    outside allowable bounds.

    Parameters
    ----------
    error : None | float
        error > 0.
    dimension : int
        dimension > 0.
    volume : float
        volume > 0.
    stdev : None | float
        stdev > 0.
    conf : float
         0 < conf < 1
    samplemethod : monaco.mc_enums.SampleMethod
        Either 'random' or 'sobol'.
    """
    if (error is not None) and (error < 0):
        raise ValueError(f'{error=} must be positive')
    if dimension < 1:
        raise ValueError(f'{dimension=} must be a positive integer')
    if volume <= 0:
        raise ValueError(f'{volume=} must be positive')
    if (stdev is not None) and (stdev < 0):
        raise ValueError(f'{stdev=} must be positive')
    if not 0 < conf < 1:
        raise ValueError(f'{conf=} must be between 0 and 1')
    if samplemethod not in (SampleMethod.RANDOM, SampleMethod.SOBOL):
        raise ValueError(f'{samplemethod=} must be either ' +
                         f'{SampleMethod.RANDOM} or {SampleMethod.SOBOL}')


def max_variance(low  : float,
                 high : float,
                 ) -> float:
    """
    Calculates the maximum possible variance of a list of points that span the
    range [low, high].
    max_variance(0, 1) = 0.25

    Parameters
    ----------
    low : float
        The low end of the range.
    high : float
        The high end of the range.

    Returns
    -------
    maxvar: float
        The maximum possible variance of a list of points in the input range.
    """
    maxvar = (high-low)**2 / 4
    return maxvar


def max_stdev(low  : float,
              high : float,
              ) -> float:
    """
    Calculates the maximum possible variance of a list of points that span the
    range [low, high].
    max_stdev(0, 1) = 0.5

    Parameters
    ----------
    low : float
        The low end of the range.
    high : float
        The high end of the range.

    Returns
    -------
    maxstd: float
        The maximum possible standard deviation of a list of points in the
        input range.
    """
    maxstd = np.sqrt(max_variance(high=high, low=low))  # maxstd = (high-low)/2
    return maxstd
