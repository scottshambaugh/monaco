# integration_statistics.py

import numpy as np
from scipy.optimize import root_scalar
from typing import Union
from Monaco.gaussian_statistics import pct2sig

def integration_error(nums         : list[float],
                      volume       : float            = 1,  # By default, returns an unscaled error
                      conf         : float            = 0.95,
                      dimension    : Union[None, int] = None, # required only for samplemethod='sobol'
                      samplemethod : str              = 'random', # 'random' or 'sobol'
                      runningError : bool             = False,
                      ) -> Union[float, list[float]]:
    
    integration_args_check(error=None, volume=volume, stdev=None, conf=conf, samplemethod=samplemethod, dimension=dimension)

    n = len(nums)
    if n == 1:
        error1sig = volume
    
    elif not runningError:
        stdev = np.std(nums, ddof=1)
        if samplemethod == 'random':
            error1sig = volume*stdev/np.sqrt(n)
        elif samplemethod == 'sobol':
            error1sig = volume*stdev*np.log(n)**dimension/n
    
    else:
        # Use Welford's algorithm to calculate the running variance
        M = np.zeros(n) # Running mean
        S = np.zeros(n) # Sum of variances
        M[0] = nums[0]
        for i in range(1,n):
            M[i] = M[i-1] + (nums[i]-M[i-1])/(i+1)
            S[i] = S[i-1] + (nums[i]-M[i-1])*(nums[i]-M[i])
        variances = np.zeros(n)
        variances[1:] = S[1:]/np.arange(1, n)
        stdevs = np.sqrt(variances)
                
        if samplemethod == 'random':
            error1sig = volume*stdevs/np.sqrt(np.arange(1, n+1))
        elif samplemethod == 'sobol':
            error1sig = volume*stdevs*np.log(np.arange(1, n+1))**dimension/np.arange(1, n+1)
        
        error1sig[error1sig == 0] = max(error1sig) # Leading zeros will throw off plots, fill with reasonable dummy data

    
    error = error1sig*pct2sig(conf)
    return error


def integration_n_from_err(error        : float,
                           volume       : float,
                           stdev        : float, 
                           conf         : float            = 0.95,
                           dimension    : Union[None, int] = None, # required only for samplemethod='sobol'
                           samplemethod : str              = 'random', # 'random' or 'sobol'
                           ) -> int:
    # We generally do not know a-priori what the standard deviation will be, so
    # best practice is to set to the max range of values on the interval, and 
    # then calculate a better stdev on a lower number of cases, which can then 
    # be subsituted in here to bootleg a more efficient computation.
    # For sobol sampling, remember to round n to the next power of 2 for balance. 
    # helper_functions.next_power_of_2(n) can help with this. 
    
    integration_args_check(error=error, volume=volume, stdev=stdev, conf=conf, samplemethod=samplemethod, dimension=dimension)
    
    if samplemethod == 'random':
        n = (volume*pct2sig(conf)*stdev/error)**2
    elif samplemethod == 'sobol':
        def f(n):
            return volume*stdev*pct2sig(conf)*np.log(n)**dimension/n - error
        try:
            rootResults = root_scalar(f, method='brentq', bracket=[2**8, 2**31-1], xtol=0.1, maxiter=int(1e3))
            n = rootResults.root
        except Exception:
            raise ValueError(f"Cannot reach error tolerance of ±{error}. Adjust error tolerance or fall back to samplemethod='random'")
        
    n = int(np.ceil(n))
    return n


def integration_args_check(error        : float,
                           volume       : float,
                           stdev        : float, 
                           conf         : float,
                           samplemethod : str,
                           dimension    : Union[None, int],
                          ):
    if (not error is None) and (error < 0):
        raise ValueError(f"{error=} must be positive")
    if volume <= 0:
        raise ValueError(f"{volume=} must be positive")
    if (not stdev is None) and (stdev < 0):
        raise ValueError(f"{stdev=} must be positive")
    if not 0 < conf < 1:
        raise ValueError(f"{conf=} must be between 0 and 1")
    if samplemethod not in ('random', 'sobol'):
        raise ValueError(f"{samplemethod=} must be either 'random', or 'sobol'")
    if samplemethod == 'sobol' and (dimension is None or dimension < 1):
        raise ValueError(f'{dimension=} must be a positive integer defined for {samplemethod=}')


def max_variance(low  : float,
                 high : float,
                 ) -> float:
    maxvar = (high-low)**2 / 4
    return maxvar


def max_stdev(low  : float,
              high : float,
              ) -> float:
    maxstd = np.sqrt(max_variance(high=high, low=low))  # maxstd = (high-low)/2
    return maxstd

