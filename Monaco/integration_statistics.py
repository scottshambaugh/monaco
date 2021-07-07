# integration_statistics.py

import numpy as np
from typing import Union
from Monaco.gaussian_statistics import pct2sig

def integration_error(nums         : list[float],
                      volume       : float     = 1,  # By default, returns an unscaled error
                      runningError : bool      = False,
                      conf         : float     = 0.95,
                      ) -> Union[float, list[float]]:
    
    n = len(nums)
    if n == 1:
        error1sig = volume
    
    elif not runningError:
        stdev = np.std(nums, ddof=1)
        error1sig = volume*stdev/np.sqrt(n)
    
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
        
        stdevs[stdevs == 0] = max(stdevs) # Leading zeros will throw off plots, fill with reasonable dummy data
        error1sig = volume*stdevs/np.sqrt(np.arange(1, n+1))
    
    error = error1sig*pct2sig(conf)
    return error


def integration_n_from_err(error  : float,
                           volume : float,
                           stdev  : float, 
                           conf   : float = 0.95,
                           ) -> int:
    # We generally do not know a-priori what the standard deviation will be, so
    # best practice is to set to the max range of values on the interval, and 
    # then calculate a better stdev on a lower number of cases, which can then 
    # be subsituted in here to bootleg a more efficient computation.
    # Note that sobol sampling should converge much faster than this, at a rate
    # of log(n)^d/n, rather than the 1/sqrt(n) given by random sampling. For 
    # sobol sampling, remember to round n to the next power of 2. 
    # helper_functions.next_power_of_2(n) can help with this 
    
    n = int(np.ceil((volume*pct2sig(conf)*stdev/error)**2))
    
    return n


def max_variance(low  : float,
                 high : float,
                 ) -> float:
    maxvar = (high-low)**2 / 4
    return maxvar


def max_stdev(low  : float,
              high : float,
              ) -> float:
    maxstd = np.sqrt(max_variance(high=high, low=low))
    return maxstd

