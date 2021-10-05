# mc_sampling.py

import scipy.stats
import numpy as np
from functools import lru_cache
import warnings
import sys
from monaco.MCEnums import SampleMethod


def mc_sampling(ndraws     : int, 
                method     : SampleMethod = SampleMethod.SOBOL_RANDOM,
                ninvar     : int          = None,
                ninvar_max : int          = None,
                seed       : int          = np.random.get_state(legacy=False)['state']['key'][0],
                ) -> np.ndarray:
    if ninvar_max is None:
        ninvar_max = ninvar

    if method == SampleMethod.RANDOM:
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed)
    
    elif method in (SampleMethod.SOBOL, SampleMethod.SOBOL_RANDOM, SampleMethod.HALTON, SampleMethod.HALTON_RANDOM, SampleMethod.LATIN_HYPERCUBE):
        if ninvar is None:
            raise ValueError(f'{ninvar=} must defined for the {method} method')           
        elif (not 1 <= ninvar <= 21201) and method in (SampleMethod.SOBOL, SampleMethod.SOBOL_RANDOM):
            raise ValueError(f'{ninvar=} must be between 1 and 21201 for the {method} method')
        
        scramble = False
        if method in (SampleMethod.SOBOL_RANDOM, SampleMethod.HALTON_RANDOM):
            scramble = True
        elif method in (SampleMethod.SOBOL, SampleMethod.HALTON):
            seed = 0 # These do not use randomness, so keep seed constant for caching
            
        all_pcts = cached_pcts(ndraws=ndraws, method=method, ninvar_max=ninvar_max, scramble=scramble, seed=seed)
        pcts = all_pcts[:,ninvar-1] # ninvar will always be >= 1

    else:
        raise ValueError("".join([f'{method=} must be one of the following: ',
                                  f'{SampleMethod.RANDOM}, {SampleMethod.SOBOL}, {SampleMethod.SOBOL_RANDOM}, '
                                  f'{SampleMethod.HALTON}, {SampleMethod.HALTON_RANDOM}, {SampleMethod.LATIN_HYPERCUBE}']))
    
    return pcts

@lru_cache(maxsize=1)
def cached_pcts(ndraws     : int,
                method     : str, 
                ninvar_max : int,
                scramble   : bool, 
                seed       : int,
                ) -> np.ndarray:
    if method in (SampleMethod.SOBOL, SampleMethod.SOBOL_RANDOM):
        sampler = scipy.stats.qmc.Sobol(d=ninvar_max, scramble=scramble, seed=seed)
    elif method in (SampleMethod.HALTON, SampleMethod.HALTON_RANDOM):
        sampler = scipy.stats.qmc.Halton(d=ninvar_max, scramble=scramble, seed=seed)
    elif method == SampleMethod.LATIN_HYPERCUBE:
        sampler = scipy.stats.qmc.LatinHypercube(d=ninvar_max, seed=seed)
    
    if not sys.warnoptions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) # Suppress the power of 2 warning for sobol / halton sequences
            points = sampler.random(n=ndraws)
    
    all_pcts = np.array(points)
    
    return all_pcts

