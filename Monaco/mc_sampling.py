# mc_sampling.py

import scipy.stats
import numpy as np
from functools import lru_cache
import warnings
import sys

def mc_sampling(ndraws     : int, 
                method     : str = 'sobol_random', 
                ninvar     : int = None,
                ninvar_max : int = None,
                seed       : int = np.random.get_state()[1][0],
                ) -> np.ndarray:
    if ninvar_max is None:
        ninvar_max = ninvar

    if method == 'random':
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed)
    
    elif method in ('sobol', 'sobol_random', 'halton', 'halton_random', 'latin_hypercube'):
        if ninvar is None:
            raise ValueError(f'{ninvar=} must defined for the {method} method')           
        elif (not 1 <= ninvar <= 21201) and method in ('sobol', 'sobol_random'):
            raise ValueError(f'{ninvar=} must be between 1 and 21201 for the {method} method')
        
        scramble = False
        if method in ('sobol_random', 'halton_random'):
            scramble = True
        elif method in ('sobol', 'halton'):
            seed = 0 # These do not use randomness, so keep seed constant for caching
            
        all_pcts = cached_pcts(ndraws=ndraws, method=method, ninvar_max=ninvar_max, scramble=scramble, seed=seed)
        pcts = all_pcts[:,ninvar-1] # ninvar will always be >= 1

    else:
        raise ValueError(f'{method=} must be one of the following: ',
                         "'random', 'sobol', 'sobol_random', 'halton', 'halton_random', 'latin_hypercube'")
    
    return pcts

@lru_cache(maxsize=1)
def cached_pcts(ndraws     : int,
                method     : str, 
                ninvar_max : int,
                scramble   : bool, 
                seed       : int,
                ) -> np.ndarray:
    if method in ('sobol', 'sobol_random'):
        sampler = scipy.stats.qmc.Sobol(d=ninvar_max, scramble=scramble, seed=seed)
    elif method in ('halton', 'halton_random'):
        sampler = scipy.stats.qmc.Halton(d=ninvar_max, scramble=scramble, seed=seed)
    elif method == 'latin_hypercube':
        sampler = scipy.stats.qmc.LatinHypercube(d=ninvar_max, seed=seed)
    
    if not sys.warnoptions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) # Suppress the power of 2 warning for sobol / halton sequences
            points = sampler.random(n=ndraws)
    
    all_pcts = np.array(points)
    
    return all_pcts

