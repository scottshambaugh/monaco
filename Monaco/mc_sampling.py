# mc_sampling.py

import scipy.stats
import numpy as np
from functools import lru_cache

def mc_sampling(ndraws     : int, 
                method     : str = 'random', 
                ninvar     : int = None,
                ninvar_max : int = None,
                seed       : int = np.random.get_state()[1][0],
                ) -> list[float]:
    
    if method in ('sobol', 'sobol_random'):
        if not 1 <= ninvar <= 21201:
            raise ValueError(f'{ninvar=} must be between 1 and 21201 for the {method} method')

    if method == 'random':
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed)
    
    elif method in ('sobol', 'sobol_random', 'halton', 'halton_random', 'latin_hypercube'):
        if (ninvar is None) or (ninvar_max is None):
            raise ValueError(f'{ninvar=} and {ninvar_max=} must defined for the {method} method')           
            
        scramble = False
        if method in ('sobol_random', 'halton_random'):
            scramble = True
        elif method in ('sobol', 'halton', 'latin_hypercube'):
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
                ):
    if method in ('sobol', 'sobol_random'):
        sampler = scipy.stats.qmc.Sobol(d=ninvar_max, scramble=scramble, seed=seed)
    elif method in ('halton', 'halton_random'):
        sampler = scipy.stats.qmc.Halton(d=ninvar_max, scramble=scramble, seed=seed)
    elif method == 'latin_hypercube':
        sampler = scipy.stats.qmc.LatinHypercube(d=ninvar_max)
    
    points = sampler.random(n=ndraws)
    all_pcts = np.array(points)
    
    return all_pcts



'''
### Test ###
if __name__ == '__main__':
    
    def plot_sampling_test(ndraws, method, seeds):
        import matplotlib.pyplot as plt
        
        pcts = np.array([mc_sampling(ndraws=ndraws, method=method, ninvar_max=2, ninvar=1, seed=seeds[0]), 
                         mc_sampling(ndraws=ndraws, method=method, ninvar_max=2, ninvar=2, seed=seeds[1])])
        
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f'Sampling Method: {method}', fontweight='bold')
        fig.set_dpi(96)
        fig.set_size_inches(16, 5)

        
        axs[0].set_title('Uniform')
        axs[0].scatter(pcts[0], pcts[1], alpha=0.5)
        square = plt.Rectangle((0,0), 1, 1, color='k', fill=False)
        axs[0].add_patch(square)
        axs[0].set_xlim([-0.1,1.1])
        axs[0].set_ylim([-0.1,1.1])
        axs[0].set_aspect('equal')
        
        axs[1].set_title('Normal')
        axs[1].scatter(scipy.stats.norm.ppf(pcts[0]), scipy.stats.norm.ppf(pcts[1]), alpha=0.5)
        circle1 = plt.Circle((0, 0), 1, color='k', fill=False)
        circle2 = plt.Circle((0, 0), 2, color='k', fill=False)
        circle3 = plt.Circle((0, 0), 3, color='k', fill=False)
        axs[1].add_patch(circle1)
        axs[1].add_patch(circle2)
        axs[1].add_patch(circle3)
        axs[1].set_xlim([-3.5,3.5])
        axs[1].set_ylim([-3.5,3.5])
        axs[1].set_aspect('equal')
        
        axs[2].set_title('Frequency Spectra')
        ndraws_freq = 1000000
        n_freq_grid = 2**8
        pcts_freq = np.array([mc_sampling(ndraws=ndraws_freq, method=method, ninvar=1, seed=seeds[0]), 
                              mc_sampling(ndraws=ndraws_freq, method=method, ninvar=2, seed=seeds[1])])
        pcts_freq_int = np.round(pcts_freq*(n_freq_grid - 1)).astype(int)
        S = np.zeros([n_freq_grid, n_freq_grid])
        for i in range(ndraws_freq):
            S[pcts_freq_int[0,i], pcts_freq_int[1,i]] += 1
        FS = np.fft.fft2(S)
        axs[2].imshow(np.log(np.abs(np.fft.fftshift(FS))**2), cmap='Blues_r', extent=[-n_freq_grid/2,n_freq_grid/2,-n_freq_grid/2,n_freq_grid/2], aspect="equal")
        
        #fig.savefig(f'../docs/{method}_sampling.png')
        
    
    generator = np.random.RandomState(744948050)
    seeds = generator.randint(0, 2**31-1, size=10)
    ndraws = 512
    plot_sampling_test(ndraws=ndraws, method='random', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='sobol', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='sobol_random', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='halton', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='halton_random', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='latin_hypercube', seeds=seeds)
    print(cached_pcts.cache_info())

#'''
