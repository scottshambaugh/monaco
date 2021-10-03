# test_mc_sampling.py

import pytest
import numpy as np
from monaco.mc_sampling import mc_sampling
from monaco.MCEnums import SampleMethod

generator = np.random.RandomState(744948050)
seeds = generator.randint(0, 2**31-1, size=10)

@pytest.mark.parametrize("method,ninvar,seed,ans", [
    (SampleMethod.RANDOM,       None, seeds[0], 0.5424051),
    (SampleMethod.SOBOL,           2, seeds[1], 0.0      ),
    (SampleMethod.SOBOL_RANDOM,    2, seeds[2], 0.0952614),
    (SampleMethod.HALTON,          2, seeds[3], 0.0      ),
    (SampleMethod.HALTON_RANDOM,   2, seeds[4], 0.9198435),
    (SampleMethod.LATIN_HYPERCUBE, 2, seeds[5], 0.3522243),
])
def test_mcsampling(method, ninvar, seed, ans):
    pcts = mc_sampling(ndraws=512, method=method, ninvar=ninvar, ninvar_max=None, seed=seed)
    assert pcts[0] == pytest.approx(ans)

def test_mcsampling_error():
    with pytest.raises(ValueError):
        mc_sampling(ndraws=512, method=SampleMethod.SOBOL, ninvar=None, ninvar_max=None, seed=seeds[0])


### Inline Testing ###
# Can run here or copy into bottom of main file
def inline_testing():
    def plot_sampling_test(ndraws, method, seeds, genplot=True):
        import matplotlib.pyplot as plt
        import scipy.stats
        
        pcts = np.array([mc_sampling(ndraws=ndraws, method=method, ninvar=1, ninvar_max=2, seed=seeds[0]), 
                         mc_sampling(ndraws=ndraws, method=method, ninvar=2, ninvar_max=2, seed=seeds[1])])
        
        if genplot:
            fig, axs = plt.subplots(1, 3)
            fig.suptitle(f"samplemethod = '{method}'", fontweight='bold')
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
            ndraws_freq = 10000
            #ndraws_freq = 1000000  # For better frequency plots
            n_freq_grid = 2**8
            pcts_freq = np.array([mc_sampling(ndraws=ndraws_freq, method=method, ninvar=1, ninvar_max=2, seed=seeds[0]), 
                                  mc_sampling(ndraws=ndraws_freq, method=method, ninvar=2, ninvar_max=2, seed=seeds[1])])
            pcts_freq_int = np.round(pcts_freq*(n_freq_grid - 1)).astype(int)
            S = np.zeros([n_freq_grid, n_freq_grid])
            for i in range(ndraws_freq):
                S[pcts_freq_int[0,i], pcts_freq_int[1,i]] += 1
            FS = np.fft.fft2(S)
            axs[2].imshow(np.log(np.abs(np.fft.fftshift(FS))**2), cmap='Blues_r', extent=[-n_freq_grid/2,n_freq_grid/2,-n_freq_grid/2,n_freq_grid/2], aspect="equal")
    
            #fig.savefig(f'../docs/{method}_sampling.png')
        
        return pcts

    generator = np.random.RandomState(744948050)
    seeds = generator.randint(0, 2**31-1, size=10)
    ndraws = 512
    plot_sampling_test(ndraws=ndraws, method=SampleMethod.RANDOM,          seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method=SampleMethod.SOBOL,           seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method=SampleMethod.SOBOL_RANDOM,    seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method=SampleMethod.HALTON,          seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method=SampleMethod.HALTON_RANDOM,   seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method=SampleMethod.LATIN_HYPERCUBE, seeds=seeds)
    #print(cached_pcts.cache_info())  # Can only check caching when run in main file

if __name__ == '__main__':
    inline_testing()
