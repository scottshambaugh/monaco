import scipy.stats
import numpy as np
import sobol

def mc_sampling(ndraws : int, 
                method : str = 'random', 
                ninvar : int = None,
                seed   : int = np.random.get_state()[1][0],
                ) -> list[float]:
    
    if method in ('sobol', 'sobol_random'):
        if (ninvar is None) or (not 1 <= ninvar <= 1111):
            raise ValueError(f'{ninvar=} must be between 1 and 1111 for the sobol or sobol_random method')

    if method == 'random':
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed)
        
    elif method == 'sobol':
        sobol_points = sobol.sample(dimension=ninvar, n_points=ndraws, skip=0)
        pcts = np.array(sobol_points)[:,ninvar-1] # ninvar will always be >= 1
        
    elif method == 'sobol_random':
        # TODO: Replace this with proper Owen's Scrambling, ,currently doesn't help freq spectra
        sobol_skip = int(seed % 2**10)
        sobol_points = sobol.sample(dimension=ninvar, n_points=ndraws, skip=sobol_skip)
        pcts = np.array(sobol_points)[:,ninvar-1] # ninvar will always be >= 1
                
    else:
        raise ValueError(f'{method=} must be one of the following: ',
                         "'random', 'sobol', 'sobol_random")
    
    return pcts



'''
### Test ###
if __name__ == '__main__':
    
    def plot_sampling_test(ndraws, method, seeds):
        import matplotlib.pyplot as plt
        
        pcts = np.array([mc_sampling(ndraws=ndraws, method=method, ninvar=1, seed=seeds[0]), 
                         mc_sampling(ndraws=ndraws, method=method, ninvar=2, seed=seeds[1])])
        
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
        
    
    generator = np.random.RandomState(74494850)
    seeds = generator.randint(0, 2**31-1, size=10)
    ndraws = 500
    plot_sampling_test(ndraws=ndraws, method='random', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='sobol', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='sobol_random', seeds=seeds)
    
#'''
