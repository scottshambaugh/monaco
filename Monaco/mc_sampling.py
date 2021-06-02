import scipy.stats
import numpy as np
import sobol

def mc_sampling(ndraws : int, 
                method : str = 'random', 
                ninvar : int = None,
                seed   : int = np.random.get_state()[1][0],
                ) -> list[float]:
    
    if method == 'random':
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed)
    
    elif method == 'sobol':
        if (ninvar is None) or (not 1 <= ninvar <= 1111):
            raise ValueError(f'{ninvar=} must be between 1 and 1111 for the sobol method')
        
        # Note: Randomization using seeds not necessary for sobol, but it's nice
        #       to be able to get slightly different results with different seeds
        sobol_skip = int(seed % 2**10)  
        sobol_points = sobol.sample(dimension=ninvar, n_points=ndraws, skip=sobol_skip)
        pcts = np.array(sobol_points)[:,ninvar-1] # ninvar will always be >= 1
        
    else:
        raise ValueError(f'{method=} must be one of the following: ',
                         "'random', 'sobol'")
    
    return pcts



'''
### Test ###
if __name__ == '__main__':
    
    def plot_sampling_test(ndraws, method, seeds):
        import matplotlib.pyplot as plt
        
        random_pcts = [mc_sampling(ndraws=ndraws, method=method, ninvar=1, seed=seeds[0]), 
                       mc_sampling(ndraws=ndraws, method=method, ninvar=2, seed=seeds[1])]
        
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'Sampling Method: {method}')
        axs[0].set_title('Uniform')
        axs[0].scatter(random_pcts[0], random_pcts[1], alpha=0.5)
        square = plt.Rectangle((0,0), 1, 1, color='k', fill=False)
        axs[0].add_patch(square)
        axs[0].set_xlim([-0.05,1.05])
        axs[0].set_ylim([-0.05,1.05])
        
        axs[1].set_title('Normal')
        axs[1].scatter(scipy.stats.norm.ppf(random_pcts[0]), scipy.stats.norm.ppf(random_pcts[1]), alpha=0.5)
        circle1 = plt.Circle((0, 0), 1, color='k', fill=False)
        circle2 = plt.Circle((0, 0), 2, color='k', fill=False)
        circle3 = plt.Circle((0, 0), 3, color='k', fill=False)
        axs[1].add_patch(circle1)
        axs[1].add_patch(circle2)
        axs[1].add_patch(circle3)
        axs[1].set_xlim([-3.5,3.5])
        axs[1].set_ylim([-3.5,3.5])
        
    
    generator = np.random.RandomState(74494861)
    seeds = generator.randint(0, 2**31-1, size=10)
    ndraws = 500
    plot_sampling_test(ndraws=ndraws, method='random', seeds=seeds)
    plot_sampling_test(ndraws=ndraws, method='sobol', seeds=seeds)
    
#'''
