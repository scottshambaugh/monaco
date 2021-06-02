import scipy.stats
import numpy as np

def mc_sampling(ndraws : int, 
                method : str = 'random', 
                seed   : int = np.random.get_state()[1][0],
                ) -> list[float]:
    pcts = []
    if method == 'random':
        pcts = scipy.stats.uniform.rvs(size=ndraws, random_state=seed).tolist()
    else:
        raise ValueError(f"{method=} must be one of the following: ",
                         "'random'")

    return pcts


'''
### Test ###
if __name__ == '__main__':
    
    def plot_sampling_test(ndraws, method, seeds):
        import matplotlib.pyplot as plt
        
        random_pcts = [mc_sampling(ndraws=ndraws, method=method, seed=seeds[0]), mc_sampling(ndraws=ndraws, method=method, seed=seeds[1])]
        
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'Sampling Method: {method}')
        axs[0].scatter(random_pcts[0], random_pcts[1])
        axs[0].set_title('Uniform')
        axs[1].scatter(scipy.stats.norm.ppf(random_pcts[0]), scipy.stats.norm.ppf(random_pcts[1]))
        circle1 = plt.Circle((0, 0), 1, color='k', fill=False)
        circle2 = plt.Circle((0, 0), 2, color='k', fill=False)
        circle3 = plt.Circle((0, 0), 3, color='k', fill=False)
        axs[1].add_patch(circle1)
        axs[1].add_patch(circle2)
        axs[1].add_patch(circle3)
        axs[1].set_title('Normal')
        
    
    generator = np.random.RandomState(74494861)
    seeds = generator.randint(0, 2**31-1, size=10)
    ndraws = 200
    plot_sampling_test(ndraws=ndraws, method='random', seeds=seeds)
    
#'''
