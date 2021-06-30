# test_mc_multi_plot.py

#import pytest

### Inline Testing ###
# Can run here or copy into bottom of main file
#'''
if __name__ == '__main__':
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from Monaco.MCVar import MCInVar
    from Monaco.mc_multi_plot import mc_multi_plot
    
    plt.close('all')
    
    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)
        
    mcinvars = dict()
    mcinvars['norm1'] = MCInVar('norm1', ndraws=1000, dist=norm, distkwargs={'loc':1, 'scale':5}, seed=invarseeds[0])
    mcinvars['norm2'] = MCInVar('norm2', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1])
    
    mc_multi_plot(mcinvars['norm1'], mcinvars['norm2'], highlight_cases=range(10,30), rug_plot=True, cov_plot=True, cov_p=0.95, title='test')  # MCPlot2DScatter
#'''
