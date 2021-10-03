# test_mc_multi_plot.py

#import pytest

### Inline Testing ###
# Can run here or copy into bottom of main file
def inline_testing():
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from monaco.MCVar import MCInVar
    from monaco.mc_multi_plot import mc_multi_plot
    from monaco.MCEnums import SampleMethod

    plt.close('all')
    
    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)
        
    mcinvars = dict()
    mcinvars['norm1'] = MCInVar('norm1', ndraws=1000, dist=norm, distkwargs={'loc':1, 'scale':5}, seed=invarseeds[0], samplemethod=SampleMethod.RANDOM)
    mcinvars['norm2'] = MCInVar('norm2', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod=SampleMethod.RANDOM)
    
    mc_multi_plot(mcinvars['norm1'], mcinvars['norm2'], highlight_cases=range(10,30), rug_plot=True, cov_plot=True, cov_p=0.95, title='test')  # MCPlot2DScatter

if __name__ == '__main__':
    inline_testing()
