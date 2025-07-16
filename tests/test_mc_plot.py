# test_mc_plot.py

import pytest
from monaco.mc_plot import manage_invar_space, plot_2d_scatter
from monaco.mc_enums import InVarSpace


def test_manage_invar_space():
    assert manage_invar_space(invar_space=InVarSpace.NUMS, nvars=2) == ['nums', 'nums']
    assert manage_invar_space(invar_space=['pcts', 'nums'], nvars=2) == ['pcts', 'nums']
    with pytest.raises(ValueError, match='expected length'):
        manage_invar_space(invar_space=['pcts', 'nums'], nvars=3)


# Does not test the plot appearances, but does check that the codepaths can run
def test_gen_plots():
    plot_testing(show=False)
    assert True

### Plot Testing ###
def plot_testing(show=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import randint, norm
    from monaco.mc_var import InVar, OutVar
    from monaco.mc_plot import (plot, plot_hist, plot_cdf,
                                plot_cov_corr, plot_integration_convergence,
                                plot_integration_error, plot_sensitivities)
    from monaco.mc_enums import SampleMethod

    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)
    plt.close('all')

    invars = dict()
    nummap = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    ndraws = 1000
    invars['randint'] = InVar(name='randint', ndraws=ndraws,
                              dist=randint, distkwargs={'low': 1, 'high': 6},
                              nummap=nummap,
                              samplemethod=SampleMethod.RANDOM, seed=invarseeds[0])
    invars['norm'] = InVar(name='norm', ndraws=ndraws,
                           dist=norm, distkwargs={'loc': 10, 'scale': 4},
                           samplemethod=SampleMethod.RANDOM, seed=invarseeds[1])
    invars['norm'].addVarStat(stat='orderstatTI',
                              statkwargs={'p': 0.75, 'c': 0.50, 'bound': '2-sided'})
    invars['norm'].addVarStat(stat='orderstatP',
                              statkwargs={'p': 0.5, 'c': 0.9999, 'bound': 'all'})
    invars['norm2'] = InVar(name='norm2', ndraws=ndraws,
                            dist=norm, distkwargs={'loc': 10, 'scale': 4},
                            samplemethod=SampleMethod.RANDOM, seed=invarseeds[2])
    invars['custom_invar'] = InVar(name='custom_invar', ndraws=ndraws,
                                   vals=['a']*ndraws, nummap=nummap,
                                   samplemethod=None, seed=invarseeds[3])
    outvars = dict()
    outvars['test'] = OutVar(name='test', vals=[1, 0, 2, 2], firstcaseismedian=True)
    outvars['test'].sensitivity_ratios = {'a': 0.15, 'b': 0.35, 'c': 0.4}

    f, (ax1, ax2) = plt.subplots(2, 1)
    plot_hist(invars['randint'], ax=ax1, orientation='horizontal')        # plot_hist
    plot_cdf(invars['randint'], ax=ax2)                                   # plot_cdf
    plot(invars['norm'], title='norm', invar_space=InVarSpace.PCTS)       # plot_hist
    plot_hist(outvars['test'], orientation='horizontal', rug_plot=False)  # plot_hist
    plot_cdf(invars['norm'], orientation='horizontal')                    # plot_cdf
    plot_cdf(invars['custom_invar'], orientation='horizontal')            # plot_cdf
    plot_cdf(outvars['test'])                                             # plot_cdf

    plot_sensitivities(outvars['test'], sensitivities='ratios', sort=True)  # plot_sensitivities

    plot(invars['randint'], invars['norm'],
         cases=None, highlight_cases=range(10, 30),
         rug_plot=True, cov_plot=True, cov_p=[0.90, 0.95, 0.99])  # plot_2d_scatter
    plot(invars['randint'], invars['norm'], invars['norm2'],
         cases=[], highlight_cases=range(10, 30))  # plot_3d_scatter
    plot_2d_scatter(invars['randint'], invars['norm'], invars['norm2'])  # plot_2d_scatter

    v = np.array([-2, -1, 2, 3, 4, 5])
    var0 = OutVar(name='testscalar', vals=[1, 2, 3, 4, 5], firstcaseismedian=True)
    var1 = OutVar(name='testx', vals=[v, v, v, v, v], firstcaseismedian=True)
    var2 = OutVar(name='testy', vals=[1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    var3 = OutVar(name='testz', vals=[1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    var2.addVarStat(stat='sigma', statkwargs={'sig': 3})
    var2.addVarStat(stat='sigma', statkwargs={'sig': -3})
    var2.addVarStat(stat='orderstatTI', statkwargs={'p': 0.6, 'c': 0.50, 'bound': '2-sided'})
    var2.addVarStat(stat='mean')

    plot(var2, highlight_cases=None)               # plot_2d_line
    plot(var1, var2, highlight_cases=[0, 1])       # plot_2d_line
    plot(var0, var1)                               # plot_2p5d_line
    plot(var1, var2, var3, highlight_cases=[0, ])  # plot_3d_line
    plot(var0, var1, var2)                         # plot_2p5d_line

    m = np.eye(3)
    var4 = OutVar(name='testm', vals=[1*m, 2*m, 0*m, -1*m, -2*m])
    plot(var4, highlight_cases=[])                 # plot_3d_line

    n = np.eye(2)
    var5 = OutVar(name='testn', vals=[1*n, 2*n, 0*n, -1*n, -2*n])
    plot(var0, var5)                               # plot_3d_line

    plot_cov_corr(np.array([[2, 0.1111, np.nan], [-0.19, -1, np.nan], [np.nan, np.nan, np.nan]]),
                  ['Test1', 'Test2', 'Test3'])

    invars['randint2'] = InVar(name='randint2', ndraws=1000,
                               dist=randint, distkwargs={'low': 0, 'high': 2},
                               samplemethod=SampleMethod.RANDOM, seed=invarseeds[3])
    plot_integration_convergence(invars['randint2'], volume=1, dimension=1,
                                 refval=0.5, conf=0.95)
    plot_integration_error(invars['randint2'], volume=1, dimension=1,
                           refval=0.5, conf=0.95)

    if show:
        plt.show(block=True)


if __name__ == '__main__':
    plot_testing(show=True)
