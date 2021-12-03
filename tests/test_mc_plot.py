# test_mc_plot.py

import pytest
from monaco.mc_plot import get_cases

@pytest.mark.parametrize("ncases, cases, ans", [
    (3,   None, (0, 1, 2)),
    (3,      1, (1,)),
    (3, (1, 2), (1, 2)),
    (3,     [], ()),
])
def test_get_cases(ncases, cases, ans):
    assert get_cases(ncases=ncases, cases=cases) == pytest.approx(ans)


### Plot Testing ###
def plot_testing():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import randint, norm
    from monaco.mc_var import MCInVar, MCOutVar
    from monaco.mc_plot import (mc_plot, mc_plot_hist, mc_plot_cdf,
                                mc_plot_cov_corr, mc_plot_integration_convergence,
                                mc_plot_integration_error)
    from monaco.mc_enums import SampleMethod

    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)
    plt.close('all')

    mcinvars = dict()
    nummap = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    mcinvars['randint'] = MCInVar(name='randint', ndraws=1000,
                                  dist=randint, distkwargs={'low': 1, 'high': 6},
                                  nummap=nummap,
                                  samplemethod=SampleMethod.RANDOM, seed=invarseeds[0])
    mcinvars['norm'] = MCInVar(name='norm', ndraws=1000,
                               dist=norm, distkwargs={'loc': 10, 'scale': 4},
                               samplemethod=SampleMethod.RANDOM, seed=invarseeds[1])
    mcinvars['norm'].addVarStat(stattype='orderstatTI',
                                statkwargs={'p': 0.75, 'c': 0.50, 'bound': '2-sided'})
    mcinvars['norm'].addVarStat(stattype='orderstatP',
                                statkwargs={'p': 0.5, 'c': 0.9999, 'bound': 'all'})
    mcinvars['norm2'] = MCInVar(name='norm2', ndraws=1000,
                                dist=norm, distkwargs={'loc': 10, 'scale': 4},
                                samplemethod=SampleMethod.RANDOM, seed=invarseeds[2])
    mcoutvars = dict()
    mcoutvars['test'] = MCOutVar(name='test', vals=[1, 0, 2, 2], firstcaseismedian=True)

    f, (ax1, ax2) = plt.subplots(2, 1)
    mc_plot_hist(mcinvars['randint'], ax=ax1, orientation='horizontal')        # mc_plot_hist
    mc_plot_cdf(mcinvars['randint'], ax=ax2)                                   # mc_plot_cdf
    mc_plot(mcinvars['norm'], title='norm')                                    # mc_plot_hist
    mc_plot_hist(mcoutvars['test'], orientation='horizontal', rug_plot=False)  # mc_plot_hist
    mc_plot_cdf(mcinvars['norm'], orientation='horizontal')                    # mc_plot_cdf
    mc_plot_cdf(mcoutvars['test'])                                             # mc_plot_cdf

    mc_plot(mcinvars['randint'], mcinvars['norm'],
            cases=None, highlight_cases=range(10, 30),
            rug_plot=True, cov_plot=True, cov_p=[0.90, 0.95, 0.99])  # mc_plot_2d_scatter
    mc_plot(mcinvars['randint'], mcinvars['norm'], mcinvars['norm2'],
            cases=[], highlight_cases=range(10, 30))  # mc_plot_3d_scatter

    v = np.array([-2, -1, 2, 3, 4, 5])
    var1 = MCOutVar(name='testx', vals=[v, v, v, v, v], firstcaseismedian=True)
    var2 = MCOutVar(name='testy', vals=[1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    var3 = MCOutVar(name='testz', vals=[1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    var2.addVarStat(stattype='sigma', statkwargs={'sig': 3})
    var2.addVarStat(stattype='sigma', statkwargs={'sig': -3})
    var2.addVarStat(stattype='orderstatTI', statkwargs={'p': 0.6, 'c': 0.50, 'bound': '2-sided'})
    var2.addVarStat(stattype='mean')

    mc_plot(var2, highlight_cases=None)              # mc_plot_2d_line
    mc_plot(var1, var2, highlight_cases=[0, 1])       # mc_plot_2d_line
    mc_plot(var1, var2, var3, highlight_cases=[0, ])  # mc_plot_3d_line

    mc_plot_cov_corr(np.array([[2, 0.1111, np.nan], [-0.19, -1, np.nan], [np.nan, np.nan, np.nan]]),
                              ['Test1', 'Test2', 'Test3'])

    mcinvars['randint2'] = MCInVar(name='randint2', ndraws=1000,
                                   dist=randint, distkwargs={'low': 0, 'high': 2},
                                   samplemethod=SampleMethod.RANDOM, seed=invarseeds[3])
    mc_plot_integration_convergence(mcinvars['randint2'], volume=1, dimension=1,
                                    refval=0.5, conf=0.95)
    mc_plot_integration_error(mcinvars['randint2'], volume=1, dimension=1,
                              refval=0.5, conf=0.95)


if __name__ == '__main__':
    plot_testing()
