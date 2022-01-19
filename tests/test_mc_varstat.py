# test_mc_varstat.py

import pytest
import numpy as np
from monaco.mc_varstat import VarStat
from monaco.gaussian_statistics import sig2pct
from monaco.mc_enums import SampleMethod, StatBound, VarStatType

@pytest.fixture
def invar():
    from scipy.stats import norm
    from monaco.mc_var import InVar
    seed = 74494861
    return InVar('norm', ndraws=100000,
                 dist=norm, distkwargs={'loc': 0, 'scale': 1},
                 samplemethod=SampleMethod.RANDOM, seed=seed)


bound = StatBound.ONESIDED
@pytest.mark.parametrize("stat, statkwargs, vals", [
    (VarStatType.ORDERSTATTI, {'p': sig2pct( 3, bound), 'c': 0.50, 'bound': bound}        ,  3.0060487),  # noqa: E501
    (VarStatType.ORDERSTATP , {'p': sig2pct( 3, bound), 'c': 0.50, 'bound': StatBound.ALL}, [2.9965785, 3.0069887, 3.0209262]),  # noqa: E501
    (VarStatType.GAUSSIANP  , {'p': sig2pct(-3, bound), 'bound': bound}                   , -3.0021598),  # noqa: E501
    (VarStatType.SIGMA      , {'sig': 3, 'bound': bound}                                  ,  3.0020925),  # noqa: E501
    (VarStatType.MEAN       , dict()                                                      , -3.361e-05),  # noqa: E501
    (np.mean                , dict()                                                      , -3.361e-05),  # noqa: E501
])
def test_invarstat(stat, statkwargs, vals, invar):
    invarstat = VarStat(invar, stat=stat, statkwargs=statkwargs)
    assert np.allclose(invarstat.vals, vals)


def test_varstat_setName(invar):
    bound = StatBound.ONESIDED
    invarstat = VarStat(invar, stat=VarStatType.ORDERSTATTI,
                        statkwargs={'p': sig2pct(3, bound=bound), 'c': 0.50, 'bound': bound})
    assert invarstat.name == '1-sided P99.865/50.0% Confidence Interval'


v = np.array([-2, -1, 2, 3, 4, 5])

def test_outvarstat_2d():
    from monaco.mc_var import OutVar
    outvar = OutVar('test', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    outvarstat1 = VarStat(outvar, stat=VarStatType.ORDERSTATTI,
                          statkwargs={'p': 0.6, 'c': 0.50, 'bound': StatBound.ALL})
    outvarstat2 = VarStat(outvar, stat=VarStatType.MIN)
    assert(np.allclose(outvarstat1.vals, [[-4, -2, 4], [-2, -1, 2], [-4, -2, 4],
                                          [-6, -3, 6], [-8, -4, 8], [-10, -5, 10]]))
    assert(np.allclose(outvarstat2.vals, [ -4, -2, -4, -6, -8, -10.]))


def test_outvarstat_2d_irregular():
    from monaco.mc_var import OutVar
    outvar = OutVar('test', [[0, 0], 1*v, 2*v, 0*v, -1*v, [0, 0]], firstcaseismedian=True)
    outvarstat = VarStat(outvar, stat=VarStatType.MIN)
    assert(np.allclose(outvarstat.vals, [-4, -2, -2, -3, -4, -5.]))
