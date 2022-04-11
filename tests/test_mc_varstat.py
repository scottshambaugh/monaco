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
    return InVar('norm', ndraws=10000,
                 dist=norm, distkwargs={'loc': 0, 'scale': 1},
                 samplemethod=SampleMethod.RANDOM, seed=seed)


bound = StatBound.ONESIDED
@pytest.mark.parametrize("stat, statkwargs, vals", [
    (VarStatType.ORDERSTATTI, {'p': sig2pct( 3, bound), 'c': 0.50, 'bound': bound}        ,  3.0069887),  # noqa: E501
    (VarStatType.ORDERSTATP , {'p': sig2pct( 3, bound), 'c': 0.50, 'bound': StatBound.ALL}, [2.9472576, 3.0069887, 3.0265706]),  # noqa: E501
    (VarStatType.GAUSSIANP  , {'p': sig2pct(-3, bound), 'bound': bound}                   , -2.9930052),  # noqa: E501
    (VarStatType.SIGMA      , {'sig': 3, 'bound': bound}                                  ,  3.0002124),  # noqa: E501
    (VarStatType.MEAN       , dict()                                                      ,  0.0036036),  # noqa: E501
    (np.mean                , dict()                                                      ,  0.0036036),  # noqa: E501
])
def test_invarstat(stat, statkwargs, vals, invar):
    invarstat = VarStat(invar, stat=stat, statkwargs=statkwargs,
                        bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    print(invarstat.vals)
    assert np.allclose(invarstat.vals, vals)


def test_varstat_setName(invar):
    bound = StatBound.ONESIDED
    invarstat = VarStat(invar, stat=VarStatType.ORDERSTATTI,
                        statkwargs={'p': sig2pct(3, bound=bound), 'c': 0.50, 'bound': bound},
                        bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert invarstat.name == 'norm 1-sided P99.865/50.0% Confidence Interval'


@pytest.fixture
def v():
    v = np.array([-2, -1, 2, 3, 4, 5])
    return v


def test_outvarstat_2d(v):
    from monaco.mc_var import OutVar
    outvar = OutVar('test', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    outvarstat1 = VarStat(outvar, stat=VarStatType.ORDERSTATTI,
                          statkwargs={'p': 0.6, 'c': 0.50, 'bound': StatBound.ALL},
                          bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    outvarstat2 = VarStat(outvar, stat=VarStatType.MIN,
                          bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert(np.allclose(outvarstat1.vals, [[-4, -2, 4], [-2, -1, 2], [-4, -2, 4],
                                          [-6, -3, 6], [-8, -4, 8], [-10, -5, 10]]))
    assert(np.allclose(outvarstat2.vals, [ -4, -2, -4, -6, -8, -10.]))


def test_outvarstat_2d_irregular(v):
    from monaco.mc_var import OutVar
    outvar = OutVar('test', [[0, 0], 1*v, 2*v, 0*v, -1*v, [0, 0]], firstcaseismedian=True)
    outvarstat = VarStat(outvar, stat=VarStatType.MIN,
                         bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert(np.allclose(outvarstat.vals, [-4, -2, -2, -3, -4, -5.]))
