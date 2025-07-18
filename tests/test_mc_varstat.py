# test_mc_varstat.py

import pytest
import numpy as np
from scipy.stats import randint
from monaco.mc_var import InVar, OutVar
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


@pytest.fixture
def v():
    v = np.array([-2, -1, 2, 3, 4, 5])
    return v


bound = StatBound.ONESIDED
@pytest.mark.parametrize("stat, statkwargs, vals, cilow, cihigh", [
    (VarStatType.ORDERSTATTI, {'p': sig2pct( 3, bound), 'c': 0.50, 'bound': bound},
     3.0069887, [], []),
    (VarStatType.ORDERSTATP , {'p': sig2pct( 3, bound), 'c': 0.50, 'bound': StatBound.ALL},
     [2.9472576, 3.0069887, 3.0265706], [], []),
    (VarStatType.GAUSSIANP  , {'p': sig2pct(-3, bound), 'bound': bound},
     -2.9930052, -3.0343254, -2.9629306),
    (VarStatType.SIGMA      , {'sig': 3, 'bound': bound}, 3.0002124, 2.9653332, 3.0297001),
    (VarStatType.PERCENTILE , {'p': 0.5}, -0.0052538, -0.02621756, 0.0278626),
    (VarStatType.MOMENT     , {'n': 2}, 0.9977405, 0.9770223, 1.0184109),
    (VarStatType.KURTOSIS   , dict(), 0.0363506, -0.0734930, 0.1519812),
    (VarStatType.SKEWNESS   , dict(), -0.0144037, -0.0541356, 0.0288986),
    (VarStatType.VARIANCE   , dict(), 0.9977405, 0.9770223, 1.0184109),
    (VarStatType.MEAN       , dict(), 0.0036036, -0.0084521, 0.0287600),
    ('mean'                 , dict(), 0.0036036, -0.0084521, 0.0287600),
    (np.mean                , dict(), 0.0036036, -0.0084521, 0.0287600),
])
def test_invarstat(stat, statkwargs, vals, cihigh, cilow, invar):
    invarstat = VarStat(invar, stat=stat, statkwargs=statkwargs,
                        bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert np.allclose(invarstat.vals, vals)
    assert np.allclose(invarstat.confidence_interval_low_vals, cilow)
    assert np.allclose(invarstat.confidence_interval_high_vals, cihigh)


def test_varstat_setName(invar):
    bound = StatBound.ONESIDED
    invarstat = VarStat(invar, stat=VarStatType.ORDERSTATTI,
                        statkwargs={'p': sig2pct(3, bound=bound), 'c': 0.50, 'bound': bound},
                        bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert invarstat.name == 'norm 1-sided P99.865/50.0% Confidence Interval'


def test_varstat_cases(invar):
    cases = np.where(np.array(invar.nums) > 0)[0]
    # make sure warnings is issued
    with pytest.warns(UserWarning, match="Only using 4983 of 10000 cases for VarStat norm Mean"):
        invarstat = VarStat(invar, stat=VarStatType.MEAN,
                            cases=cases,
                            bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert invarstat.ncases == len(cases)
    assert np.allclose(invarstat.vals, 0.8050436)


def test_varstat_nummap():
    # Scalar variable
    nummap = {1: 'a', 2: 'b'}
    invar = InVar('test', ndraws=3, dist=randint, distkwargs={'low': 1, 'high': 3},
                  nummap=nummap, samplemethod=SampleMethod.RANDOM, seed=12345)
    invarstat = VarStat(invar, stat=VarStatType.MEDIAN, bootstrap=True, seed=0)
    assert invarstat.vals == 'a'

    with pytest.raises(ValueError, match='not in nummap'):
        invarstat = VarStat(invar, stat=VarStatType.MEAN, bootstrap=True, seed=0)

    # 1-D variable
    valmap = {'a': 1, 'b': 2}
    outvar = OutVar('test', [['a', 'b', 'a'], ['b', 'a', 'b'], ['a', 'b', 'a']],
                    valmap=valmap)
    outvarstat = VarStat(outvar, stat=VarStatType.MEDIAN, bootstrap=True, conf=0.1, seed=0)
    assert outvarstat.vals == ['a', 'b', 'a']

    with pytest.raises(ValueError, match='not in nummap'):
        outvarstat = VarStat(outvar, stat=VarStatType.MEAN, bootstrap=True, seed=0)


def test_outvarstat_2d(v):
    outvar = OutVar('test', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    outvarstat1 = VarStat(outvar, stat=VarStatType.ORDERSTATTI,
                          statkwargs={'p': 0.6, 'c': 0.50, 'bound': StatBound.ALL},
                          bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    outvarstat2 = VarStat(outvar, stat=VarStatType.MIN,
                          bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)

    assert np.allclose(outvarstat1.vals, [[-4, -2, 4], [-2, -1, 2], [-4, -2, 4],
                                          [-6, -3, 6], [-8, -4, 8], [-10, -5, 10]])
    assert outvarstat1.confidence_interval_low_nums is None
    assert outvarstat1.confidence_interval_high_nums is None

    assert np.allclose(outvarstat2.vals, [ -4, -2, -4, -6, -8, -10.])
    assert np.allclose(outvarstat2.confidence_interval_low_nums,
                       [-8.0, -4.0, -7.5, -11.25, -15.0, -18.75])
    assert np.allclose(outvarstat2.confidence_interval_high_nums,
                       [-4.0, -2.0, -4.0, -6.0, -8.0, -10.0])


def test_outvarstat_2d_irregular(v):
    outvar = OutVar('test', [[0, 0], 1*v, 2*v, 0*v, -1*v, [0, 0]], firstcaseismedian=True)
    outvarstat1 = VarStat(outvar, stat=VarStatType.MIN,
                          bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert np.allclose(outvarstat1.vals, [-4, -2, -2, -3, -4, -5.])
    assert np.allclose(outvarstat1.confidence_interval_low_nums, [-8, -4, -6, -9, -12, -15.])
    assert np.allclose(outvarstat1.confidence_interval_high_nums, [-4, -2, -2, -3, -4, -5.])

    outvarstat2 = VarStat(outvar, stat=VarStatType.PERCENTILE, statkwargs={'p': [0.25, 0.75]},
                          bootstrap=True, bootstrap_k=10, conf=0.95, seed=0)
    assert np.allclose(outvarstat2.vals,
                       [[-1.5, 0], [-0.75, 0], [-0.5, 2.5], [-0.75, 3.75], [-1., 5.], [-1.25, 6.25]])  # noqa: E501
    assert np.allclose(outvarstat2.confidence_interval_low_nums,
                       [[-3, -1.875], [-1.5, -0.9375], [-4.25, 1], [-6.375, 1.5 ], [-8.5, 2], [-10.625, 2.5]])  # noqa: E501
    assert np.allclose(outvarstat2.confidence_interval_high_nums,
                       [[0.5, 0.875], [0.25, 0.4375], [1, 6.25], [1.5, 9.375], [2, 12.5], [2.5, 15.625]])  # noqa: E501


def test_outvarstat_1d_orderstatp(v):
    # Create a 1-D OutVar with 6 cases
    outvar = OutVar('test_1d', [v, 2*v, 0.5*v, -v, 0.2*v, 3*v], firstcaseismedian=False)
    outvarstat = VarStat(outvar, stat=VarStatType.ORDERSTATP,
                         statkwargs={'p': 0.5, 'c': 0.50, 'bound': StatBound.NEAREST},
                         bootstrap=False, seed=0)
    assert all(outvarstat.vals == np.array([-0.4, -0.2, 4.0, 6.0, 8.0, 10.0]))
