# test_mc_var.py

import pytest
import numpy as np
from scipy.stats import rv_discrete, randint, norm, lognorm
from monaco.mc_var import InVar, OutVar
from monaco.mc_enums import SampleMethod
import matplotlib.pyplot as plt

generator = np.random.RandomState(74494861)
invarseeds = generator.randint(0, 2**31-1, size=10)

xk = np.array([1, 5, 6])
pk = np.ones(len(xk))/len(xk)
custom_dist = rv_discrete(name='custom', values=(xk, pk))
lognorm_sigma, lognorm_mu = 1, 2


@pytest.fixture
def invar_norm_random():
    return InVar('norm', ndraws=1000,
                 dist=norm, distkwargs={'loc': 10, 'scale': 4},
                 seed=invarseeds[1], samplemethod=SampleMethod.RANDOM,
                 firstcaseismedian=False)

@pytest.fixture
def invar_lognorm_random():
    return InVar('lognorm', ndraws=1000,
                 dist=lognorm, distkwargs={'s': lognorm_sigma, 'scale': np.exp(lognorm_mu)},
                 seed=invarseeds[1], samplemethod=SampleMethod.RANDOM,
                 firstcaseismedian=False)

@pytest.fixture
def invar_custom_dist():
    return InVar('custom', ndraws=1000, dist=custom_dist, distkwargs=dict(),
                 ninvar=1, samplemethod=SampleMethod.RANDOM, seed=invarseeds[2],
                 firstcaseismedian=False)

def test_invar_getitem(invar_norm_random):
    assert invar_norm_random[0].val == pytest.approx(14.186734703770963)

def test_invar_norm_sobol_warning():
    with pytest.warns(UserWarning, match='Infinite value'):
        InVar('norm', ndraws=1000, dist=norm, distkwargs={'loc': 10, 'scale': 4},
                ninvar=1, samplemethod=SampleMethod.SOBOL, seed=invarseeds[1],
                firstcaseismedian=False)

def test_invar_discrete():
    invar = InVar('randint', ndraws=1000, dist=randint, distkwargs={'low': 1, 'high': 5},
                    seed=invarseeds[0], samplemethod=SampleMethod.RANDOM)
    assert invar.stats().mean == pytest.approx(2.538)

def test_invar_continuous(invar_norm_random):
    assert invar_norm_random.stats().mean == pytest.approx(9.9450812)

def test_invar_custom(invar_custom_dist):
    assert invar_custom_dist.stats().mean == pytest.approx(4.105)

def test_invar_addvarstat(invar_norm_random):
    invar_norm_random.addVarStat(stat='orderstatTI',
                                 statkwargs={'p': 0.75, 'c': 0.95, 'bound': '2-sided'})
    assert invar_norm_random.varstats[0].vals == pytest.approx([5.10075, 14.75052273])

def test_invar_getVal(invar_custom_dist):
    assert invar_custom_dist.getVal(0).val == 5

def test_invar_getMean(invar_lognorm_random):
    pytest.approx(invar_lognorm_random.getDistMean(), lognorm_mu + lognorm_sigma**2 / 2)

def test_invar_getMedian(invar_lognorm_random):
    pytest.approx(invar_lognorm_random.getDistMedian(), lognorm_mu)

def test_invar_nummap():
    invar = InVar('map', ndraws=10, dist=custom_dist, distkwargs=dict(),
                  ninvar=1, nummap={1: 'a', 5: 'e', 6: 'f'},
                  samplemethod=SampleMethod.RANDOM, seed=invarseeds[3],
                  firstcaseismedian=False)
    assert invar.vals == ['f', 'e', 'f', 'f', 'a', 'e', 'e', 'a', 'e', 'e']

def test_invar_custom_vals():
    invar = InVar('custom', ndraws=3, vals=[1, 2, 3],
                  ninvar=1, samplemethod=None, seed=invarseeds[3], firstcaseismedian=False)
    assert invar.pcts == [0.0, 0.5, 1.0]
    assert invar.nums == [1, 2, 3]
    assert invar.vals == [1, 2, 3]

    # With nummap
    nummap = {2: 'a', 3: 'b', 4: 'c', 5: 'd'}
    invar = InVar('custom', ndraws=3, vals=['a', 'b', 'c', 'd'],
                  ninvar=1, nummap=nummap, samplemethod=None, seed=invarseeds[3],
                  firstcaseismedian=True)
    assert invar.pcts == [0.5, 0, 0.5+1e-12, 1.0]
    assert invar.nums == [2, 3, 4, 5]
    assert invar.vals == ['a', 'b', 'c', 'd']

    # Without nummap, and also test custom pcts
    invar = InVar('custom', ndraws=3, vals=['b', 'a', 'c', 'd'],
                  pcts=[0.5, 0.2, 0.3, 0.4],
                  ninvar=1, samplemethod=None, seed=invarseeds[3],
                  firstcaseismedian=True)
    assert invar.vals == ['b', 'a', 'c', 'd']
    assert invar.nums == [1, 0, 2, 3]
    assert invar.pcts == [0.5, 0.2, 0.3, 0.4]

def test_invar_custom_vals_errors():
    with pytest.raises(ValueError, match="Either 'dist' or 'vals' must be provided"):
        InVar('custom', ndraws=3,
              ninvar=1, samplemethod=None, seed=invarseeds[3], firstcaseismedian=False)

    with pytest.raises(ValueError, match="Cannot provide both 'dist' and 'vals'"):
        InVar('custom', dist=custom_dist, distkwargs=dict(), ndraws=3, vals=[1, 2, 3],
              ninvar=1, samplemethod=None, seed=invarseeds[3], firstcaseismedian=False)

    with pytest.raises(ValueError,
                       match=r"Length of 'vals' \(3\) must match ncases \(10\)"):
        InVar('custom', ndraws=10, vals=[1, 2, 3],
              ninvar=1, samplemethod=None, seed=invarseeds[3], firstcaseismedian=False)

    with pytest.raises(ValueError,
                       match=r"Length of 'vals' \(3\) must match ncases \(11\)"):
        InVar('custom', ndraws=10, vals=[1, 2, 3],
              ninvar=1, samplemethod=None, seed=invarseeds[3], firstcaseismedian=True)


def test_invar_custom_pcts():
    invar = InVar('custom', ndraws=3, dist=randint, distkwargs={'low': 1, 'high': 5},
                  pcts=[0.01, 0.5, 0.99], ninvar=1, samplemethod=None, seed=invarseeds[3],
                  firstcaseismedian=False)
    assert invar.pcts == [0.01, 0.5, 0.99]
    assert invar.nums == [1.0, 2.0, 4.0]
    assert invar.vals == [1.0, 2.0, 4.0]


def test_invar_custom_pcts_errors():
    with pytest.raises(ValueError, match=r"Length of 'pcts' \(3\) must match "):
        InVar('custom', ndraws=5, dist=randint, distkwargs={'low': 1, 'high': 5},
              pcts=[0.01, 0.5, 0.99], ninvar=1, samplemethod=None, seed=invarseeds[3],
              firstcaseismedian=False)

    with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
        InVar('custom', ndraws=3, dist=randint, distkwargs={'low': 1, 'high': 5},
              pcts=[0.01, 0.5, 1.01], ninvar=1, samplemethod=None, seed=invarseeds[3],
              firstcaseismedian=False)

    with pytest.raises(ValueError, match="If firstcaseismedian is True, then the first " +
                                         "percentile must be 0.5"):
        InVar('custom', ndraws=3, dist=randint, distkwargs={'low': 1, 'high': 5},
              pcts=[0.01, 0.5, 0.75, 0.99], ninvar=1, samplemethod=None, seed=invarseeds[3],
              firstcaseismedian=True)


def test_outvar():
    outvar = OutVar('test', [1, 0, 2, 2], firstcaseismedian=True)
    assert outvar.getVal(1).val == 0
    assert outvar.stats().mean == pytest.approx(1.25)
    assert outvar.getMedianVal().val == 1

@pytest.mark.parametrize("vals, maxdim", [
    (                  [None, ], 0),
    (                     [0, ], 0),
    (                     [[0]], 1),
    (                  [[0, 0]], 1),
    ([[[0, 0], [0, 0], [0, 0]]], 2),
    (   [[[0]], [[0, 0]], [[]]], 2),
])
def test_outvar_genMaxDim(vals, maxdim):
    outvar = OutVar('test', vals)
    assert outvar.maxdim == maxdim

def test_outvar_extractValMap():
    outvar = OutVar('test', ['a', 'b', 'c', ['b']], firstcaseismedian=True)
    assert outvar.valmap == {'a': 0, 'b': 1, 'c': 2}

    # Mixed types
    outvar = OutVar('test', [1, 2, None, 4], firstcaseismedian=True)
    assert outvar.valmap == {None: 0, 1: 1, 2: 2, 4: 3}

@pytest.fixture
def outvars_split():
    outvars = dict()
    v = np.array([[1, 1], [2, 2], [3, 3]])
    outvars['test'] = OutVar('test', [v, v, v, v, v])
    outvars['test'].addVarStat(stat='orderstatTI',
                               statkwargs={'p': 0.33, 'c': 0.50, 'bound': '1-sided'})
    outvars.update(outvars['test'].split())
    return outvars

def test_outvar_split(outvars_split):
    assert np.array_equal(outvars_split['test [0]'].nums,
                          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    assert np.array_equal(outvars_split['test [1]'].nums,
                          [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

def test_outvar_split_orderstat(outvars_split):
    assert outvars_split['test [0]'].varstats[0].vals == pytest.approx([1, 1])


# Does not test the plot appearances, but does check that the codepaths can run
def test_gen_plots():
    plot_testing(show=False)
    assert True


def plot_testing(show=False):
    from scipy.stats import norm
    invar = InVar('norm', ndraws=1000,
                  dist=norm, distkwargs={'loc': 10, 'scale': 4},
                  seed=invarseeds[1], samplemethod=SampleMethod.RANDOM,
                  firstcaseismedian=False)

    fig, ax = invar.plot()
    if show:
        plt.show(block=True)


if __name__ == '__main__':
    plot_testing(show=True)
