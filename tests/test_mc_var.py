# test_mc_var.py

import pytest
import numpy as np
from scipy.stats import rv_discrete
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
    from scipy.stats import norm
    return InVar('norm', ndraws=1000,
                   dist=norm, distkwargs={'loc': 10, 'scale': 4},
                   seed=invarseeds[1], samplemethod=SampleMethod.RANDOM,
                   firstcaseismedian=False)

@pytest.fixture
def invar_lognorm_random():
    from scipy.stats import lognorm
    return InVar('lognorm', ndraws=1000,
                   dist=lognorm, distkwargs={'s': lognorm_sigma, 'scale': np.exp(lognorm_mu)},
                   seed=invarseeds[1], samplemethod=SampleMethod.RANDOM,
                   firstcaseismedian=False)

@pytest.fixture
def invar_custom_dist():
    return InVar('custom', ndraws=1000, dist=custom_dist, distkwargs=dict(),
                   ninvar=1, samplemethod=SampleMethod.RANDOM, seed=invarseeds[2],
                   firstcaseismedian=False)

def test_invar_norm_sobol_warning():
    from scipy.stats import norm
    with pytest.warns(UserWarning, match='Infinite value'):
        InVar('norm', ndraws=1000, dist=norm, distkwargs={'loc': 10, 'scale': 4},
                ninvar=1, samplemethod=SampleMethod.SOBOL, seed=invarseeds[1],
                firstcaseismedian=False)

def test_invar_discrete():
    from scipy.stats import randint
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


### Plot Testing ###
def plot_testing():
    from scipy.stats import norm
    invar = InVar('norm', ndraws=1000,
                  dist=norm, distkwargs={'loc': 10, 'scale': 4},
                  seed=invarseeds[1], samplemethod=SampleMethod.RANDOM,
                  firstcaseismedian=False)
    fig, ax = invar.plot()
    plt.show()


if __name__ == '__main__':
    plot_testing()
