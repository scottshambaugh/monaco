# test_MCVar.py

import pytest
import numpy as np
from scipy.stats import rv_discrete
from monaco.MCVar import MCInVar, MCOutVar
from monaco.MCEnums import SampleMethod

generator = np.random.RandomState(74494861)
invarseeds = generator.randint(0, 2**31-1, size=10)

xk = np.array([1, 5, 6])
pk = np.ones(len(xk))/len(xk)
custom_dist = rv_discrete(name='custom', values=(xk, pk))
lognorm_sigma, lognorm_mu = 1, 2


@pytest.fixture
def mcinvar_norm_random():
    from scipy.stats import norm
    return MCInVar('norm', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod=SampleMethod.RANDOM, firstcaseismedian=False)

@pytest.fixture
def mcinvar_lognorm_random():
    from scipy.stats import lognorm
    return MCInVar('lognorm', ndraws=1000, dist=lognorm, distkwargs={'s':lognorm_sigma, 'scale':np.exp(lognorm_mu)}, seed=invarseeds[1], samplemethod=SampleMethod.RANDOM, firstcaseismedian=False)

@pytest.fixture
def mcinvar_custom_dist():
    return MCInVar('custom', ndraws=1000, dist=custom_dist, distkwargs=dict(), ninvar=1, samplemethod=SampleMethod.RANDOM, seed=invarseeds[2], firstcaseismedian=False)

def test_mcinvar_norm_sobol_warning():
    from scipy.stats import norm
    with pytest.warns(UserWarning, match='Infinite value'):
        MCInVar('norm', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod=SampleMethod.SOBOL, ninvar=1, firstcaseismedian=False)

def test_mcinvar_discrete():
    from scipy.stats import randint
    invar = MCInVar('randint', ndraws=1000, dist=randint, distkwargs={'low':1, 'high':5}, seed=invarseeds[0], samplemethod=SampleMethod.RANDOM)
    assert invar.stats().mean == pytest.approx(2.538)

def test_mcinvar_continuous(mcinvar_norm_random):
    assert mcinvar_norm_random.stats().mean == pytest.approx(9.9450812)

def test_mcinvar_custom(mcinvar_custom_dist):
    assert mcinvar_custom_dist.stats().mean == pytest.approx(4.105)
    
def test_mcinvar_addvarstat(mcinvar_norm_random):
    mcinvar_norm_random.addVarStat(stattype='orderstatTI', statkwargs={'p':0.75, 'c':0.95, 'bound':'2-sided'})
    assert mcinvar_norm_random.mcvarstats[0].vals == pytest.approx([5.10075, 14.75052273])

def test_mcinvar_getVal(mcinvar_custom_dist):
    assert mcinvar_custom_dist.getVal(0).val == 5

def test_mcinvar_getMean(mcinvar_lognorm_random):
    assert pytest.approx(mcinvar_lognorm_random.getDistMean(), lognorm_mu + lognorm_sigma**2 / 2)

def test_mcinvar_getMedian(mcinvar_lognorm_random):
    assert pytest.approx(mcinvar_lognorm_random.getDistMedian(), lognorm_mu)

def test_mcinvar_nummap():
    invar = MCInVar('map', ndraws=10, dist=custom_dist, distkwargs=dict(), ninvar=1, nummap={1:'a',5:'e',6:'f'}, samplemethod=SampleMethod.RANDOM, seed=invarseeds[3], firstcaseismedian=False)
    assert invar.vals == ['f', 'e', 'f', 'f', 'a', 'e', 'e', 'a', 'e', 'e']



def test_mcoutvar():
    outvar = MCOutVar('test', [1, 0, 2, 2], firstcaseismedian=True)
    assert outvar.getVal(1).val == 0
    assert outvar.stats().mean == pytest.approx(1.25)
    assert outvar.getMedianVal().val == 1

def test_mcoutvar_extractValMap():
    outvar = MCOutVar('test', ['a', 'b', 'c', 'b'], firstcaseismedian=True)
    assert outvar.valmap == {'a':0, 'b':1, 'c':2}
    
@pytest.fixture
def mcoutvars_split():
    mcoutvars = dict()
    v = np.array([[1,1],[2,2],[3,3]])
    mcoutvars['test'] = MCOutVar('test', [v, v, v, v, v])
    mcoutvars['test'].addVarStat(stattype='orderstatTI', statkwargs={'p':0.33, 'c':0.50, 'bound':'1-sided'})
    mcoutvars.update(mcoutvars['test'].split())
    return mcoutvars

def test_mcoutvar_split(mcoutvars_split):
    assert np.array_equal(mcoutvars_split['test [0]'].nums, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    assert np.array_equal(mcoutvars_split['test [1]'].nums, [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])


def test_mcoutvar_split_orderstat(mcoutvars_split):
    assert mcoutvars_split['test [0]'].mcvarstats[0].vals == pytest.approx([1, 1])
