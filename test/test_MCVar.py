# test_MCVar.py

import pytest
import numpy as np
from scipy.stats import norm, randint, rv_discrete
from Monaco.MCVar import MCInVar, MCOutVar

generator = np.random.RandomState(74494861)
invarseeds = generator.randint(0, 2**31-1, size=10)

xk = np.array([1, 5, 6])
pk = np.ones(len(xk))/len(xk)
custom_dist = rv_discrete(name='custom', values=(xk, pk))


@pytest.fixture
def mcinvar_norm_random():
    return MCInVar('norm', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod='random')

@pytest.fixture
def mcinvar_custom_dist():
    return MCInVar('custom', ndraws=1000, dist=custom_dist, distkwargs=dict(), ninvar=1, samplemethod='random', seed=invarseeds[2])

def test_mcinvar_norm_sobol_warning():
    with pytest.warns(UserWarning, match='Infinite value'):
        MCInVar('norm', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod='sobol', ninvar=1)

def test_mcinvar_discrete():
    invar = MCInVar('randint', ndraws=1000, dist=randint, distkwargs={'low':1, 'high':5}, seed=invarseeds[0], samplemethod='random')
    assert invar.stats().mean == pytest.approx(2.5374625)

def test_mcinvar_continuous(mcinvar_norm_random):
    assert mcinvar_norm_random.stats().mean == pytest.approx(9.9451361)

def test_mcinvar_custom(mcinvar_custom_dist):
    assert mcinvar_custom_dist.stats().mean == pytest.approx(4.1058941)
    
def test_mcinvar_addvarstat(mcinvar_norm_random):
    mcinvar_norm_random.addVarStat(stattype='orderstatTI', statkwargs={'p':0.75, 'c':0.95, 'bound':'2-sided'})
    assert mcinvar_norm_random.mcvarstats[0].vals == pytest.approx([5.10075, 14.75052273])

def test_mcinvar_getVal(mcinvar_custom_dist):
    assert mcinvar_custom_dist.getVal(0).val == 5

def test_mcinvar_nummap():
    invar = MCInVar('map', ndraws=10, dist=custom_dist, distkwargs=dict(), ninvar=1, nummap={1:'a',5:'e',6:'f'}, samplemethod='random', seed=invarseeds[3])
    assert invar.vals == ['e', 'f', 'e', 'f', 'f', 'a', 'e', 'e', 'a', 'e', 'e']



def test_mcoutvar():
    outvar = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)
    assert outvar.getVal(1).val == 0
    assert outvar.stats().mean == pytest.approx(1.25)

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

    
### Inline Testing ###
'''
if __name__ == '__main__':
    from scipy.stats import norm, randint
    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)
    
    mcinvars = dict()
    mcinvars['randint'] = MCInVar('randint', ndraws=1000, dist=randint, distkwargs={'low':1, 'high':5}, seed=invarseeds[0], samplemethod='random')
    print(mcinvars['randint'].stats()) # expected: DescribeResult(nobs=1001, minmax=(1.0, 4.0), mean=2.5374625374625372, variance=1.218845154845155, skewness=-0.04361558166066395, kurtosis=-1.3248511805059544)
    mcinvars['norm1'] = MCInVar('norm', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod='sobol', ninvar=1) # Expected: Warning
    mcinvars['norm2'] = MCInVar('norm', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1], samplemethod='random') 
    print(mcinvars['norm2'].stats()) # expected: DescribeResult(nobs=1001, minmax=(-3.2763755735803652, 21.713592332532034), mean=9.94513614069452, variance=15.792150741321997, skewness=-0.0353388726779112, kurtosis=-0.08122682085492805)
    mcinvars['norm2'].addVarStat(stattype='orderstatTI', statkwargs={'p':0.75, 'c':0.95, 'bound':'2-sided'})
    print(mcinvars['norm2'].mcvarstats[0].vals) # expected: [ 5.10075    14.75052273]
    xk = np.array([1, 5, 6])
    pk = np.ones(len(xk))/len(xk)
    custom_dist = rv_discrete(name='custom', values=(xk, pk))
    mcinvars['custom'] = MCInVar('custom', ndraws=1000, dist=custom_dist, distkwargs=dict(), ninvar=1, samplemethod='random', seed=invarseeds[2])
    print(mcinvars['custom'].stats()) # expected: DescribeResult(nobs=1001, minmax=(1.0, 6.0), mean=4.105894105894106, variance=4.444775224775225, skewness=-0.7129149182621393, kurtosis=-1.3236396700106972)
    print(mcinvars['custom'].vals[1:10]) # expected: [5, 1, 1, 6, 6, 5, 5, 5, 5]
    print(mcinvars['custom'].getVal(0).val) # expected: 5.0
    mcinvars['map'] = MCInVar('map', ndraws=10, dist=custom, distkwargs=dict(), ninvar=1, nummap={1:'a',5:'e',6:'f'}, samplemethod='random', seed=invarseeds[3])
    print(mcinvars['map'].vals) # expected: ['e', 'f', 'e', 'f', 'f', 'a', 'e', 'e', 'a', 'e', 'e']
    print(mcinvars['map'].stats()) # expected: DescribeResult(nobs=11, minmax=(1.0, 6.0), mean=4.545454545454546, variance=3.2727272727272734, skewness=-1.405456737852613, kurtosis=0.38611111111111107)
    
    mcoutvars = dict()
    mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)
    print(mcoutvars['test'].getVal(1).val) # expected: 0
    print(mcoutvars['test'].stats()) # expected: DescribeResult(nobs=4, minmax=(0, 2), mean=1.25, variance=0.9166666666666666, skewness=-0.49338220021815865, kurtosis=-1.371900826446281)
    
    v = np.array([[1,1],[2,2],[3,3]])
    mcoutvars['test2'] = MCOutVar('test2', [v, v, v, v, v])
    mcoutvars['test2'].addVarStat(stattype='orderstatTI', statkwargs={'p':0.33, 'c':0.50, 'bound':'1-sided'})
    mcoutvars.update(mcoutvars['test2'].split())
    print(mcoutvars['test2 [0]'].nums) # expected: [array([1, 1]), array([1, 1]), array([1, 1]), array([1, 1]), array([1, 1])]
    print(mcoutvars['test2 [0]'].mcvarstats[0].vals) # expected: [1. 1.]
#'''
