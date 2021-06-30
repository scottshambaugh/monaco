# test_MCVarStat.py

import pytest
import numpy as np
from Monaco.MCVarStat import MCVarStat
from Monaco.gaussian_statistics import sig2pct

@pytest.fixture
def mcinvar():
    from scipy.stats import norm
    from Monaco.MCVar import MCInVar
    seed = 74494861
    return MCInVar('norm', ndraws=100000, dist=norm, distkwargs={'loc':0, 'scale':1}, samplemethod='random', seed=seed)


bound='1-sided'
@pytest.mark.parametrize("stattype,statkwargs,vals", [
    ('orderstatTI', {'p':sig2pct( 3, bound=bound), 'c':0.50, 'bound':bound},  3.0060487),
    ( 'orderstatP', {'p':sig2pct( 3, bound=bound), 'c':0.50, 'bound':'all'}, [2.9965785, 3.0069887, 3.0209262]),
    (  'gaussianP', {'p':sig2pct(-3, bound=bound), 'bound':bound}          , -3.0021598),
    (     'sigmaP', {'sig':3, 'bound':bound}                               ,  3.0020925),
    (       'mean', dict()                                                 , -3.361e-05),
])
def test_mcinvarstat(stattype,statkwargs,vals, mcinvar):
    mcinvarstat = MCVarStat(mcinvar, stattype=stattype, statkwargs=statkwargs)
    assert np.allclose(mcinvarstat.vals, vals)


def test_mcvarstat_setName(mcinvar):
    bound='1-sided'
    mcinvarstat = MCVarStat(mcinvar, stattype='orderstatTI', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':bound})
    assert mcinvarstat.name == '1-sided P99.865/50.0% Confidence Interval'


v = np.array([-2, -1, 2, 3, 4, 5])

def test_mcoutvarstat_2d():
    from Monaco.MCVar import MCOutVar
    outvar = MCOutVar('test', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
    mcoutvarstat1 = MCVarStat(outvar, stattype='orderstatTI', statkwargs={'p':0.6, 'c':0.50, 'bound':'all'})
    mcoutvarstat2 = MCVarStat(outvar, stattype='min')
    assert(np.allclose(mcoutvarstat1.vals, [[-4, -2, 4], [-2, -1, 2], [-4, -2, 4], [-6, -3, 6], [-8, -4, 8], [-10, -5, 10]]))
    assert(np.allclose(mcoutvarstat2.vals,[ -4, -2, -4, -6, -8, -10.]))
    
    
def test_mcoutvarstat_2d_irregular():
    from Monaco.MCVar import MCOutVar
    outvar = MCOutVar('test', [1*v, 2*v, 0*v, -1*v, [0,0]], firstcaseisnom=True)
    mcoutvarstat = MCVarStat(outvar, stattype='min')
    assert(np.allclose(mcoutvarstat.vals,[-4, -2, -2, -3, -4, -5.]))



### Inline Testing ###
# Can run here or copy into bottom of main file
#'''
if __name__ == '__main__':
    from scipy.stats import norm
    from Monaco.MCVar import MCInVar, MCOutVar
    seed = 74494861

    mcinvar2 = MCInVar('norm', ndraws=100000, dist=norm, distkwargs={'loc':0, 'scale':1}, samplemethod='random', seed=seed)
    bound='1-sided'
    mcinvarstat1 = MCVarStat(mcinvar2, stattype='orderstatTI', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':bound})
    mcinvarstat2 = MCVarStat(mcinvar2, stattype='orderstatP', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':'all'})
    mcinvarstat3 = MCVarStat(mcinvar2, stattype='gaussianP', statkwargs={'p':sig2pct(-3, bound=bound), 'bound':bound})
    mcinvarstat4 = MCVarStat(mcinvar2, stattype='sigmaP', statkwargs={'sig':3, 'bound':bound})
    mcinvarstat5 = MCVarStat(mcinvar2, stattype='mean')
    print(mcinvarstat1.name) # expected: 1-sided P99.865/50.0% Confidence Interval
    print(mcinvarstat1.k)    # expected: 135
    print(mcinvarstat1.vals) # expected: 3.006048706515629
    print(mcinvarstat2.k)    # expected: 8
    print(mcinvarstat2.vals) # expected: [2.99657859 3.00698873 3.02092621]
    print(mcinvarstat3.vals) # expected: -3.00215981041833
    print(mcinvarstat4.vals) # expected: 3.0020925785191706
    print(mcinvarstat5.vals) # expected: -3.361594957953134e-05
    
    v = np.array([-2, -1, 2, 3, 4, 5])
    var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
    mcoutvarstat1 = MCVarStat(var2, stattype='orderstatTI', statkwargs={'p':0.6, 'c':0.50, 'bound':'all'})
    mcoutvarstat2 = MCVarStat(var2, stattype='min')
    var3 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, [0,0]], firstcaseisnom=True)
    mcoutvarstat3 = MCVarStat(var3, stattype='min')
    print(mcoutvarstat1.name) # expected: 2-sided P60.0/50.0% Confidence Interval
    print(mcoutvarstat1.vals) # expected: [[ -4.  -2.   4.] [ -2.  -1.   2.] [ -4.  -2.   4.] [ -6.  -3.   6.] [ -8.  -4.   8.] [-10.  -5.  10.]]
    print(mcoutvarstat2.vals) # expected: [ -4.  -2.  -4.  -6.  -8. -10.]
    print(mcoutvarstat3.vals) # expected: [-4. -2. -2. -3. -4. -5.]
#'''
