# test_MCVarStat.py

import pytest
import numpy as np
from monaco.MCVarStat import MCVarStat
from monaco.gaussian_statistics import sig2pct
from monaco.MCEnums import SampleMethod, StatBound, VarStat

@pytest.fixture
def mcinvar():
    from scipy.stats import norm
    from monaco.MCVar import MCInVar
    seed = 74494861
    return MCInVar('norm', ndraws=100000, dist=norm, distkwargs={'loc':0, 'scale':1}, samplemethod=SampleMethod.RANDOM, seed=seed)


bound=StatBound.ONESIDED
@pytest.mark.parametrize("stattype,statkwargs,vals", [
    (VarStat.ORDERSTATTI, {'p':sig2pct( 3, bound=bound), 'c':0.50, 'bound':bound}        ,  3.0060487),
    (VarStat.ORDERSTATP , {'p':sig2pct( 3, bound=bound), 'c':0.50, 'bound':StatBound.ALL}, [2.9965785, 3.0069887, 3.0209262]),
    (VarStat.GAUSSIANP  , {'p':sig2pct(-3, bound=bound), 'bound':bound}                  , -3.0021598),
    (VarStat.SIGMA      , {'sig':3, 'bound':bound}                                       ,  3.0020925),
    (VarStat.MEAN       , dict()                                                         , -3.361e-05),
])
def test_mcinvarstat(stattype,statkwargs,vals, mcinvar):
    mcinvarstat = MCVarStat(mcinvar, stattype=stattype, statkwargs=statkwargs)
    assert np.allclose(mcinvarstat.vals, vals)


def test_mcvarstat_setName(mcinvar):
    bound=StatBound.ONESIDED
    mcinvarstat = MCVarStat(mcinvar, stattype=VarStat.ORDERSTATTI, statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':bound})
    assert mcinvarstat.name == '1-sided P99.865/50.0% Confidence Interval'


v = np.array([-2, -1, 2, 3, 4, 5])

def test_mcoutvarstat_2d():
    from monaco.MCVar import MCOutVar
    outvar = MCOutVar('test', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseismedian=True)
    mcoutvarstat1 = MCVarStat(outvar, stattype=VarStat.ORDERSTATTI, statkwargs={'p':0.6, 'c':0.50, 'bound':StatBound.ALL})
    mcoutvarstat2 = MCVarStat(outvar, stattype=VarStat.MIN)
    assert(np.allclose(mcoutvarstat1.vals, [[-4, -2, 4], [-2, -1, 2], [-4, -2, 4], [-6, -3, 6], [-8, -4, 8], [-10, -5, 10]]))
    assert(np.allclose(mcoutvarstat2.vals,[ -4, -2, -4, -6, -8, -10.]))
    
    
def test_mcoutvarstat_2d_irregular():
    from monaco.MCVar import MCOutVar
    outvar = MCOutVar('test', [1*v, 2*v, 0*v, -1*v, [0,0]], firstcaseismedian=True)
    mcoutvarstat = MCVarStat(outvar, stattype=VarStat.MIN)
    assert(np.allclose(mcoutvarstat.vals,[-4, -2, -2, -3, -4, -5.]))
