# test_MCSim.py

import pytest
from Monaco.MCSim import MCSim
from scipy.stats import norm, randint
import numpy as np

seed = 74494861
def dummyfcn(*args):
    return 1

@pytest.fixture
def sim():
    sim = MCSim(name='Sim', ndraws=100, fcns={'preprocess':dummyfcn, 'run':dummyfcn, 'postprocess':dummyfcn}, firstcaseisnom=True, verbose=False, samplemethod='random', seed=seed)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low':1, 'high':6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc':10, 'scale':4})
    sim.genCases()
    return sim

def test_mcsim_dist_draws(sim):
    assert sim.mccases[0].mcinvals['Var1'].val == pytest.approx(3)
    assert sim.mccases[1].mcinvals['Var2'].val == pytest.approx(9.0195133)

def test_mcsim_corr_cov(sim):
    # We convert to numpy arrays because pytest.approx doesn't work on nested lists
    assert np.array(sim.corr()[0]) == pytest.approx(np.array([[ 1., -0.07565637], [-0.07565637,  1.]]))
    assert np.array(sim.cov()[0]) == pytest.approx(np.array([[ 2.27009901, -0.43928375], [-0.43928375, 14.85095717]]))


### Inline Testing ###
'''
if __name__ == '__main__':
    def dummyfcn(*args):
        return 1
    from scipy.stats import norm, randint
    seed = 74494861
    sim = MCSim(name='Sim', ndraws=100, fcns={'preprocess':dummyfcn, 'run':dummyfcn, 'postprocess':dummyfcn}, firstcaseisnom=True, samplemethod='random', seed=seed)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low':1, 'high':6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc':10, 'scale':4})
    sim.genCases()
    print(sim.seed)
    print(sim.mcinvars['Var1'].name)           # expected: Var1
    print(sim.mccases[0].mcinvals['Var1'].val) # expected: 3.0
    print(sim.mcinvars['Var2'].name)           # expected: Var2
    print(sim.mccases[1].mcinvals['Var2'].val) # expected: 9.019513324531903
    print(sim.corr())                          # expected: (array([[ 1., -0.07565637], [-0.07565637,  1.]]), ['Var1', 'Var2'])
    print(sim.cov())                           # expected: (array([[ 2.27009901, -0.43928375], [-0.43928375, 14.85095717]]), ['Var1', 'Var2'])
#'''