# test_MCSim.py

import pytest
from monaco.MCSim import MCSim
from monaco.MCEnums import MCFunctions, SampleMethod
from test.mcsim_testing_fcns import dummyfcn
import numpy as np

seed = 74494861

@pytest.fixture
def sim():
    from scipy.stats import norm, randint
    fcns = {MCFunctions.PREPROCESS:dummyfcn, MCFunctions.RUN:dummyfcn, MCFunctions.POSTPROCESS:dummyfcn}
    sim = MCSim(name='Sim', ndraws=100, fcns=fcns, firstcaseisnom=True, verbose=False, samplemethod=SampleMethod.RANDOM, seed=seed, debug=True, cores=1)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low':1, 'high':6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc':10, 'scale':4})
    sim.drawVars()
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
# Can run here or copy into bottom of main file
def inline_testing():
    def dummyfcn2(*args):
        return 1

    from scipy.stats import norm, randint
    seed = 74494861
    fcns2 = {MCFunctions.PREPROCESS:dummyfcn2, MCFunctions.RUN:dummyfcn2, MCFunctions.POSTPROCESS:dummyfcn2}
    sim2 = MCSim(name='Sim2', ndraws=100, fcns=fcns2, firstcaseisnom=True, samplemethod=SampleMethod.RANDOM, seed=seed, debug=True, cores=1)
    sim2.addInVar(name='Var1', dist=randint, distkwargs={'low':1, 'high':6})
    sim2.addInVar(name='Var2', dist=norm, distkwargs={'loc':10, 'scale':4})
    sim2.drawVars()
    sim2.genCases()
    print(sim2.seed)
    print(sim2.mcinvars['Var1'].name)           # expected: Var1
    print(sim2.mccases[0].mcinvals['Var1'].val) # expected: 3.0
    print(sim2.mcinvars['Var2'].name)           # expected: Var2
    print(sim2.mccases[1].mcinvals['Var2'].val) # expected: 9.019513324531903
    print(sim2.corr())                          # expected: (array([[ 1., -0.07565637], [-0.07565637,  1.]]), ['Var1', 'Var2'])
    print(sim2.cov())                           # expected: (array([[ 2.27009901, -0.43928375], [-0.43928375, 14.85095717]]), ['Var1', 'Var2'])
    print(MCFunctions.PREPROCESS == 'preprocess')

if __name__ == '__main__':
    inline_testing()
