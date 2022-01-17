# test_mc_sim.py

import pytest
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions, SampleMethod
import numpy as np

seed = 74494861

def dummyfcn():
    return 1

@pytest.fixture
def sim():
    from scipy.stats import norm, randint
    fcns = {SimFunctions.PREPROCESS : dummyfcn,
            SimFunctions.RUN        : dummyfcn,
            SimFunctions.POSTPROCESS: dummyfcn}
    sim = Sim(name='Sim', ndraws=100, fcns=fcns, firstcaseismedian=True,
              verbose=False, samplemethod=SampleMethod.RANDOM,
              seed=seed, debug=True, cores=1)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc': 10, 'scale': 4})
    sim.drawVars()
    sim.genCases()
    return sim

def test_sim_dist_draws(sim):
    assert sim.cases[0].invals['Var1'].val == pytest.approx(3)
    assert sim.cases[1].invals['Var2'].val == pytest.approx(9.98228884)

def test_sim_corr_cov(sim):
    # We convert to numpy arrays because pytest.approx doesn't work on nested lists
    assert np.array(sim.corr()[0]) \
           == pytest.approx(np.array([[1, 0.06495995], [0.06495995, 1]]))
    assert np.array(sim.cov()[0]) \
           == pytest.approx(np.array([[1.91643564,  0.37543008], [0.37543008, 17.42900278]]))
