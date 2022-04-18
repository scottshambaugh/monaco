# test_mc_sim.py

import pytest
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions, SampleMethod
import numpy as np

seed = 74494861

@pytest.fixture
def sim():
    from scipy.stats import norm, randint

    def testing_preprocess(case):
        return ([True, ])

    def testing_run(inputs):
        return (True)

    def testing_postprocess(case, output):
        case.addOutVal('casenum', case.ncase)

    def fcns():
        fcns = {SimFunctions.PREPROCESS : testing_preprocess,
                SimFunctions.RUN        : testing_run,
                SimFunctions.POSTPROCESS: testing_postprocess}
        return fcns

    sim = Sim(name='Sim', ndraws=16, fcns=fcns(), firstcaseismedian=True,
              verbose=False, samplemethod=SampleMethod.RANDOM,
              seed=seed, debug=True, singlethreaded=True, daskkwargs=dict(),
              savesimdata=False, savecasedata=False)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc': 10, 'scale': 4})
    return sim

@pytest.fixture
def sim_singlethreaded(sim):
    sim.runSim()
    return sim

@pytest.fixture
def sim_parallel(sim):
    sim.singlethreaded = False
    sim.initDaskClient()
    sim.runSim()
    return sim


def test_sim_dist_draws(sim_singlethreaded, sim_parallel):
    for sim in (sim_singlethreaded, sim_parallel):
        assert sim.cases[0].invals['Var1'].val == pytest.approx(3)
        assert sim.cases[1].invals['Var2'].val == pytest.approx(9.98228884)


def test_sim_scalaroutvars(sim_singlethreaded, sim_parallel):
    for sim in (sim_singlethreaded, sim_parallel):
        assert len(sim.scalarOutVars()) == 1


def test_sim_corr_cov(sim_singlethreaded, sim_parallel):
    for sim in (sim_singlethreaded, sim_parallel):
        # Convert to numpy arrays because pytest.approx doesn't work on nested lists
        assert np.array(sim.corr()[0]) \
            == pytest.approx(np.array([[ 1,           0.44720757, -0.18114221],
                                       [ 0.44720757,  1,          -0.01869903],
                                       [-0.18114221, -0.01869903,  1]]))
        assert np.array(sim.cov()[0]) \
            == pytest.approx(np.array([[2.05882353,  2.44162274, -1.3125],
                                       [2.44162274, 14.47837096, -0.35929331],
                                       [-1.3125   , -0.35929331, 25.5]]))
