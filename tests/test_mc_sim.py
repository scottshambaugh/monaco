# test_mc_sim.py

import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, randint
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions, SampleMethod


def sim_testing_preprocess(case):
    return ([case.ncase, ])

def sim_testing_run(casenum_in):
    casenum_out = casenum_in
    return (casenum_out)

def sim_testing_postprocess(case, casenum_out):
    case.addOutVal('casenum_out', casenum_out)

def sim_testing_fcns():
    fcns = {SimFunctions.PREPROCESS : sim_testing_preprocess,
            SimFunctions.RUN        : sim_testing_run,
            SimFunctions.POSTPROCESS: sim_testing_postprocess}
    return fcns

def sim_testing_preprocess_failure(case):
    if case.ncase == 0:
        raise Exception(f'Preprocess testing failed for case {case.ncase}')
    else:
        return ([case.ncase, ])

def sim_testing_run_failure(casenum_in):
    casenum_out = casenum_in
    if casenum_out == 0:
        raise Exception(f'Run testing failed for case {casenum_in}')
    else:
        return (casenum_out)

def sim_testing_postprocess_failure(case, casenum_out):
    if casenum_out == 0:
        raise Exception(f'Postprocess testing failed for case {case.ncase}')
    else:
        case.addOutVal('casenum_out', casenum_out)


@pytest.fixture
def sim():
    seed = 74494861
    sim = Sim(name='Sim', ndraws=16, fcns=sim_testing_fcns(), firstcaseismedian=True,
              verbose=False, samplemethod=SampleMethod.RANDOM,
              seed=seed, debug=True, singlethreaded=True, daskkwargs=dict(),
              savesimdata=False, savecasedata=False)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc': 10, 'scale': 4})
    return sim

@pytest.fixture
def sim_singlethreaded(sim):
    sim.name = 'Sim single-threaded'
    sim.runSim()
    return sim

@pytest.fixture
def sim_parallel(sim):
    sim.name = 'Sim parallel (dask)'
    sim.singlethreaded = False
    sim.initDaskClient()
    sim.runSim()
    return sim

@pytest.fixture
def sim_parallel_expanded(sim):
    sim.name = 'Sim parallel expanded (dask)'
    sim.singlethreaded = False
    sim.initDaskClient()
    sim.drawVars()
    sim.genCases()
    sim.preProcessCases()
    sim.runCases()
    sim.postProcessCases()
    sim.genOutVars()
    return sim


def test_sim_getitem(sim_singlethreaded):
    # Test case number
    assert sim_singlethreaded[0].invals['Var1'].val == pytest.approx(3)
    # Test variable name
    assert sim_singlethreaded['Var1'].vals[0] == pytest.approx(3)


def test_sim_dist_draws(sim_singlethreaded, sim_parallel, sim_parallel_expanded):
    for sim in (sim_singlethreaded, sim_parallel, sim_parallel_expanded):
        assert sim.cases[0].invals['Var1'].val == pytest.approx(3)
        assert sim.cases[1].invals['Var2'].val == pytest.approx(9.98228884)


def test_sim_scalaroutvars(sim_singlethreaded, sim_parallel, sim_parallel_expanded):
    for sim in (sim_singlethreaded, sim_parallel, sim_parallel_expanded):
        assert len(sim.scalarOutVars()) == 1


def test_sim_custom_vals():
    seed = 74494861
    sim = Sim(name='Sim', ndraws=3, fcns=sim_testing_fcns(), firstcaseismedian=False,
              verbose=False, samplemethod=SampleMethod.RANDOM,
              seed=seed, debug=True, singlethreaded=True, daskkwargs=dict(),
              savesimdata=False, savecasedata=False)
    sim.addInVar(name='Var1', vals=[1, 2, 3])
    sim.addInVar(name='Var2', vals=['a', 'b', 'c'], nummap={3: 'a', 4: 'b', 5: 'c'})
    sim.drawVars()
    sim.genCases()

    assert sim.cases[0].invals['Var1'].val == 1
    assert sim.cases[0].invals['Var2'].val == 'a'
    assert sim.cases[0].invals['Var2'].num == 3

    with pytest.raises(ValueError, match="Cannot provide both 'dist' and 'vals'"):
        sim.addInVar(name='Var3', dist=norm, distkwargs={'loc': 10, 'scale': 4},
                     vals=[1, 2, 3])


def test_sim_corr_cov(sim_singlethreaded, sim_parallel, sim_parallel_expanded):
    for sim in (sim_singlethreaded, sim_parallel, sim_parallel_expanded):
        # Convert to numpy arrays because pytest.approx doesn't work on nested lists
        assert np.array(sim.corr()[0]) \
            == pytest.approx(np.array([[ 1,           0.44720757, -0.18114221],
                                       [ 0.44720757,  1,          -0.01869903],
                                       [-0.18114221, -0.01869903,  1]]))
        assert np.array(sim.cov()[0]) \
            == pytest.approx(np.array([[2.05882353,  2.44162274, -1.3125],
                                       [2.44162274, 14.47837096, -0.35929331],
                                       [-1.3125   , -0.35929331, 25.5]]))


def test_sim_preprocess_failure(sim_singlethreaded, sim_parallel, sim_parallel_expanded):
    for sim in (sim_singlethreaded, sim_parallel, sim_parallel_expanded):
        fcns = {SimFunctions.PREPROCESS : sim_testing_preprocess_failure,
                SimFunctions.RUN        : sim_testing_run,
                SimFunctions.POSTPROCESS: sim_testing_postprocess}
        sim.fcns = fcns
        sim.debug = True
        sim.clearResults()
        with pytest.raises(Exception, match='Preprocess testing failed for case 0'):
            sim.runSim()

        sim.debug = False
        sim.clearResults()
        sim.runSim()
        assert len(sim.casespreprocessed) == sim.ncases - 1
        assert len(sim.casesrun) == sim.ncases - 1
        assert len(sim.casespostprocessed) == sim.ncases - 1


def test_sim_run_failure(sim_singlethreaded, sim_parallel, sim_parallel_expanded):
    for sim in (sim_singlethreaded, sim_parallel, sim_parallel_expanded):
        fcns = {SimFunctions.PREPROCESS : sim_testing_preprocess,
                SimFunctions.RUN        : sim_testing_run_failure,
                SimFunctions.POSTPROCESS: sim_testing_postprocess}
        sim.fcns = fcns
        sim.debug = True
        sim.clearResults()
        with pytest.raises(Exception, match='Run testing failed for case 0'):
            sim.runSim()

        sim.debug = False
        sim.clearResults()
        sim.runSim()
        assert len(sim.casespreprocessed) == sim.ncases
        assert len(sim.casesrun) == sim.ncases - 1
        assert len(sim.casespostprocessed) == sim.ncases - 1


def test_sim_postprocess_failure(sim_singlethreaded, sim_parallel, sim_parallel_expanded):
    for sim in (sim_singlethreaded, sim_parallel, sim_parallel_expanded):
        fcns = {SimFunctions.PREPROCESS : sim_testing_preprocess,
                SimFunctions.RUN        : sim_testing_run,
                SimFunctions.POSTPROCESS: sim_testing_postprocess_failure}
        sim.fcns = fcns
        sim.debug = True
        sim.clearResults()
        with pytest.raises(Exception, match='Postprocess testing failed for case 0'):
            sim.runSim()

        sim.debug = False
        sim.clearResults()
        sim.runSim()
        assert len(sim.casespreprocessed) == sim.ncases
        assert len(sim.casesrun) == sim.ncases
        assert len(sim.casespostprocessed) == sim.ncases - 1


# Does not test the plot appearances, but does check that the codepaths can run
def test_gen_plots():
    plot_testing(show=False)
    assert True


def plot_testing(show=False):
    seed = 74494861
    sim = Sim(name='Sim', ndraws=16, fcns=sim_testing_fcns(), firstcaseismedian=True,
              verbose=False, samplemethod=SampleMethod.RANDOM,
              seed=seed, debug=True, singlethreaded=True, daskkwargs=dict(),
              savesimdata=False, savecasedata=False)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc': 10, 'scale': 4})
    sim.addInVar(name='Var3', vals=['a']*11 + ['b']*4 + ['c']*2,
                 nummap={3: 'a', 5: 'b', 4: 'c'})
    sim.runSim()

    fig, axs = sim.plot()
    if show:
        plt.show(block=True)


if __name__ == '__main__':
    plot_testing(show=True)
    plt.show()
