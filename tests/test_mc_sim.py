# test_mc_sim.py

import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, randint
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions, SampleMethod

try:
    import dask  # noqa: F401
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SIM_FIXTURES = [
    "sim_singlethreaded",
    "sim_multiprocessing",
    "sim_multiprocessing_expanded",
    "sim_dask",
    "sim_dask_expanded"
]

def sim_testing_preprocess(case):
    return ([case.ncase, ])

def sim_testing_run(casenum_in):
    casenum_out = casenum_in
    return (casenum_out)

def sim_testing_postprocess(case, casenum_out):
    case.addOutVal('casenum_out', casenum_out)
    case.addOutVal('casenum_list', [casenum_out] * (case.ncase + 1))
    case.addOutVal('casenum_array', np.array([casenum_out] * (case.ncase + 1)))
    if HAS_PANDAS:
        case.addOutVal('casenum_series', pd.Series([casenum_out] * (case.ncase + 1)))
        case.addOutVal('casenum_index', pd.Index([casenum_out] * (case.ncase + 1)))

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
def sim_base():
    seed = 74494861
    sim = Sim(name='Sim', ndraws=16, fcns=sim_testing_fcns(), firstcaseismedian=True,
              verbose=False, samplemethod=SampleMethod.RANDOM,
              seed=seed, debug=True, singlethreaded=True, daskkwargs=dict(),
              savesimdata=False, savecasedata=False)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc': 10, 'scale': 4})
    return sim

@pytest.fixture
def sim_singlethreaded(sim_base):
    sim = sim_base
    sim.name = 'Sim single-threaded'
    sim.runSim()
    return sim

@pytest.fixture
def sim_multiprocessing(sim_base):
    sim = sim_base
    sim.name = 'Sim multiprocessing'
    sim.singlethreaded = False
    sim.usedask = False
    sim.runSim()
    return sim

@pytest.fixture
def sim_multiprocessing_expanded(sim_base):
    sim = sim_base
    sim.name = 'Sim multiprocessing expanded'
    sim.singlethreaded = False
    sim.usedask = False
    sim.drawVars()
    sim.genCases()
    sim.initMultiprocessingPool()
    sim.preProcessCases()
    sim.runCases()
    sim.postProcessCases()
    sim.genOutVars()
    return sim

@pytest.fixture
def sim_dask(sim_base):
    if not HAS_DASK:
        return None
    sim = sim_base
    sim.name = 'Sim dask'
    sim.singlethreaded = False
    sim.usedask = True
    sim.runSim()
    return sim

@pytest.fixture
def sim_dask_expanded(sim_base):
    if not HAS_DASK:
        return None
    sim = sim_base
    sim.name = 'Sim dask expanded'
    sim.singlethreaded = False
    sim.usedask = True
    sim.drawVars()
    sim.genCases()
    sim.initDaskClient()
    sim.preProcessCases()
    sim.runCases()
    sim.postProcessCases()
    sim.genOutVars()
    return sim


@pytest.fixture
def sim_fixture(request):
    """Fixture that provides specific sim fixtures based on parametrization"""
    return request.getfixturevalue(request.param)


def test_sim_singlethreaded_fixture(sim_singlethreaded):
    assert sim_singlethreaded.singlethreaded

def test_sim_multiprocessing_fixture(sim_multiprocessing):
    assert not sim_multiprocessing.singlethreaded
    assert not sim_multiprocessing.usedask

def test_sim_multiprocessing_expanded_fixture(sim_multiprocessing_expanded):
    assert not sim_multiprocessing_expanded.singlethreaded
    assert not sim_multiprocessing_expanded.usedask

def test_sim_dask_fixture(sim_dask):
    if sim_dask is None:
        pytest.skip("Dask is not installed, skipping parallel tests")
    assert not sim_dask.singlethreaded

def test_sim_dask_expanded_fixture(sim_dask_expanded):
    if sim_dask_expanded is None:
        pytest.skip("Dask is not installed, skipping parallel tests")
    assert not sim_dask_expanded.singlethreaded


def test_sim_getitem(sim_singlethreaded):
    # Test case number
    assert sim_singlethreaded[0].invals['Var1'].val == pytest.approx(3)
    # Test variable name
    assert sim_singlethreaded['Var1'].vals[0] == pytest.approx(3)


@pytest.mark.parametrize("sim_fixture", SIM_FIXTURES, indirect=True)
def test_sim_dist_draws(sim_fixture):
    if sim_fixture is None:
        pytest.skip("Dask is not installed, skipping parallel tests")

    assert sim_fixture.cases[0].invals['Var1'].val == pytest.approx(3)
    assert sim_fixture.cases[1].invals['Var2'].val == pytest.approx(9.98228884)


@pytest.mark.parametrize("sim_fixture", SIM_FIXTURES, indirect=True)
def test_sim_scalaroutvars(sim_fixture):
    if sim_fixture is None:
        pytest.skip("Dask is not installed, skipping parallel tests")

    assert len(sim_fixture.scalarOutVars()) == 1


def test_sim_extendoutvars(sim_singlethreaded):
    sim = sim_singlethreaded
    assert len(sim.outvars['casenum_list'][0].val) == 1
    assert len(sim.outvars['casenum_list'][0].num) == 1
    assert len(sim.outvars['casenum_array'][0].val) == 1
    if HAS_PANDAS:
        assert len(sim.outvars['casenum_series'][0].val) == 1
        assert len(sim.outvars['casenum_index'][0].val) == 1
    sim.extendOutVars()
    for i in range(sim.ncases):
        assert len(sim.outvars['casenum_list'][i].val) == sim.ncases
        assert len(sim.outvars['casenum_list'][i].num) == sim.ncases
        assert len(sim.outvars['casenum_array'][i].val) == sim.ncases
        if HAS_PANDAS:
            assert len(sim.outvars['casenum_series'][i].val) == sim.ncases
            assert len(sim.outvars['casenum_index'][i].val) == sim.ncases


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


@pytest.mark.parametrize("sim_fixture", SIM_FIXTURES, indirect=True)
def test_sim_corr_cov(sim_fixture):
    if sim_fixture is None:
        pytest.skip("Dask is not installed, skipping parallel tests")

    sim = sim_fixture
    # Convert to numpy arrays because pytest.approx doesn't work on nested lists
    assert np.array(sim.corr()[0]) \
        == pytest.approx(np.array([[ 1,           0.44720757, -0.18114221],
                                   [ 0.44720757,  1,          -0.01869903],
                                   [-0.18114221, -0.01869903,  1]]))
    assert np.array(sim.cov()[0]) \
        == pytest.approx(np.array([[2.05882353,  2.44162274, -1.3125],
                                   [2.44162274, 14.47837096, -0.35929331],
                                   [-1.3125   , -0.35929331, 25.5]]))


@pytest.mark.parametrize("sim_fixture", SIM_FIXTURES, indirect=True)
def test_sim_preprocess_failure(sim_fixture):
    if sim_fixture is None:
        pytest.skip("Dask is not installed, skipping parallel tests")

    sim = sim_fixture
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


@pytest.mark.parametrize("sim_fixture", SIM_FIXTURES, indirect=True)
def test_sim_run_failure(sim_fixture):
    if sim_fixture is None:
        pytest.skip("Dask is not installed, skipping parallel tests")

    sim = sim_fixture
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


@pytest.mark.parametrize("sim_fixture", SIM_FIXTURES, indirect=True)
def test_sim_postprocess_failure(sim_fixture):
    if sim_fixture is None:
        pytest.skip("Dask is not installed, skipping parallel tests")

    sim = sim_fixture
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
