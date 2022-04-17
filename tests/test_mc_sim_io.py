# test_mc_sim_io.py

import pytest
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions
import cloudpickle
import os
import warnings


ndraws = 16
seed = 12362398

@pytest.fixture
def sim(tmp_path):
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

    sim = Sim(name='sim_io_test', ndraws=ndraws, fcns=fcns(),
              firstcaseismedian=False, seed=seed, cores=2, verbose=True,
              resultsdir=tmp_path)
    sim.runSim()
    return sim


@pytest.fixture
def sim_without_1_2(sim):
    os.remove(sim.resultsdir / 'sim_io_test_1.mccase')
    os.remove(sim.resultsdir / 'sim_io_test_2.mccase')
    with open(sim.resultsdir / 'sim_io_test.mcsim', 'rb') as file:
        with pytest.warns(UserWarning) as log:
            sim = cloudpickle.load(file)
    return (sim, log)


@pytest.fixture
def sim_with_extra_files(sim):
    for filename in ('dummyfile.mcsim', 'dummyfile.mccase', 'dummyfile.txt'):
        with open(sim.resultsdir / filename, 'wb'):
            pass
    return sim


def test_sim_load_partial(sim_without_1_2):
    (sim, log) = sim_without_1_2
    assert len(log) == 4
    assert 'sim_io_test_1.mccase expected but not found' in log[0].message.args[0]
    assert 'sim_io_test_2.mccase expected but not found' in log[1].message.args[0]
    assert 'The following cases were not loaded: [1, 2]' in log[2].message.args[0]
    assert 'The following cases have not been postprocessed: [1, 2]' in log[3].message.args[0]


def test_sim_run_partial(sim_without_1_2):
    (sim, log) = sim_without_1_2
    sim.runSim([1, 2])
    assert sim.casesrun == set(range(ndraws))
    assert sim.casespostprocessed == set(range(ndraws))


def test_sim_load_stale(sim_without_1_2):
    (sim, log) = sim_without_1_2
    sim.runSim([1, 2])
    with open(sim.resultsdir / 'sim_io_test.mcsim', 'rb') as file:
        with pytest.warns(UserWarning) as log:
            cloudpickle.load(file)
            expectedwarning = 'The following cases were loaded but may be stale: ' + \
                              '[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]'
            assert expectedwarning in log[-1].message.args[0]


def test_sim_run_incomplete(sim_without_1_2):
    (sim, log) = sim_without_1_2
    sim.runIncompleteSim()
    assert sim.casesrun == set(range(ndraws))
    assert sim.casespostprocessed == set(range(ndraws))


def test_sim_find_extra_files(sim_with_extra_files):
    with open(sim_with_extra_files.resultsdir / 'sim_io_test.mcsim', 'rb') as file:
        expectedwarning = 'The following extra .mcsim and .mccase files were found in the ' + \
                          'results directory, run removeExtraResultsFiles() to clean them up: ' + \
                          '[dummyfile.mccase, dummyfile.mcsim]'
        with pytest.warns(UserWarning) as log:
            cloudpickle.load(file)
            assert expectedwarning in log[0].message.args[0]


def test_sim_remove_extra_files(sim_with_extra_files):
    sim_with_extra_files.removeExtraResultsFiles()
    with open(sim_with_extra_files.resultsdir / 'sim_io_test.mcsim', 'rb') as file:
        with warnings.catch_warnings() as log:
            cloudpickle.load(file)
            assert not log
