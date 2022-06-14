# test_mc_sim_io.py

import pytest
import cloudpickle
import os
import warnings
import hashlib
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions
from scipy.stats import norm, randint


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
              firstcaseismedian=False, seed=seed, singlethreaded=True,
              savecasedata=True, savesimdata=True,
              verbose=True, resultsdir=tmp_path)
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6})
    sim.addInVar(name='Var2', dist=norm, distkwargs={'loc': 10, 'scale': 4})
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


def test_sim_export_invar_nums(sim):
    with pytest.raises(ValueError):
        sim.exportInVarNums('invars')

    hashes = ('f5280c4fb5340d9ba657190751a4c367', 'c55f69893622935ae8d9bd1974adb1f2')
    for i, filename in enumerate(['invars.csv', 'invars.json']):
        sim.exportInVarNums(filename)
        hash = hashlib.md5()
        with open(sim.resultsdir / filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        assert hashes[i] == hash.hexdigest()


def test_sim_import_outvals(sim):
    for filename in ['invars.csv', 'invars.json']:
        filepath = sim.resultsdir / filename
        sim.exportInVarNums(filename)
        sim.clearResults()
        sim.importOutVals(filepath)
        assert sim.invars['Var1'].nums == sim.outvars['Var1'].nums
        assert sim.invars['Var2'].nums == sim.outvars['Var2'].nums
        assert sim.outvars['Var1'].datasource == str(filepath.resolve())
