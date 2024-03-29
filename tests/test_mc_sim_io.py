# test_mc_sim_io.py

import pytest
import cloudpickle
import os
import warnings
import hashlib
import numpy as np
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
    var1_nummap = {1.0: 'a', 2.0: 'b', 3.0: 'c', 4.0: 'd', 5.0: 'e'}
    sim.addInVar(name='Var1', dist=randint, distkwargs={'low': 1, 'high': 6}, nummap=var1_nummap)
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


@pytest.mark.parametrize("filename, expected_hash", [
    ('invars.csv', 'aa6f66046f235fe1a05743b34a87ca87'),
    ('invars.json', '8cbe1328474349a7bf2ab5145a31ea46'),
])
def test_sim_export_invars(sim, filename, expected_hash):
    with pytest.raises(ValueError):
        sim.exportInVars('invars')

    sim.exportInVars(filename)
    hash = hashlib.md5()
    with open(sim.resultsdir / filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    assert expected_hash == hash.hexdigest()


@pytest.mark.parametrize("filename, expected_hash", [
    ('outvars.csv', '51a78ae66543a9655499fe5a17781e0c'),
    ('outvars.json', '4443c042ec0ebec1489745f85aa90d6d'),
])
def test_sim_export_outvars(sim, filename, expected_hash):
    with pytest.raises(ValueError):
        sim.exportOutVars('outvars')

    sim.exportOutVars(filename)
    hash = hashlib.md5()
    with open(sim.resultsdir / filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    assert expected_hash == hash.hexdigest()


@pytest.mark.parametrize("filename", ['invars.csv', 'invars.json'])
def test_sim_import_invars(sim, filename):
    filepath = sim.resultsdir / filename
    var1 = sim.invars['Var1']
    var2 = sim.invars['Var2']
    dists = [var1.dist, var2.dist]
    distskwargs = [var1.distkwargs, var2.distkwargs]
    nummaps = [var1.nummap, None]

    sim.exportInVars(filename)
    sim.reset()
    sim.importInVars(filepath, dists=dists, distskwargs=distskwargs, nummaps=nummaps)
    assert sim.invars['Var1'].nums == var1.nums
    assert sim.invars['Var2'].nums == var2.nums
    assert sim.invars['Var1'].vals == var1.vals
    assert sim.invars['Var2'].vals == var2.vals
    # The pcts for the imported Var1 will be different from the original because
    # the distribution is discrete
    assert np.allclose(sim.invars['Var1'].pcts, np.ceil(5*np.array(var1.pcts))/5)
    assert np.allclose(sim.invars['Var2'].pcts, var2.pcts)
    assert sim.invars['Var1'].datasource == str(filepath.resolve())

    sim.runSim()
    assert sim.outvars['casenum'].vals == list(range(sim.ncases))


@pytest.mark.parametrize("filename", ['invars.csv', 'invars.json'])
def test_sim_import_outvars(sim, filename):
    filepath = sim.resultsdir / filename
    var1 = sim.invars['Var1']
    var2 = sim.invars['Var2']
    nummaps = [var1.nummap, None]

    sim.exportInVars(filename)
    with pytest.raises(ValueError):
        sim.importOutVars(filepath)

    sim.reset()
    sim.importOutVars(filepath, nummaps=nummaps)
    assert sim.outvars['Var1'].nums == var1.nums
    assert sim.outvars['Var2'].nums == var2.nums
    assert sim.outvars['Var1'].vals == var1.vals
    assert sim.outvars['Var2'].vals == var2.vals
    assert sim.outvars['Var1'].datasource == str(filepath.resolve())
