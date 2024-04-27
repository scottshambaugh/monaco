# test_mc_sim_io.py

import pytest
import cloudpickle
import os
import warnings
import csv
import json
import numpy as np
from monaco.mc_sim import Sim
from monaco.mc_enums import SimFunctions
from scipy.stats import norm, randint


ndraws = 16
seed = 12362398
expected_data = {
    "Var1": [4.0, 2.0, 1.0, 3.0, 4.0, 1.0, 3.0, 5.0, 5.0, 3.0, 2.0, 4.0, 4.0, 1.0, 2.0, 5.0],
    "Var2": [
        3.6032930546602886, 12.842558315597907, 7.713622164768125, 10.16861392776618,
        8.941121864635, 11.680281868640574, 5.684657237716662, 16.015476112872,
        8.420645518481285, 10.802167161581739, 5.250984893760062, 13.723048169120581,
        6.6765074149514145, 20.746921589883062, 9.58112220045209, 12.396945336261876
    ],
    "casenum": list(range(ndraws))
}

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


def test_sim_export_csv(sim):
    with pytest.raises(ValueError):
        sim.exportInVars('invars')
    with pytest.raises(ValueError):
        sim.exportOutVars('outvars')

    def read_csv(filename):
        with open(sim.resultsdir / filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            data_dict = {header: [] for header in headers}
            # Populate the dictionary with data converted to float
            for row in reader:
                for i, header in enumerate(headers):
                    data_dict[header].append(float(row[i]))
        return data_dict

    atol = 1e-12
    filename = 'invars.csv'
    sim.exportInVars(filename)
    invars = read_csv(filename)
    np.testing.assert_allclose(invars['Var1'], expected_data['Var1'], atol=atol)
    np.testing.assert_allclose(invars['Var2'], expected_data['Var2'], atol=atol)

    filename = 'outvars.csv'
    sim.exportOutVars(filename)
    outvars = read_csv(filename)
    np.testing.assert_allclose(outvars['casenum'], expected_data['casenum'], atol=atol)


def test_sim_export_json(sim):
    with pytest.raises(ValueError):
        sim.exportInVars('invars')
    with pytest.raises(ValueError):
        sim.exportOutVars('outvars')

    def read_json(filename):
        with open(sim.resultsdir / filename, 'r') as file:
            data = json.load(file)
        return data

    filename = 'invars.json'
    sim.exportInVars(filename)
    invars = read_json(filename)
    np.testing.assert_allclose(invars['Var1'], expected_data['Var1'])
    np.testing.assert_allclose(invars['Var2'], expected_data['Var2'])

    filename = 'outvars.json'
    sim.exportOutVars(filename)
    outvars = read_json(filename)
    np.testing.assert_allclose(outvars['casenum'], expected_data['casenum'])


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
