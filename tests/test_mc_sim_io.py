# test_mc_sim_io.py

import pytest
from monaco.mc_sim import Sim
import dill
import os

from sim_testing_fcns import fcns

ndraws = 16
seed = 12362398

@pytest.fixture
def sim(tmp_path):
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
            sim = dill.load(file)
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
            dill.load(file)
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
            dill.load(file)
            assert expectedwarning in log[0].message.args[0]


def test_sim_remove_extra_files(sim_with_extra_files):
    sim_with_extra_files.removeExtraResultsFiles()
    with open(sim_with_extra_files.resultsdir / 'sim_io_test.mcsim', 'rb') as file:
        with pytest.warns(None) as log:
            dill.load(file)
            assert not log


### Inline Testing ###
def sim_io_test_example_sim(resultsdir):
    sim = Sim(name='sim_io_test', ndraws=ndraws, fcns=fcns(),
              firstcaseismedian=False, seed=seed, cores=2, verbose=True,
              resultsdir=resultsdir, debug=False)
    sim.runSim()

    results_dir = sim.resultsdir

    print('\n -------- 1 \n', flush=True)
    os.remove(results_dir / 'sim_io_test_1.mccase')
    os.remove(results_dir / 'sim_io_test_2.mccase')
    with open(results_dir / 'sim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)
        # Expected: 14/16 cases loaded from disk,
        #           UserWarning: The following cases were not loaded: [1, 2],
        #           UserWarning: The following cases have not been postprocessed: [1, 2]

    print('\n -------- 2 \n', flush=True)
    sim.runSim([1, 2])  # Expected: 2/2 case run

    print('\n -------- 3 \n', flush=True)
    os.remove(results_dir / 'sim_io_test_1.mccase')
    os.remove(results_dir / 'sim_io_test_2.mccase')
    with open(results_dir / 'sim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)
        # Expected: 14/16 cases loaded,
        #           UserWarning: The following cases were not loaded: [1, 2],
        #           UserWarning: The following cases have not been postprocessed:
        #                        [1, 2]
        #           UserWarning: The following cases were loaded but may be stale:
        #                        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    print('\n -------- 4 \n', flush=True)
    sim.runIncompleteSim()
    # Expected: Resuming incomplete 'sim_io_test' Monte Carlo simulation with
    #           2/16 cases remaining to run, and 2/16 cases remaining to post process...

    print('\n -------- 5 \n', flush=True)
    sim.runSim()
    with open(results_dir / 'dummyfile.mcsim', 'wb') as file:
        pass
    with open(results_dir / 'dummyfile.mccase', 'wb') as file:
        pass
    with open(results_dir / 'dummyfile.txt', 'wb') as file:
        pass

    with open(results_dir / 'sim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)
        # Expected: UserWarning: The following extra .mcsim and .mccase files
        #           were found in the results directory, run removeExtraResultsFiles()
        #           to clean them up: [dummyfile.mcsim, dummyfile.mccase]
    sim.removeExtraResultsFiles()

    print('\n -------- 6 \n', flush=True)
    with open(results_dir / 'sim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)  # Expected: Data for 16/16 cases loaded from disk
    try:
        os.remove(results_dir / 'dummyfile.txt')
    except Exception:
        pass

    return sim


if __name__ == '__main__':
    sim_io_test_example_sim(None)
