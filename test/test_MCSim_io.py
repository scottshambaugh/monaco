# test_MCSim_io.py

import pytest 
from Monaco.MCSim import MCSim
import dill
import os

from test.mcsim_testing_fcns import fcns

ndraws = 16
seed=12362398

@pytest.fixture
def mcsim(tmp_path):
    mcsim = MCSim(name='mcsim_io_test', ndraws=ndraws, fcns=fcns(), firstcaseisnom=False, seed=seed, cores=2, verbose=True, resultsdir=tmp_path)
    mcsim.runSim()
    return mcsim

@pytest.fixture
def mcsim_without_1_2(mcsim):
    os.remove(mcsim.resultsdir / 'mcsim_io_test_1.mccase')
    os.remove(mcsim.resultsdir / 'mcsim_io_test_2.mccase')
    with open(mcsim.resultsdir / 'mcsim_io_test.mcsim', 'rb') as file:
        with pytest.warns(UserWarning) as log:
            mcsim = dill.load(file)
    return (mcsim, log)


@pytest.fixture
def mcsim_with_extra_files(mcsim):
    for filename in ('dummyfile.mcsim', 'dummyfile.mccase','dummyfile.txt'):
        with open(mcsim.resultsdir / filename, 'wb'):
            pass
    return mcsim


def test_mcsim_load_partial(mcsim_without_1_2):
    (mcsim, log) = mcsim_without_1_2
    assert len(log) == 4
    assert 'mcsim_io_test_1.mccase expected but not found' in log[0].message.args[0]  
    assert 'mcsim_io_test_2.mccase expected but not found' in log[1].message.args[0]  
    assert 'The following cases were not loaded: [1, 2]' in log[2].message.args[0]  
    assert 'The following cases have not been postprocessed: [1, 2]' in log[3].message.args[0]  


def test_mcsim_run_partial(mcsim_without_1_2):
    (mcsim, log) = mcsim_without_1_2
    mcsim.runSim([1,2])
    assert mcsim.casesrun == set(range(ndraws))
    assert mcsim.casespostprocessed == set(range(ndraws))


def test_mcsim_load_stale(mcsim_without_1_2):
    (mcsim, log) = mcsim_without_1_2
    mcsim.runSim([1,2])
    with open(mcsim.resultsdir / 'mcsim_io_test.mcsim', 'rb') as file:
        with pytest.warns(UserWarning) as log:
            dill.load(file)
            assert 'The following cases were loaded but may be stale: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]' in log[-1].message.args[0] 


def test_mcsim_run_incomplete(mcsim_without_1_2):
    (mcsim, log) = mcsim_without_1_2
    mcsim.runIncompleteSim()
    assert mcsim.casesrun == set(range(ndraws))
    assert mcsim.casespostprocessed == set(range(ndraws))


def test_mcsim_find_extra_files(mcsim_with_extra_files):
    with open(mcsim_with_extra_files.resultsdir / 'mcsim_io_test.mcsim', 'rb') as file:
        expectedwarning = "The following extra .mcsim and .mccase files were found in the results directory, run removeExtraResultsFiles() to clean them up: ['dummyfile.mccase', 'dummyfile.mcsim']"
        with pytest.warns(UserWarning) as log:
            dill.load(file)
            assert expectedwarning in log[0].message.args[0] 
    

def test_mcsim_remove_extra_files(mcsim_with_extra_files):
    mcsim_with_extra_files.removeExtraResultsFiles()
    with open(mcsim_with_extra_files.resultsdir / 'mcsim_io_test.mcsim', 'rb') as file:
        with pytest.warns(None) as log:
            dill.load(file)
            assert not log


### Inline Testing ###
def mcsim_io_test_example_sim(resultsdir):
    sim = MCSim(name='mcsim_io_test', ndraws=ndraws, fcns=fcns(), firstcaseisnom=False, seed=seed, cores=2, verbose=True, resultsdir=resultsdir)
    sim.runSim()
    
    results_dir = sim.resultsdir
     
    print('\n -------- 1 \n', flush=True)
    os.remove(results_dir / 'mcsim_io_test_1.mccase')
    os.remove(results_dir / 'mcsim_io_test_2.mccase')
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)  # Expected: 14/16 cases loaded from disk, 
                               #           UserWarning: The following cases were not loaded: [1, 2], 
                               #           UserWarning: The following cases have not been postprocessed: [1, 2]

    print('\n -------- 2 \n', flush=True)
    sim.runSim([1,2])          # Expected: 2/2 case run

    print('\n -------- 3 \n', flush=True)
    os.remove(results_dir / 'mcsim_io_test_1.mccase')
    os.remove(results_dir / 'mcsim_io_test_2.mccase')
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)  # Expected: 14/16 cases loaded, 
                               #           UserWarning: The following cases were not loaded: [1, 2], 
                               #           UserWarning: The following cases have not been postprocessed: [1, 2]
                               #           UserWarning: The following cases were loaded but may be stale: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    print('\n -------- 4 \n', flush=True)
    sim.runIncompleteSim()     # Expected: Resuming incomplete 'mcsim_io_test' Monte Carlo simulation with 2/16 cases remaining to run, and 2/16 cases remaining to post process...
    
    print('\n -------- 5 \n', flush=True)
    sim.runSim()    
    with open(results_dir / 'dummyfile.mcsim', 'wb') as file:
        pass
    with open(results_dir / 'dummyfile.mccase', 'wb') as file:
        pass
    with open(results_dir / 'dummyfile.txt', 'wb') as file:
        pass

    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)  # Expected: UserWarning: The following extra .mcsim and .mccase files were found in the results directory, run removeExtraResultsFiles() to clean them up: ['dummyfile.mcsim', 'dummyfile.mccase']
    sim.removeExtraResultsFiles()
    
    print('\n -------- 6 \n', flush=True)
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)  # Expected: Data for 16/16 cases loaded from disk
    try:
        os.remove(results_dir / 'dummyfile.txt')
    except:
        pass

    return sim


if __name__ == '__main__':
    #'''
    resultsdir = 'mcsim_io_test'
    sim = mcsim_io_test_example_sim(resultsdir)
    #'''
    '''
    import tempfile
    with tempfile.TemporaryDirectory() as resultsdir:
        sim = mcsim_io_test_example_sim(resultsdir)
    #'''
    