from Monaco.MCSim import MCSim
import dill
import os

from mytest_example_sim import mytest_example_sim
from mytest_example_preprocess import mytest_example_preprocess
from mytest_example_postprocess import mytest_example_postprocess
fcns ={'preprocess' :mytest_example_preprocess,   \
       'run'        :mytest_example_sim,          \
       'postprocess':mytest_example_postprocess}

ndraws = 16
seed=12362398

def mcsim_io_mcsim_io_test_example_run_script():
    sim = MCSim(name='mcsim_io_test', ndraws=ndraws, fcns=fcns, firstcaseisnom=False, seed=seed, cores=2, verbose=True)
    sim.runSim()
    
    results_dir = sim.resultsdir
    
    os.remove(results_dir / 'mcsim_io_test_1.mccase')
    os.remove(results_dir / 'mcsim_io_test_2.mccase')
    
    print('\n -------- \n')
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)

    print('\n -------- \n')
    sim.runSim([1,2])    
    os.remove(results_dir / 'mcsim_io_test_1.mccase')
    os.remove(results_dir / 'mcsim_io_test_2.mccase')

    print('\n -------- \n')
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)

    print('\n -------- \n')
    sim.runIncompleteSim()
    
    print('\n -------- \n')
    sim.runSim()    
    with open(results_dir / 'dummyfile.mcsim', 'wb') as file:
        pass
    with open(results_dir / 'dummyfile.mccase', 'wb') as file:
        pass
    with open(results_dir / 'dummyfile.txt', 'wb') as file:
        pass
    
    print('\n -------- \n')
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)
    sim.removeExtraResultsFiles()
    
    print('\n -------- \n')
    with open(results_dir / 'mcsim_io_test.mcsim', 'rb') as file:
        sim = dill.load(file)
    try:
        os.remove(results_dir / 'dummyfile.txt')
    except:
        pass

    return sim

if __name__ == '__main__':
    sim = mcsim_io_mcsim_io_test_example_run_script()
    