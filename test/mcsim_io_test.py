from Monaco.MCSim import MCSim
import dill
import os
from time import sleep

ndraws = 16
seed=12362398

### Functions 
def testing_preprocess(mccase):
    return ([True,])

def testing_run(inputs):
    sleep(0.01)
    return (True)

def testing_postprocess(mccase, output):
    mccase.addOutVal('casenum', mccase.ncase)

fcns ={'preprocess' :testing_preprocess,   \
       'run'        :testing_run,          \
       'postprocess':testing_postprocess}

### Main Sim 
def mcsim_io_test_example_sim():
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
    sim = mcsim_io_test_example_sim()
    os.remove(sim.resultsdir)
    