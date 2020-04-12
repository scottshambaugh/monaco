from scipy.stats import uniform
from PyMonteCarlo.MCSim import MCSim
from PyMonteCarlo.MCPlot import MCPlot

from rocket_example_sim import rocket_example_sim
from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
fcns ={'preprocess' :rocket_example_preprocess,   \
       'run'        :rocket_example_sim,          \
       'postprocess':rocket_example_postprocess}

ndraws = 8
seed=12362398

def rocket_example_run_script():
    sim = MCSim(name='Rocket', ndraws=ndraws, fcns=fcns, firstcaseisnom=True, seed=seed, cores=1)
    
    sim.addInVar(name='windazi', dist=uniform, distargs=(0, 360))
    sim.addInVar(name='windspd', dist=uniform, distargs=(0, 2))
    
    sim.runSim()
    
    print(sim.runtime)
    
    #print(sim.mcoutvars['Landing Dist'].stats())
    #MCPlot(sim.mcoutvars['Time'], sim.mcoutvars['Distance'])
    #MCPlot(sim.mcoutvars['Landing Dist'])
    #MCPlot(sim.mcoutvars['Landing E'], sim.mcoutvars['Landing N'])
    MCPlot(sim.mcoutvars['Position'])
    return sim

if __name__ == '__main__':
    sim = rocket_example_run_script()
    