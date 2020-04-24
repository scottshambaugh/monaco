from scipy.stats import uniform, rv_discrete
from PyMonteCarlo.MCSim import MCSim
from PyMonteCarlo.MCPlot import MCPlot
from PyMonteCarlo.MCMultiPlot import MCMultiPlot

from rocket_example_sim import rocket_example_sim
from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
fcns ={'preprocess' :rocket_example_preprocess,   \
       'run'        :rocket_example_sim,          \
       'postprocess':rocket_example_postprocess}

ndraws = 100
seed=12362398

def rocket_example_run_script():
    sim = MCSim(name='Rocket', ndraws=ndraws, fcns=fcns, firstcaseisnom=True, seed=seed, cores=1)
    
    sim.addInVar(name='Wind Azi [deg]', dist=uniform, distargs=(0, 360))
    sim.addInVar(name='Wind Speed [m/s]', dist=uniform, distargs=(0, 2))
    
    para_fail_dist = rv_discrete(name='para_fail_dist', values=([1, 2], [0.8, 0.2]))
    para_fail_nummap = {1:False, 2:True}
    sim.addInVar(name='Parachute Failure', dist=para_fail_dist, distargs=(), nummap=para_fail_nummap)

    sim.runSim()
    
    print(sim.runtime)
    
    #print(sim.mcoutvars['Landing Dist [m]'].stats())
    #MCPlot(sim.mcoutvars['Time [s]'], sim.mcoutvars['Distance [m]'])
    #MCPlot(sim.mcoutvars['Time [s]'], sim.mcoutvars['|Velocity| [m/s]'])
    #MCPlot(sim.mcoutvars['Landing Dist [m]'])
    #MCPlot(sim.mcoutvars['Landing E [m]'], sim.mcoutvars['Landing N [m]'])
    #MCPlot(sim.mcoutvars['Time [s]'], sim.mcoutvars['Flight Stage'])    
    #MCPlot(sim.mcoutvars['Position [m]'])
    MCPlot(sim.mcoutvars['Easting [m]'], sim.mcoutvars['Northing [m]'], sim.mcoutvars['Altitude [m]'], title='Model Rocket Trajectory')
    #import matplotlib.pyplot as plt
    #plt.savefig('rocket_trajectory.png')
    MCMultiPlot(sim.mcoutvars['Landing Dist [m]'], sim.mcinvars['Wind Speed [m/s]'], title='Wind Speed vs Landing Distance')
    #plt.savefig('wind_vs_landing.png')
    
    return sim

if __name__ == '__main__':
    sim = rocket_example_run_script()
    