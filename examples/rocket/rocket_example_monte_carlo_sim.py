from scipy.stats import uniform, rv_discrete
from Monaco.MCSim import MCSim, MCFunctions
from Monaco.mc_plot import mc_plot
from Monaco.mc_multi_plot import mc_multi_plot

from rocket_example_run import rocket_example_run
from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
fcns ={MCFunctions.PREPROCESS :rocket_example_preprocess,   \
       MCFunctions.RUN        :rocket_example_run,          \
       MCFunctions.POSTPROCESS:rocket_example_postprocess}

ndraws = 50
seed=12362398

def rocket_example_monte_carlo_sim():
    sim = MCSim(name='Rocket', ndraws=ndraws, fcns=fcns, firstcaseisnom=True, seed=seed, cores=4, verbose=True, debug=False)
    
    sim.addInVar(name='Wind Azi [deg]', dist=uniform, distkwargs={'loc':0, 'scale':360})
    sim.addInVar(name='Wind Speed [m/s]', dist=uniform, distkwargs={'loc':0, 'scale':2})
    
    para_fail_dist = rv_discrete(name='para_fail_dist', values=([1, 2], [0.8, 0.2]))
    para_fail_nummap = {1:False, 2:True}
    sim.addInVar(name='Parachute Failure', dist=para_fail_dist, distkwargs=dict(), nummap=para_fail_nummap)

    sim.runSim()
    
    sim.mcoutvars['Distance [m]'].addVarStat(stattype='gaussianP', statkwargs={'p':0.90, 'c':0.50})
    sim.mcoutvars['Distance [m]'].addVarStat(stattype='gaussianP', statkwargs={'p':0.10, 'c':0.50})
    #print(sim.mcoutvars['Landing Dist [m]'].stats())
    mc_plot(sim.mcoutvars['Time [s]'], sim.mcoutvars['Distance [m]'], highlight_cases=0)
    #mc_plot(sim.mcoutvars['Time [s]'], sim.mcoutvars['|Velocity| [m/s]'], highlight_cases=0)
    #mc_plot(sim.mcoutvars['Landing Dist [m]'], highlight_cases=0)
    #mc_plot(sim.mcoutvars['Landing E [m]'], sim.mcoutvars['Landing N [m]'], highlight_cases=0)
    #mc_plot(sim.mcoutvars['Time [s]'], sim.mcoutvars['Flight Stage'], highlight_cases=0)
    #mc_plot(sim.mcoutvars['Position [m]'], highlight_cases=0)
    mc_plot(sim.mcoutvars['Easting [m]'], sim.mcoutvars['Northing [m]'], sim.mcoutvars['Altitude [m]'], title='Model Rocket Trajectory', highlight_cases=0)
    #import matplotlib.pyplot as plt
    #plt.savefig('rocket_trajectory.png')
    mc_multi_plot(sim.mcinvars['Wind Speed [m/s]'], sim.mcoutvars['Landing Dist [m]'], cov_p=0.95, title='Wind Speed vs Landing Distance w/ 95% CI', highlight_cases=0)
    #plt.savefig('wind_vs_landing.png')
    
    return sim

if __name__ == '__main__':
    sim = rocket_example_monte_carlo_sim()
    