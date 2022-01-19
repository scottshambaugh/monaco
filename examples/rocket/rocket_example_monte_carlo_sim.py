from scipy.stats import uniform, rv_discrete, norm
import monaco as mc
# import matplotlib.pyplot as plt

from rocket_example_run import rocket_example_run
from rocket_example_preprocess import rocket_example_preprocess
from rocket_example_postprocess import rocket_example_postprocess
fcns = {'preprocess' : rocket_example_preprocess,
        'run'        : rocket_example_run,
        'postprocess': rocket_example_postprocess}

ndraws = 50
seed = 12362398

def rocket_example_monte_carlo_sim():
    sim = mc.Sim(name='Rocket', ndraws=ndraws, fcns=fcns, firstcaseismedian=True,
                 seed=seed, cores=1, verbose=True, debug=False)

    sim.addInVar(name='Wind Azi [deg]', dist=uniform, distkwargs={'loc': 0, 'scale': 360})
    sim.addInVar(name='Wind Speed [m/s]', dist=uniform, distkwargs={'loc': 0, 'scale': 2})
    sim.addInVar(name='Thrust Factor', dist=norm, distkwargs={'loc': 1, 'scale': 0.1})

    para_fail_dist = rv_discrete(name='para_fail_dist', values=([1, 2], [0.8, 0.2]))
    para_fail_nummap = {1: False, 2: True}
    sim.addInVar(name='Parachute Failure', dist=para_fail_dist, distkwargs=dict(),
                 nummap=para_fail_nummap)

    sim.runSim()

    sim.outvars['Distance [m]'].addVarStat(stat='gaussianP',
                                           statkwargs={'p': 0.90, 'c': 0.50})
    sim.outvars['Distance [m]'].addVarStat(stat='gaussianP',
                                           statkwargs={'p': 0.10, 'c': 0.50})
    # print(sim.outvars['Landing Dist [m]'].stats())
    mc.plot(sim.outvars['Time [s]'], sim.outvars['Distance [m]'], highlight_cases=0)
    # mc.plot(sim.outvars['Time [s]'], sim.outvars['|Velocity| [m/s]'], highlight_cases=0)
    mc.plot(sim.outvars['Time [s]'], sim.outvars['Acceleration [m/s^2] [2]'],
            highlight_cases=0)
    # mc.plot(sim.outvars['Time [s]'], sim.outvars['Thrust [N]'], highlight_cases=0)
    # mc.plot(sim.outvars['Landing Dist [m]'], highlight_cases=0)
    # mc.plot(sim.outvars['Landing E [m]'], sim.outvars['Landing N [m]'], highlight_cases=0)
    # mc.plot(sim.outvars['Time [s]'], sim.outvars['Flight Stage'], highlight_cases=0)
    # mc.plot(sim.outvars['Position [m]'], highlight_cases=0)
    mc.plot(sim.outvars['Easting [m]'], sim.outvars['Northing [m]'],
            sim.outvars['Altitude [m]'], title='Model Rocket Trajectory', highlight_cases=0)
    # plt.savefig('rocket_trajectory.png')
    mc.multi_plot([sim.invars['Wind Speed [m/s]'], sim.outvars['Landing Dist [m]']],
                  cov_p=0.95, title='Wind Speed vs Landing Distance w/ 95% CI', highlight_cases=0)
    # plt.savefig('wind_vs_landing.png')

    return sim


if __name__ == '__main__':
    sim = rocket_example_monte_carlo_sim()
