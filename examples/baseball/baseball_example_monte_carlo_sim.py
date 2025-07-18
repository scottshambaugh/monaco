from scipy.stats import uniform, norm
import monaco as mc
import matplotlib.pyplot as plt
import numpy as np

from baseball_example_run import baseball_example_run, plot_baseball_field
from baseball_example_preprocess import baseball_example_preprocess
from baseball_example_postprocess import baseball_example_postprocess
fcns = {'preprocess' : baseball_example_preprocess,
        'run'        : baseball_example_run,
        'postprocess': baseball_example_postprocess}

## Constants
ndraws = 100
seed = 78547876
in2m = 0.0254
mph2mps = 0.44704

def baseball_example_monte_carlo_sim():
    ## Define sim
    sim = mc.Sim(name='baseball', ndraws=ndraws, fcns=fcns, firstcaseismedian=True,
                 seed=seed, singlethreaded=False, usedask=False, verbose=True, debug=True,
                 savecasedata=False, savesimdata=False)

    ## Define input variables
    sim.addInVar(name='Y Init [m]', dist=uniform,
                 distkwargs={'loc': -19.94*in2m/2, 'scale': 19.94*in2m})
    sim.addInVar(name='Z Init [m]', dist=uniform,
                 distkwargs={'loc': 18.29*in2m, 'scale': 25.79*in2m})
    sim.addInVar(name='Speed Init [m/s]', dist=norm,
                 distkwargs={'loc': 90*mph2mps, 'scale': 5*mph2mps})
    sim.addInVar(name='Launch Angle [deg]', dist=norm, distkwargs={'loc': 10, 'scale': 20})
    sim.addInVar(name='Side Angle [deg]', dist=norm, distkwargs={'loc': 0, 'scale': 30})
    sim.addInVar(name='Topspin [rpm]', dist=norm, distkwargs={'loc': 80, 'scale': 500})
    sim.addInVar(name='Mass [kg]', dist=uniform, distkwargs={'loc': 0.142, 'scale': 0.007})
    sim.addInVar(name='Diameter [m]', dist=uniform, distkwargs={'loc': 0.073, 'scale': 0.003})
    sim.addInVar(name='Wind Speed [m/s]', dist=uniform, distkwargs={'loc': 0, 'scale': 10})
    sim.addInVar(name='Wind Azi [deg]', dist=uniform, distkwargs={'loc': 0, 'scale': 360})
    sim.addInVar(name='CD', dist=norm, distkwargs={'loc': 0.300, 'scale': 0.015})

    ## Run sim
    sim.runSim()

    ## Extend outvars to have the same length so that stats account for all cases
    sim.extendOutVars()

    ## Generate stats and plots
    homerun_indices = np.where(sim.outvars['Home Run'].vals)[0]
    # foul_indices = np.where(sim.outvars['Foul Ball'].vals)[0]

    # sim.outvars['Landing Dist [m]'].addVarStat(stat='gaussianP',
    #                                            statkwargs={'p': 0.90, 'c': 0.50})
    # sim.outvars['Landing Dist [m]'].addVarStat(stat='gaussianP',
    #                                            statkwargs={'p': 0.10, 'c': 0.50})
    # print(sim.outvars['Landing Dist [m]'].stats())
    # mc.plot(sim.outvars['Time [s]'], sim.outvars['Distance [m]'], highlight_cases=homerun_indices)
    # mc.plot(sim.outvars['Time [s]'], sim.outvars['Speed [m/s]'],
    #         highlight_cases=homerun_indices)
    fig, ax = mc.plot(sim.outvars['X [m]'], sim.outvars['Y [m]'],
                      sim.outvars['Z [m]'], title='Baseball Trajectory',
                      highlight_cases=homerun_indices, plotkwargs={'zorder': 10})
    plot_baseball_field(ax)
    ax.scatter(sim.outvars['Landing X [m]'].nums, sim.outvars['Landing Y [m]'].nums, 0,
               s=2, c='k', alpha=0.9, marker='o')
    # fig.set_size_inches(7.0, 7.0)
    # plt.savefig('baseball_trajectory.png', dpi=100)

    sim.plot(highlight_cases=homerun_indices)

    fig, axs = mc.multi_plot([sim.invars['Launch Angle [deg]'], sim.outvars['Landing Dist [m]']],
                             title='Launch Angle vs Landing Distance', cov_plot=False,
                             highlight_cases=homerun_indices)
    # fig.set_size_inches(8.8, 6.6)
    # plt.savefig('launch_angle_vs_landing.png', dpi=100)
    plt.show(block=False)

    ## Calculate and plot sensitivity indices
    sim.calcSensitivities('Home Run')
    fig, ax = sim.outvars['Home Run'].plotSensitivities()
    # fig.set_size_inches(10, 4)
    # plt.savefig('landing_dist_sensitivities.png', dpi=100)
    plt.show()

    return sim


if __name__ == '__main__':
    sim = baseball_example_monte_carlo_sim()
