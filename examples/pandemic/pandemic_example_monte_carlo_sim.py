from scipy.stats import uniform
from monaco.mc_sim import MCSim
from monaco.mc_plot import mc_plot
from monaco.mc_multi_plot import mc_multi_plot
from monaco.order_statistics import order_stat_TI_n
from math import ceil


from pandemic_example_run import pandemic_example_run
from pandemic_example_preprocess import pandemic_example_preprocess
from pandemic_example_postprocess import pandemic_example_postprocess
fcns = {'preprocess' : pandemic_example_preprocess,
        'run'        : pandemic_example_run,
        'postprocess': pandemic_example_postprocess}

k = 2
p = 0.95
c = 0.90
bound = '1-sided'
ndraws = order_stat_TI_n(k=k, p=p, c=c, bound=bound)  # 77
ndraws = ceil(ndraws/10)*10  # 80
seed = 12362398

def pandemic_example_monte_carlo_sim():

    sim = MCSim(name='pandemic', ndraws=ndraws, fcns=fcns,
                firstcaseismedian=True, seed=seed, cores=4,
                verbose=True, debug=False)

    sim.addInVar(name='Probability of Infection',
                 dist=uniform, distkwargs={'loc': 0.28, 'scale': 0.04})

    sim.runSim()

    sim.mcoutvars['Proportion Infected'].addVarStat(stattype='orderstatTI',
                                                    statkwargs={'p': p, 'c': c, 'bound': bound})
    sim.mcoutvars['Superspreader Degree'].addVarStat(stattype='orderstatTI',
                                                     statkwargs={'p': 0.5, 'c': 0.5,
                                                                 'bound': 'all'})

    mc_plot(sim.mcoutvars['Timestep'], sim.mcoutvars['Superspreader Degree'])
    mc_plot(sim.mcoutvars['Max Superspreader Degree'], highlight_cases=0)
    mc_plot(sim.mcoutvars['Herd Immunity Threshold'], highlight_cases=0)

    # import matplotlib.pyplot as plt
    mc_plot(sim.mcoutvars['Timestep'], sim.mcoutvars['Proportion Infected'], highlight_cases=0)
    # plt.savefig('cum_infections_vs_time.png')
    mc_multi_plot([sim.mcinvars['Probability of Infection'],
                   sim.mcoutvars['Herd Immunity Threshold']],
                  cov_plot=False, highlight_cases=0)
    # plt.savefig('p_infection_vs_herd_immunity.png')

    return sim


if __name__ == '__main__':
    sim = pandemic_example_monte_carlo_sim()
