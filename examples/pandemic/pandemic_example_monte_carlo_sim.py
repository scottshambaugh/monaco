from scipy.stats import uniform
import monaco as mc
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

    sim = mc.Sim(name='pandemic', ndraws=ndraws, fcns=fcns,
                 firstcaseismedian=True, seed=seed, cores=4,
                 verbose=True, debug=False)

    sim.addInVar(name='Probability of Infection',
                 dist=uniform, distkwargs={'loc': 0.28, 'scale': 0.04})

    sim.runSim()

    sim.outvars['Proportion Infected'].addVarStat(stat='orderstatTI',
                                                  statkwargs={'p': p, 'c': c, 'bound': bound})
    sim.outvars['Superspreader Degree'].addVarStat(stat='orderstatTI',
                                                   statkwargs={'p': 0.5, 'c': 0.5,
                                                               'bound': 'all'})

    mc.plot(sim.outvars['Timestep'], sim.outvars['Superspreader Degree'])
    mc.plot(sim.outvars['Max Superspreader Degree'], highlight_cases=0)
    mc.plot(sim.outvars['Herd Immunity Threshold'], highlight_cases=0)

    '''
    import matplotlib.pyplot as plt
    fig, ax = mc.plot(sim.outvars['Timestep'], sim.outvars['Proportion Infected'],
                      highlight_cases=0)
    fig.set_size_inches(8.8, 6.0)
    plt.savefig('cum_infections_vs_time.png', dpi=100)

    fig, axs = mc.multi_plot([sim.invars['Probability of Infection'],
                              sim.outvars['Herd Immunity Threshold']],
                             cov_plot=False, highlight_cases=0)
    fig.set_size_inches(8.8, 6.0)
    plt.savefig('p_infection_vs_herd_immunity.png', dpi=100)
    #'''

    return sim


if __name__ == '__main__':
    sim = pandemic_example_monte_carlo_sim()
