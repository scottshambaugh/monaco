from scipy.stats import norm, uniform
from Monaco.MCSim import MCSim
from Monaco.mc_plot import mc_plot, mc_plot_cov_corr
import matplotlib.pyplot as plt
import numpy as np

from retirement_example_sim import retirement_example_sim
from retirement_example_preprocess import retirement_example_preprocess
from retirement_example_postprocess import retirement_example_postprocess
fcns ={'preprocess' :retirement_example_preprocess,   \
       'run'        :retirement_example_sim,          \
       'postprocess':retirement_example_postprocess}

ndraws = 1000
seed=12362397

def retirement_example_run_script():

    sim = MCSim(name='retirement', ndraws=ndraws, fcns=fcns, firstcaseisnom=True, seed=seed, cores=4, savecasedata=False, verbose=True, debug=True)
    
    sp500_mean = 0.114
    sp500_stdev = 0.197
    inflation = 0.02
    nyears = 30
    
    for i in range(nyears):
        sim.addInVar(name=f'Year {i} Returns', dist=norm, distkwargs={'loc':(sp500_mean - inflation), 'scale':sp500_stdev})
    
    sim.addInVar(name='Beginning Balance', dist=uniform, distkwargs={'loc':1000000, 'scale':100000})
    sim.addConstVal(name='nyears', val=nyears)    
    
    sim.runSim()
    
    wentbrokecases = [i for i, e in enumerate(sim.mcoutvars['Went Broke'].vals) if e == 'Yes']
    
    mc_plot(sim.mcinvars['Year 0 Returns'], sim.mcoutvars['Final Balance'], highlight_cases=0)
    mc_plot(sim.mcoutvars['Date'], sim.mcoutvars['Ending Balance'], highlight_cases=wentbrokecases)
    mc_plot(sim.mcoutvars['Went Broke'])
    
    sim.genCovarianceMatrix()
    plt.figure()
    yearly_return_broke_corr = []
    for i in range(nyears):
        yearly_return_broke_corr.append(sim.covs[0][sim.covvarlist.index(f'Year {i} Returns')])
    plt.plot(range(nyears), yearly_return_broke_corr)
    plt.plot(range(nyears), np.zeros(nyears), 'k--')

    return sim


if __name__ == '__main__':
    sim = retirement_example_run_script()
    