from scipy.stats import uniform
from Monaco.MCSim import MCSim
from Monaco.helper_functions import next_power_of_2
from Monaco.integration_statistics import integration_error, integration_n_from_err, max_stdev
import numpy as np

# Define our functions
def integration_example_preprocess(mccase):
    x = mccase.mcinvals['x'].val
    y = mccase.mcinvals['y'].val    
    return (x, y)

def integration_example_run(x, y):
    # Note that you can return True/False for simple area integrals (since these
    # vals automatically get valmapped to the nums 1/0), but best practice is to 
    # return the value of the function at each sample point.
    isUnderCurve = x**2 + y**2 < 1
    return isUnderCurve

def integration_example_postprocess(mccase, isUnderCurve):
    mccase.addOutVal('pi_est', isUnderCurve)

fcns ={'preprocess' :integration_example_preprocess,   \
       'run'        :integration_example_run,          \
       'postprocess':integration_example_postprocess}

# We need to know the limits of integration beforehand, and for integration these should always be uniform dists
xrange = [-1, 1]
yrange = [-1, 1]
totalArea = (xrange[1] - xrange[0])*(yrange[1] - yrange[0])

# Maximum Error bound:
# Sobol sampling will give much faster convergence than random. But for random 
# sampling, we can determine how many samples we would need to reach an absolute
# error of 0.01 at 95% confidence. 
# A-priori, we do not know the standard deviation, so best practice is to use 
# the maximum possible gives the low and high values in your range, and the 
# a-posteriori error will be better than this since the actual variance is known.
# You can bootstrap the standard deviation by running a shorter sim and extracting
# stdev from there
error = 0.01
conf = 0.95
stdev = max_stdev(low=0, high=1)
nIfRandom = integration_n_from_err(error=error, volume=totalArea, stdev=stdev, conf=conf)
print(f'Number of samples needed to reach an error ≤ ±{error} at {round(conf*100, 2)}% confidence if using random sampling: {nIfRandom}')

# Integration best practices:
savecasedata = False                 # File I/O will crush performance, so recommended not to save case data
samplemethod = 'sobol'               # Use 'sobol' over 'sobol_random' for a speedup, since all our dists are uniform
ndraws = next_power_of_2(nIfRandom)  # The sobol methods need to be a power of 2 for best performance and balance
firstcaseisnom = False               # Since we want a power of 2, we should not run a 'nominal' case which would add 1

seed=12362397

def integration_example_monte_carlo_sim():

    sim = MCSim(name='integration', ndraws=ndraws, fcns=fcns, firstcaseisnom=firstcaseisnom, samplemethod=samplemethod, seed=seed, cores=4, savecasedata=savecasedata, verbose=True, debug=True)
    
    sim.addInVar(name='x', dist=uniform, distkwargs={'loc':xrange[0], 'scale':(xrange[1] - xrange[0])}) # -1 <= x <= 1
    sim.addInVar(name='y', dist=uniform, distkwargs={'loc':yrange[0], 'scale':(yrange[1] - yrange[0])}) # -1 <= y <= 1
    
    sim.runSim()
    
    underCurvePct = sum(sim.mcoutvars['pi_est'].nums)/ndraws # Note that (True,False) vals are automatically valmapped to the nums (1,0)
    err = integration_error(sim.mcoutvars['pi_est'].nums, volume=totalArea, runningError=False, conf=conf)

    resultsstr = f'π ≈ {underCurvePct*totalArea:0.5f}, n = {ndraws}, {round(conf*100, 2)}% error = ±{err:0.5f}'
    print(resultsstr)
    
    '''
    from Monaco.mc_plot import mc_plot, mc_plot_integration_convergence, mc_plot_integration_error
    import matplotlib.pyplot as plt
    indices_under_curve = [i for i, x in enumerate(sim.mcoutvars['pi_est'].vals) if x]
    fig, ax = mc_plot(sim.mcinvars['x'], sim.mcinvars['y'], highlight_cases=indices_under_curve)
    ax.axis('equal')
    plt.title(resultsstr)
    
    fig, ax = mc_plot_integration_convergence(sim.mcoutvars['pi_est'], refval=np.pi, volume=totalArea, conf=0.95, title='Approx. value of π')
    ax.set_ylim((3.10, 3.18))
    fig, ax = mc_plot_integration_error(sim.mcoutvars['pi_est'], refval=np.pi, volume=totalArea, conf=0.95, title='Approx. error of π')
    #'''
    
    return sim


if __name__ == '__main__':
    sim = integration_example_monte_carlo_sim()
    