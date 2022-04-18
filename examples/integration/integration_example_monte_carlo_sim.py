from scipy.stats import uniform
import monaco as mc
import numpy as np

# Define our functions
def integration_example_preprocess(case):
    x = case.invals['x'].val
    y = case.invals['y'].val
    return (x, y)


def integration_example_run(x, y):
    # Note that you can return True/False for simple area integrals (since these
    # vals automatically get valmapped to the nums 1/0), but best practice is to
    # return the value of the function at each sample point.
    isUnderCurve = x**2 + y**2 < 1
    return isUnderCurve


def integration_example_postprocess(case, isUnderCurve):
    case.addOutVal('pi_est', isUnderCurve)


fcns = {'preprocess' : integration_example_preprocess,
        'run'        : integration_example_run,
        'postprocess': integration_example_postprocess}

# We need to know the limits of integration beforehand, and for integration
# these should always be uniform dists
xrange = [-1, 1]
yrange = [-1, 1]
totalArea = (xrange[1] - xrange[0])*(yrange[1] - yrange[0])
dimension = 2

# Integration best practices:
# File I/O will crush performance, so recommended not to save case data
savecasedata = False
# Use SOBOL over SOBOL_RANDOM for a speedup, since all our dists are uniform
samplemethod = 'sobol'
# Since we want a power of 2, we should not run a 'median' case which would add 1
firstcaseismedian = False

# Maximum Error bound:
# Sobol sampling will give much faster convergence than random.
# A-priori, we do not know the standard deviation, so best practice is to use
# the maximum possible gives the low and high values in your range, and the
# a-posteriori error will be better than this since the actual variance is known.
# You can bootstrap the standard deviation by running a shorter sim and extracting
# stdev from there.
error = 0.01
conf = 0.95
stdev = mc.max_stdev(low=0, high=1)
print(f'Maximum possible standard deviation: {stdev:0.3f}')

nRandom = mc.integration_n_from_err(error=error, dimension=dimension, volume=totalArea,
                                    stdev=stdev, conf=conf, samplemethod='random')
nSobol  = mc.integration_n_from_err(error=error, dimension=dimension, volume=totalArea,
                                    stdev=stdev, conf=conf, samplemethod='sobol')
print(f'Number of samples needed to reach an error ≤ ±{error} at {round(conf*100, 2)}% ' +
      f'confidence if using random vs sobol sampling: {nRandom} vs {nSobol}')

# The sobol methods need to be a power of 2 for best performance and balance
ndraws = mc.next_power_of_2(nSobol)
print(f'Rounding up to next power of 2: {ndraws} samples')

seed = 123639

def integration_example_monte_carlo_sim():

    sim = mc.Sim(name='integration', ndraws=ndraws, fcns=fcns,
                 firstcaseismedian=firstcaseismedian, samplemethod=samplemethod,
                 seed=seed, singlethreaded=False,
                 savecasedata=savecasedata, savesimdata=False,
                 verbose=True, debug=True)

    sim.addInVar(name='x', dist=uniform,
                 distkwargs={'loc': xrange[0], 'scale': (xrange[1] - xrange[0])})  # -1 <= x <= 1
    sim.addInVar(name='y', dist=uniform,
                 distkwargs={'loc': yrange[0], 'scale': (yrange[1] - yrange[0])})  # -1 <= y <= 1

    sim.runSim()

    # Note that (True,False) vals are automatically valmapped to the nums (1,0)
    underCurvePct = sum(sim.outvars['pi_est'].nums)/ndraws
    err = mc.integration_error(sim.outvars['pi_est'].nums, dimension=dimension,
                               volume=totalArea, conf=conf,
                               samplemethod=samplemethod, runningerror=False)
    stdev = np.std(sim.outvars['pi_est'].nums, ddof=1)

    resultsstr = f'π ≈ {underCurvePct*totalArea:0.5f}, n = {ndraws}, ' + \
                 f'{round(conf*100, 2)}% error = ±{err:0.5f}, stdev={stdev:0.3f}'
    print(resultsstr)

    '''
    import matplotlib.pyplot as plt
    indices_under_curve = [i for i, x in enumerate(sim.outvars['pi_est'].vals) if x]
    fig, ax = mc.plot(sim.invars['x'], sim.invars['y'], highlight_cases=indices_under_curve)
    ax.axis('equal')
    plt.title(resultsstr)
    fig.set_size_inches(8.8, 6.0)
    plt.savefig('circle_integration.png', dpi=100)

    fig, ax = mc.plot_integration_convergence(sim.outvars['pi_est'], dimension=dimension,
                                              volume=totalArea, refval=np.pi, conf=0.95,
                                              title='Approx. value of π',
                                              samplemethod=samplemethod)
    ax.set_ylim((3.10, 3.18))
    fig.set_size_inches(8.8, 6.0)
    plt.savefig('pi_convergence.png', dpi=100)

    fig, ax = mc.plot_integration_error(sim.outvars['pi_est'], dimension=dimension,
                                        volume=totalArea, refval=np.pi, conf=0.95,
                                        title='Approx. error of π', samplemethod=samplemethod)
    fig.set_size_inches(8.8, 6.0)
    plt.savefig('pi_error.png', dpi=100)

    plt.show()
    #'''

    return sim


if __name__ == '__main__':
    sim = integration_example_monte_carlo_sim()
