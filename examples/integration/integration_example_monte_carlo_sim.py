from scipy.stats import uniform
from Monaco.MCSim import MCSim
from Monaco.helper_functions import next_power_of_2

# Define our functions
def integration_example_preprocess(mccase):
    x = mccase.mcinvals['x'].val
    y = mccase.mcinvals['y'].val    
    return (x, y)

def integration_example_run(x, y):
    is_under_curve = x**2 + y**2 < 1
    return is_under_curve

def integration_example_postprocess(mccase, is_under_curve):
    mccase.addOutVal('is_under_curve', is_under_curve)

fcns ={'preprocess' :integration_example_preprocess,   \
       'run'        :integration_example_run,          \
       'postprocess':integration_example_postprocess}

# Integration best practices:
savecasedata = False           # File I/O will crush performance, so recommended not to save case data
samplemethod = 'sobol'         # Use 'sobol' over 'sobol_random' for a speedup, since all our dists are uniform
ndraws = next_power_of_2(1e6)  # The sobol methods need to be a power of 2 for best performance and balance
firstcaseisnom = False         # Since we want a power of 2, we should not run a 'nominal' case which would add 1

seed=12362397

def integration_example_monte_carlo_sim():

    sim = MCSim(name='integration', ndraws=ndraws, fcns=fcns, firstcaseisnom=firstcaseisnom, samplemethod=samplemethod, seed=seed, cores=4, savecasedata=savecasedata, verbose=True, debug=True)
    
    # We do need to know the limits of integration beforehand, and for integration these should always be uniform dists
    xrange = 2
    yrange = 2
    sim.addInVar(name='x', dist=uniform, distkwargs={'loc':-1, 'scale':xrange}) # -1 <= x <= 1
    sim.addInVar(name='y', dist=uniform, distkwargs={'loc':-1, 'scale':yrange}) # -1 <= y <= 1
    
    sim.runSim()
    
    total_area = xrange*yrange
    under_curve_pct = sum(sim.mcoutvars['is_under_curve'].nums)/ndraws # Note that (True,False) vals are automatically valmapped to the nums (1,0)

    resultsstr = f'π ≈ {under_curve_pct*total_area}, n = {ndraws}'
    print(resultsstr)
    
    '''
    from Monaco.mc_plot import mc_plot
    import matplotlib.pyplot as plt
    indices_under_curve = [i for i, x in enumerate(sim.mcoutvars['is_under_curve'].vals) if x]
    fig, ax = mc_plot(sim.mcinvars['x'], sim.mcinvars['y'], highlight_cases=indices_under_curve)
    plt.title(resultsstr)
    #'''
    
    return sim


if __name__ == '__main__':
    sim = integration_example_monte_carlo_sim()
    