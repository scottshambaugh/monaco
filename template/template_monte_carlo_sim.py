# template_monte_carlo_sim.py

import monaco as mc

# Import the statistical distributions from scipy.stats that you will be using.
# These must be rv_discrete or rv_continuous functions.
# See https://docs.scipy.org/doc/scipy/reference/stats.html for a complete list.
from scipy.stats import randint, rv_discrete

# Import the preprocess, sim, and postprocess function handles that you have created.
from template_functions import template_preprocess, template_run, template_postprocess

# These get packaged in the following format for Sim to consume:
fcns = {'preprocess' : template_preprocess,
        'run'        : template_run,
        'postprocess': template_postprocess}

# Set the number of random draws you wish to make.
# If firstcaseismedian is True, then case 0 will be run with the expected value of
# each statistical distribution as a 'nominal' run. The total number of cases
# will then be ndraws+1
ndraws = 500
firstcaseismedian = False

# Setting a known random seed is recommended for repeatability of random draws.
seed = 12362398

# Whether to run the simulation in a single thread. If True then the simulation
# will run single-threaded in a 'for' loop, and if False then the dask parallel
# processing module will be used for running concurrent threads. The overhead of
# dask may make it slower than single-threaded execution for runs that execute
# quickly.
singlethreaded = True

# If you want, you can save all the results from each case to file, or just the
# postprocessed simulation results. This can be incredibly useful for examining
# time-consuming sim data after the fact without having to rerun the whole thing.
# On the other hand, if your run function generates a lot of data, you may not have
# enough storage space to save all of the raw case data. Plus, the file I/O may
# be the limiting factor that slows down your simulation. Try setting these flags
# to False and seeing how much faster the sim runs.
savecasedata = True
savesimdata = True

def template_monte_carlo_sim():
    # We first initialize the sim with a name of our choosing
    sim = mc.Sim(name='Coin Flip', ndraws=ndraws, fcns=fcns,
                 firstcaseismedian=firstcaseismedian, seed=seed,
                 singlethreaded=singlethreaded,
                 savecasedata=savecasedata, savesimdata=savesimdata,
                 verbose=True, debug=False)

    # We now add input variables, with their associated distributions
    # Out first variable will be the person flipping a coin - Sam and Alex will
    # do so with even odds.
    # For even odds between 0 and 1 we want to call scipy.stats.randint(0,2)
    # The dist argument is therefor randint, and we look up its arguments in the
    # documentation:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html
    # The kwargs are {'low':0, 'high':2}, which we package in a keyword dictionary.
    # Since our inputs here are strings rather than numbers, we need to map the
    # numbers from our random draws to those imputs with a nummap dictionary
    nummap = {0: 'Sam', 1: 'Alex'}
    sim.addInVar(name='flipper', dist=randint, distkwargs={'low': 0, 'high': 2}, nummap=nummap)

    # If we want to generate custom odds, we can create our own distribution, see:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    # Let's assume that we have a weighted coin where heads comes up 70% of the time.
    # Our 'template_run' run function reads 0 as heads and 1 as tails.
    flip_dist = rv_discrete(name='flip_dist', values=([0, 1], [0.7, 0.3]))
    # If the distribution is custom then it doesn't take arguments, so pass in
    # an empty dictionary
    sim.addInVar(name='flip', dist=flip_dist, distkwargs=dict())

    # Once all input variables are initialized, we run the sim.
    # For each case, the preprocessing function will pull in the random variables
    # and any other data the run fuction needs, the run function executes, and
    # its output is passed to the postprocessing function for extracting the
    # desired output values from each case.
    # The cases are collected, and the individual output values are collected into
    # an output variable stored by the sim object.
    sim.runSim()

    # Once the sim is run, we have access to its member variables.
    print(f'{sim.name} Runtime: {sim.runtime}')

    # From here we can perform further postprocessing on our results. The outvar
    # names were assigned in our postprocessing function. We expect the heads
    # bias to be near the 70% we assigned up in flip_dist.
    bias = sim.outvars['Flip Result'].vals.count('heads')/sim.ncases*100
    print(f'Average heads bias: {bias}%')

    # We can also quickly make some plots of our invars and outvars. The mc.plot
    # function will automatically try to figure out which type of plot is most
    # appropriate based on the number and dimension of the variables.
    # This will make a histogram of the results:
    mc.plot(sim.outvars['Flip Result'])
    # And this scatter plot will show that the flips were random over time:
    mc.plot(sim.outvars['Flip Number'], sim.outvars['Flip Result'])

    # We can also look at the correlation between all scalar input and output
    # vars to see which are most affecting the others. This shows that the input
    # and output flip information is identical, and that the flipper and flip
    # number had no effect on the coin landing heads or tails.
    mc.plot_cov_corr(*sim.corr())

    # Alternatively, you can return the sim object and work with it elsewhere.
    return sim


if __name__ == '__main__':
    sim = template_monte_carlo_sim()
