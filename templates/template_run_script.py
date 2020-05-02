from PyMonteCarlo.MCSim import MCSim
from PyMonteCarlo.mc_plot import mc_plot, mc_plot_cov_corr

# Import the statistical distributions from scipy.stats that you will be using
# These must be rv_discrete or rv_continuous functions
# See https://docs.scipy.org/doc/scipy/reference/stats.html for a complete list
from scipy.stats import randint, rv_discrete

# Import the preprocess, sim, and postprocess function handles that you have created
from template_preprocess import template_preprocess
from template_sim import template_sim
from template_postprocess import template_postprocess

# These get packaged in the following format for MCSim to consume:
fcns ={'preprocess' :template_preprocess,   \
       'run'        :template_sim,          \
       'postprocess':template_postprocess}

# Set the number of random draws you wish to make
# If firstcaseisnom is True, then case 0 will be run with the mean of each 
# statistical distribution as a 'nominal' run. The total number of cases will 
# then be ndraws+1
ndraws = 500
firstcaseisnom = False

# Setting a known random seed is recommended for repeatability of random draws
seed=12362398

# The number of processor cores to run on. If 1 then the simulation will run 
# single-threaded in a for loop, and if more than 1 then the pathos multiprocessing
# module will be used for running concurrent threads. The overhead of multithreading
# may make it slower than single-threaded execution
cores = 1

def template_run_script():
    # We first initialize the sim with a name of our choosing
    sim = MCSim(name='Coin Flip', ndraws=ndraws, fcns=fcns, firstcaseisnom=firstcaseisnom, seed=seed, cores=cores)
    
    # We now add input variables, with their associated distributions
    # Out first variable will be the person flipping a coin - Sam and Alex will 
    # do so with even odds
    # For even odds between 0 and 1 we want to call scipy.stats.randint(0,2)
    # The dist argument is therefor randint, and its arguments are (0,2),
    # packaged in a tuple
    # Since our inputs here are strings rather than numbers, we need to map the
    # numbers from our random draws to those imputs with a nummap dictionary
    flipper_nummap = {0:'Sam', 1:'Alex'}
    sim.addInVar(name='flipper', dist=randint, distargs=(0, 2), nummap=flipper_nummap)
    
    # If we want to generate custom odds, we can create our own distribution
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    # Let's assume that we have a weighted coin where heads comes up 70% of the time
    # Here, our run function takes in 0 as heads and 1 as tails
    flip_dist = rv_discrete(name='flip_dist', values=([0, 1], [0.7, 0.3]))
    # If the distribution is custom then it doesn't take arguments, so pass in
    # an empty tuple
    sim.addInVar(name='flip', dist=flip_dist, distargs=())

    # Once all input variables are initialized, we run the sim
    # The preprocessing function will pull in the random variables and any other
    # data the run fuction needs, the run function executes, and that output is 
    # passed to the postprocessing function for extracting the desired output
    # from that single case
    sim.runSim()

    # Once the sim is run, we have access to its member variables
    print(f'{sim.name} Runtime: {sim.runtime}')
    
    # From here we can perform further postprocessing on our results. The outvar
    # names were assigned in our postprocessing function
    bias = sim.mcoutvars['Flip Result'].vals.count('heads')/sim.ncases*100
    print(f'Average heads bias: {bias}%')
    
    # We can also quickly make some plots of our invars and outvars. The MCPlot 
    # function will automatically try to figure out which type of plot is most 
    # appropriate based on the number and dimension of the variables
    # The cases argument here is a list of cases to highlight on the plots
    # This will make a histogram of the results:
    mc_plot(sim.mcoutvars['Flip Result'], cases=[])
    # And this scatter plot will show that the flips were random over time
    mc_plot(sim.mcoutvars['Flip Number'], sim.mcoutvars['Flip Result'], cases=[])
    
    # We can also look at the correlation between all scalar input and output
    # vars to see which are most affecting the others. This shows that the input 
    # and output flip information is identical, and that the flipper and flip 
    # number had no effect on the coin landing heads or tails
    mc_plot_cov_corr(*sim.corr())
    
    # Alternatively, you can return the sim object and work with it elsewhere
    return sim

if __name__ == '__main__':
    sim = template_run_script()
    