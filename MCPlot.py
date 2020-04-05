import numpy as np
from scipy.stats import rv_continuous, rv_discrete, mode
import matplotlib.pyplot as plt
from MCVar import MCInVar, MCOutVar


def MCPlot(mcvarx, mcvary = None, mcvarz = None):
    
    # Single Variable Plots
    if mcvary == None and mcvarz == None:
        if len([mcvarx.vals[0],]) == 1: # TODO: fix up
            MCPlotHist(mcvarx)


def MCPlotHist(mcvar):
    if isinstance(mcvar, MCInVar): 
        fig, ax = plt.subplots(1, 1)
        
        # Histogram generation
        counts, bins = np.histogram(mcvar.vals, bins='auto')
        binwidth = mode(np.diff(bins))[0]
        bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
        counts, bins = np.histogram(mcvar.vals, bins=bins)

        # Continuous distribution
        if isinstance(mcvar.dist, rv_continuous):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=True, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/100)
            dist = mcvar.dist(*mcvar.distargs)
            plt.plot(x, dist.pdf(x), color='k', alpha=0.9)
        
        # Discrete distribution
        elif isinstance(mcvar.dist, rv_discrete):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.concatenate(([xlim[0]], bins, [xlim[1]]))
            dist = mcvar.dist(*mcvar.distargs)
            pdf = np.diff(dist.cdf(x))
            plt.step(x[1:], pdf, color='k', alpha=0.9)

        plt.xlabel(mcvar.name)
        plt.ylabel('Probability Density')
        
    elif isinstance(mcvar, MCOutVar): 
        fig, ax = plt.subplots(1, 1)
        
        # Histogram generation
        counts, bins = np.histogram(mcvar.vals, bins='auto')
        binwidth = mode(np.diff(bins))[0]
        bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
        counts, bins = np.histogram(mcvar.vals, bins=bins)

        plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, histtype='bar', facecolor='k', alpha=0.5)
        if mcvar.firstcaseisnom:
            plt.plot([mcvar.getNom(), mcvar.getNom()], ax.get_ylim(), 'k-')

        plt.xlabel(mcvar.name)
        plt.ylabel('Probability Density')




'''
### Test ###
np.random.seed(74494861)
from scipy.stats import randint, norm
mcinvars = dict()
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000)
MCPlot(mcinvars['randint'])
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
MCPlot(mcinvars['norm'])

mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)
MCPlot(mcoutvars['test'])

#'''
