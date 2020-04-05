import numpy as np
from scipy.stats import rv_continuous, rv_discrete, mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MCVar import MCInVar, MCOutVar


def MCPlot(mcvarx, mcvary = None, mcvarz = None):
    
    # Single Variable Plots
    if mcvary == None and mcvarz == None:
        if len([mcvarx.vals[0],]) == 1: # TODO: fix up
            MCPlotHist(mcvarx)

    # Two Variable Plots
    elif mcvarz == None:
        if len([mcvarx.vals[0],]) == 1 and len([mcvary.vals[0],]) == 1: # TODO: fix up
            MCPlot2DScatter(mcvarx, mcvary)
            
        elif len([mcvarx.vals[0],]) > 1 and len([mcvary.vals[0],]) > 1: # TODO: fix up
            MCPlot2DLine(mcvarx, mcvary)
            
    # Three Variable Plots
    else:
        if len([mcvarx.vals[0],]) == 1 and len([mcvary.vals[0],]) == 1 and len([mcvarz.vals[0],]) == 1: # TODO: fix up
            MCPlot3DScatter(mcvarx, mcvary, mcvarz)
            
        elif len([mcvarx.vals[0],]) > 1 and len([mcvary.vals[0],]) > 1: # TODO: fix up
            MCPlot3DLine(mcvarx, mcvary, mcvarz)



def MCPlotHist(mcvar):
    fig, ax = plt.subplots(1, 1)

    # Histogram generation
    counts, bins = np.histogram(mcvar.vals, bins='auto')
    binwidth = mode(np.diff(bins))[0]
    bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
    counts, bins = np.histogram(mcvar.vals, bins=bins)
    
    if isinstance(mcvar, MCInVar): 
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
        
    elif isinstance(mcvar, MCOutVar): 
        plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, histtype='bar', facecolor='k', alpha=0.5)
    
    if mcvar.firstcaseisnom:
        plt.plot([mcvar.getNom(), mcvar.getNom()], ax.get_ylim(), 'k-')

    plt.xlabel(mcvar.name)
    plt.ylabel('Probability Density')



def MCPlot2DScatter(mcvarx, mcvary):
    fig, ax = plt.subplots(1, 1)
    colorblack = [[0,0,0],]
    
    if not mcvarx.firstcaseisnom or mcvarx.ndraws > 0:
        plt.scatter(mcvarx.vals, mcvary.vals, edgecolors=None, c=colorblack, alpha=0.5)
    if mcvarx.firstcaseisnom:
        plt.scatter(mcvarx.vals[0], mcvary.vals[0], edgecolors=None, c=colorblack, alpha=1)        

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)


def MCPlot2DLine(mcvarx, mcvary):
    fig, ax = plt.subplots(1, 1)
    
    if not mcvarx.firstcaseisnom or mcvarx.ndraws > 0:
        for i in range(mcvarx.ncases):
            plt.plot(mcvarx.vals[i], mcvary.vals[i], 'k-', alpha=0.5)
    if mcvarx.firstcaseisnom:
        plt.plot(mcvarx.vals[0], mcvary.vals[0], 'k-', alpha=1)     

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)


def MCPlot3DScatter(mcvarx, mcvary, mcvarz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colorblack = [[0,0,0],]
    
    if not mcvarx.firstcaseisnom or mcvarx.ndraws > 0:
        ax.scatter(mcvarx.vals, mcvary.vals,  mcvarz.vals, edgecolors=None, c=colorblack, alpha=0.5)
    if mcvarx.firstcaseisnom:
        ax.scatter(mcvarx.vals[0], mcvary.vals[0], mcvarz.vals[0], edgecolors=None, c=colorblack, alpha=1)        

    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)


def MCPlot3DLine(mcvarx, mcvary, mcvarz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if not mcvarx.firstcaseisnom or mcvarx.ndraws > 0:
        for i in range(mcvarx.ncases):
            ax.plot(mcvarx.vals[i], mcvary.vals[i], mcvarz.vals[i], 'k-', alpha=0.5)
    if mcvarx.firstcaseisnom:
        ax.plot(mcvarx.vals[0], mcvary.vals[0], mcvarz.vals[0], 'k-', alpha=0.5)
        
    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)




'''
### Test ###
np.random.seed(74494861)
from scipy.stats import randint, norm
mcinvars = dict()
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000)
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
mcinvars['norm2'] = MCInVar('norm2', norm, (10, 4), 1000)
mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)

MCPlotHist(mcinvars['randint'])
MCPlotHist(mcinvars['norm'])
MCPlotHist(mcoutvars['test'])

MCPlot2DScatter(mcinvars['randint'], mcinvars['norm'])
MCPlot3DScatter(mcinvars['randint'], mcinvars['norm'],  mcinvars['norm2'])

v = np.array([-2, -1, 2, 3, 4, 5])
var1 = MCOutVar('test', [v, v, v, v, v], firstcaseisnom=True)
var2 = MCOutVar('test', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)
var3 = MCOutVar('test', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)

MCPlot2DLine(var1, var2)
MCPlot3DLine(var1, var2, var3)
#'''
