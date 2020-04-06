import numpy as np
from scipy.stats import rv_continuous, rv_discrete, mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MCVar import MCInVar, MCOutVar
from copy import copy


def MCPlot(mcvarx, mcvary = None, mcvarz = None):
    # Split
    if mcvary == None and mcvarz == None:
        if mcvarx.size[0] not in (1, 2, 3):
            raise ValueError(f'Invalid variable size at index 0: ({mcvarx.size[0]},{mcvarx.size[1]})')
        elif mcvarx.size[0] == 2:
            mcvary = MCOutVar(mcvarx.name + ' [1]', mcvarx.vals[1], mcvarx.ndraws, mcvarx.firstcaseisnom)
            mcvarx = MCOutVar(mcvarx.name + ' [0]', mcvarx.vals[0], mcvarx.ndraws, mcvarx.firstcaseisnom)
        elif mcvarx.size[0] == 3:
            mcvarz = MCOutVar(mcvarx.name + ' [2]', mcvarx.vals[2], mcvarx.ndraws, mcvarx.firstcaseisnom)
            mcvary = MCOutVar(mcvarx.name + ' [1]', mcvarx.vals[1], mcvarx.ndraws, mcvarx.firstcaseisnom)
            mcvarx = MCOutVar(mcvarx.name + ' [0]', mcvarx.vals[0], mcvarx.ndraws, mcvarx.firstcaseisnom)
            
    elif mcvarz == None:
        if mcvarx.size[0] + mcvary.size[0] not in (2, 3):
            raise ValueError(f'Invalid variable sizes at index 0: ({mcvarx.size[0]},{mcvarx.size[1]}), ({mcvary.size[0]},{mcvary.size[1]})')
        elif mcvarx.size[0] == 2:
            mcvarz = mcvary
            mcvary = MCOutVar(mcvarx.name + ' [1]', mcvarx.vals[1], mcvarx.ndraws, mcvarx.firstcaseisnom)
            mcvarx = MCOutVar(mcvarx.name + ' [0]', mcvarx.vals[0], mcvarx.ndraws, mcvarx.firstcaseisnom)
        elif mcvary.size[0] == 2:
            mcvarz = MCOutVar(mcvary.name + ' [2]', mcvary.vals[1], mcvary.ndraws, mcvary.firstcaseisnom)
            mcvary = MCOutVar(mcvary.name + ' [1]', mcvary.vals[0], mcvary.ndraws, mcvary.firstcaseisnom)

    # Single Variable Plots
    if mcvary == None and mcvarz == None:
        if mcvarx.size[1] == 1:
            MCPlotHist(mcvarx)
        else:
            mcvart = copy(mcvarx)
            mcvart.name = 'Simulation Steps'
            vals = []
            vals.extend([[*range(mcvarx.size[1])] for i in range(mcvart.ncases)])
            mcvart.vals = vals
            MCPlot2DLine(mcvart, mcvarx)

    # Two Variable Plots
    elif mcvarz == None:
        if mcvarx.size[1] == 1 and mcvary.size[1] == 1:
            MCPlot2DScatter(mcvarx, mcvary)
            
        elif mcvarx.size[1] > 1 and mcvary.size[1] > 1:
            MCPlot2DLine(mcvarx, mcvary)
            
    # Three Variable Plots
    else:
        if mcvarx.size[1] == 1 and mcvary.size[1] == 1 and mcvarz.size[1] == 1:
            MCPlot3DScatter(mcvarx, mcvary, mcvarz)
            
        elif mcvarx.size[1] > 1 and mcvary.size[1] > 1 and mcvarz.size[1] > 1:
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
        plt.plot([mcvar.getNom(), mcvar.getNom()], ax.get_ylim(), 'r-')

    plt.xlabel(mcvar.name)
    plt.ylabel('Probability Density')



def MCPlot2DScatter(mcvarx, mcvary):
    fig, ax = plt.subplots(1, 1)
    colorblack = [[0,0,0],]
    colorred = [[1,0,0],]
    idx = int(mcvarx.firstcaseisnom)

    if not mcvarx.firstcaseisnom or mcvarx.ndraws > 0:
        plt.scatter(mcvarx.vals[idx:-1], mcvary.vals[idx:-1], edgecolors=None, c=colorblack, alpha=0.5)
    if mcvarx.firstcaseisnom:
        plt.scatter(mcvarx.vals[0], mcvary.vals[0], edgecolors=None, c=colorred, alpha=1)        

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)



def MCPlot2DLine(mcvarx, mcvary):
    fig, ax = plt.subplots(1, 1)
    
    if mcvarx.ndraws > 0:
        for i in range(int(mcvarx.firstcaseisnom), mcvarx.ncases):
            plt.plot(mcvarx.vals[i], mcvary.vals[i], 'k-', alpha=0.5)
    if mcvarx.firstcaseisnom:
        plt.plot(mcvarx.vals[0], mcvary.vals[0], 'r-', alpha=1)     

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)



def MCPlot3DScatter(mcvarx, mcvary, mcvarz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colorblack = [[0,0,0],]
    colorred = [[1,0,0],]
    idx = int(mcvarx.firstcaseisnom)
    
    if mcvarx.ndraws > 0:
        ax.scatter(mcvarx.vals[idx:-1], mcvary.vals[idx:-1],  mcvarz.vals[idx:-1], edgecolors=None, c=colorblack, alpha=0.5)
    if mcvarx.firstcaseisnom:
        ax.scatter(mcvarx.vals[0], mcvary.vals[0], mcvarz.vals[0], edgecolors=None, c=colorred, alpha=1)        

    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)



def MCPlot3DLine(mcvarx, mcvary, mcvarz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if mcvarx.ndraws > 0:
        for i in range(int(mcvarx.firstcaseisnom), mcvarx.ncases):
            ax.plot(mcvarx.vals[i], mcvary.vals[i], mcvarz.vals[i], 'k-', alpha=0.5)
    if mcvarx.firstcaseisnom:
        ax.plot(mcvarx.vals[0], mcvary.vals[0], mcvarz.vals[0], 'r-', alpha=1)
        
    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)




'''
### Test ###
np.random.seed(74494861)
from scipy.stats import randint, norm
from IPython import get_ipython
mcinvars = dict()
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000)
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
mcinvars['norm2'] = MCInVar('norm2', norm, (10, 4), 1000)
mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)

MCPlot(mcinvars['randint'])  # MCPlotHist
MCPlot(mcinvars['norm'])  # MCPlotHist
MCPlot(mcoutvars['test'])  # MCPlotHist

MCPlot(mcinvars['randint'], mcinvars['norm'])  # MCPlot2DScatter
MCPlot(mcinvars['randint'], mcinvars['norm'],  mcinvars['norm2'])  # MCPlot3DScatter

v = np.array([-2, -1, 2, 3, 4, 5])
var1 = MCOutVar('testx', [v, v, v, v, v], firstcaseisnom=True)
var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)
var3 = MCOutVar('testz', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)

MCPlot(var2)  # MCPlot2DLine
MCPlot(var1, var2)  # MCPlot2DLine
MCPlot(var1, var2, var3)  # MCPlot3DLine
#'''
