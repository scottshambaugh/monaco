import numpy as np
from scipy.stats import rv_continuous, rv_discrete, mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyMonteCarlo.MCVar import MCInVar, MCOutVar
from copy import copy


def MCPlot(mcvarx, mcvary = None, mcvarz = None):
    # Split larger vars
    if mcvary == None and mcvarz == None:
        if mcvarx.size[0] not in (1, 2, 3):
            raise ValueError(f'Invalid variable size at index 0: ({mcvarx.size[0]},{mcvarx.size[1]})')
        elif mcvarx.size[0] in (2, 3):
            mcvarx_split = mcvarx.split()
            origname = mcvarx.name
            origsize = mcvarx.size
            mcvarx = mcvarx_split[origname + ' [0]']
            mcvary = mcvarx_split[origname + ' [1]']
            if origsize[0] == 3:
                mcvarz = mcvarx_split[origname + ' [2]']
            
    elif mcvarz == None:
        if    (mcvarx.size[0] not in (1, 2)) \
          and (mcvary.size[0] not in (1, 2)) \
          and (mcvarx.size[0] + mcvary.size[0] not in (2, 3)):
            raise ValueError(f'Invalid variable sizes at index 0: ({mcvarx.size[0]},{mcvarx.size[1]}), ({mcvary.size[0]},{mcvary.size[1]})')
        elif mcvarx.size[0] == 2:
            mcvarx_split = mcvarx.split()
            origname = mcvarx.name
            mcvarz = mcvary
            mcvarx = mcvarx_split[origname + ' [0]']
            mcvary = mcvarx_split[origname + ' [1]']
        elif mcvary.size[0] == 2:
            mcvary_split = mcvary.split()
            origname = mcvary.name
            mcvary = mcvary_split[origname + ' [0]']
            mcvarz = mcvary_split[origname + ' [1]']

    # Single Variable Plots
    if mcvary == None and mcvarz == None:
        if mcvarx.size[1] == 1:
            MCPlotHist(mcvarx)
        else:
            mcvary = copy(mcvarx)
            mcvarx = copy(mcvarx) # don't overwrite the underlying object
            mcvarx.name = 'Simulation Steps'
            nums = []
            nums.extend([[*range(mcvary.size[1])] for i in range(mcvarx.ncases)])
            mcvarx.nums = nums
            mcvarx.nummap = None
            MCPlot2DLine(mcvarx, mcvary)

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
    
    

def MCPlotHist(mcvar, cumulative=False):
    fig, ax = plt.subplots(1, 1)

    # Histogram generation
    counts, bins = np.histogram(mcvar.nums, bins='auto')
    binwidth = mode(np.diff(bins))[0]
    bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
    counts, bins = np.histogram(mcvar.nums, bins=bins)
    
    if isinstance(mcvar, MCInVar): 
        # Continuous distribution
        if isinstance(mcvar.dist, rv_continuous):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=True, cumulative=cumulative, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/100)
            dist = mcvar.dist(*mcvar.distargs)
            if cumulative:
                plt.plot(x, dist.cdf(x), color='k', alpha=0.9)
            else:
                plt.plot(x, dist.pdf(x), color='k', alpha=0.9)
        
        # Discrete distribution
        elif isinstance(mcvar.dist, rv_discrete):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, cumulative=cumulative, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.concatenate(([xlim[0]], bins, [xlim[1]]))
            dist = mcvar.dist(*mcvar.distargs)
            if cumulative:
                plt.step(x, dist.cdf(x), color='k', alpha=0.9)
            else:
                pdf = np.diff(dist.cdf(x))
                plt.step(x[1:], pdf, color='k', alpha=0.9)
        
    elif isinstance(mcvar, MCOutVar): 
        plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, cumulative=cumulative, histtype='bar', facecolor='k', alpha=0.5)
    
    if mcvar.firstcaseisnom:
        plt.plot([mcvar.getNom(), mcvar.getNom()], ax.get_ylim(), 'r-')

    plt.xlabel(mcvar.name)
    if cumulative:
        plt.ylabel('Cumulative Probability')
    else:
        plt.ylabel('Probability Density')
    applyCategoryLabels(ax, mcvar)
        


def MCPlotCDF(mcvar):
    MCPlotHist(mcvar, cumulative=True)



def MCPlot2DScatter(mcvarx, mcvary):
    fig, ax = plt.subplots(1, 1)
    colorblack = [[0,0,0],]
    colorred = [[1,0,0],]
    idx = int(mcvarx.firstcaseisnom)

    if not mcvarx.firstcaseisnom or mcvarx.ndraws > 0:
        plt.scatter(mcvarx.nums[idx:-1], mcvary.nums[idx:-1], edgecolors=None, c=colorblack, alpha=0.4)
    if mcvarx.firstcaseisnom:
        plt.scatter(mcvarx.nums[0], mcvary.nums[0], edgecolors=None, c=colorred, alpha=1)        

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    applyCategoryLabels(ax, mcvarx, mcvary)



def MCPlot2DLine(mcvarx, mcvary):
    fig, ax = plt.subplots(1, 1)
    
    if mcvarx.ndraws > 0:
        for i in range(int(mcvarx.firstcaseisnom), mcvarx.ncases):
            plt.plot(mcvarx.nums[i], mcvary.nums[i], 'k-', alpha=0.3)
    if mcvarx.firstcaseisnom:
        plt.plot(mcvarx.nums[0], mcvary.nums[0], 'r-', alpha=1)     

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    applyCategoryLabels(ax, mcvarx, mcvary)



def MCPlot3DScatter(mcvarx, mcvary, mcvarz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colorblack = [[0,0,0],]
    colorred = [[1,0,0],]
    idx = int(mcvarx.firstcaseisnom)
    
    if mcvarx.ndraws > 0:
        ax.scatter(mcvarx.nums[idx:-1], mcvary.nums[idx:-1],  mcvarz.nums[idx:-1], edgecolors=None, c=colorblack, alpha=0.4)
    if mcvarx.firstcaseisnom:
        ax.scatter(mcvarx.nums[0], mcvary.nums[0], mcvarz.nums[0], edgecolors=None, c=colorred, alpha=1)        

    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    applyCategoryLabels(ax, mcvarx, mcvary, mcvarz)



def MCPlot3DLine(mcvarx, mcvary, mcvarz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if mcvarx.ndraws > 0:
        for i in range(int(mcvarx.firstcaseisnom), mcvarx.ncases):
            ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], 'k-', alpha=0.3)
    if mcvarx.firstcaseisnom:
        ax.plot(mcvarx.nums[0], mcvary.nums[0], mcvarz.nums[0], 'r-', alpha=1)
        
    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    applyCategoryLabels(ax, mcvarx, mcvary, mcvarz)



def MCPlotCorr(corrcoeff, corrvarnames):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(corrcoeff, cmap="RdBu", vmin=-1, vmax=1)
    n = corrcoeff.shape[1]
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corrvarnames)
    ax.set_yticklabels(corrvarnames)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor')
    
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(n+1)-.5, minor=True)
    ax.set_yticks(np.arange(n+1)-.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    threshold = 0.6
    textcolors = ["black", "white"]
    kw = {'horizontalalignment':'center', 'verticalalignment':'center'}

    texts = []
    for i in range(n):
        for j in range(n):
            kw.update(color=textcolors[int(abs(corrcoeff[i, j]) > threshold)])
            text = im.axes.text(j, i, f'{corrcoeff[i, j]:.2f}', **kw)
            texts.append(text)



def applyCategoryLabels(ax, mcvarx, mcvary=None, mcvarz=None):
    # Wrapped in try statements in case some categories aren't printable
    if mcvarx.nummap != None:
        try:
            plt.xticks(list(mcvarx.nummap.keys()), list(mcvarx.nummap.values()))
        except:
            pass
    if mcvary != None and mcvary.nummap != None:
        try:
            plt.yticks(list(mcvary.nummap.keys()), list(mcvary.nummap.values()))
        except:
            pass
    if mcvarz != None and mcvarz.nummap != None:
        try:
            ax.set_zticks(list(mcvarz.nummap.keys()))
            ax.set_zticklabels(list(mcvarz.nummap.values()))
        except:
            pass



'''
### Test ###
np.random.seed(74494861)
from scipy.stats import randint, norm
mcinvars = dict()
nummap={1:'a',2:'b',3:'c',4:'d',5:'e'}
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000, nummap=nummap)
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
mcinvars['norm2'] = MCInVar('norm2', norm, (10, 4), 1000)
mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)

MCPlot(mcinvars['randint'])  # MCPlotHist
MCPlot(mcinvars['norm'])  # MCPlotHist
MCPlot(mcoutvars['test'])  # MCPlotHist
MCPlotCDF(mcinvars['randint'])  # MCPlotCDF
MCPlotCDF(mcinvars['norm'])  # MCPlotCDF
MCPlotCDF(mcoutvars['test'])  # MCPlotCDF

MCPlot(mcinvars['randint'], mcinvars['norm'])  # MCPlot2DScatter
MCPlot(mcinvars['randint'], mcinvars['norm'],  mcinvars['norm2'])  # MCPlot3DScatter

v = np.array([-2, -1, 2, 3, 4, 5])
var1 = MCOutVar('testx', [v, v, v, v, v], firstcaseisnom=True)
var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)
var3 = MCOutVar('testz', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)

MCPlot(var2)  # MCPlot2DLine
MCPlot(var1, var2)  # MCPlot2DLine
MCPlot(var1, var2, var3)  # MCPlot3DLine

MCPlotCorr(np.array([[1, 0.1111],[-0.1111, -1]]), ['Test1', 'Test2'])
#'''
