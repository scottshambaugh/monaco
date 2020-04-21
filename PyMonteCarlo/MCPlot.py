import numpy as np
from scipy.stats import rv_continuous, rv_discrete, mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyMonteCarlo.MCVar import MCInVar, MCOutVar
from copy import copy
from helper_functions import get_iterable, slice_by_index


def MCPlot(mcvarx, mcvary = None, mcvarz = None, cases=0, ax=None):
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
            fig, ax = MCPlotHist(mcvar=mcvarx, cases=cases, ax=ax)
        else:
            mcvary = copy(mcvarx)
            mcvarx = copy(mcvarx) # don't overwrite the underlying object
            mcvarx.name = 'Simulation Steps'
            nums = []
            nums.extend([[*range(mcvary.size[1])] for i in range(mcvarx.ncases)])
            mcvarx.nums = nums
            mcvarx.nummap = None
            fig, ax = MCPlot2DLine(mcvarx=mcvarx, mcvary=mcvary, cases=cases, ax=ax)

    # Two Variable Plots
    elif mcvarz == None:
        if mcvarx.size[1] == 1 and mcvary.size[1] == 1:
            fig, ax = MCPlot2DScatter(mcvarx=mcvarx, mcvary=mcvary, cases=cases, ax=ax)
            
        elif mcvarx.size[1] > 1 and mcvary.size[1] > 1:
            fig, ax = MCPlot2DLine(mcvarx=mcvarx, mcvary=mcvary, cases=cases, ax=ax)
            
    # Three Variable Plots
    else:
        if mcvarx.size[1] == 1 and mcvary.size[1] == 1 and mcvarz.size[1] == 1:
            fig, ax = MCPlot3DScatter(mcvarx=mcvarx, mcvary=mcvary, mcvarz=mcvarz, cases=cases, ax=ax)
            
        elif mcvarx.size[1] > 1 and mcvary.size[1] > 1 and mcvarz.size[1] > 1:
            fig, ax = MCPlot3DLine(mcvarx=mcvarx, mcvary=mcvary, mcvarz=mcvarz, cases=cases, ax=ax)
    
    return fig, ax



def MCPlotHist(mcvar, cases=0, cumulative=False, ax=None):
    fig, ax = setAxis(ax, is3d=False)

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
    
    highlighted_cases = get_iterable(cases)
    for i in highlighted_cases:
        plt.plot([mcvar.nums[i], mcvar.nums[i]], ax.get_ylim(), 'r-')

    plt.xlabel(mcvar.name)
    if cumulative:
        plt.ylabel('Cumulative Probability')
    else:
        plt.ylabel('Probability Density')
    applyCategoryLabels(ax, mcvar)
    
    return fig, ax
        


def MCPlotCDF(mcvar, cases=0, ax=None):
    return MCPlotHist(mcvar=mcvar, cases=cases, cumulative=True, ax=ax)



def MCPlot2DScatter(mcvarx, mcvary, cases=0, ax=None):
    fig, ax = setAxis(ax, is3d=False)
    colorblack = [[0,0,0],]
    colorred = [[1,0,0],]

    reg_cases = set(range(mcvarx.ncases)) - set(get_iterable(cases))
    highlighted_cases = get_iterable(cases)
    if reg_cases:
        plt.scatter(slice_by_index(mcvarx.nums, reg_cases), slice_by_index(mcvary.nums, reg_cases), edgecolors=None, c=colorblack, alpha=0.4)
    if highlighted_cases:
        plt.scatter(slice_by_index(mcvarx.nums, highlighted_cases), slice_by_index(mcvary.nums, highlighted_cases), edgecolors=None, c=colorred, alpha=1)        

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    applyCategoryLabels(ax, mcvarx, mcvary)

    return fig, ax



def MCPlot2DLine(mcvarx, mcvary, cases=0, ax=None):
    fig, ax = setAxis(ax, is3d=False)
    
    reg_cases = set(range(mcvarx.ncases)) - set(get_iterable(cases))
    highlighted_cases = get_iterable(cases)
    for i in reg_cases:
        plt.plot(mcvarx.nums[i], mcvary.nums[i], 'k-', alpha=0.3)
    for i in highlighted_cases:
        plt.plot(mcvarx.nums[i], mcvary.nums[i], 'r-', alpha=1)     

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    applyCategoryLabels(ax, mcvarx, mcvary)

    return fig, ax



def MCPlot3DScatter(mcvarx, mcvary, mcvarz, cases=0, ax=None):
    fig, ax = setAxis(ax, is3d=True)
    colorblack = [[0,0,0],]
    colorred = [[1,0,0],]
    
    reg_cases = set(range(mcvarx.ncases)) - set(get_iterable(cases))
    highlighted_cases = get_iterable(cases)
    if reg_cases:
        ax.scatter(slice_by_index(mcvarx.nums, reg_cases), slice_by_index(mcvary.nums, reg_cases), \
                   slice_by_index(mcvarz.nums, reg_cases), edgecolors=None, c=colorblack, alpha=0.4)
    if highlighted_cases:
        ax.scatter(slice_by_index(mcvarx.nums, highlighted_cases), slice_by_index(mcvary.nums, highlighted_cases), \
                   slice_by_index(mcvarz.nums, highlighted_cases), edgecolors=None, c=colorred, alpha=1)        

    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    applyCategoryLabels(ax, mcvarx, mcvary, mcvarz)

    return fig, ax



def MCPlot3DLine(mcvarx, mcvary, mcvarz, cases=0, ax=None):
    fig, ax = setAxis(ax, is3d=True)
    
    reg_cases = set(range(mcvarx.ncases)) - set(get_iterable(cases))
    highlighted_cases = get_iterable(cases)
    for i in reg_cases:
        ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], 'k-', alpha=0.3)
    for i in highlighted_cases:
        ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], 'r-', alpha=1)
        
    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    applyCategoryLabels(ax, mcvarx, mcvary, mcvarz)

    return fig, ax



def MCPlotCorr(corrcoeff, corrvarnames, ax=None):
    fig, ax = setAxis(ax, is3d=False)
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

    return fig, ax



def setAxis(ax, is3d=False):
    if not ax:
        if is3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    plt.sca(ax)
    return fig, ax



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
plt.close('all')

mcinvars = dict()
nummap={1:'a',2:'b',3:'c',4:'d',5:'e'}
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000, nummap=nummap)
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
mcinvars['norm2'] = MCInVar('norm2', norm, (10, 4), 1000)
mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)

f, (ax1, ax2) = plt.subplots(2, 1)
MCPlot(mcinvars['randint'], ax=ax1)  # MCPlotHist
MCPlot(mcinvars['norm'])  # MCPlotHist
MCPlot(mcoutvars['test'])  # MCPlotHist
MCPlotCDF(mcinvars['randint'], ax=ax2)  # MCPlotCDF
MCPlotCDF(mcinvars['norm'])  # MCPlotCDF
MCPlotCDF(mcoutvars['test'])  # MCPlotCDF

MCPlot(mcinvars['randint'], mcinvars['norm'], cases=range(10,30))  # MCPlot2DScatter
MCPlot(mcinvars['randint'], mcinvars['norm'],  mcinvars['norm2'])  # MCPlot3DScatter

v = np.array([-2, -1, 2, 3, 4, 5])
var1 = MCOutVar('testx', [v, v, v, v, v], firstcaseisnom=True)
var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)
var3 = MCOutVar('testz', [1*v, 2*v, 0*v, -1*v, -1*v], firstcaseisnom=True)

MCPlot(var2, cases=None)  # MCPlot2DLine
MCPlot(var1, var2, cases=[0,1])  # MCPlot2DLine
MCPlot(var1, var2, var3)  # MCPlot3DLine

MCPlotCorr(np.array([[1, 0.1111],[-0.1111, -1]]), ['Test1', 'Test2'])
#'''
