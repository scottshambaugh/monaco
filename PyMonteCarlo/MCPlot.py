import numpy as np
from scipy.stats import rv_continuous, rv_discrete, mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyMonteCarlo.MCVar import MCInVar, MCOutVar
from copy import copy
from helper_functions import get_iterable, slice_by_index


def MCPlot(mcvarx, mcvary = None, mcvarz = None, cases=0, ax=None, title=''):
    # Split larger vars
    if mcvary == None and mcvarz == None:
        if mcvarx.size[0] not in (1, 2, 3):
            raise ValueError(f'Invalid variable size at index 0: {mcvarx.name} ({mcvarx.size[0]},{mcvarx.size[1]})')
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
          or (mcvary.size[0] not in (1, 2)) \
          or (mcvarx.size[0] + mcvary.size[0] not in (2, 3)):
            raise ValueError(f'Invalid variable sizes at indices 0: {mcvarx.name} ({mcvarx.size[0]},{mcvarx.size[1]}), {mcvary.name} ({mcvary.size[0]},{mcvary.size[1]})')
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
            fig, ax = MCPlotHist(mcvar=mcvarx, cases=cases, ax=ax, title=title)
        else:
            mcvary = copy(mcvarx)
            mcvarx = copy(mcvarx) # don't overwrite the underlying object
            mcvarx.name = 'Simulation Steps'
            nums = []
            nums.extend([[*range(mcvary.size[1])] for i in range(mcvarx.ncases)])
            mcvarx.nums = nums
            mcvarx.nummap = None
            fig, ax = MCPlot2DLine(mcvarx=mcvarx, mcvary=mcvary, cases=cases, ax=ax, title=title)

    # Two Variable Plots
    elif mcvarz == None:
        if mcvarx.size[1] == 1 and mcvary.size[1] == 1:
            fig, ax = MCPlot2DScatter(mcvarx=mcvarx, mcvary=mcvary, cases=cases, ax=ax, title=title)
            
        elif mcvarx.size[1] > 1 and mcvary.size[1] > 1:
            fig, ax = MCPlot2DLine(mcvarx=mcvarx, mcvary=mcvary, cases=cases, ax=ax, title=title)
            
    # Three Variable Plots
    else:
        if mcvarx.size[1] == 1 and mcvary.size[1] == 1 and mcvarz.size[1] == 1:
            fig, ax = MCPlot3DScatter(mcvarx=mcvarx, mcvary=mcvary, mcvarz=mcvarz, cases=cases, ax=ax, title=title)
            
        elif mcvarx.size[1] > 1 and mcvary.size[1] > 1 and mcvarz.size[1] > 1:
            fig, ax = MCPlot3DLine(mcvarx=mcvarx, mcvary=mcvary, mcvarz=mcvarz, cases=cases, ax=ax, title=title)
    
    return fig, ax



def MCPlotHist(mcvar, cases=0, cumulative=False, orientation='vertical', ax=None, title=''):
    fig, ax = manage_axis(ax, is3d=False)

    # Histogram generation
    counts, bins = np.histogram(mcvar.nums, bins='auto')
    binwidth = mode(np.diff(bins))[0]
    bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
    counts, bins = np.histogram(mcvar.nums, bins=bins)
    
    if isinstance(mcvar, MCInVar):          
        # Continuous distribution
        if isinstance(mcvar.dist, rv_continuous):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=True, cumulative=cumulative, orientation=orientation, histtype='bar', facecolor='k', alpha=0.5)
            lim = get_hist_lim(orientation, ax)
            x = np.arange(lim[0], lim[1], (lim[1] - lim[0])/100)
            dist = mcvar.dist(*mcvar.distargs)
            if cumulative:
                ydata = dist.cdf(x)
            else:
                ydata = dist.pdf(x)
            if orientation == 'vertical':
                plt.plot(x, ydata, color='k', alpha=0.9)
            elif orientation == 'horizontal':
                plt.plot(ydata, x, color='k', alpha=0.9)            
        
        # Discrete distribution
        elif isinstance(mcvar.dist, rv_discrete):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, orientation=orientation, cumulative=cumulative, histtype='bar', facecolor='k', alpha=0.5)
            lim = get_hist_lim(orientation, ax)
            x = np.concatenate(([lim[0]], bins, [lim[1]]))
            dist = mcvar.dist(*mcvar.distargs)
            if cumulative:
                xdata = x
                ydata = dist.cdf(x)
            else:
                xdata = x[1:]
                ydata = np.diff(dist.cdf(x)) # manual pdf
            if orientation == 'vertical':
                plt.step(xdata, ydata, color='k', alpha=0.9, where='post')
            elif orientation == 'horizontal':
                plt.step(ydata, xdata, color='k', alpha=0.9, where='post') 
        
    elif isinstance(mcvar, MCOutVar): 
        plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, orientation=orientation, cumulative=cumulative, histtype='bar', facecolor='k', alpha=0.5)
    
    highlighted_cases = get_iterable(cases)
    if cumulative:
        ylabeltext = 'Cumulative Probability'
    else:
        ylabeltext = 'Probability Density'

    if orientation == 'vertical':
        ylim = ax.get_ylim()
        for i in highlighted_cases:
            plt.plot([mcvar.nums[i], mcvar.nums[i]], ylim, 'r-')
        plt.xlabel(mcvar.name)
        plt.ylabel(ylabeltext)
        apply_category_labels(ax, mcvarx=mcvar)
    elif orientation == 'horizontal':
        xlim = ax.get_xlim()
        for i in highlighted_cases:
            plt.plot(xlim, [mcvar.nums[i], mcvar.nums[i]], 'r-')
        plt.ylabel(mcvar.name)
        plt.xlabel(ylabeltext)
        apply_category_labels(ax, mcvary=mcvar)
    plt.title(title)
    
    return fig, ax
        


def MCPlotCDF(mcvar, cases=0, orientation='vertical', ax=None, title=''):
    return MCPlotHist(mcvar=mcvar, cases=cases, orientation=orientation, cumulative=True, ax=ax, title=title)



def MCPlot2DScatter(mcvarx, mcvary, cases=0, ax=None, title=''):
    fig, ax = manage_axis(ax, is3d=False)
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
    apply_category_labels(ax, mcvarx, mcvary)
    plt.title(title)

    return fig, ax



def MCPlot2DLine(mcvarx, mcvary, cases=0, ax=None, title=''):
    fig, ax = manage_axis(ax, is3d=False)
    
    reg_cases = set(range(mcvarx.ncases)) - set(get_iterable(cases))
    highlighted_cases = get_iterable(cases)
    for i in reg_cases:
        plt.plot(mcvarx.nums[i], mcvary.nums[i], 'k-', alpha=0.3)
    for i in highlighted_cases:
        plt.plot(mcvarx.nums[i], mcvary.nums[i], 'r-', alpha=1)     

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    apply_category_labels(ax, mcvarx, mcvary)
    plt.title(title)

    return fig, ax



def MCPlot3DScatter(mcvarx, mcvary, mcvarz, cases=0, ax=None, title=''):
    fig, ax = manage_axis(ax, is3d=True)
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
    apply_category_labels(ax, mcvarx, mcvary, mcvarz)
    plt.title(title)

    return fig, ax



def MCPlot3DLine(mcvarx, mcvary, mcvarz, cases=0, ax=None, title=''):
    fig, ax = manage_axis(ax, is3d=True)
    
    reg_cases = set(range(mcvarx.ncases)) - set(get_iterable(cases))
    highlighted_cases = get_iterable(cases)
    for i in reg_cases:
        ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], 'k-', alpha=0.3)
    for i in highlighted_cases:
        ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], 'r-', alpha=1)
        
    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    apply_category_labels(ax, mcvarx, mcvary, mcvarz)
    plt.title(title)

    return fig, ax



def MCPlotCovCorr(matrix, varnames, ax=None, title=''):
    fig, ax = manage_axis(ax, is3d=False)
    scale = np.nanmax(np.abs(matrix)) # for a correlation matrix this will always be 1 from diagonal
    im = ax.imshow(matrix, cmap="RdBu", vmin=-scale, vmax=scale)
    n = matrix.shape[1]
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(varnames)
    ax.set_yticklabels(varnames)
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
            kw.update(color=textcolors[int(abs(matrix[i, j]/scale) > threshold)])
            text = im.axes.text(j, i, f'{matrix[i, j]:.2f}', **kw)
            texts.append(text)
    plt.title(title)

    return fig, ax



def manage_axis(ax, is3d=False):
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



def apply_category_labels(ax, mcvarx=None, mcvary=None, mcvarz=None):
    # Wrapped in try statements in case some categories aren't printable
    if mcvarx != None and mcvarx.nummap != None:
        try:
            ax.set_xticks(list(mcvarx.nummap.keys()))
            ax.set_xticklabels(list(mcvarx.nummap.values()))
        except:
            pass
    if mcvary != None and mcvary.nummap != None:
        try:
            ax.set_yticks(list(mcvary.nummap.keys()))
            ax.set_yticklabels(list(mcvary.nummap.values()))
        except:
            pass
    if mcvarz != None and mcvarz.nummap != None:
        try:
            ax.set_zticks(list(mcvarz.nummap.keys()))
            ax.set_zticklabels(list(mcvarz.nummap.values()))
        except:
            pass



def get_hist_lim(orientation, ax):
    if orientation == 'vertical':
        lim = ax.get_xlim()
    elif orientation == 'horizontal':
        lim = ax.get_ylim()
    return lim


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
MCPlotHist(mcinvars['randint'], ax=ax1, orientation='horizontal') # MCPlotHist
MCPlot(mcinvars['norm'], title='norm')                            # MCPlotHist
MCPlotHist(mcoutvars['test'], orientation='horizontal')           # MCPlotHist
MCPlotCDF(mcinvars['randint'], ax=ax2)                            # MCPlotCDF
MCPlotCDF(mcinvars['norm'], orientation='horizontal')             # MCPlotCDF
MCPlotCDF(mcoutvars['test'])                                      # MCPlotCDF

MCPlot(mcinvars['randint'], mcinvars['norm'], cases=range(10,30))  # MCPlot2DScatter
MCPlot(mcinvars['randint'], mcinvars['norm'],  mcinvars['norm2'])  # MCPlot3DScatter

v = np.array([-2, -1, 2, 3, 4, 5])
var1 = MCOutVar('testx', [v, v, v, v, v], firstcaseisnom=True)
var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
var3 = MCOutVar('testz', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)

MCPlot(var2, cases=None)         # MCPlot2DLine
MCPlot(var1, var2, cases=[0,1])  # MCPlot2DLine
MCPlot(var1, var2, var3)         # MCPlot3DLine

MCPlotCovCorr(np.array([[2, 0.1111],[-0.19, -1]]), ['Test1', 'Test2'])
#'''
