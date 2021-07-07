# mc_plot.py

import numpy as np
from scipy.stats import rv_continuous, rv_discrete, chi2, mode
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from Monaco.MCVar import MCVar, MCInVar, MCOutVar
from copy import copy
from Monaco.helper_functions import get_iterable, slice_by_index, length
from Monaco.gaussian_statistics import conf_ellipsoid_sig2pct
from Monaco.integration_statistics import integration_error
from typing import Union


# If cases or highlight_cases are None, will plot all. Set to [] to plot none.
def mc_plot(mcvarx   : MCVar, 
            mcvary   : Union[None, MCVar] = None, 
            mcvarz   : Union[None, MCVar] = None, 
            cases           : Union[None, int, list[int], set[int]] = None, # All cases 
            highlight_cases : Union[None, int, list[int], set[int]] = [],   # No cases 
            rug_plot : bool               = False, 
            cov_plot : bool               = False,
            cov_p    : Union[None, float, list[float], set[float]]  = None,
            ax       : Union[None, Axes]  = None, 
            title    : str                = '',
            ):
    
    # Split larger vars
    if mcvary is None and mcvarz is None:
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
            
    elif mcvarz is None:
        if   (mcvarx.size[0] not in (1, 2)) \
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
    if mcvary is None and mcvarz is None:
        if mcvarx.size[1] == 1:
            fig, ax = mc_plot_hist(mcvar=mcvarx, highlight_cases=highlight_cases, rug_plot=rug_plot, ax=ax, title=title)
        else:
            mcvary = copy(mcvarx)
            mcvarx = copy(mcvarx) # don't overwrite the underlying object
            mcvarx.name = 'Simulation Steps'
            nums = []
            nums.extend([[*range(mcvary.size[1])] for i in range(mcvarx.ncases)])
            mcvarx.nums = nums
            mcvarx.nummap = None
            fig, ax = mc_plot_2d_line(mcvarx=mcvarx, mcvary=mcvary, highlight_cases=highlight_cases, ax=ax, title=title)

    # Two Variable Plots
    elif mcvarz is None:
        if mcvarx.size[1] != mcvary.size[1]:
            raise ValueError(f'Variables have inconsistent lengths: {mcvarx.name}:{mcvarx.size[1]}, {mcvary.name}:{mcvary.size[1]}')            
       
        if mcvarx.size[1] == 1:
            fig, ax = mc_plot_2d_scatter(mcvarx=mcvarx, mcvary=mcvary, cases=cases, highlight_cases=highlight_cases, rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p, ax=ax, title=title)
            
        elif mcvarx.size[1] > 1:
            fig, ax = mc_plot_2d_line(mcvarx=mcvarx, mcvary=mcvary, cases=cases, highlight_cases=highlight_cases, ax=ax, title=title)
            
    # Three Variable Plots
    else:
        if (mcvarx.size[1] != mcvary.size[1]) or (mcvarx.size[1] != mcvarz.size[1]) or (mcvary.size[1] != mcvarz.size[1]):
            raise ValueError(f'Variables have inconsistent lengths: {mcvarx.name}:{mcvarx.size[1]}, {mcvary.name}:{mcvary.size[1]}, {mcvarz.name}:{mcvarz.size[1]}')            

        if mcvarx.size[1] == 1:
            fig, ax = mc_plot_3d_scatter(mcvarx=mcvarx, mcvary=mcvary, mcvarz=mcvarz, cases=cases, highlight_cases=highlight_cases, ax=ax, title=title)
            
        elif mcvarx.size[1] > 1:
            fig, ax = mc_plot_3d_line(mcvarx=mcvarx, mcvary=mcvary, mcvarz=mcvarz, cases=cases, highlight_cases=highlight_cases, ax=ax, title=title)
    
    return fig, ax



def mc_plot_hist(mcvar       : MCVar, 
                 highlight_cases : Union[None, int, list[int], set[int]] = [], # No cases
                 cumulative  : bool              = False,
                 orientation : str               = 'vertical', # 'vertical' or 'horizontal'
                 rug_plot    : bool              = True,
                 ax          : Union[None, Axes] = None, 
                 title       : str               = '',
                 ):

    fig, ax = manage_axis(ax, is3d=False)

    # Histogram generation
    highlight_cases = get_cases(mcvar.ncases, highlight_cases)
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
            dist = mcvar.dist(**mcvar.distkwargs)
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
            dist = mcvar.dist(**mcvar.distkwargs)
            if cumulative:
                xdata = x - binwidth
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
    
    if cumulative:
        ylabeltext = 'Cumulative Probability'
    else:
        ylabeltext = 'Probability Density'
    
    if rug_plot:
        plot_rug_marks(ax, orientation=orientation, nums=mcvar.nums)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Highlight cases and MCVarStats
    if orientation == 'vertical':
        for i in highlight_cases:
            plt.plot([mcvar.nums[i], mcvar.nums[i]], [ylim[0], ylim[0]+(ylim[1]-ylim[0])*0.20], linestyle='-', linewidth=1, color='red')
        for mcvarstat in mcvar.mcvarstats:
            nums = get_iterable(mcvarstat.nums)
            if length(nums) == 1:
                plt.plot([nums[0],nums[0]], ylim, linestyle='-', color='blue')
            elif length(nums) == 3:
                plt.plot([nums[1],nums[1]], ylim, linestyle='-', color='blue')
            if length(nums) in (2, 3):
                ax.fill_between([nums[0],nums[-1]], [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='blue', alpha=0.2)
        plt.xlabel(mcvar.name)
        plt.ylabel(ylabeltext)
        apply_category_labels(ax, mcvarx=mcvar)
        
    elif orientation == 'horizontal':
        for i in highlight_cases:
            plt.plot([xlim[0], xlim[0]+(xlim[1]-xlim[0])*0.20], [mcvar.nums[i], mcvar.nums[i]], linestyle='-', linewidth=1, color='red')
        for mcvarstat in mcvar.mcvarstats:
            nums = get_iterable(mcvarstat.nums)
            if length(nums) == 1:
                plt.plot(xlim, [nums[0],nums[0]], linestyle='-', color='blue')
            elif length(nums) == 3:
                plt.plot(xlim, [nums[1],nums[1]], linestyle='-', color='blue')
            if length(nums) in (2,3):
                ax.fill_between(xlim, nums[0], nums[-1], color='blue', alpha=0.2)
        plt.ylabel(mcvar.name)
        plt.xlabel(ylabeltext)
        apply_category_labels(ax, mcvary=mcvar)
                
    plt.title(title)
    
    return fig, ax
        


def mc_plot_cdf(mcvar       : MCVar, 
                highlight_cases : Union[None, int, list[int], set[int]] = [], # No cases
                orientation : str               = 'vertical', # 'vertical' or 'horizontal'
                rug_plot    : bool              = True,
                ax          : Union[None, Axes] = None, 
                title       : str               = '',
                ):
    return mc_plot_hist(mcvar=mcvar, highlight_cases=highlight_cases, cumulative=True, orientation=orientation, rug_plot=rug_plot, ax=ax, title=title)



def mc_plot_2d_scatter(mcvarx   : MCVar, 
                       mcvary   : MCVar, 
                       cases           : Union[None, int, list[int], set[int]] = None,  # All cases 
                       highlight_cases : Union[None, int, list[int], set[int]] = [],    # No cases
                       rug_plot : bool               = False,
                       cov_plot : bool               = False,
                       cov_p    : Union[None, float, list[float], set[float]]  = None,
                       ax       : Union[None, Axes]  = None, 
                       title    : str                = '',
                       ):
    fig, ax = manage_axis(ax, is3d=False)

    cases = get_cases(mcvarx.ncases, cases)
    highlight_cases = get_cases(mcvarx.ncases, highlight_cases)
    reg_cases = set(cases) - set(highlight_cases)
    if reg_cases:
        plt.scatter(slice_by_index(mcvarx.nums, reg_cases), slice_by_index(mcvary.nums, reg_cases), edgecolors=None, c='k', alpha=0.4)
    if highlight_cases:
        plt.scatter(slice_by_index(mcvarx.nums, highlight_cases), slice_by_index(mcvary.nums, highlight_cases), edgecolors=None, c='r', alpha=1)

    if cov_plot:
        if cov_p is None:
            cov_p = conf_ellipsoid_sig2pct(3.0, df=2); # 3-sigma for 2D gaussian
        cov_p = get_iterable(cov_p)
        for p in cov_p:
            plot_2d_cov_ellipse(ax=ax, mcvarx=mcvarx, mcvary=mcvary, p=p)

    if rug_plot:
        all_cases = set(cases) | set(highlight_cases)
        plot_rug_marks(ax, orientation='vertical', nums=slice_by_index(mcvarx.nums, all_cases))
        plot_rug_marks(ax, orientation='horizontal', nums=slice_by_index(mcvary.nums, all_cases))

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    apply_category_labels(ax, mcvarx, mcvary)
    plt.title(title)

    return fig, ax



def mc_plot_2d_line(mcvarx : MCVar, 
                    mcvary : MCVar, 
                    cases           : Union[None, int, list[int], set[int]] = None, # All cases 
                    highlight_cases : Union[None, int, list[int], set[int]] = [],   # No cases
                    ax     : Union[None, Axes] = None, 
                    title  : str               = '',
                    ):
    fig, ax = manage_axis(ax, is3d=False)
    
    cases = get_cases(mcvarx.ncases, cases)
    highlight_cases = get_cases(mcvarx.ncases, highlight_cases)
    reg_cases = set(cases) - set(highlight_cases)
    for i in reg_cases:
        plt.plot(mcvarx.nums[i], mcvary.nums[i], linestyle='-', color='black', alpha=0.2)
    for i in highlight_cases:
        plt.plot(mcvarx.nums[i], mcvary.nums[i], linestyle='-', color='red', alpha=1)     

    for mcvarstat in mcvary.mcvarstats:
        if length(mcvarstat.nums[0]) == 1:
            plt.plot(mcvarx.nums[0], mcvarstat.nums[:], linestyle='-', color='blue')
        elif length(mcvarstat.nums[0]) == 3:
            plt.plot(mcvarx.nums[0], mcvarstat.nums[:,1], linestyle='-', color='blue')
        if length(mcvarstat.nums[0]) in (2,3):
            ax.fill_between(mcvarx.nums[0], mcvarstat.nums[:,0], mcvarstat.nums[:,-1], color='blue', alpha=0.3)
                     

    plt.xlabel(mcvarx.name)
    plt.ylabel(mcvary.name)
    apply_category_labels(ax, mcvarx, mcvary)
    plt.title(title)

    return fig, ax



def mc_plot_3d_scatter(mcvarx : MCVar, 
                       mcvary : MCVar, 
                       mcvarz : MCVar, 
                       cases           : Union[None, int, list[int], set[int]] = None, # All cases 
                       highlight_cases : Union[None, int, list[int], set[int]] = [],   # No cases
                       ax     : Union[None, Axes] = None, 
                       title  : str               = '',
                       ):
    fig, ax = manage_axis(ax, is3d=True)
    
    cases = get_cases(mcvarx.ncases, cases)
    highlight_cases = get_cases(mcvarx.ncases, highlight_cases)
    reg_cases = set(cases) - set(highlight_cases)
    if reg_cases:
        ax.scatter(slice_by_index(mcvarx.nums, reg_cases), slice_by_index(mcvary.nums, reg_cases), \
                   slice_by_index(mcvarz.nums, reg_cases), edgecolors=None, c='k', alpha=0.4)
    if highlight_cases:
        ax.scatter(slice_by_index(mcvarx.nums, highlight_cases), slice_by_index(mcvary.nums, highlight_cases), \
                   slice_by_index(mcvarz.nums, highlight_cases), edgecolors=None, c='r', alpha=1)

    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    apply_category_labels(ax, mcvarx, mcvary, mcvarz)
    plt.title(title)

    return fig, ax



def mc_plot_3d_line(mcvarx : MCVar, 
                    mcvary : MCVar, 
                    mcvarz : MCVar, 
                    cases           : Union[None, int, list[int], set[int]] = None, # All cases 
                    highlight_cases : Union[None, int, list[int], set[int]] = [],   # No cases
                    ax     : Union[None, Axes] = None, 
                    title  : str               = '',
                    ):
    fig, ax = manage_axis(ax, is3d=True)
    
    cases = get_cases(mcvarx.ncases, cases)
    highlight_cases = get_cases(mcvarx.ncases, highlight_cases)
    reg_cases = set(cases) - set(highlight_cases)
    for i in reg_cases:
        ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], linestyle='-', color='black', alpha=0.3)
    for i in highlight_cases:
        ax.plot(mcvarx.nums[i], mcvary.nums[i], mcvarz.nums[i], linestyle='-', color='red', alpha=1)
        
    ax.set_xlabel(mcvarx.name)
    ax.set_ylabel(mcvary.name)
    ax.set_zlabel(mcvarz.name)
    apply_category_labels(ax, mcvarx, mcvary, mcvarz)
    plt.title(title)

    return fig, ax



def mc_plot_cov_corr(matrix    : np.ndarray, 
                     varnames  : list[str],
                     ax        : Union[None, Axes] = None, 
                     title     : str               = '',
                     ):
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



def mc_plot_integration_convergence(mcoutvar     : MCOutVar,
                                    volume       : float,
                                    refval       : Union[int, float] = None,
                                    conf         : float             = 0.95,
                                    dimension    : int               = None,
                                    samplemethod : str               = 'random', # 'random' or 'sobol'
                                    ax           : Union[None, Axes] = None, 
                                    title        : str               = '',
                                    ):
    fig, ax = manage_axis(ax, is3d=False)

    if not refval is None:
        ax.axhline(refval, color='k')

    cummean = volume*np.cumsum(mcoutvar.nums)/np.arange(1, mcoutvar.ncases+1)
    err = integration_error(nums=mcoutvar.nums, volume=volume, conf=conf, dimension=dimension, samplemethod=samplemethod, runningError=True)
    ax.plot(cummean,'r')
    ax.plot(cummean+err, 'b')
    ax.plot(cummean-err, 'b')
    
    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'Convergence of {mcoutvar.name} Integral')
    ax.set_title(title)

    return fig, ax



def mc_plot_integration_error(mcoutvar     : MCOutVar,
                              volume       : float,
                              refval       : float,
                              conf         : float             = 0.95,
                              dimension    : int               = None,
                              samplemethod : str               = 'random', # 'random' or 'sobol'
                              ax           : Union[None, Axes] = None, 
                              title        : str               = '',
                              ):
    fig, ax = manage_axis(ax, is3d=False)

    cummean = volume*np.cumsum(mcoutvar.nums)/np.arange(1, mcoutvar.ncases+1)
    err = integration_error(nums=mcoutvar.nums, volume=volume, conf=conf, dimension=dimension, samplemethod=samplemethod, runningError=True)
    ax.loglog(err, 'b')
    ax.plot(np.abs(cummean - refval), 'r')
    
    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'{mcoutvar.name} {round(conf*100, 2)}% Confidence Error')
    plt.title(title)

    return fig, ax



def manage_axis(ax   : Union[None, Axes], 
                is3d : bool = False,
                ):
    if not ax is None:
        fig = ax.figure
    else:
        if is3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(1, 1)
    plt.sca(ax)
    return fig, ax



def apply_category_labels(ax : Axes, 
                          mcvarx : MCVar = None, 
                          mcvary : MCVar = None, 
                          mcvarz : MCVar = None,
                          ):
    # Wrapped in try statements in case some categories aren't printable
    if mcvarx is not None and mcvarx.nummap is not None:
        try:
            ax.set_xticks(list(mcvarx.nummap.keys()))
            ax.set_xticklabels(list(mcvarx.nummap.values()))
        except Exception:
            pass
    if mcvary is not None and mcvary.nummap is not None:
        try:
            ax.set_yticks(list(mcvary.nummap.keys()))
            ax.set_yticklabels(list(mcvary.nummap.values()))
        except Exception:
            pass
    if mcvarz is not None and mcvarz.nummap is not None:
        try:
            ax.set_zticks(list(mcvarz.nummap.keys()))
            ax.set_zticklabels(list(mcvarz.nummap.values()))
        except Exception:
            pass



def get_hist_lim(orientation : str,  # 'vertical' or 'horizontal' , 
                 ax          : Axes, 
                 ):
    if orientation == 'vertical':
        lim = ax.get_xlim()
    elif orientation == 'horizontal':
        lim = ax.get_ylim()
    return lim



def plot_rug_marks(ax          : Union[None, Axes], 
                   orientation : str, # 'vertical' or 'horizontal' 
                   nums        : Union[list[int], list[float]]
                   ):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if orientation == 'vertical':
        for num in nums:
            plt.plot([num,num], [ylim[0], ylim[0] + 0.02*(ylim[1]-ylim[0])], color='black', linewidth=0.5, alpha=0.5)
    elif orientation == 'horizontal':
        for num in nums:
            plt.plot([xlim[0], xlim[0] + 0.02*(xlim[1]-xlim[0])], [num,num], color='black', linewidth=0.5, alpha=0.5)



def plot_2d_cov_ellipse(ax     : Union[None, Axes], 
                        mcvarx : MCVar, 
                        mcvary : MCVar, 
                        p      : float,
                        ):
    # See https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    allnums = [mcvarx.nums, mcvary.nums]
    center = [np.mean(mcvarx.nums), np.mean(mcvary.nums)]
    
    covs = np.cov(np.array(allnums))
    eigvals, eigvecs = np.linalg.eigh(covs) # Use eigh over eig since covs is guaranteed symmetric
    inds = (-eigvals).argsort() # sort from largest to smallest
    eigvals = eigvals[inds]
    eigvecs = eigvecs[inds]
    angle = np.arctan2(eigvecs[0][1], eigvecs[0][0])*180/np.pi
    
    scalefactor = chi2.ppf(p, df=2)
    ellipse_axis_radii = np.sqrt(scalefactor*eigvals)
    
    ellipse = Ellipse(center, 2*ellipse_axis_radii[0], 2*ellipse_axis_radii[1], angle=angle, fill=False, edgecolor='k')
    ax.add_patch(ellipse)

    # For now plot both eigenaxes
    plt.axline(xy1=(center[0], center[1]), slope=eigvecs[0][1]/eigvecs[0][0], color='k')
    plt.axline(xy1=(center[0], center[1]), slope=eigvecs[1][1]/eigvecs[1][0], color='k')



def get_cases(ncases : int, 
              cases  : Union[None, int, list[int], set[int]],
              ):
    if cases is None:
        cases = list(range(ncases))
    cases = get_iterable(cases)
    return cases
        
