# mc_plot.py
from __future__ import annotations

import numpy as np
from scipy.stats import rv_continuous, rv_discrete, chi2, mode
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from monaco.mc_var import InVar, OutVar
from monaco.helper_functions import get_list, slice_by_index, length, empty_list
from monaco.gaussian_statistics import conf_ellipsoid_sig2pct
from monaco.integration_statistics import integration_error
from monaco.mc_enums import SampleMethod, PlotOrientation
from copy import copy
from typing import Optional, Iterable


# If cases or highlight_cases are None, will plot all. Set to [] to plot none.
def plot(varx   : InVar | OutVar,
         vary   : InVar | OutVar = None,
         varz   : InVar | OutVar = None,
         cases           : None | int | Iterable[int] = None,
         highlight_cases : None | int | Iterable[int] = empty_list(),
         rug_plot : bool           = False,
         cov_plot : bool           = False,
         cov_p    : None | float | Iterable[float] = None,
         ax       : Optional[Axes] = None,
         title    : str            = '',
         ) -> tuple[Figure, Axes]:
    """
    Umbrella function to make single plots of a single Monte-Carlo variable or
    pairs or triplets of variables.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar, default: None
        The y variable to plot.
    varz : monaco.mc_var.InVar | monaco.mc_var.OutVar, default: None
        The z variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool, default: True
        Whether to plot rug marks.
    cov_plot : bool, default: False
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : None | float | Iterable[float], default: None
        The gaussian percentiles for the covariance plot.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    # Split larger vars
    if vary is None and varz is None:
        if varx.maxdim not in (0, 1, 2):
            raise ValueError(f'Invalid variable dimension: {varx.name} ({varx.maxdim})')
        elif varx.maxdim == 2 and isinstance(varx, OutVar):
            varx_split = varx.split()  # split only defined for OutVar
            origname = varx.name
            varx = varx_split[origname + ' [0]']
            vary = varx_split[origname + ' [1]']
            if len(varx_split) == 3:
                varz = varx_split[origname + ' [2]']

    elif vary is not None and varz is None:
        if varx.maxdim == 1 and vary.maxdim == 0 and isinstance(varx, OutVar):
            varx_split = varx.split()  # split only defined for OutVar
            origname = varx.name
            varz = vary
            varx = varx_split[origname + ' [0]']
            vary = varx_split[origname + ' [1]']
        elif varx.maxdim == 0 and vary.maxdim == 1 and isinstance(vary, OutVar):
            vary_split = vary.split()  # split only defined for OutVar
            origname = vary.name
            vary = vary_split[origname + ' [0]']
            varz = vary_split[origname + ' [1]']

    # Single Variable Plots
    if vary is None and varz is None:
        if varx.maxdim == 0:
            fig, ax = plot_hist(var=varx, cases=cases, highlight_cases=highlight_cases,
                                rug_plot=rug_plot, ax=ax, title=title)
        else:
            vary = copy(varx)
            varx = copy(varx)   # don't overwrite the underlying object
            varx.name = 'Simulation Steps'
            steps = np.arange(max(len(num) for num in vary.nums))
            varx.nums = [steps for _ in range(varx.ncases)]
            varx.nummap = None
            fig, ax = plot_2d_line(varx=varx, vary=vary,
                                   highlight_cases=highlight_cases,
                                   ax=ax, title=title)

    # Two Variable Plots
    elif vary is not None and varz is None:
        if varx.maxdim == 0 and vary.maxdim == 0:
            fig, ax = plot_2d_scatter(varx=varx, vary=vary,
                                      cases=cases, highlight_cases=highlight_cases,
                                      rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p,
                                      ax=ax, title=title)

        elif varx.maxdim == 1 and vary.maxdim == 1:
            fig, ax = plot_2d_line(varx=varx, vary=vary,
                                      cases=cases, highlight_cases=highlight_cases,
                                      ax=ax, title=title)
        else:
            raise ValueError( 'Variables have inconsistent dimensions: ' +
                             f'{varx.name}:{varx.maxdim}, ' +
                             f'{vary.name}:{vary.maxdim}')

    # Three Variable Plots
    else:
        if varx.maxdim == 0 and vary.maxdim == 0 and varz.maxdim == 0:
            fig, ax = plot_3d_scatter(varx=varx, vary=vary, varz=varz,
                                      cases=cases, highlight_cases=highlight_cases,
                                      ax=ax, title=title)

        elif varx.maxdim == 1 and vary.maxdim == 1 and varz.maxdim == 1:
            fig, ax = plot_3d_line(varx=varx, vary=vary, varz=varz,
                                   cases=cases, highlight_cases=highlight_cases,
                                   ax=ax, title=title)

        else:
            raise ValueError( 'Variables have inconsistent dimensions: ' +
                             f'{varx.name}:{varx.maxdim}, ' +
                             f'{vary.name}:{vary.maxdim}, ' +
                             f'{varz.name}:{varz.maxdim}')

    return fig, ax



def plot_hist(var         : InVar | OutVar,
              cases           : None | int | Iterable[int] = None,
              highlight_cases : None | int | Iterable[int] = empty_list(),
              cumulative  : bool            = False,
              orientation : PlotOrientation = PlotOrientation.VERTICAL,
              rug_plot    : bool            = True,
              ax          : Optional[Axes]  = None,
              title       : str             = '',
              ) -> tuple[Figure, Axes]:
    """
    Plot a histogram of a single variable.

    Parameters
    ----------
    var : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    cumulative : bool, default: False
        Whether to plot the histograms as cumulative distribution functions.
    orientation : monaco.mc_enums.PlotOrientation, default: 'vertical'
        The orientation of the histogram. Either 'vertical' or 'horizontal'.
    rug_plot : bool, default: True
        Whether to plot rug marks.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=False)

    # Histogram generation
    cases_list = get_cases(var.ncases, cases)
    highlight_cases_list = get_cases(var.ncases, highlight_cases)
    nums = slice_by_index(var.nums, cases_list)
    counts, bins = np.histogram(nums, bins='auto')
    binwidth = mode(np.diff(bins))[0]
    bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
    counts, bins = np.histogram(nums, bins=bins)

    if isinstance(var, InVar):
        # Continuous distribution
        if isinstance(var.dist, rv_continuous):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=True,
                     cumulative=cumulative, orientation=orientation, histtype='bar',
                     facecolor='k', alpha=0.5)
            lim = get_hist_lim(ax, orientation)
            x = np.arange(lim[0], lim[1], (lim[1] - lim[0])/100)
            dist = var.dist(**var.distkwargs)
            if cumulative:
                ydata = dist.cdf(x)
            else:
                ydata = dist.pdf(x)
            if orientation == PlotOrientation.VERTICAL:
                plt.plot(x, ydata, color='k', alpha=0.9)
            elif orientation == PlotOrientation.HORIZONTAL:
                plt.plot(ydata, x, color='k', alpha=0.9)

        # Discrete distribution
        elif isinstance(var.dist, rv_discrete):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False,
                     orientation=orientation, cumulative=cumulative, histtype='bar',
                     facecolor='k', alpha=0.5)
            lim = get_hist_lim(ax, orientation)
            x = np.concatenate(([lim[0]], bins, [lim[1]]))
            dist = var.dist(**var.distkwargs)
            if cumulative:
                xdata = x - binwidth
                ydata = dist.cdf(x)
            else:
                xdata = x[1:]
                ydata = np.diff(dist.cdf(x))  # manual pdf
            if orientation == PlotOrientation.VERTICAL:
                plt.step(xdata, ydata, color='k', alpha=0.9, where='post')
            elif orientation == PlotOrientation.HORIZONTAL:
                plt.step(ydata, xdata, color='k', alpha=0.9, where='post')

    elif isinstance(var, OutVar):
        plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False,
                 orientation=orientation, cumulative=cumulative, histtype='bar',
                 facecolor='k', alpha=0.5)

    if cumulative:
        ylabeltext = 'Cumulative Probability'
    else:
        ylabeltext = 'Probability Density'

    if rug_plot:
        plot_rug_marks(ax, orientation=orientation, nums=np.array(var.nums))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Highlight cases and MCVarStats
    if orientation == PlotOrientation.VERTICAL:
        for i in highlight_cases_list:
            plt.plot([var.nums[i], var.nums[i]],
                     [ylim[0], ylim[0] + (ylim[1] - ylim[0])*0.20],
                     linestyle='-', linewidth=1, color='red')
        for varstat in var.varstats:
            nums = get_list(varstat.nums)
            if length(nums) == 1:
                plt.plot([nums[0], nums[0]], ylim, linestyle='-', color='blue')
            elif length(nums) == 3:
                plt.plot([nums[1], nums[1]], ylim, linestyle='-', color='blue')
            if length(nums) in (2, 3):
                ax.fill_between([nums[0], nums[-1]],
                                [ylim[0], ylim[0]],
                                [ylim[1], ylim[1]],
                                color='blue', alpha=0.2)
        plt.xlabel(var.name)
        plt.ylabel(ylabeltext)
        apply_category_labels(ax, varx=var)

    elif orientation == PlotOrientation.HORIZONTAL:
        for i in highlight_cases_list:
            plt.plot([xlim[0], xlim[0] + (xlim[1] - xlim[0])*0.20],
                     [var.nums[i], var.nums[i]],
                     linestyle='-', linewidth=1, color='red')
        for varstat in var.varstats:
            nums = get_list(varstat.nums)
            if length(nums) == 1:
                plt.plot(xlim, [nums[0], nums[0]], linestyle='-', color='blue')
            elif length(nums) == 3:
                plt.plot(xlim, [nums[1], nums[1]], linestyle='-', color='blue')
            if length(nums) in (2, 3):
                ax.fill_between(xlim, nums[0], nums[-1], color='blue', alpha=0.2)
        plt.ylabel(var.name)
        plt.xlabel(ylabeltext)
        apply_category_labels(ax, vary=var)

    plt.title(title)

    return fig, ax



def plot_cdf(var       : InVar | OutVar,
             cases           : None | int | Iterable[int] = None,
             highlight_cases : None | int | Iterable[int] = empty_list(),
             orientation : PlotOrientation = PlotOrientation.VERTICAL,
             rug_plot    : bool            = True,
             ax          : Optional[Axes]  = None,
             title       : str             = '',
             ) -> tuple[Figure, Axes]:
    """
    Plot a cumulative distribution of a single variable.

    Parameters
    ----------
    var : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    orientation : monaco.mc_enums.PlotOrientation, default: 'vertical'
        The orientation of the histogram. Either 'vertical' or 'horizontal'.
    rug_plot : bool, default: True
        Whether to plot rug marks.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    return plot_hist(var=var, cases=cases, highlight_cases=highlight_cases, cumulative=True,
                     orientation=orientation, rug_plot=rug_plot, ax=ax, title=title)



def plot_2d_scatter(varx   : InVar | OutVar,
                    vary   : InVar | OutVar,
                    cases           : None | int | Iterable[int] = None,
                    highlight_cases : None | int | Iterable[int] = empty_list(),
                    rug_plot : bool           = False,
                    cov_plot : bool           = False,
                    cov_p    : None | float | Iterable[float] = None,
                    ax       : Optional[Axes] = None,
                    title    : str            = '',
                    ) -> tuple[Figure, Axes]:
    """
    Plot a scatter plot of two variables.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are highlighted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool, default: True
        Whether to plot rug marks.
    cov_plot : bool, default: False
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : None | float | Iterable[float], default: None
        The gaussian percentiles for the covariance plot.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=False)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    if reg_cases:
        plt.scatter(slice_by_index(varx.nums, reg_cases),
                    slice_by_index(vary.nums, reg_cases),
                    edgecolors=None, c='k', alpha=0.4)
    if highlight_cases_list:
        plt.scatter(slice_by_index(varx.nums, highlight_cases_list),
                    slice_by_index(vary.nums, highlight_cases_list),
                    edgecolors=None, c='r', alpha=1)

    if cov_plot:
        if cov_p is None:
            cov_p = conf_ellipsoid_sig2pct(3.0, df=2)  # 3-sigma for 2D gaussian
        cov_p_list = get_list(cov_p)
        for p in cov_p_list:
            plot_2d_cov_ellipse(ax=ax, varx=varx, vary=vary, p=p)

    if rug_plot:
        all_cases = set(cases_list) | set(highlight_cases_list)
        plot_rug_marks(ax, orientation=PlotOrientation.VERTICAL,
                       nums=slice_by_index(varx.nums, all_cases))
        plot_rug_marks(ax, orientation=PlotOrientation.HORIZONTAL,
                       nums=slice_by_index(vary.nums, all_cases))

    plt.xlabel(varx.name)
    plt.ylabel(vary.name)
    apply_category_labels(ax, varx, vary)
    plt.title(title)

    return fig, ax



def plot_2d_line(varx : InVar | OutVar,
                 vary : InVar | OutVar,
                 cases           : None | int | Iterable[int] = None,
                 highlight_cases : None | int | Iterable[int] = empty_list(),
                 ax     : Optional[Axes] = None,
                 title  : str            = '',
                 ) -> tuple[Figure, Axes]:
    """
    Plot an ensemble of 2D lines for two nonscalar variables.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are highlighted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=False)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    for i in reg_cases:
        plt.plot(varx.nums[i], vary.nums[i], linestyle='-', color='black', alpha=0.2)
    for i in highlight_cases_list:
        plt.plot(varx.nums[i], vary.nums[i], linestyle='-', color='red', alpha=1)

    for varstat in vary.varstats:
        if length(varstat.nums[0]) == 1:
            plt.plot(varx.nums[0], varstat.nums[:], linestyle='-', color='blue')
        elif length(varstat.nums[0]) == 3:
            plt.plot(varx.nums[0], varstat.nums[:, 1], linestyle='-', color='blue')
        if length(varstat.nums[0]) in (2, 3):
            ax.fill_between(varx.nums[0], varstat.nums[:, 0], varstat.nums[:, -1],
                            color='blue', alpha=0.3)

    plt.xlabel(varx.name)
    plt.ylabel(vary.name)
    apply_category_labels(ax, varx, vary)
    plt.title(title)

    return fig, ax



def plot_3d_scatter(varx : InVar | OutVar,
                    vary : InVar | OutVar,
                    varz : InVar | OutVar,
                    cases           : None | int | Iterable[int] = None,
                    highlight_cases : None | int | Iterable[int] = empty_list(),
                    ax     : Optional[Axes] = None,
                    title  : str            = '',
                    ) -> tuple[Figure, Axes]:
    """
    Plot a scatter plot of three variables in 3D space.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable to plot.
    varz : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The z variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are highlighted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=True)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    if reg_cases:
        ax.scatter(slice_by_index(varx.nums, reg_cases),
                   slice_by_index(vary.nums, reg_cases),
                   slice_by_index(varz.nums, reg_cases),
                   edgecolors=None, c='k', alpha=0.4)
    if highlight_cases_list:
        ax.scatter(slice_by_index(varx.nums, highlight_cases_list),
                   slice_by_index(vary.nums, highlight_cases_list),
                   slice_by_index(varz.nums, highlight_cases_list),
                   edgecolors=None, c='r', alpha=1)

    ax.set_xlabel(varx.name)
    ax.set_ylabel(vary.name)
    ax.set_zlabel(varz.name)
    apply_category_labels(ax, varx, vary, varz)
    plt.title(title)

    return fig, ax



def plot_3d_line(varx : InVar | OutVar,
                 vary : InVar | OutVar,
                 varz : InVar | OutVar,
                 cases           : None | int | Iterable[int] = None,
                 highlight_cases : None | int | Iterable[int] = empty_list(),
                 ax     : Optional[Axes] = None,
                 title  : str            = '',
                 ) -> tuple[Figure, Axes]:
    """
    Plot an ensemble of 3D lines for three nonscalar variables.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable to plot.
    varz : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The z variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are highlighted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=True)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    for i in reg_cases:
        ax.plot(varx.nums[i], vary.nums[i], varz.nums[i],
                linestyle='-', color='black', alpha=0.3)
    for i in highlight_cases_list:
        ax.plot(varx.nums[i], vary.nums[i], varz.nums[i],
                linestyle='-', color='red', alpha=1)

    ax.set_xlabel(varx.name)
    ax.set_ylabel(vary.name)
    ax.set_zlabel(varz.name)
    apply_category_labels(ax, varx, vary, varz)
    plt.title(title)

    return fig, ax



def plot_cov_corr(matrix    : np.ndarray,
                  varnames  : list[str],
                  ax        : Optional[Axes] = None,
                  title     : str            = '',
                  ) -> tuple[Figure, Axes]:
    """
    Plot either a covariance or correlation matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        The covariance or correlation matrix.
    varnames : list[str]
        A list of the variable names.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=False)
    # for a correlation matrix scale will always be 1 from diagonal
    scale = np.nanmax(np.abs(matrix))
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
    kw = {'horizontalalignment': 'center', 'verticalalignment': 'center'}

    texts = []
    for i in range(n):
        for j in range(n):
            kw.update(color=textcolors[int(abs(matrix[i, j]/scale) > threshold)])
            text = im.axes.text(j, i, f'{matrix[i, j]:.2f}', **kw)
            texts.append(text)
    plt.title(title)

    return fig, ax



def plot_integration_convergence(outvar     : OutVar,
                                 dimension    : int,
                                 volume       : float,
                                 refval       : float          = None,
                                 conf         : float          = 0.95,
                                 samplemethod : SampleMethod   = SampleMethod.RANDOM,
                                 ax           : Optional[Axes] = None,
                                 title        : str            = '',
                                 ) -> tuple[Figure, Axes]:
    """
    For a Monte-Carlo integration, plot the running integration estimate along
    with error bars for a given confidence level.

    Parameters
    ----------
    outvar : monaco.mc_var.OutVar
        The variable representing the integration estimate.
    dimension : int
        The number of dimensions over which the integration was performed.
    volume : float
        The total integration volume.
    refval : float, default: None
        If known a-priori, the reference value for the integration.
    conf : float, default: 0.95
        The confidence level for the error estimate. 0.95 corresponds to 95%.
    samplemethod : monaco.mc_enums.SampleMethod
        The sample method used for integration. Either 'random' or 'sobol'.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=False)

    if refval is not None:
        ax.axhline(refval, color='k')

    cummean = volume*np.cumsum(outvar.nums)/np.arange(1, outvar.ncases+1)
    err = integration_error(nums=np.array(outvar.nums), dimension=dimension, volume=volume,
                            conf=conf, samplemethod=samplemethod, runningerror=True)
    ax.plot(cummean, 'r')
    ax.plot(cummean + err, 'b')
    ax.plot(cummean - err, 'b')

    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'Convergence of {outvar.name} Integral')
    plt.title(title)

    return fig, ax



def plot_integration_error(outvar     : OutVar,
                           dimension    : int,
                           volume       : float,
                           refval       : float,
                           conf         : float          = 0.95,
                           samplemethod : SampleMethod   = SampleMethod.RANDOM,
                           ax           : Optional[Axes] = None,
                           title        : str            = '',
                           ) -> tuple[Figure, Axes]:
    """
    For a Monte-Carlo integration where the reference value is known, plot the
    running integration error along with error bounds for a given confidence
    level.

    Parameters
    ----------
    outvar : monaco.mc_var.OutVar
        The variable representing the integration estimate.
    dimension : int
        The number of dimensions over which the integration was performed.
    volume : float
        The total integration volume.
    refval : float
        The reference value for the integration.
    conf : float, default: 0.95
        The confidence level for the error estimate. 0.95 corresponds to 95%.
    samplemethod : monaco.mc_enums.SampleMethod
        The sample method used for integration. Either 'random' or 'sobol'.
    ax : matplotlib.axes.Axes, default: None
        The axes handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    fig, ax = manage_axis(ax, is3d=False)

    cummean = volume*np.cumsum(outvar.nums)/np.arange(1, outvar.ncases+1)
    err = integration_error(nums=np.array(outvar.nums), dimension=dimension, volume=volume,
                            conf=conf, samplemethod=samplemethod, runningerror=True)
    ax.loglog(err, 'b')
    ax.plot(np.abs(cummean - refval), 'r')

    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'{outvar.name} {round(conf*100, 2)}% Confidence Error')
    plt.title(title)

    return fig, ax



def manage_axis(ax   : Optional[Axes],
                is3d : bool = False,
                ) -> tuple[Figure, Axes]:
    """
    Set the target axis, either by making a new figure or setting active an
    existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axis. If None, a new figure is created.
    is3d : bool, default: False
        If creating a new figure, whether the plot is a 3D plot.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        fig is the figure handle for the plot.
        ax is the axes handle for the plot.
    """
    if ax is not None:
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
                          varx : InVar | OutVar = None,
                          vary : InVar | OutVar = None,
                          varz : InVar | OutVar = None,
                          ) -> None:
    """
    For nonnumeric Monte-Carlo variables, use the `nummap` to label the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axis.
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar, default: None
        The x variable.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar, default: None
        The y variable.
    varz : monaco.mc_var.InVar | monaco.mc_var.OutVar, default: None
        The z variable.
    """
    # Wrapped in try statements in case some categories aren't printable
    if varx is not None and varx.nummap is not None:
        try:
            ax.set_xticks(list(varx.nummap.keys()))
            ax.set_xticklabels(list(varx.nummap.values()))
        except Exception:
            pass
    if vary is not None and vary.nummap is not None:
        try:
            ax.set_yticks(list(vary.nummap.keys()))
            ax.set_yticklabels(list(vary.nummap.values()))
        except Exception:
            pass
    if varz is not None and varz.nummap is not None:
        try:
            ax.set_zticks(list(varz.nummap.keys()))
            ax.set_zticklabels(list(varz.nummap.values()))
        except Exception:
            pass



def get_hist_lim(ax          : Axes,
                 orientation : PlotOrientation,
                 ) -> tuple[float, float]:
    """
    Get the axis limits for a histogram.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axis.
    orientation : PlotOrientation
        The orientation of the histogram plot, either 'vertical' or
        'horizontal'.

    Returns
    -------
    lim : (float, float)
        Returns the (low, high) limits of the axis.
    """
    if orientation == PlotOrientation.VERTICAL:
        lim = ax.get_xlim()
    elif orientation == PlotOrientation.HORIZONTAL:
        lim = ax.get_ylim()
    return lim



def plot_rug_marks(ax          : Axes,
                   orientation : PlotOrientation,
                   nums        : Iterable[float]
                   ) -> None:
    """
    Plot rug marks for a histogram or scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axis.
    orientation : PlotOrientation
        The orientation of the plot, either 'vertical' or 'horizontal'.
    nums : Iterable[float]
        The numbers to plot the rug marks at.
    """
    if ax is None:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if orientation == PlotOrientation.VERTICAL:
        for num in nums:
            plt.plot([num, num], [ylim[0], ylim[0] + 0.02*(ylim[1] - ylim[0])],
                     color='black', linewidth=0.5, alpha=0.5)
    elif orientation == PlotOrientation.HORIZONTAL:
        for num in nums:
            plt.plot([xlim[0], xlim[0] + 0.02*(xlim[1] - xlim[0])], [num, num],
                     color='black', linewidth=0.5, alpha=0.5)



def plot_2d_cov_ellipse(ax     : Axes,
                        varx : InVar | OutVar,
                        vary : InVar | OutVar,
                        p      : float,
                        ) -> None:
    """
    Add a covariance ellipse to a 2D scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axis.
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable.
    p : float
        Coviariance percentile, assuming a gaussian distribution.
    """
    if ax is None:
        return

    # See https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    allnums = [np.array(varx.nums), np.array(vary.nums)]
    center = [np.mean(varx.nums), np.mean(vary.nums)]

    covs = np.cov(np.array(allnums))
    eigvals, eigvecs = np.linalg.eigh(covs)  # Use eigh over eig since covs is guaranteed symmetric
    inds = (-eigvals).argsort()  # sort from largest to smallest
    eigvals = eigvals[inds]
    eigvecs = eigvecs[inds]
    angle = np.arctan2(eigvecs[0][1], eigvecs[0][0])*180/np.pi

    scalefactor = chi2.ppf(p, df=2)
    ellipse_axis_radii = np.sqrt(scalefactor*eigvals)

    ellipse = Ellipse(center, 2*ellipse_axis_radii[0], 2*ellipse_axis_radii[1], angle=angle,
                      fill=False, edgecolor='k')
    ax.add_patch(ellipse)

    # For now plot both eigenaxes
    plt.axline(xy1=(center[0], center[1]), slope=eigvecs[0][1]/eigvecs[0][0], color='k')
    plt.axline(xy1=(center[0], center[1]), slope=eigvecs[1][1]/eigvecs[1][0], color='k')



def get_cases(ncases : int,
              cases  : None | int | Iterable[int],
              ) -> list[int]:
    """
    Parse the `cases` input for plotting functions. If None, return a list of
    all the cases. Otherwise, return a list of all the specified cases.

    Parameters
    ----------
    ncases : int
        The total number of cases.
    cases : None | int | Iterable[int]
        The cases to downselect to.

    Returns
    -------
    cases_list : list[int]
        The cases.
    """
    if cases is None:
        cases = list(range(ncases))
    cases_list = get_list(cases)
    return cases_list
