# mc_plot.py
from __future__ import annotations

# Somewhat hacky type checking to avoid circular imports:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from monaco.mc_var import InVar, OutVar
import monaco.mc_var

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import rv_continuous, rv_discrete, chi2, mode
from copy import copy, deepcopy
from typing import Optional, Iterable
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from monaco.helper_functions import get_list, slice_by_index, length, empty_list
from monaco.gaussian_statistics import conf_ellipsoid_sig2pct
from monaco.integration_statistics import integration_error
from monaco.mc_enums import SampleMethod, PlotOrientation, InVarSpace, Sensitivities


# If cases or highlight_cases are None, will plot all. Set to [] to plot none.
def plot(varx   : InVar | OutVar,
         vary   : InVar | OutVar = None,
         varz   : InVar | OutVar = None,
         cases           : None | int | Iterable[int] = None,
         highlight_cases : None | int | Iterable[int] = empty_list(),
         rug_plot    : bool           = False,
         cov_plot    : bool           = False,
         cov_p       : None | float | Iterable[float] = None,
         invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
         ax          : Optional[Axes] = None,
         title       : str            = '',
         plotkwargs  : dict           = dict(),
         ) -> tuple[Figure, Axes]:
    """
    Umbrella function to make single plots of a single Monte Carlo variable or
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    # Split single vars that are 2D or 3D vectors
    if vary is None and varz is None:
        if varx.maxdim not in (0, 1, 2):
            raise ValueError(f'Invalid variable dimension: {varx.name}: {varx.maxdim}')

        elif varx.maxdim == 2 and isinstance(varx, monaco.mc_var.OutVar):
            varx_split = varx.split()  # split only defined for OutVar
            origname = varx.name
            varx = varx_split[origname + ' [0]']
            vary = varx_split[origname + ' [1]']
            if len(varx_split) == 3:
                varz = varx_split[origname + ' [2]']
            elif len(varx_split) > 3:
                raise ValueError('Can only split a single variable into 3 vectors:' +
                                 f'{origname}: {len(varx_split)}')

    # Split one of two vars that is a 2D vector
    elif vary is not None and varz is None:
        if varx.maxdim == 2 and vary.maxdim == 0 and isinstance(varx, monaco.mc_var.OutVar):
            varx_split = varx.split()  # split only defined for OutVar
            origname = varx.name
            varz = vary
            varx = varx_split[origname + ' [0]']
            vary = varx_split[origname + ' [1]']
            if len(varx_split) > 2:
                raise ValueError('Can only split one of two variables into 2 vectors:' +
                                 f'{origname}: {len(varx_split)}')

        elif varx.maxdim == 0 and vary.maxdim == 2 and isinstance(vary, monaco.mc_var.OutVar):
            vary_split = vary.split()  # split only defined for OutVar
            origname = vary.name
            vary = vary_split[origname + ' [0]']
            varz = vary_split[origname + ' [1]']
            if len(vary_split) > 2:
                raise ValueError('Can only split one of two variables into 2 vectors:' +
                                 f'{origname}: {len(vary_split)}')

    # Single Variable Plots
    if vary is None and varz is None:
        if varx.maxdim == 0:
            fig, ax = plot_hist(var=varx, cases=cases, highlight_cases=highlight_cases,
                                rug_plot=rug_plot, invar_space=invar_space, ax=ax, title=title,
                                plotkwargs=plotkwargs)
        else:
            var1 = copy(varx)
            var0 = get_var_steps(varx)
            fig, ax = plot_2d_line(varx=var0, vary=var1,
                                   highlight_cases=highlight_cases,
                                   invar_space=invar_space,
                                   ax=ax, title=title, plotkwargs=plotkwargs)

    # Two Variable Plots
    elif vary is not None and varz is None:
        if varx.maxdim == 0 and vary.maxdim == 0:
            fig, ax = plot_2d_scatter(varx=varx, vary=vary,
                                      cases=cases, highlight_cases=highlight_cases,
                                      rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p,
                                      invar_space=invar_space,
                                      ax=ax, title=title, plotkwargs=plotkwargs)

        elif varx.maxdim == 1 and vary.maxdim == 1:
            fig, ax = plot_2d_line(varx=varx, vary=vary,
                                   cases=cases, highlight_cases=highlight_cases,
                                   invar_space=invar_space,
                                   ax=ax, title=title, plotkwargs=plotkwargs)

        elif varx.maxdim in (0, 1) and vary.maxdim in (0, 1):
            var2 = copy(vary)
            if varx.maxdim == 1:
                var1 = varx
                var0 = get_var_steps(varx)
            elif vary.maxdim == 1:
                var1 = get_var_steps(vary)
                var0 = varx
            fig, ax = plot_2p5d_line(varx=var0, vary=var1, varz=var2,
                                     cases=cases, highlight_cases=highlight_cases,
                                     invar_space=invar_space,
                                     ax=ax, title=title, plotkwargs=plotkwargs)

        else:
            raise ValueError( 'Invalid variable dimension: ' +
                             f'{varx.name}: {varx.maxdim}, ' +
                             f'{vary.name}: {vary.maxdim}')

    # Three Variable Plots
    else:
        if varx.maxdim == 0 and vary.maxdim == 0 and varz.maxdim == 0:
            fig, ax = plot_3d_scatter(varx=varx, vary=vary, varz=varz,
                                      cases=cases, highlight_cases=highlight_cases,
                                      invar_space=invar_space,
                                      ax=ax, title=title, plotkwargs=plotkwargs)

        elif varx.maxdim == 1 and vary.maxdim == 1 and varz.maxdim == 1:
            fig, ax = plot_3d_line(varx=varx, vary=vary, varz=varz,
                                   cases=cases, highlight_cases=highlight_cases,
                                   invar_space=invar_space,
                                   ax=ax, title=title, plotkwargs=plotkwargs)

        elif varx.maxdim in (0, 1) and vary.maxdim in (0, 1) and varz.maxdim in (0, 1):
            fig, ax = plot_2p5d_line(varx=varx, vary=vary, varz=varz,
                                     cases=cases, highlight_cases=highlight_cases,
                                     invar_space=invar_space,
                                     ax=ax, title=title, plotkwargs=plotkwargs)

        else:
            raise ValueError( 'Invalid variable dimension: ' +
                             f'{varx.name}: {varx.maxdim}, ' +
                             f'{vary.name}: {vary.maxdim}, ' +
                             f'{varz.name}: {varz.maxdim}')

    return fig, ax



def plot_hist(var         : InVar | OutVar,
              cases           : None | int | Iterable[int] = None,
              highlight_cases : None | int | Iterable[int] = empty_list(),
              cumulative  : bool            = False,
              orientation : PlotOrientation = PlotOrientation.VERTICAL,
              rug_plot    : bool            = True,
              invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
              ax          : Optional[Axes]  = None,
              title       : str             = '',
              plotkwargs  : dict            = dict(),
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    invar_space = manage_invar_space(invar_space=invar_space, nvars=1)

    bins : str | np.ndarray = 'auto'
    if 'bins' in plotkwargs:
        bins = plotkwargs['bins']
        del plotkwargs['bins']

    # Histogram generation
    cases_list = get_cases(var.ncases, cases)
    highlight_cases_list = get_cases(var.ncases, highlight_cases)
    points = get_plot_points(var, invar_space[0])
    nums = slice_by_index(points, cases_list)
    counts, bins = np.histogram(nums, bins=bins)
    binwidth = mode(np.diff(bins), keepdims=False)[0]
    bins = np.append(bins - binwidth/2, bins[-1] + binwidth/2)
    counts, bins = np.histogram(nums, bins=bins)

    if isinstance(var, monaco.mc_var.InVar):
        # Loaded from file
        if var.dist is None:
            plt.hist(bins[:-1], bins=bins, weights=counts/sum(counts), density=True,
                     cumulative=cumulative, orientation=orientation, histtype='bar',
                     facecolor='k', alpha=0.5, **plotkwargs)

        # Continuous distribution
        if isinstance(var.dist, rv_continuous):
            plt.hist(bins[:-1], bins=bins, weights=counts/sum(counts), density=True,
                     cumulative=cumulative, orientation=orientation, histtype='bar',
                     facecolor='k', alpha=0.5, **plotkwargs)
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
            plt.hist(bins[:-1], bins=bins, weights=counts/sum(counts), density=False,
                     orientation=orientation, cumulative=cumulative, histtype='bar',
                     facecolor='k', alpha=0.5, **plotkwargs)
            lim = get_hist_lim(ax, orientation)
            x = np.arange(int(np.floor(lim[0])), int(np.ceil(lim[1])))
            dist = var.dist(**var.distkwargs)
            xdata = x
            if cumulative:
                ydata = dist.cdf(x)
                if orientation == PlotOrientation.VERTICAL:
                    plt.step(xdata, ydata, color='k', alpha=0.9, where='post')
                elif orientation == PlotOrientation.HORIZONTAL:
                    plt.step(ydata, xdata, color='k', alpha=0.9, where='post')
            else:
                ydata = dist.pmf(x)
                if orientation == PlotOrientation.VERTICAL:
                    plt.vlines(xdata, 0, ydata, color='k', alpha=0.9)
                elif orientation == PlotOrientation.HORIZONTAL:
                    plt.hlines(xdata, 0, ydata, color='k', alpha=0.9)

    elif isinstance(var, monaco.mc_var.OutVar):
        plt.hist(bins[:-1], bins=bins, weights=counts/sum(counts), density=False,
                 orientation=orientation, cumulative=cumulative, histtype='bar',
                 facecolor='k', alpha=0.5, **plotkwargs)

    if cumulative:
        ylabeltext = 'Cumulative Probability'
    else:
        ylabeltext = 'Probability Density'

    if rug_plot:
        plot_rug_marks(ax, orientation=orientation, nums=np.array(points))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Highlight cases and VarStats
    if orientation == PlotOrientation.VERTICAL:
        for i in highlight_cases_list:
            plt.plot([points[i], points[i]],
                     [ylim[0], ylim[0] + (ylim[1] - ylim[0])*0.20],
                     linestyle='-', linewidth=1, color='C1', alpha=1,
                     **plotkwargs)
        for i, varstat in enumerate(var.varstats):
            nums = get_list(varstat.nums)
            if length(nums) == 1:  # Single Statistic
                plt.plot([nums[0], nums[0]], ylim,
                         linestyle='-', color='C0', alpha=1, **plotkwargs)
            elif length(nums) == 3:  # Sided Order Statistics
                plt.plot([nums[1], nums[1]], ylim,
                         linestyle='-', color='C0', alpha=1, **plotkwargs)
            if varstat.confidence_interval_high_nums is not None:
                ax.fill_between([varstat.confidence_interval_low_nums,
                                 varstat.confidence_interval_high_nums],
                                [ylim[0], ylim[0]],
                                [ylim[1], ylim[1]],
                                color='C0', alpha=0.3)
            if length(nums) in (2, 3):  # Sided Order Statistics
                ax.fill_between([nums[0], nums[-1]],
                                [ylim[0], ylim[0]],
                                [ylim[1], ylim[1]],
                                color='C0', alpha=0.3)
        plt.xlabel(var.name)
        plt.ylabel(ylabeltext)
        apply_category_labels(ax, varx=var)

    elif orientation == PlotOrientation.HORIZONTAL:
        for i in highlight_cases_list:
            plt.plot([xlim[0], xlim[0] + (xlim[1] - xlim[0])*0.20],
                     [points[i], points[i]],
                     linestyle='-', linewidth=1, color='C1', alpha=1, **plotkwargs)
        for varstat in var.varstats:
            nums = get_list(varstat.nums)
            if length(nums) == 1:  # Single Statistic
                plt.plot(xlim, [nums[0], nums[0]],
                         linestyle='-', color='C0', alpha=1, **plotkwargs)
            elif length(nums) == 3:  # Sided Order Statistics
                plt.plot(xlim, [nums[1], nums[1]],
                         linestyle='-', color='C0', alpha=1, **plotkwargs)
            if varstat.confidence_interval_high_nums is not None:
                ax.fill_between(xlim,
                                [varstat.confidence_interval_low_nums,
                                 varstat.confidence_interval_high_nums],
                                color='C0', alpha=0.3)
            if length(nums) in (2, 3):  # Sided Order Statistics
                ax.fill_between(xlim, nums[0], nums[-1], color='C0', alpha=0.3)
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
             invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
             ax          : Optional[Axes]  = None,
             title       : str             = '',
             plotkwargs  : dict            = dict(),
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
                     orientation=orientation, rug_plot=rug_plot, invar_space=invar_space,
                     ax=ax, title=title, plotkwargs=plotkwargs)



def plot_2d_scatter(varx   : InVar | OutVar,
                    vary   : InVar | OutVar,
                    varz   : InVar | OutVar = None,
                    cases           : None | int | Iterable[int] = None,
                    highlight_cases : None | int | Iterable[int] = empty_list(),
                    rug_plot : bool           = False,
                    cov_plot : bool           = False,
                    cov_p    : None | float | Iterable[float] = None,
                    invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                    ax       : Optional[Axes] = None,
                    title    : str            = '',
                    plotkwargs : dict         = dict(),
                    ) -> tuple[Figure, Axes]:
    """
    Plot a scatter plot of two variables.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable to plot.
    varz : monaco.mc_var.InVar | monaco.mc_var.OutVar, default: None
        If not None, then sets the values for the underlying contour plot.
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    nvars = 2
    if varz is not None:
        nvars = 3

    fig, ax = manage_axis(ax, is3d=False)
    invar_space = manage_invar_space(invar_space=invar_space, nvars=nvars)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    varx_points = get_plot_points(varx, invar_space[0])
    vary_points = get_plot_points(vary, invar_space[1])

    if varz is not None:
        ngrid = 200
        cmap = 'viridis'
        levels = 21

        xi = np.linspace(min(varx_points), max(varx_points), ngrid)
        yi = np.linspace(min(vary_points), max(vary_points), ngrid)
        varz_points = get_plot_points(varz, invar_space[2])
        zi = griddata((varx_points, vary_points), varz_points,
                      (xi[None, :], yi[:, None]), method='linear')
        contour = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap)
        fig.colorbar(contour, ax=ax, label=varz.name)

    if reg_cases:
        plt.scatter(slice_by_index(varx_points, reg_cases),
                    slice_by_index(vary_points, reg_cases),
                    edgecolors=None, c='k', alpha=0.4, **plotkwargs)
    if highlight_cases_list:
        plt.scatter(slice_by_index(varx_points, highlight_cases_list),
                    slice_by_index(vary_points, highlight_cases_list),
                    edgecolors=None, c='C1', alpha=0.9, **plotkwargs)

    if cov_plot:
        if cov_p is None:
            cov_p = conf_ellipsoid_sig2pct(3.0, df=2)  # 3-sigma for 2D gaussian
        cov_p_list = get_list(cov_p)
        for p in cov_p_list:
            plot_2d_cov_ellipse(ax=ax, varx=varx, vary=vary, p=p)

    if rug_plot:
        all_cases = set(cases_list) | set(highlight_cases_list)
        plot_rug_marks(ax, orientation=PlotOrientation.VERTICAL,
                       nums=slice_by_index(varx_points, all_cases))
        plot_rug_marks(ax, orientation=PlotOrientation.HORIZONTAL,
                       nums=slice_by_index(vary_points, all_cases))

    plt.xlabel(varx.name)
    plt.ylabel(vary.name)
    apply_category_labels(ax, varx, vary)
    plt.title(title)

    return fig, ax



def plot_2d_line(varx : InVar | OutVar,
                 vary : InVar | OutVar,
                 cases           : None | int | Iterable[int] = None,
                 highlight_cases : None | int | Iterable[int] = empty_list(),
                 invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                 ax     : Optional[Axes] = None,
                 title  : str            = '',
                 plotkwargs : dict       = dict(),
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    invar_space = manage_invar_space(invar_space=invar_space, nvars=2)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    varx_points = get_plot_points(varx, invar_space[0])
    vary_points = get_plot_points(vary, invar_space[1])
    for i in reg_cases:
        plt.plot(varx_points[i], vary_points[i],
                 linestyle='-', color='black', alpha=0.2, **plotkwargs)
    for i in highlight_cases_list:
        plt.plot(varx_points[i], vary_points[i],
                 linestyle='-', color='C1', alpha=0.9, **plotkwargs)

    for varstat in vary.varstats:
        varx_points_max = max(varx_points, key=len)
        if length(varstat.nums[0]) == 1:  # Single Statistic
            plt.plot(varx_points_max, varstat.nums[:],
                     linestyle='-', color='C0', alpha=0.9, **plotkwargs)
        elif length(varstat.nums[0]) == 3:  # Sided Order Statistic
            plt.plot(varx_points_max, varstat.nums[:, 1],
                     linestyle='-', color='C0', alpha=0.9, **plotkwargs)

        if varstat.confidence_interval_high_nums is not None:
            ax.fill_between(varx_points_max,
                            varstat.confidence_interval_low_nums,
                            varstat.confidence_interval_high_nums,
                            color='C0', alpha=0.3)
        if length(varstat.nums[0]) in (2, 3):  # Sided Order Statistic
            ax.fill_between(varx_points_max,
                            np.array(varstat.nums)[:, 0],
                            np.array(varstat.nums)[:, -1],
                            color='C0', alpha=0.3)

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
                    invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                    ax     : Optional[Axes] = None,
                    title  : str            = '',
                    plotkwargs : dict       = dict(),
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    invar_space = manage_invar_space(invar_space=invar_space, nvars=3)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    varx_points = get_plot_points(varx, invar_space[0])
    vary_points = get_plot_points(vary, invar_space[1])
    varz_points = get_plot_points(varz, invar_space[2])
    if reg_cases:
        ax.scatter(slice_by_index(varx_points, reg_cases),
                   slice_by_index(vary_points, reg_cases),
                   slice_by_index(varz_points, reg_cases),
                   edgecolors=None, c='k', alpha=0.4, **plotkwargs)
    if highlight_cases_list:
        ax.scatter(slice_by_index(varx_points, highlight_cases_list),
                   slice_by_index(vary_points, highlight_cases_list),
                   slice_by_index(varz_points, highlight_cases_list),
                   edgecolors=None, c='C1', alpha=0.9, **plotkwargs)

    ax.set_xlabel(varx.name)
    ax.set_ylabel(vary.name)
    ax.set_zlabel(varz.name)
    apply_category_labels(ax, varx, vary, varz)
    plt.title(title)

    return fig, ax



def plot_2p5d_line(varx : InVar | OutVar,
                   vary : InVar | OutVar,
                   varz : InVar | OutVar,
                   cases           : None | int | Iterable[int] = None,
                   highlight_cases : None | int | Iterable[int] = empty_list(),
                   invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                   ax     : Optional[Axes] = None,
                   title  : str            = '',
                   plotkwargs : dict       = dict(),
                   ) -> tuple[Figure, Axes]:
    """
    Plot an ensemble of 2.5D lines for one scalar and two nonscalar variables.

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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    var0 = deepcopy(varx)
    var1 = deepcopy(vary)
    var2 = deepcopy(varz)
    npoints = 0
    for var in (var0, var1, var2):
        if var.maxdim == 1:
            npoints = max(npoints, max(len(num) for num in var.nums))
    for var in (var0, var1, var2):
        if var.maxdim == 0:
            for i in range(var.ncases):
                var.nums[i] = np.array([var.nums[i] for _ in range(npoints)])
                if isinstance(var, monaco.mc_var.InVar):
                    var.pcts[i] = [float(var.pcts[i]) for _ in range(npoints)]
            var.maxdim = 1

    fig, ax = plot_3d_line(varx=var0, vary=var1, varz=var2,
                           cases=cases, highlight_cases=highlight_cases,
                           invar_space=invar_space,
                           ax=ax, title=title, plotkwargs=plotkwargs)

    return fig, ax



def plot_3d_line(varx : InVar | OutVar,
                 vary : InVar | OutVar,
                 varz : InVar | OutVar,
                 cases           : None | int | Iterable[int] = None,
                 highlight_cases : None | int | Iterable[int] = empty_list(),
                 invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                 ax     : Optional[Axes] = None,
                 title  : str            = '',
                 plotkwargs : dict       = dict(),
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
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
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
    invar_space = manage_invar_space(invar_space=invar_space, nvars=3)

    cases_list = get_cases(varx.ncases, cases)
    highlight_cases_list = get_cases(varx.ncases, highlight_cases)
    reg_cases = set(cases_list) - set(highlight_cases_list)
    varx_points = get_plot_points(varx, invar_space[0])
    vary_points = get_plot_points(vary, invar_space[1])
    varz_points = get_plot_points(varz, invar_space[2])
    for i in reg_cases:
        ax.plot(varx_points[i], vary_points[i], varz_points[i],
                linestyle='-', color='black', alpha=0.3, **plotkwargs)
    for i in highlight_cases_list:
        ax.plot(varx_points[i], vary_points[i], varz_points[i],
                linestyle='-', color='C1', alpha=0.9, **plotkwargs)

    ax.set_xlabel(varx.name)
    ax.set_ylabel(vary.name)
    ax.set_zlabel(varz.name)
    apply_category_labels(ax, varx, vary, varz)
    plt.title(title)

    return fig, ax



def plot_cov_corr(matrix     : np.ndarray,
                  varnames   : list[str],
                  ax         : Optional[Axes] = None,
                  title      : str            = '',
                  plotkwargs : dict           = dict(),
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
    im = ax.imshow(matrix, cmap="RdBu", vmin=-scale, vmax=scale, **plotkwargs)
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



def plot_integration_convergence(outvar       : OutVar,
                                 dimension    : int,
                                 volume       : float,
                                 refval       : float          = None,
                                 conf         : float          = 0.95,
                                 samplemethod : SampleMethod   = SampleMethod.RANDOM,
                                 ax           : Optional[Axes] = None,
                                 title        : str            = '',
                                 plotkwargs   : dict           = dict(),
                                 ) -> tuple[Figure, Axes]:
    """
    For a Monte Carlo integration, plot the running integration estimate along
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
    ax.plot(cummean, 'C1')
    ax.plot(cummean + err, 'C0')
    ax.plot(cummean - err, 'C0')

    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'Convergence of {outvar.name} Integral')
    plt.title(title)

    return fig, ax



def plot_integration_error(outvar       : OutVar,
                           dimension    : int,
                           volume       : float,
                           refval       : float,
                           conf         : float          = 0.95,
                           samplemethod : SampleMethod   = SampleMethod.RANDOM,
                           ax           : Optional[Axes] = None,
                           title        : str            = '',
                           plotkwargs   : dict           = dict(),
                           ) -> tuple[Figure, Axes]:
    """
    For a Monte Carlo integration where the reference value is known, plot the
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
    ax.loglog(err, 'C0')
    ax.plot(np.abs(cummean - refval), 'C1')

    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'{outvar.name} {round(conf*100, 2)}% Confidence Error')
    plt.title(title)

    return fig, ax


def plot_sensitivities(outvar        : OutVar,
                       sensitivities : Sensitivities  = Sensitivities.RATIOS,
                       sort          : bool           = True,
                       ax            : Optional[Axes] = None,
                       title         : str            = '',
                       plotkwargs    : dict           = dict(),
                       ) -> tuple[Figure, Axes]:
    """
    For a Monte Carlo integration where the reference value is known, plot the
    running integration error along with error bounds for a given confidence
    level.

    Parameters
    ----------
    outvar : monaco.mc_var.OutVar
        The variable representing the integration estimate.
    sensitivities : monaco.mc_enums.Sensitivities, default: 'ratios'
        The sensitivities to plot. Either 'ratios' or 'indices'.
    sort : bool, default: True
        Whether to show the sensitivity indices sorted from greatest to least.
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
    fig.set_layout_engine('tight')

    if sensitivities == Sensitivities.RATIOS:
        sensitivities_dict = outvar.sensitivity_ratios
        ax.set_xlabel(f"Sensitivity Ratio for '{outvar.name}'")

    elif sensitivities == Sensitivities.INDICES:
        sensitivities_dict = outvar.sensitivity_indices
        ax.set_xlabel(f"Sensitivity Index for '{outvar.name}'")

    y_pos = np.arange(len(sensitivities_dict))
    if sort:
        sens_tuples = sorted(sensitivities_dict.items(), key=lambda item: item[1])
    else:
        sens_tuples = list(sensitivities_dict.items())
        y_pos = np.flipud(y_pos)
    invarnames = [invarname for invarname, _ in sens_tuples]
    sensitivities_vals = [val for _, val in sens_tuples]

    ax.barh(y_pos, sensitivities_vals, facecolor='k', alpha=0.5, tick_label=invarnames)
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



def apply_category_labels(ax   : Axes,
                          varx : InVar | OutVar | None = None,
                          vary : InVar | OutVar | None = None,
                          varz : InVar | OutVar | None = None,
                          ) -> None:
    """
    For nonnumeric Monte Carlo variables, use the `nummap` to label the axes.

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



def plot_2d_cov_ellipse(ax   : Axes,
                        varx : InVar | OutVar,
                        vary : InVar | OutVar,
                        p    : float,
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



def manage_invar_space(invar_space : InVarSpace | Iterable[InVarSpace],
                       nvars       : int,
                       ) -> list[InVarSpace]:
    '''
    Parse the `invarspace` input for plotting functions. If already an iterable
    of the right length, passes through. If a single value, returns a list of
    the invarspace of the right length.

    Parameters
    ----------
    invar_space : monaco.InVarSpace | Iterable[InVarSpace]
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
    nvars : int
        The length of the list to generate.

    Returns
    -------
    invar_space_list : list[InVarSpace]
        A list of length `nvars` for the desired InVarSpace's.
    '''
    if isinstance(invar_space, InVarSpace):
        invar_space = [invar_space for _ in range(nvars)]

    if len(invar_space) != nvars:
        raise ValueError(f'Argument invar_space = {invar_space} expected length {nvars}')
    invar_space_list = list(invar_space)

    return invar_space_list



def get_plot_points(var         : InVar | OutVar,
                    invar_space : InVarSpace,
                    ) -> list[float]:
    '''
    Get the points to plot based on the invar_space. Returns var.nums, unless
    var is an InVar and invar_space is 'pcts' in which case it returns
    var.pcts.

    Parameters
    ----------
    nvars : InVar | OutVar
        The target variable.
    invar_space : monaco.InVarSpace
        The space to plot the invars in.

    Returns
    -------
    plot_points : list[float]
        The points to plot.
    '''
    plot_points = var.nums
    if isinstance(var, monaco.mc_var.InVar) and invar_space == InVarSpace.PCTS:
        plot_points = var.pcts
    return plot_points



def get_var_steps(var : InVar | OutVar) -> OutVar:
    '''
    For a 1D variable, get an OutVar that has as values the simulation sets.

    Parameters
    ----------
    var : InVar | OutVar
        The target variable.

    Returns
    -------
    varsteps : OutVar
        The variable with simulation steps as values.
    '''
    vals = [np.arange(len(num)) for num in var.nums]
    varsteps = monaco.mc_var.OutVar(name='Simulation Steps', vals=vals)
    return varsteps
