# mc_multi_plot.py
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from monaco.mc_plot import plot_hist, plot_2d_scatter
from monaco.mc_var import InVar, OutVar
from monaco.mc_enums import PlotOrientation, InVarSpace
from monaco.helper_functions import empty_list, get_list
from typing import Optional, Iterable


def multi_plot(vars        : list[InVar | OutVar],
               cases           : None | int | Iterable[int] = None,
               highlight_cases : None | int | Iterable[int] = empty_list(),
               rug_plot    : bool   = False,
               cov_plot    : bool   = False,
               cov_p       : None | float | Iterable[float] = None,
               invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
               fig         : Figure | None = None,
               title       : str    = '',
               plotkwargs  : dict   = dict(),
               ) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Umbrella function to make more complex plots of Monte Carlo variables.

    Parameters
    ----------
    vars : list[monaco.mc_var.InVar | monaco.mc_var.OutVar]
        The variables to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool, default: False
        Whether to plot rug marks.
    cov_plot : bool, default: False
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : None | float | Iterable[float], default: None
        The gaussian percentiles for the covariance plot.
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
    fig : matplotlib.figure.Figure, default: None
        The figure handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, axes) : (matplotlib.figure.Figure, (matplotlib.axes.Axes, ...))
        fig is the figure handle for the plot.
        axes is a tuple of the axes handles for the plots.
    """
    # Split larger vars
    if len(vars) == 1:
        if isinstance(vars[0], OutVar) and vars[0].maxdim == 1:
            var_split = vars[0].split()
            origname = vars[0].name
            varx = var_split[origname + ' [0]']
            vary = var_split[origname + ' [1]']
            vars = [varx, vary]
        else:
            raise ValueError(f'Invalid vars[0] dimension: {vars[0].maxdim}')

    # Two Variable Plots
    if len(vars) == 2:
        if vars[0].isscalar and vars[1].isscalar:
            fig, axs = multi_plot_2d_scatter_hist(varx=vars[0], vary=vars[1],
                                                  cases=cases, highlight_cases=highlight_cases,
                                                  rug_plot=rug_plot,
                                                  cov_plot=cov_plot, cov_p=cov_p,
                                                  cumulative=False,
                                                  invar_space=invar_space,
                                                  fig=fig, title=title, plotkwargs=plotkwargs)
        else:
            raise ValueError( 'Invalid variable dimensions: ' +
                             f'{varx.name} {varx.maxdim}, ' +
                             f'{vary.name} {vary.maxdim}')

    # Many Variable Plots
    elif len(vars) > 2:
        scalarvars = [var for var in vars if var.isscalar]
        fig, axs = multi_plot_grid_tri(vars=scalarvars,
                                       cases=cases, highlight_cases=highlight_cases,
                                       rug_plot=rug_plot,
                                       cov_plot=cov_plot, cov_p=cov_p,
                                       cumulative=False,
                                       invar_space=invar_space,
                                       fig=fig, title=title, plotkwargs=plotkwargs)

    return fig, axs



def multi_plot_2d_scatter_hist(varx     : InVar | OutVar,
                               vary     : InVar | OutVar,
                               cases           : None | int | Iterable[int] = None,
                               highlight_cases : None | int | Iterable[int] = empty_list(),
                               rug_plot   : bool   = False,
                               cov_plot   : bool   = True,
                               cov_p      : None | float | Iterable[float] = None,
                               cumulative : bool   = False,
                               invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                               fig        : Figure | None = None,
                               title      : str    = '',
                               plotkwargs : dict   = dict(),
                               ) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Plot two variables against each other with a central scatterplot and two
    histograms along the x and y axes.

    Parameters
    ----------
    varx : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The x variable to plot.
    vary : monaco.mc_var.InVar | monaco.mc_var.OutVar
        The y variable to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool, default: False
        Whether to plot rug marks.
    cov_plot : bool, default: False
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : None | float | Iterable[float], default: None
        The gaussian percentiles for the covariance plot.
    cumulative : bool, default: False
        Whether to plot the histograms as cumulative distribution functions.
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
    fig : matplotlib.figure.Figure, default: None
        The figure handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
    (fig, (ax1, ax2, ax3)) : (matplotlib.figure.Figure,
    (matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes))
        fig is the figure handle for the plot.
        (ax1, ax2, ax3) are the axes handles for the central, y-axis, and
        x-axis plots, respectively.
    """
    fig = handle_fig(fig)

    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[3, 1:4])
    ax2 = fig.add_subplot(gs[0:3, 0])
    ax3 = fig.add_subplot(gs[0:3, 1:4], sharex=ax1, sharey=ax2)

    plot_hist(varx, cases=cases, highlight_cases=highlight_cases,
              cumulative=cumulative, rug_plot=False, invar_space=invar_space,
              ax=ax1, title='', plotkwargs=plotkwargs)
    plot_hist(vary, cases=cases, highlight_cases=highlight_cases,
              cumulative=cumulative, rug_plot=False, invar_space=invar_space,
              ax=ax2, title='', plotkwargs=plotkwargs,
              orientation=PlotOrientation.HORIZONTAL)
    plot_2d_scatter(varx, vary,
                    cases=cases, highlight_cases=highlight_cases,
                    rug_plot=rug_plot,
                    cov_plot=cov_plot, cov_p=cov_p,
                    invar_space=invar_space,
                    ax=ax3, title='', plotkwargs=plotkwargs)

    ax1.set_ylabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.xaxis.set_tick_params(labelbottom=False)
    ax3.yaxis.set_tick_params(labelbottom=False)

    plt.suptitle(title, y=.93)

    return fig, (ax1, ax2, ax3)



def multi_plot_grid_tri(vars     : list[InVar | OutVar],
                        cases           : None | int | Iterable[int] = None,
                        highlight_cases : None | int | Iterable[int] = empty_list(),
                        rug_plot   : bool   = False,
                        cov_plot   : bool   = False,
                        cov_p      : None | float | Iterable[float] = None,
                        cumulative : bool   = False,
                        invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                        fig        : Figure | None = None,
                        title      : str    = '',
                        plotkwargs : dict   = dict(),
                        ) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Plot multiple variables against each other in a triangular grid. The
    off-diagonal grid locations show scatterplots of the two corresponding
    variables. The plots along the diagonal show histograms for the
    corresponding variables.

    Parameters
    ----------
    vars : list[monaco.mc_var.InVar | monaco.mc_var.OutVar]
        The variables to plot.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool, default: False
        Whether to plot rug marks.
    cov_plot : bool, default: False
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : None | float | Iterable[float], default: None
        The gaussian percentiles for the covariance plot.
    cumulative : bool, default: False
        Whether to plot the histograms as cumulative distribution functions.
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
    fig : matplotlib.figure.Figure, default: None
        The figure handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
        (fig, axes) : (matplotlib.figure.Figure, (matplotlib.axes.Axes, ...))
            fig is the figure handle for the plot.
            axes is a tuple of the axes handles for the plots, starting from
            the top-left corner and working left-to-right, then top-to-bottom.
    """
    fig = handle_fig(fig)

    nvars = len(vars)
    axs = fig.subplots(nvars, nvars, sharex='col')
    for i in range(nvars):
        row_scatter_axs = [ax for k, ax in enumerate(axs[i]) if k < i]
        for j in range(nvars):
            ax = axs[i][j]
            if i == j:
                plot_hist(vars[i], cases=cases, highlight_cases=highlight_cases,
                          cumulative=cumulative, rug_plot=False, invar_space=invar_space,
                          ax=ax, title='', plotkwargs=plotkwargs)
                ax.yaxis.tick_right()
            elif j > i:
                fig.delaxes(ax)
            else:
                # Don't link the histogram y axis
                ax.get_shared_y_axes()._grouper.join(*row_scatter_axs)
                plot_2d_scatter(vars[j], vars[i],
                                cases=cases, highlight_cases=highlight_cases,
                                rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p,
                                invar_space=invar_space,
                                ax=ax, title='', plotkwargs=plotkwargs)
                if j > 0:
                    ax.set_yticklabels([])

            ax.set_xlabel('')
            ax.set_ylabel('')
            if j == 0:
                ax.set_ylabel(vars[i].name)
            if i == nvars-1:
                ax.set_xlabel(vars[j].name)

    fig.subplots_adjust(wspace=0.075, hspace=0.075)
    fig.suptitle(title)

    return fig, axs


def multi_plot_grid_rect(varsx     : InVar | OutVar | list[InVar | OutVar],
                         varsy     : InVar | OutVar | list[InVar | OutVar] = None,
                         cases           : None | int | Iterable[int] = None,
                         highlight_cases : None | int | Iterable[int] = empty_list(),
                         rug_plot   : bool   = False,
                         cov_plot   : bool   = False,
                         cov_p      : None | float | Iterable[float] = None,
                         cumulative : bool   = False,
                         invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
                         fig        : Figure | None = None,
                         title      : str    = '',
                         plotkwargs : dict   = dict(),
                         ) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Plot multiple variables against each other in a rectangual grid. The
    left column and bottom row show histograms for each variable, and the middle
    grid locations show the scatter plots between each pari of variables.

    Parameters
    ----------
    varsx : monaco.mc_var.InVar | monaco.mc_var.OutVar | list[InVar | OutVar]
        The variables to plot along the X axis.
    varsy : monaco.mc_var.InVar | monaco.mc_var.OutVar | list[InVar | OutVar]
        The variables to plot along the Y axis.
    cases : None | int | Iterable[int], default: None
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : None | int | Iterable[int], default: []
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool, default: False
        Whether to plot rug marks.
    cov_plot : bool, default: False
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : None | float | Iterable[float], default: None
        The gaussian percentiles for the covariance plot.
    cumulative : bool, default: False
        Whether to plot the histograms as cumulative distribution functions.
    invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
        The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
        specifies this individually for each of varx, vary, and varz.
    fig : matplotlib.figure.Figure, default: None
        The figure handle to plot in. If None, a new figure is created.
    title : str, default: ''
        The figure title.

    Returns
    -------
        (fig, axes) : (matplotlib.figure.Figure, (matplotlib.axes.Axes, ...))
            fig is the figure handle for the plot.
            axes is a tuple of the axes handles for the plots, starting from
            the top-left corner and working left-to-right, then top-to-bottom.
    """
    fig = handle_fig(fig)

    varsx = get_list(varsx)
    varsy = get_list(varsy)
    nvarsx = len(varsx)
    nvarsy = len(varsy)

    axs = fig.subplots(nvarsy+1, nvarsx+1, sharex='col')
    for i in range(nvarsy+1):
        row_scatter_axs = [ax for k, ax in enumerate(axs[i]) if k < nvarsy+1]
        for j in range(nvarsx+1):
            ax = axs[i][j]
            if i == nvarsy and j == 0:
                fig.delaxes(ax)
            elif i == nvarsy:
                plot_hist(varsx[j-1], cases=cases, highlight_cases=highlight_cases,
                          cumulative=cumulative, rug_plot=False, invar_space=invar_space,
                          orientation=PlotOrientation.VERTICAL,
                          ax=ax, title='', plotkwargs=plotkwargs)
            elif j == 0:
                plot_hist(varsy[i], cases=cases, highlight_cases=highlight_cases,
                          cumulative=cumulative, rug_plot=False, invar_space=invar_space,
                          orientation=PlotOrientation.HORIZONTAL,
                          ax=ax, title='', plotkwargs=plotkwargs)
                ax.set_xticklabels([])
            else:
                # Don't link the histogram y axis
                ax.get_shared_y_axes()._grouper.join(*row_scatter_axs)
                plot_2d_scatter(varsx[j-1], varsy[i],
                                cases=cases, highlight_cases=highlight_cases,
                                rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p,
                                invar_space=invar_space,
                                ax=ax, title='', plotkwargs=plotkwargs)
            if j > 0:
                ax.set_yticklabels([])

            ax.set_xlabel('')
            ax.set_ylabel('')
            if i < nvarsy and j == 0:
                ax.set_ylabel(varsy[i].name)
            if i == nvarsy and j != 0:
                ax.set_xlabel(varsx[j-1].name)

    fig.subplots_adjust(wspace=0.075, hspace=0.075)
    fig.suptitle(title)

    return fig, axs



def handle_fig(fig : Optional[Figure]) -> Figure:
    """
    Set the target figure, either by making a new figure or setting active an
    existing one.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The target figure. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The active figure.
    """
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.num)
    plt.clf()
    return fig
