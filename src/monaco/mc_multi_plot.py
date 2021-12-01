# mc_multi_plot.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from monaco.mc_plot import mc_plot_hist, mc_plot_2d_scatter
from monaco.MCVar import MCInVar, MCOutVar
from monaco.MCEnums import PlotOrientation
from monaco.helper_functions import empty_list
from typing import Union, Optional, Iterable


def mc_multi_plot(mcvars   : list[Union[MCInVar, MCOutVar]],
                  cases           : Union[None, int, Iterable[int]] = None,
                  highlight_cases : Union[None, int, Iterable[int]] = empty_list(),
                  rug_plot : bool   = True,
                  cov_plot : bool   = True,
                  cov_p    : Union[None, float, Iterable[float]] = None,
                  fig      : Figure = None, 
                  title    : str    = '',
                  ) -> tuple[Figure, tuple[Axes]]:
    """
    Umbrella function to make more complex plots of Monte-Carlo variables. 

    Parameters
    ----------
    mcvars : list[{monaco.MCVar.MCInVar, monaco.MCVar.MCOutVar}]
        The variables to plot.
    cases : {None, int, Iterable[int]} (default: None)
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : {None, int, Iterable[int]} (default: [])
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool (default: True)
        Whether to plot rug marks.
    cov_plot : bool (default: False)
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : {None, float, Iterable[float]} (default: None)
        The gaussian percentiles for the covariance plot.
    fig : matplotlib.figure.Figure (default: None)
        The figure handle to plot in. If None, a new figure is created.
    title : str (default: '')
        The figure title.
    
    Returns
    -------
        (fig, axes) : (matplotlib.figure.Figure, (matplotlib.axes.Axes,))
            fig is the figure handle for the plot.
            axes is a tuple of the axes handles for the plots.
    """
    # Split larger vars
    if len(mcvars) == 1:
        if isinstance(mcvars[0], MCOutVar) and mcvars[0].size[0] == 2:
            mcvar_split = mcvars[0].split()
            origname = mcvars[0].name
            mcvarx = mcvar_split[origname + ' [0]']
            mcvary = mcvar_split[origname + ' [1]']
            mcvars = [mcvarx, mcvary]
        else:
            raise ValueError(f'Invalid mcvars[0] size at index 0: ({mcvars[0].size[0]},{mcvars[0].size[1]})')
    # Two Variable Plots
    if len(mcvars) == 2:
        if mcvars[1].size[0] != 1:
            raise ValueError(f'Invalid mcvars[1] size at index 0: ({mcvars[1].size[0]},{mcvars[1].size[1]})')
        
        if mcvars[0].size[1] == 1 and mcvars[1].size[1] == 1:
            fig, axs = mc_multi_plot_2d_scatter_hist(mcvarx=mcvars[0], mcvary=mcvars[1], cases=cases, highlight_cases=highlight_cases, rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p, cumulative=False, fig=fig, title=title)
    
    # Many Variable Plots
    elif len(mcvars) > 2:
        scalarmcvars = [mcvar for mcvar in mcvars if mcvar.isscalar]
        fig, axs =  mc_multi_plot_2d_scatter_grid(mcvars=scalarmcvars, cases=cases, highlight_cases=highlight_cases, rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p, cumulative=False, fig=fig, title=title)
    
    return fig, axs



def mc_multi_plot_2d_scatter_hist(mcvarx     : Union[MCInVar, MCOutVar], 
                                  mcvary     : Union[MCInVar, MCOutVar],
                                  cases           : Union[None, int, Iterable[int]] = None,
                                  highlight_cases : Union[None, int, Iterable[int]] = empty_list(),
                                  rug_plot   : bool   = True,
                                  cov_plot   : bool   = True,
                                  cov_p      : Union[None, float, Iterable[float]] = None,
                                  cumulative : bool   = False,
                                  fig        : Figure = None, 
                                  title      : str    = '',
                                  ) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plot two variables against each other with a central scatterplot and two
    histograms along the x and y axes.

    Parameters
    ----------
    mcvarx : {monaco.MCVar.MCInVar, monaco.MCVar.MCOutVar}
        The x variable to plot.
    mcvary : {monaco.MCVar.MCInVar, monaco.MCVar.MCOutVar}
        The y variable to plot.
    cases : {None, int, Iterable[int]} (default: None)
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : {None, int, Iterable[int]} (default: [])
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool (default: True)
        Whether to plot rug marks.
    cov_plot : bool (default: False)
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : {None, float, Iterable[float]} (default: None)
        The gaussian percentiles for the covariance plot.
    cumulative : bool (default: False)
        Whether to plot the histograms as cumulative distribution functions.
    fig : matplotlib.figure.Figure (default: None)
        The figure handle to plot in. If None, a new figure is created.
    title : str (default: '')
        The figure title.
    
    Returns
    -------
        (fig, (ax1, ax2, ax3)) : (matplotlib.figure.Figure, 
                                 (matplotlib.axes.Axes, matplotlib.axes.Axes,
                                  matplotlib.axes.Axes))
            fig is the figure handle for the plot.
            (ax1, ax2, ax3) are the axes handles for the central, y-axis, and
            x-axis plots, respectively.
    """
    fig = handle_fig(fig)
    
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[3, 1:4])
    ax2 = fig.add_subplot(gs[0:3, 0])
    ax3 = fig.add_subplot(gs[0:3, 1:4], sharex=ax1, sharey=ax2)

    mc_plot_hist(mcvarx, highlight_cases=highlight_cases, rug_plot=False, ax=ax1, title='', cumulative=cumulative)
    mc_plot_hist(mcvary, highlight_cases=highlight_cases, rug_plot=False, ax=ax2, title='', cumulative=cumulative, orientation=PlotOrientation.HORIZONTAL)
    mc_plot_2d_scatter(mcvarx, mcvary, cases=cases, highlight_cases=highlight_cases, rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p, ax=ax3, title='')
    
    ax1.set_ylabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.xaxis.set_tick_params(labelbottom=False)
    ax3.yaxis.set_tick_params(labelbottom=False)
    
    plt.suptitle(title, y=.93)

    return fig, (ax1, ax2, ax3)



def mc_multi_plot_2d_scatter_grid(mcvars     : list[Union[MCInVar, MCOutVar]], 
                                  cases           : Union[None, int, Iterable[int]] = None,
                                  highlight_cases : Union[None, int, Iterable[int]] = empty_list(),
                                  rug_plot   : bool   = True,
                                  cov_plot   : bool   = True,
                                  cov_p      : Union[None, float, Iterable[float]] = None,
                                  cumulative : bool   = False,
                                  fig        : Figure = None, 
                                  title      : str    = '',
                                  ) -> tuple[Figure, tuple[Axes]]:
    """
    Plot multiple variables against each other in a grid. The off-diagonal grid
    locations show scatterplots of the two corresponding variables. The
    plots along the diagonal show histograms for the corresponding variables. 

    Parameters
    ----------
    mcvars : list[{monaco.MCVar.MCInVar, monaco.MCVar.MCOutVar}]
        The variables to plot.
    cases : {None, int, Iterable[int]} (default: None)
        The cases to plot. If None, then all cases are plotted.
    highlight_cases : {None, int, Iterable[int]} (default: [])
        The cases to highlight. If [], then no cases are highlighted.
    rug_plot : bool (default: True)
        Whether to plot rug marks.
    cov_plot : bool (default: False)
        Whether to plot a covariance ellipse at a certain gaussian percentile
        level.
    cov_p : {None, float, Iterable[float]} (default: None)
        The gaussian percentiles for the covariance plot.
    cumulative : bool (default: False)
        Whether to plot the histograms as cumulative distribution functions.
    fig : matplotlib.figure.Figure (default: None)
        The figure handle to plot in. If None, a new figure is created.
    title : str (default: '')
        The figure title.
    
    Returns
    -------
        (fig, axes) : (matplotlib.figure.Figure, (matplotlib.axes.Axes,))
            fig is the figure handle for the plot.
            axes is a tuple of the axes handles for the plots, starting from
            the top-left corner and working left-to-right, then top-to-bottom.
    """
    fig = handle_fig(fig)
    
    nvars = len(mcvars)
    axs = fig.subplots(nvars, nvars, sharex='col')
    for i in range(nvars):
        row_scatter_axs = [ax for k, ax in enumerate(axs[i]) if k != i]
        for j in range(nvars):
            ax=axs[i][j]
            if i == j: 
                mc_plot_hist(mcvars[i], highlight_cases=highlight_cases, rug_plot=False, ax=ax, title='', cumulative=cumulative)
            else:
                ax.get_shared_y_axes().join(*row_scatter_axs) # Don't link the histogram y axis
                mc_plot_2d_scatter(mcvars[j], mcvars[i], cases=cases, highlight_cases=highlight_cases, rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p, ax=ax, title='')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            if j == 0:
                ax.set_ylabel(mcvars[i].name)
            if i == nvars-1:
                ax.set_xlabel(mcvars[j].name)
    
    plt.suptitle(title)

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
