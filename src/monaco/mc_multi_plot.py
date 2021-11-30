# mc_multi_plot.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from monaco.mc_plot import mc_plot_hist, mc_plot_2d_scatter
from monaco.MCVar import MCInVar, MCOutVar
from monaco.helper_functions import empty_list
from typing import Union, Optional


def mc_multi_plot(mcvars   : list[Union[MCInVar, MCOutVar]],
                  cases           : Union[None, int, list[int], set[int]] = None, # All cases
                  highlight_cases : Union[None, int, list[int], set[int]] = empty_list(), # No cases
                  rug_plot : bool                = True,
                  cov_plot : bool                = True,
                  cov_p    : Union[None, float, list[float], set[float]]  = None,
                  fig      : Figure              = None, 
                  title    : str                 = '',
                  ) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    
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
                                  cases           : Union[None, int, list[int], set[int]] = None, # All cases
                                  highlight_cases : Union[None, int, list[int], set[int]] = empty_list(), # No cases
                                  rug_plot   : bool                = True,
                                  cov_plot   : bool                = True,
                                  cov_p      : Union[None, float, list[float], set[float]]  = None,
                                  cumulative : bool                = False,
                                  fig        : Figure              = None, 
                                  title      : str                 = '',
                                  ) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    fig = handle_fig(fig)
    
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[3, 1:4])
    ax2 = fig.add_subplot(gs[0:3, 0])
    ax3 = fig.add_subplot(gs[0:3, 1:4], sharex=ax1, sharey=ax2)

    mc_plot_hist(mcvarx, highlight_cases=highlight_cases, rug_plot=False, ax=ax1, title='', cumulative=cumulative)
    mc_plot_hist(mcvary, highlight_cases=highlight_cases, rug_plot=False, ax=ax2, title='', cumulative=cumulative, orientation='horizontal')
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
                                  cases           : Union[None, int, list[int], set[int]] = None, # All cases
                                  highlight_cases : Union[None, int, list[int], set[int]] = empty_list(), # No cases
                                  rug_plot   : bool                = True,
                                  cov_plot   : bool                = True,
                                  cov_p      : Union[None, float, list[float], set[float]]  = None,
                                  cumulative : bool                = False,
                                  fig        : Figure              = None, 
                                  title      : str                 = '',
                                  ) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
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
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.num)
    plt.clf()
    return fig
