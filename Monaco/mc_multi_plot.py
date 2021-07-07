# mc_multi_plot.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from Monaco.mc_plot import mc_plot_hist, mc_plot_2d_scatter
from Monaco.MCVar import MCVar
from typing import Union


def mc_multi_plot(mcvarx   : MCVar, 
                  mcvary   : MCVar,
                  cases           : Union[None, int, list[int], set[int]] = None, # All cases
                  highlight_cases : Union[None, int, list[int], set[int]] = [],   # No cases
                  rug_plot : bool                = True,
                  cov_plot : bool                = True,
                  cov_p    : Union[None, float, list[float], set[float]]  = None,
                  fig      : Union[None, Figure] = None, 
                  title    : str                 = '',
                  ):
    # Split larger vars
    if mcvary is None:
        if mcvarx.size[0] == 2:
            mcvarx_split = mcvarx.split()
            origname = mcvarx.name
            mcvarx = mcvarx_split[origname + ' [0]']
            mcvary = mcvarx_split[origname + ' [1]']
        else:
            raise ValueError(f'Invalid mcvarx size at index 0: ({mcvarx.size[0]},{mcvarx.size[1]})')
    elif mcvary.size[0] != 1:
        raise ValueError(f'Invalid mcvary size at index 0: ({mcvary.size[0]},{mcvary.size[1]})')

    # Two Variable Plots
    if mcvarx.size[1] == 1 and mcvary.size[1] == 1:
        fig, axs = mc_multi_plot_2d_scatter_hist(mcvarx=mcvarx, mcvary=mcvary, cases=cases, highlight_cases=highlight_cases, rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p, cumulative=False, fig=fig, title=title)
            
    return fig, axs



def mc_multi_plot_2d_scatter_hist(mcvarx     : MCVar, 
                                  mcvary     : MCVar,
                                  cases           : Union[None, int, list[int], set[int]] = None, # All cases
                                  highlight_cases : Union[None, int, list[int], set[int]] = [],   # No cases
                                  rug_plot   : bool                = True,
                                  cov_plot   : bool                = True,
                                  cov_p      : Union[None, float, list[float], set[float]]  = None,
                                  cumulative : bool                = False,
                                  fig        : Union[None, Figure] = None, 
                                  title      : str                 = '',
                                  ):
    if not fig:
        fig = plt.figure()
    else:
        plt.figure(fig.num)
    plt.clf()
    
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
    

