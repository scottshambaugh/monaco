# mc_multi_plot.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from Monaco.mc_plot import mc_plot_hist, mc_plot_2d_scatter
from Monaco.MCVar import MCVar
from typing import Union


def mc_multi_plot(mcvarx   : MCVar, 
                  mcvary   : MCVar,
                  cases                          = None, # TODO: typing
                  highlight_cases                = [],   # TODO: typing
                  rug_plot : bool                = True,
                  cov_plot : bool                = True,
                  cov_p                          = None,  # TODO: typing 
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
                                  cases                            = None, # TODO: typing
                                  highlight_cases                  = [],   # TODO: typing
                                  rug_plot   : bool                = True,
                                  cov_plot   : bool                = True,
                                  cov_p                            = None,  # TODO: typing 
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
    

'''
### Test ###
if __name__ == '__main__':
    from Monaco.MCVar import MCInVar
    from scipy.stats import norm
    import numpy as np
    plt.close('all')
    
    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)
        
    mcinvars = dict()
    mcinvars['norm1'] = MCInVar('norm1', ndraws=1000, dist=norm, distkwargs={'loc':1, 'scale':5}, seed=invarseeds[0])
    mcinvars['norm2'] = MCInVar('norm2', ndraws=1000, dist=norm, distkwargs={'loc':10, 'scale':4}, seed=invarseeds[1])
    
    mc_multi_plot(mcinvars['norm1'], mcinvars['norm2'], highlight_cases=range(10,30), rug_plot=True, cov_plot=True, cov_p=0.95, title='test')  # MCPlot2DScatter
#'''
