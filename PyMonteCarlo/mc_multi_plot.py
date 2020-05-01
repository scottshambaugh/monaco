import matplotlib.pyplot as plt
from PyMonteCarlo.mc_plot import mc_plot_hist, mc_plot_2d_scatter


def mc_multi_plot(mcvarx, mcvary=None, cases=0, fig=None, title=''):
    # Split larger vars
    if mcvary == None:
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
        fig, axs = mc_multi_plot_2d_scatter_hist(mcvarx=mcvarx, mcvary=mcvary, cases=cases, cumulative=False, fig=fig, title=title)
            
    return fig, axs



def mc_multi_plot_2d_scatter_hist(mcvarx, mcvary, cases, cumulative=False, fig=None, title=''):
    if not fig:
        fig = plt.figure()
    else:
        plt.figure(fig.num)
    plt.clf()
    
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[0:3, 1:4])
    ax2 = fig.add_subplot(gs[0:3, 0])
    ax3 = fig.add_subplot(gs[3, 1:4])

    mc_plot_2d_scatter(mcvarx, mcvary, cases=cases, ax=ax1, title='')
    mc_plot_hist(mcvary, cases=cases, ax=ax2, title='', cumulative=cumulative, orientation='horizontal')
    mc_plot_hist(mcvarx, cases=cases, ax=ax3, title='', cumulative=cumulative)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax2.set_xlabel('')
    ax3.set_ylabel('')
    
    plt.suptitle(title, y=.93)

    return fig, (ax1, ax2, ax3)
    

'''
### Test ###
from PyMonteCarlo.MCVar import MCInVar
from scipy.stats import norm
plt.close('all')

mcinvars = dict()
mcinvars['norm1'] = MCInVar('norm1', norm, (1, 5), 1000, seed=1)
mcinvars['norm2'] = MCInVar('norm2', norm, (10, 4), 1000, seed=2)

mc_multi_plot(mcinvars['norm1'], mcinvars['norm2'], cases=range(10,30), title='test')  # MCPlot2DScatter
#'''
