import matplotlib.pyplot as plt
from PyMonteCarlo.MCPlot import MCPlotHist, MCPlot2DScatter


def MCMultiPlot(mcvarx, mcvary=None, cases=0, fig=None):
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
        fig, axs = MCMultiPlot2DScatterHist(mcvarx=mcvarx, mcvary=mcvary, cases=cases, cumulative=False, fig=fig)
            
    return fig, axs



def MCMultiPlot2DScatterHist(mcvarx, mcvary, cases, cumulative=False, fig=None):
    if not fig:
        fig = plt.figure()
    else:
        plt.figure(fig.num)
    plt.clf()
    
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[0:3, 1:4])
    ax2 = fig.add_subplot(gs[0:3, 0])
    ax3 = fig.add_subplot(gs[3, 1:4])

    MCPlot2DScatter(mcvarx, mcvary, cases=cases, ax=ax1)
    MCPlotHist(mcvary, cases=cases, ax=ax2, cumulative=cumulative, orientation='horizontal')
    MCPlotHist(mcvarx, cases=cases, ax=ax3, cumulative=cumulative)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax2.set_xlabel('')
    ax3.set_ylabel('')

    return fig, (ax1, ax2, ax3)
    

'''
### Test ###
from PyMonteCarlo.MCVar import MCInVar
from scipy.stats import norm
plt.close('all')

mcinvars = dict()
mcinvars['norm1'] = MCInVar('norm1', norm, (1, 5), 1000, seed=1)
mcinvars['norm2'] = MCInVar('norm2', norm, (10, 4), 1000, seed=2)

MCMultiPlot(mcinvars['norm1'], mcinvars['norm2'], cases=range(10,30))  # MCPlot2DScatter
#'''
