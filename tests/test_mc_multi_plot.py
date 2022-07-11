# test_mc_multi_plot.py

# import pytest

### Plot Testing ###
def plot_testing():
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from monaco.mc_var import InVar
    from monaco.mc_multi_plot import multi_plot, multi_plot_grid_rect
    from monaco.mc_enums import SampleMethod

    plt.close('all')

    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31-1, size=10)

    invars = dict()
    invars['norm1'] = InVar('norm1', ndraws=1000,
                            dist=norm, distkwargs={'loc': 1, 'scale': 5},
                            seed=invarseeds[0], samplemethod=SampleMethod.RANDOM)
    invars['norm2'] = InVar('norm2', ndraws=1000,
                            dist=norm, distkwargs={'loc': 10, 'scale': 4},
                            seed=invarseeds[1], samplemethod=SampleMethod.RANDOM)
    invars['norm3'] = InVar('norm3', ndraws=1000,
                            dist=norm, distkwargs={'loc': 5, 'scale': 2},
                            seed=invarseeds[3], samplemethod=SampleMethod.RANDOM)

    fig, axs = multi_plot([invars['norm1'], invars['norm2']],
                          highlight_cases=range(10, 30),
                          rug_plot=True,
                          cov_plot=True, cov_p=0.95,
                          title='test')  # multi_plot_2d_scatter_hist

    fig, axs = multi_plot([invars['norm1'], invars['norm2'], invars['norm3']],
                           highlight_cases=range(10, 30),
                           rug_plot=True,
                           cov_plot=True, cov_p=0.95,
                           title='test')  # multi_plot_grid_tri

    fig, axs = multi_plot_grid_rect([invars['norm1'], invars['norm2']], invars['norm3'],
                                    highlight_cases=range(10, 30),
                                    rug_plot=True,
                                    cov_plot=True, cov_p=0.95,
                                    title='test')  # multi_plot_grid_rect

    plt.show(block=True)


if __name__ == '__main__':
    plot_testing()
