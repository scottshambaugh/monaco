# test_mc_multi_plot.py

import pytest


# Does not test the plot appearances, but does check that the codepaths can run
def test_gen_plots():
    plot_testing(show=False)
    assert True


def test_multi_plot_split_1d_outvar():
    import numpy as np
    import matplotlib.pyplot as plt
    from monaco.mc_var import OutVar
    from monaco.mc_multi_plot import multi_plot

    outvar = OutVar(name="pos", vals=[np.array([i, 2 * i]) for i in range(10)])
    fig, axs = multi_plot([outvar])
    labels = {ax.get_xlabel() for ax in axs} | {ax.get_ylabel() for ax in axs}
    assert {"pos [0]", "pos [1]"} <= labels
    plt.close("all")


def test_multi_plot_drops_nonscalar():
    import numpy as np
    import matplotlib.pyplot as plt
    from monaco.mc_var import OutVar
    from monaco.mc_multi_plot import multi_plot

    x = OutVar(name="x", vals=list(range(10)))
    y = OutVar(name="y", vals=[i**2 for i in range(10)])
    v = OutVar(name="v", vals=[np.array([i, i]) for i in range(10)])

    with pytest.warns(UserWarning, match="Dropping 1 non-scalar variable"):
        fig, axs = multi_plot([x, y, v])
    axs = np.ravel(axs)
    assert len(axs) == 4
    labels = {ax.get_xlabel() for ax in axs} | {ax.get_ylabel() for ax in axs}
    assert {"x", "y"} <= labels
    assert "v" not in labels

    with pytest.warns(UserWarning, match="Dropping 2"):
        with pytest.raises(ValueError, match="at least two scalar variables"):
            multi_plot([x, v, v])
    plt.close("all")


### Plot Testing ###
def plot_testing(show=False):
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from monaco.mc_var import InVar
    from monaco.mc_multi_plot import multi_plot, multi_plot_grid_rect
    from monaco.mc_enums import SampleMethod

    plt.close("all")

    generator = np.random.RandomState(74494861)
    invarseeds = generator.randint(0, 2**31 - 1, size=10)

    invars = dict()
    invars["norm1"] = InVar(
        "norm1",
        ndraws=1000,
        dist=norm,
        distkwargs={"loc": 1, "scale": 5},
        seed=invarseeds[0],
        samplemethod=SampleMethod.RANDOM,
    )
    invars["norm2"] = InVar(
        "norm2",
        ndraws=1000,
        dist=norm,
        distkwargs={"loc": 10, "scale": 4},
        seed=invarseeds[1],
        samplemethod=SampleMethod.RANDOM,
    )
    invars["norm3"] = InVar(
        "norm3",
        ndraws=1000,
        dist=norm,
        distkwargs={"loc": 5, "scale": 2},
        seed=invarseeds[3],
        samplemethod=SampleMethod.RANDOM,
    )

    fig, axs = multi_plot(
        [invars["norm1"], invars["norm2"]],
        highlight_cases=range(10, 30),
        rug_plot=True,
        cov_plot=True,
        cov_p=0.95,
        title="test",
    )  # multi_plot_2d_scatter_hist

    fig, axs = multi_plot(
        [invars["norm1"], invars["norm2"], invars["norm3"]],
        highlight_cases=range(10, 30),
        rug_plot=True,
        cov_plot=True,
        cov_p=0.95,
        title="test",
    )  # multi_plot_grid_tri

    fig, axs = multi_plot_grid_rect(
        [invars["norm1"], invars["norm2"]],
        invars["norm3"],
        highlight_cases=range(10, 30),
        rug_plot=True,
        cov_plot=True,
        cov_p=0.95,
        title="test",
    )  # multi_plot_grid_rect

    if show:
        plt.show(block=True)


if __name__ == "__main__":
    plot_testing(show=True)
