# test_mc_var.py

import pytest
import subprocess
import sys
import numpy as np
from enum import Enum
from scipy.stats import rv_discrete, randint, norm, lognorm
from monaco.mc_var import InVar, OutVar
from monaco.mc_enums import SampleMethod
import matplotlib.pyplot as plt

generator = np.random.RandomState(74494861)
invarseeds = generator.randint(0, 2**31 - 1, size=10)

xk = np.array([1, 5, 6])
pk = np.ones(len(xk)) / len(xk)
custom_dist = rv_discrete(name="custom", values=(xk, pk))
lognorm_sigma, lognorm_mu = 1, 2


@pytest.fixture
def invar_norm_random():
    return InVar(
        "norm",
        ndraws=100,
        dist=norm,
        distkwargs={"loc": 10, "scale": 4},
        seed=invarseeds[1],
        samplemethod=SampleMethod.RANDOM,
        firstcaseismedian=False,
    )


@pytest.fixture
def invar_lognorm_random():
    return InVar(
        "lognorm",
        ndraws=100,
        dist=lognorm,
        distkwargs={"s": lognorm_sigma, "scale": np.exp(lognorm_mu)},
        seed=invarseeds[1],
        samplemethod=SampleMethod.RANDOM,
        firstcaseismedian=False,
    )


@pytest.fixture
def invar_custom_dist():
    return InVar(
        "custom",
        ndraws=100,
        dist=custom_dist,
        distkwargs=dict(),
        ninvar=1,
        samplemethod=SampleMethod.RANDOM,
        seed=invarseeds[2],
        firstcaseismedian=False,
    )


def test_invar_getitem(invar_norm_random):
    assert invar_norm_random[0].val == pytest.approx(14.186734703770963)


def test_invar_norm_sobol_warning():
    with pytest.warns(UserWarning, match="Infinite value"):
        InVar(
            "norm",
            ndraws=100,
            dist=norm,
            distkwargs={"loc": 10, "scale": 4},
            ninvar=1,
            samplemethod=SampleMethod.SOBOL,
            seed=invarseeds[1],
            firstcaseismedian=False,
        )


def test_invar_discrete():
    invar = InVar(
        "randint",
        ndraws=100,
        dist=randint,
        distkwargs={"low": 1, "high": 5},
        seed=invarseeds[0],
        samplemethod=SampleMethod.RANDOM,
    )
    assert invar.stats().mean == pytest.approx(2.7)


def test_invar_continuous(invar_norm_random):
    assert invar_norm_random.stats().mean == pytest.approx(9.8637355)


def test_invar_custom(invar_custom_dist):
    assert invar_custom_dist.stats().mean == pytest.approx(4.23)


def test_invar_addvarstat(invar_norm_random):
    invar_norm_random.addVarStat(
        stat="orderstatTI", statkwargs={"p": 0.75, "c": 0.95, "bound": "2-sided"}
    )
    assert invar_norm_random.varstats[0].vals == pytest.approx([4.2292074, 14.4507394])


def test_invar_getVal(invar_custom_dist):
    assert invar_custom_dist.getVal(0).val == 5


def test_invar_getMean(invar_lognorm_random):
    assert invar_lognorm_random.getDistMean() == pytest.approx(
        np.exp(lognorm_mu + lognorm_sigma**2 / 2)
    )


@pytest.mark.parametrize(
    "dist, distkwargs, mean",
    [
        (norm, {"loc": 10, "scale": 4}, 10),
        (randint, {"low": 1, "high": 4}, 2),
        (custom_dist, dict(), 5),
    ],
)
def test_invar_getDistMean(dist, distkwargs, mean):
    invar = InVar(
        "test",
        ndraws=10,
        dist=dist,
        distkwargs=distkwargs,
        seed=invarseeds[0],
        samplemethod=SampleMethod.RANDOM,
    )
    assert invar.getDistMean() == pytest.approx(mean)


def test_invar_getMedian(invar_lognorm_random):
    assert invar_lognorm_random.getDistMedian() == pytest.approx(np.exp(lognorm_mu))


def test_invar_nummap():
    invar = InVar(
        "map",
        ndraws=10,
        dist=custom_dist,
        distkwargs=dict(),
        ninvar=1,
        nummap={1: "a", 5: "e", 6: "f"},
        samplemethod=SampleMethod.RANDOM,
        seed=invarseeds[3],
        firstcaseismedian=False,
    )
    assert invar.vals == ["f", "e", "f", "f", "a", "e", "e", "a", "e", "e"]


def test_invar_unseeded():
    a = InVar("a", ndraws=3, dist=norm, samplemethod=SampleMethod.RANDOM)
    b = InVar("b", ndraws=3, dist=norm, samplemethod=SampleMethod.RANDOM)
    assert a.seed != b.seed
    assert a.nums != b.nums


def test_invar_unseeded_repeatable():
    code = "from monaco.helper_functions import get_seed; print(get_seed(None), get_seed(None))"
    outputs = {subprocess.check_output([sys.executable, "-c", code]) for _ in range(2)}
    assert len(outputs) == 1


def test_invar_custom_vals_unhashable():
    invar = InVar("custom", ndraws=3, vals=[{"a": 1}, {"b": 2}, {"a": 1}], samplemethod=None)
    assert invar.getVal(0).val == {"a": 1}
    assert invar.getVal(1).val == {"b": 2}
    assert invar.nums == [0, 1, 0]


class Mode(Enum):
    LOW = 1
    HIGH = 2


def test_invar_custom_vals_unorderable():
    invar = InVar("custom", ndraws=4, vals=["b", 2, Mode.HIGH, Mode.LOW], samplemethod=None)
    assert invar.nummap == {0: 2, 1: "b", 2: Mode.LOW, 3: Mode.HIGH}
    assert invar.nums == [1, 0, 3, 2]


def test_invar_custom_vals():
    eps = np.finfo(float).eps
    invar = InVar(
        "custom",
        ndraws=3,
        vals=[1, 2, 3],
        ninvar=1,
        samplemethod=None,
        seed=invarseeds[3],
        firstcaseismedian=False,
    )
    assert invar.pcts == [eps, 0.5, 1.0 - eps]
    assert invar.nums == [1, 2, 3]
    assert invar.vals == [1, 2, 3]

    # With nummap
    nummap = {2: "a", 3: "b", 4: "c", 5: "d"}
    invar = InVar(
        "custom",
        ndraws=3,
        vals=["a", "b", "c", "d"],
        ninvar=1,
        nummap=nummap,
        samplemethod=None,
        seed=invarseeds[3],
        firstcaseismedian=True,
    )
    assert invar.pcts == [0.5, eps, 0.5 + eps, 1.0 - eps]
    assert invar.nums == [2, 3, 4, 5]
    assert invar.vals == ["a", "b", "c", "d"]

    # Without nummap, and also test custom pcts
    invar = InVar(
        "custom",
        ndraws=3,
        vals=["b", "a", "c", "d"],
        pcts=[0.5, 0.2, 0.3, 0.4],
        ninvar=1,
        samplemethod=None,
        seed=invarseeds[3],
        firstcaseismedian=True,
    )
    assert invar.vals == ["b", "a", "c", "d"]
    assert invar.nums == [1, 0, 2, 3]
    assert invar.pcts == [0.5, 0.2, 0.3, 0.4]


def test_invar_custom_vals_arrays():
    vals = [np.full((2, 2), 1.0), np.arange(3), np.ones(1)]
    invar = InVar("custom", ndraws=3, vals=vals, samplemethod=None, firstcaseismedian=False)
    for i, val in enumerate(vals):
        inval = invar.getVal(i).val
        assert np.shares_memory(inval, val)
        assert np.array_equal(inval, val)
        assert not inval.flags.writeable
        assert val.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        invar.getVal(1).val *= 2
    assert np.array_equal(vals[1], np.arange(3))

    invar = InVar(
        "custom", ndraws=2, vals=[[1, 2], [3, 4]], samplemethod=None, firstcaseismedian=False
    )
    assert isinstance(invar.getVal(1).val, np.ndarray)
    assert np.array_equal(invar.getVal(1).val, [3, 4])
    assert not invar.isscalar
    assert invar.maxdim == 1
    assert not invar.getVal(1).isscalar
    assert invar.getVal(1).shape == (2,)


@pytest.mark.parametrize(
    "vals",
    [
        [np.array([True, False]), np.array([False, False]), np.array([True, False])],
        [np.array(["a", "b"]), np.array(["c"]), np.array(["a", "b"])],
        [np.array([1, 2], dtype="m8[s]"), np.array([3], dtype="m8[s]"), np.array([1, 2], "m8[s]")],
        [[[1, 2], [3]], [[4]], [[1, 2], [3]]],
        [np.zeros(2000, bool), np.r_[np.zeros(1999, bool), True], np.zeros(2000, bool)],
    ],
)
def test_invar_custom_vals_nonnumeric_arrays(vals):
    # Each array is mapped to a number as a whole, and maps back to the original object
    invar = InVar("custom", ndraws=3, vals=vals, samplemethod=None, firstcaseismedian=False)
    assert invar.isscalar
    assert len(invar.nummap) == 2
    assert invar.nums[0] == invar.nums[2] != invar.nums[1]
    for i, val in enumerate(vals):
        assert invar.getVal(i).val is invar.nummap[invar.nums[i]]
        assert np.array_equal(
            np.asarray(invar.getVal(i).val, dtype=object), np.asarray(val, dtype=object)
        )


def test_invar_custom_vals_empty_arrays():
    invar = InVar(
        "custom", ndraws=2, vals=[np.array([])] * 2, samplemethod=None, firstcaseismedian=False
    )
    assert invar.nummap is None
    assert invar.getVal(1).val.shape == (0,)


def test_invar_custom_vals_strings_stay_str():
    invar = InVar(
        "custom", ndraws=3, vals=["b", "a", "b"], samplemethod=None, firstcaseismedian=False
    )
    assert type(invar.getVal(0).val) is str


def test_invar_custom_vals_errors():
    with pytest.raises(ValueError, match="Either 'dist' or 'vals' must be provided"):
        InVar(
            "custom",
            ndraws=3,
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=False,
        )

    with pytest.raises(ValueError, match="Cannot provide both 'dist' and 'vals'"):
        InVar(
            "custom",
            dist=custom_dist,
            distkwargs=dict(),
            ndraws=3,
            vals=[1, 2, 3],
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=False,
        )

    with pytest.raises(ValueError, match=r"Length of 'vals' \(3\) must match ncases \(10\)"):
        InVar(
            "custom",
            ndraws=10,
            vals=[1, 2, 3],
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=False,
        )

    with pytest.raises(ValueError, match=r"Length of 'vals' \(3\) must match ncases \(11\)"):
        InVar(
            "custom",
            ndraws=10,
            vals=[1, 2, 3],
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=True,
        )


def test_invar_custom_pcts():
    invar = InVar(
        "custom",
        ndraws=3,
        dist=randint,
        distkwargs={"low": 1, "high": 5},
        pcts=[0.01, 0.5, 0.99],
        ninvar=1,
        samplemethod=None,
        seed=invarseeds[3],
        firstcaseismedian=False,
    )
    assert invar.pcts == [0.01, 0.5, 0.99]
    assert invar.nums == [1.0, 2.0, 4.0]
    assert invar.vals == [1.0, 2.0, 4.0]


def test_invar_custom_pcts_errors():
    with pytest.raises(ValueError, match=r"Length of 'pcts' \(3\) must match "):
        InVar(
            "custom",
            ndraws=5,
            dist=randint,
            distkwargs={"low": 1, "high": 5},
            pcts=[0.01, 0.5, 0.99],
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=False,
        )

    with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
        InVar(
            "custom",
            ndraws=3,
            dist=randint,
            distkwargs={"low": 1, "high": 5},
            pcts=[0.01, 0.5, 1.01],
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=False,
        )

    with pytest.raises(
        ValueError, match="If firstcaseismedian is True, then the first " + "percentile must be 0.5"
    ):
        InVar(
            "custom",
            ndraws=3,
            dist=randint,
            distkwargs={"low": 1, "high": 5},
            pcts=[0.01, 0.5, 0.75, 0.99],
            ninvar=1,
            samplemethod=None,
            seed=invarseeds[3],
            firstcaseismedian=True,
        )


def test_outvar():
    outvar = OutVar("test", [1, 0, 2, 2], firstcaseismedian=True)
    assert outvar.getVal(1).val == 0
    assert outvar.stats().mean == pytest.approx(1.25)
    assert outvar.getMedianVal().val == 1


@pytest.mark.parametrize(
    "vals, maxdim",
    [
        (
            [
                None,
            ],
            0,
        ),
        (
            [
                0,
            ],
            0,
        ),
        ([[0]], 1),
        ([[0, 0]], 1),
        ([[[0, 0], [0, 0], [0, 0]]], 2),
        ([[[0]], [[0, 0]], [[]]], 2),
    ],
)
def test_outvar_genMaxDim(vals, maxdim):
    outvar = OutVar("test", vals)
    assert outvar.maxdim == maxdim


def test_outvar_extractValMap():
    outvar = OutVar("test", ["a", "b", "c", ["b"]], firstcaseismedian=True)
    assert outvar.valmap == {"a": 0, "b": 1, "c": 2}


def test_outvar_valmap_unorderable():
    outvar = OutVar("test", vals=[Mode.HIGH, Mode.LOW, Mode.HIGH])
    assert outvar.valmap == {Mode.LOW: 0, Mode.HIGH: 1}
    assert np.array_equal(outvar.nums, [1, 0, 1])

    outvar = OutVar("test", vals=["a", 1, None])
    assert outvar.valmap == {None: 0, 1: 1, "a": 2}
    assert np.array_equal(outvar.nums, [2, 1, 0])

    outvar = OutVar("test", vals=[np.array(["a", "b"]), [1, None]])
    assert outvar.valmap == {None: 0, 1: 1, "a": 2, "b": 3}
    assert np.array_equal(outvar.nums[0], [2, 3])
    assert np.array_equal(outvar.nums[1], [1, 0])

    outvar = OutVar("test", [1, 2, None, 4], firstcaseismedian=True)
    assert outvar.valmap is None
    assert np.array_equal(outvar.nums, [1, 2, np.nan, 4], equal_nan=True)

    outvar = OutVar("test", [np.array([1.5, 2.5]), [None, 3], None])
    assert outvar.valmap is None
    assert np.array_equal(outvar.nums[1], [np.nan, 3], equal_nan=True)


def test_outvar_valmap_nan_nat():
    outvar = OutVar("test", vals=["a", float("nan"), "a", np.float64("nan")])
    assert len(outvar.valmap) == 2
    assert np.array_equal(outvar.nums, [1, 0, 1, 0])

    outvar = OutVar("test", vals=[np.datetime64("2020-01-01"), np.datetime64("NaT", "D")])
    assert np.array_equal(outvar.nums, [1, 0])

    outvar = OutVar("test", vals=[np.array(["a", "b"]), np.array([np.nan, 1.0])])
    assert np.array_equal(outvar.nums[1], [0, 1])


@pytest.fixture
def outvars_split():
    outvars = dict()
    v = np.array([[1, 1], [2, 2], [3, 3]])
    outvars["test"] = OutVar("test", [v, v, v, v, v])
    outvars["test"].addVarStat(
        stat="orderstatTI", statkwargs={"p": 0.33, "c": 0.50, "bound": "1-sided"}
    )
    outvars.update(outvars["test"].split())
    return outvars


def test_outvar_split_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=[10, 20, 30])
    outvar = OutVar("df", [df, df])
    outvars = outvar.split()
    assert list(outvars) == ["df [0]", "df [1]", "df [2]"]
    assert np.array_equal(outvars["df [1]"].vals[0], [3, 4])

    series = pd.Series([1, 2, 3], index=[10, 20, 30])
    outvars = OutVar("s", [series, series]).split()
    assert outvars["s [0]"].vals == [1, 1]


def test_outvar_split(outvars_split):
    assert np.array_equal(outvars_split["test [0]"].nums, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    assert np.array_equal(outvars_split["test [1]"].nums, [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])


def test_outvar_split_orderstat(outvars_split):
    assert outvars_split["test [0]"].varstats[0].vals == pytest.approx([1, 1])


# Does not test the plot appearances, but does check that the codepaths can run
def test_gen_plots():
    plot_testing(show=False)
    assert True


def plot_testing(show=False):
    from scipy.stats import norm

    invar = InVar(
        "norm",
        ndraws=100,
        dist=norm,
        distkwargs={"loc": 10, "scale": 4},
        seed=invarseeds[1],
        samplemethod=SampleMethod.RANDOM,
        firstcaseismedian=False,
    )

    fig, ax = invar.plot()
    if show:
        plt.show(block=True)


if __name__ == "__main__":
    plot_testing(show=True)


def test_outvar_extractvalmap_numeric_fast_path():
    assert OutVar("test", vals=[1.5, 2, np.float64(3), np.array([1, 2])]).valmap is None
    assert OutVar("test", vals=[True, False]).valmap == {False: 0, True: 1}
    outvar = OutVar("test", vals=[["b", "c"], "a", ["c"]])
    assert outvar.valmap == {"a": 0, "b": 1, "c": 2}
