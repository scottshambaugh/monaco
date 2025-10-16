# test_mc_val.py

import pytest
import numpy as np
from monaco.mc_val import InVal, OutVal

# Only test with pandas if installed
try:
    import pandas as pd
except ImportError:
    pd = None


def test_inval():
    from scipy.stats import norm

    inval = InVal(name="test", ncase=1, pct=0.5, num=0, dist=norm, ismedian=True)
    assert inval.val == 0

    inval = InVal(name="test", ncase=1, pct=0.5, num=0, nummap={0: "a"}, dist=norm, ismedian=True)
    assert inval.val == "a"


@pytest.fixture
def outval_3_by_2():
    return OutVal(name="test", ncase=1, val=[[0, 0], [0, 0], [0, 0]], ismedian=True)


def test_outval(outval_3_by_2):
    assert outval_3_by_2.shape == (3, 2)
    assert np.array_equal(outval_3_by_2.val, [[0, 0], [0, 0], [0, 0]])


def test_outval_split(outval_3_by_2):
    outvalssplit = outval_3_by_2.split()
    assert outvalssplit["test [0]"].val == [0, 0]


def test_outval_valmap_str():
    outval = OutVal(name="test", ncase=1, val=[["a", "a"], ["b", "b"], ["a", "a"]], ismedian=True)
    assert outval.valmap == {"a": 0, "b": 1}
    assert np.array_equal(outval.num, [[0, 0], [1, 1], [0, 0]])


def test_outval_valmap_bool():
    outval = OutVal(name="test", ncase=1, val=[True, False], valmap={True: 2, False: 1})
    assert all(outval.num == [2, 1])


def test_outval_valmap_bool_default():
    outval = OutVal(name="test", ncase=1, val=[True, False])
    assert all(outval.num == [1, 0])


@pytest.mark.skipif(pd is None, reason="Requires the pandas library")
def test_outval_dataframe():
    nvals = 3
    dates = pd.date_range(start="2020-01-01", periods=nvals, freq="YS")
    df = pd.DataFrame({"vals1": range(nvals), "vals2": range(nvals)}, index=dates)
    outval1 = OutVal(name="test1", ncase=1, val=df["vals1"], ismedian=True)
    assert all(outval1.num == [0, 1, 2])

    outval2 = OutVal(name="test2", ncase=1, val=df.index, ismedian=True)
    assert outval2.val[0] == np.datetime64("2020-01-01T00:00:00.000000000")
