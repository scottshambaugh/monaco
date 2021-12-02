# test_MCVal.py

import pytest
import numpy as np
import pandas as pd
from monaco.MCVal import MCInVal, MCOutVal


def test_mcinval():
    from scipy.stats import norm
    inval = MCInVal(name='test', ncase=1, pct=0.5, num=0, dist=norm, ismedian=True)
    assert inval.val == 0

    inval = MCInVal(name='test', ncase=1, pct=0.5, num=0, nummap={0: 'a'}, dist=norm, ismedian=True)
    assert inval.val == 'a'



@pytest.fixture
def mcoutval_3_by_2():
    return MCOutVal(name='test', ncase=1, val=[[0, 0], [0, 0], [0, 0]], ismedian=True)

def test_mcoutval(mcoutval_3_by_2):
    assert mcoutval_3_by_2.size == (3, 2)
    assert np.array_equal(mcoutval_3_by_2.val, [[0, 0], [0, 0], [0, 0]])

def test_mcoutval_split(mcoutval_3_by_2):
    outvalssplit = mcoutval_3_by_2.split()
    assert outvalssplit['test [0]'].val == [0, 0]

def test_mcoutval_valmap_str():
    outval = MCOutVal(name='test', ncase=1, val=[['a', 'a'], ['b', 'b'], ['a', 'b']], ismedian=True)
    assert outval.valmap == {'a': 0, 'b': 1}
    assert np.array_equal(outval.num, [[0, 0], [1, 1], [0, 1]])

def test_mcoutval_valmap_bool():
    outval = MCOutVal(name='test', ncase=1, val=[True, False], valmap={True: 2, False: 1})
    assert outval.num == [2, 1]

def test_mcoutval_valmap_bool_default():
    outval = MCOutVal(name='test', ncase=1, val=[True, False])
    assert outval.num == [1, 0]


def test_mcoutval_dataframe():
    nvals = 3
    dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
    df = pd.DataFrame({'vals1': range(nvals), 'vals2': range(nvals)}, index=dates)
    mcoutval1 = MCOutVal(name='test1', ncase=1, val=df['vals1'], ismedian=True)
    assert all(mcoutval1.num == [0, 1, 2])

    mcoutval2 = MCOutVal(name='test2', ncase=1, val=df.index, ismedian=True)
    assert mcoutval2.val[0] == np.datetime64('2020-01-01T00:00:00.000000000')
