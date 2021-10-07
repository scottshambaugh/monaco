# test_MCVal.py

import pytest
import numpy as np
import pandas as pd
from monaco.MCVal import MCInVal, MCOutVal


def test_mcinval():
    from scipy.stats import norm
    inval = MCInVal(name='test', ncase=1, pct=0.5, num=0, dist=norm, isnom=True)
    assert inval.val == 0
    
    inval = MCInVal(name='test', ncase=1, pct=0.5, num=0, nummap={0:'a'}, dist=norm, isnom=True)
    assert inval.val == 'a'



@pytest.fixture
def mcoutval_3_by_2():
    return MCOutVal(name='test', ncase=1, val=[[0,0],[0,0],[0,0]], isnom=True)

def test_mcoutval(mcoutval_3_by_2):
    assert mcoutval_3_by_2.size == (3, 2)
    assert np.array_equal(mcoutval_3_by_2.val, [[0, 0], [0, 0], [0, 0]])

def test_mcoutval_split(mcoutval_3_by_2):
    outvalssplit = mcoutval_3_by_2.split()
    assert outvalssplit['test [0]'].val == [0, 0]

def test_mcoutval_valmap_str():
    outval = MCOutVal(name='test', ncase=1, val=[['a','a'],['b','b'],['a','b']], isnom=True)
    assert outval.valmap == {'a': 0, 'b': 1}
    assert np.array_equal(outval.num, [[0, 0], [1, 1], [0, 1]])

def test_mcoutval_valmap_bool():
    outval = MCOutVal(name='test', ncase=1, val=[True, False], valmap={True:2, False:1})
    assert outval.num == [2, 1]

def test_mcoutval_valmap_bool_default():
    outval = MCOutVal(name='test', ncase=1, val=[True, False])
    assert outval.num == [1, 0]


def test_mcoutval_dataframe():
    nvals = 3
    dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
    df = pd.DataFrame({'vals1': range(nvals), 'vals2': range(nvals)}, index = dates)
    mcoutval1 = MCOutVal(name='test1', ncase=1, val=df['vals1'], isnom=True)
    assert all(mcoutval1.num == [0, 1, 2])
    
    mcoutval2 = MCOutVal(name='test2', ncase=1, val=df.index, isnom=True)
    assert mcoutval2.val[0] == np.datetime64('2020-01-01T00:00:00.000000000')


### Inline Testing ###
# Can run here or copy into bottom of main file
def inline_testing():
    from scipy.stats import norm
    a = MCInVal(name='TestA', ncase=1, pct=0.5, num=0, dist=norm, isnom=True)
    print(a.val) # expected: 0
    b = MCInVal(name='TestB', ncase=1, pct=0.5, num=0, nummap={0:'a'}, dist=norm, isnom=True)
    print(b.val) # expected: a
    c = MCOutVal(name='TestC', ncase=1, val=[[0,0],[0,0],[0,0]], isnom=True)
    print(c.size) # expected: (3, 2)
    print(c.val) # expected: [[0, 0], [0, 0], [0, 0]]
    csplit = c.split()
    print(csplit['TestC [0]'].val) # expected: [0, 0]
    d = MCOutVal(name='TestD', ncase=1, val=[['a','a'],['b','b'],['a','b']], isnom=True)
    print(d.valmap) # expected: {'a': 0, 'b': 1}
    print(d.num) # expected: [[0, 0], [1, 1], [0, 1]]
    e = MCOutVal(name='TestE', ncase=1, val=[True, False], valmap={True:2, False:1})
    print(e.val) # expected: [True, False]
    print(e.num) # expected: [2, 1]
    f = MCOutVal(name='TestF', ncase=1, val=[True, False])
    print(f.val) # expected: [True, False]
    print(f.num) # expected: [1, 0]
    
    nvals = 3
    dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
    df = pd.DataFrame({'vals1': range(nvals), 'vals2': range(nvals)}, index = dates)
    g = MCOutVal(name='TestG', ncase=1, val=df['vals1'], isnom=True)
    print(g.num) # expected: [0 1 2]
    h = MCOutVal(name='TestH', ncase=1, val=df.index, isnom=True)
    print(h.val) # expected: ['2020-01-01T00:00:00.000000000' '2021-01-01T00:00:00.000000000' '2022-01-01T00:00:00.000000000']


if __name__ == '__main__':
    inline_testing()
