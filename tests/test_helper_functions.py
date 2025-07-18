# test_helper_functions.py

import pytest
import time
import numpy as np
from monaco.helper_functions import (next_power_of_2, hash_str_repeatable, is_num,
                                     length, get_list, slice_by_index, empty_list,
                                     flatten, get_cases, timeit, hashable_val_vectorized)

# Only test with pandas if installed
try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.mark.parametrize("num, ans", [
    (0, 0),
    (5, 8),
])
def test_next_power_of_2(num, ans):
    assert next_power_of_2(num) == ans


def test_hash_str_repeatable():
    assert hash_str_repeatable('test') == 12472987081885563334685425079619105233668272366527481043458243581788592708023738622989151050329990942934984448616851791396966092833116143876347600403212543  # noqa: E501


@pytest.mark.parametrize("val, ans", [
    ( True, False),
    (    0, True),
    (  1.0, True),
    (  '1', False),
    ([1, ], False),
])
def test_is_num(val, ans):
    assert is_num(val) == ans


@pytest.mark.parametrize("val, ans", [
    (     True, 1),
    (        0, 1),
    (      1.0, 1),
    (      '1', 1),
    (       [], 0),
    (       (), 0),
    (       {}, 0),
    (   dict(), 0),
    ([1, 2, 3], 3),
    (None, None),
])
def test_length(val, ans):
    assert length(val) == ans


@pytest.mark.parametrize("ncases, cases, ans", [
    (3,   None, (0, 1, 2)),
    (3,      1, (1,)),
    (3, (1, 2), (1, 2)),
    (3,     [], ()),
])
def test_get_cases(ncases, cases, ans):
    assert get_cases(ncases=ncases, cases=cases) == pytest.approx(ans)


@pytest.mark.parametrize("val, ans", [
    (       None, []),
    (         [], []),
    (          0, [0, ]),
    (     (0, 1), [0, 1]),
    (np.array(0), [0, ]),
    (     'abcd', ['abcd'])
])
def test_get_list(val, ans):
    assert all(get_list(get_list(val) == ans))


@pytest.mark.skipif(pd is None, reason="Requires the pandas library")
def test_get_list_pandas():
    nvals = 3
    dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
    df = pd.DataFrame({'vals1': [0, 1, 2], 'vals2': [3, 4, 5]}, index=dates)

    assert all(get_list(get_list(df) == [df]))
    assert all(get_list(get_list(df['vals2']) == [3, 4, 5]))
    assert all(get_list(get_list(df.index) == dates))


data1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
data2 = np.array(data1)
@pytest.mark.parametrize("indices, ans", [
    (     [],     []),
    (      0,    [0]),
    ( [3, 5], [3, 5]),
])
def test_slice_by_index(indices, ans):
    assert slice_by_index(data1, indices) == ans
    assert slice_by_index(data2, indices) == ans


def test_empty_list():
    assert empty_list() == []


def test_flatten():
    assert flatten(['test', [0, [1, [2], [3, 4]]], [[{(5)}]], 6, np.array([7, 8])]) \
           == ['test', 0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_timeit(capsys):
    @timeit
    def test_function(x, y=1):
        time.sleep(0.001)
        return x + y

    result = test_function(5, y=3)
    assert result == 8

    captured = capsys.readouterr()
    assert '"test_function" took' in captured.out
    assert 'ms to execute' in captured.out


def test_hashable_val_vectorized_base_path():
    vals = [1, 2, 3, 4, 5]
    result = hashable_val_vectorized(vals)
    expected = [1, 2, 3, 4, 5]
    assert result == expected


def test_hashable_val_vectorized_fallback_path():
    vals = [[1, 2], 'string', 42]
    result = hashable_val_vectorized(vals)
    expected = ['[1, 2]', 'string', 42]
    assert result == expected
