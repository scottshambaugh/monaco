# test_helper_functions.py

import pytest
from collections.abc import Iterable
import pandas as pd
from monaco.helper_functions import next_power_of_2, hash_str_repeatable, is_num, length, get_tuple, slice_by_index

@pytest.mark.parametrize("num,ans", [
    (0, 0),
    (5, 8),
])
def test_next_power_of_2(num,ans):
    assert next_power_of_2(num) == ans


def test_hash_str_repeatable():
    assert hash_str_repeatable('test') == 12472987081885563334685425079619105233668272366527481043458243581788592708023738622989151050329990942934984448616851791396966092833116143876347600403212543


@pytest.mark.parametrize("val,ans", [
    (True, False),
    (   0, True),
    ( 1.0, True),
    ( '1', False),
    ([1,], False),
])
def test_is_num(val,ans):
    assert is_num(val) == ans


@pytest.mark.parametrize("val,ans", [
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
def test_length(val,ans):
    assert length(val) == ans


nvals = 3
dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
df = pd.DataFrame({'vals1': range(nvals), 'vals2': range(nvals)}, index = dates)
@pytest.mark.parametrize("val,ans", [
    (        None, True),
    (           0, True),
    (       (0,1), True),
    (          df, True),
    ( df['vals1'], True),
])
def test_get_tuple(val,ans):
    assert isinstance(get_tuple(val), Iterable) == ans


data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
@pytest.mark.parametrize("indices,ans", [
    (    [],    []),
    (     0,   [0]),
    ( [3,5], [3,5]),
])
def test_slice_by_index(indices,ans):
    assert slice_by_index(data,indices) == ans
