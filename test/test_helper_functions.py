# test_helper_functions.py

import pytest
from collections.abc import Iterable
import pandas as pd
from Monaco.helper_functions import next_power_of_2, hash_str_repeatable, is_num, length, get_iterable, slice_by_index

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
def test_get_iterable(val,ans):
    assert isinstance(get_iterable(val), Iterable) == ans


data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
@pytest.mark.parametrize("indices,ans", [
    (    [],    []),
    (     0,   [0]),
    ( [3,5], [3,5]),
])
def test_slice_by_index(indices,ans):
    assert slice_by_index(data,indices) == ans


### Inline Testing ###
# Can run here or copy into bottom of main file
def inline_testing():
    print(next_power_of_2(0))  # Expected: 0
    print(next_power_of_2(5))  # Expected: 8
    
    print(hash_str_repeatable('test'))  # Expected 12472987081885563334685425079619105233668272366527481043458243581788592708023738622989151050329990942934984448616851791396966092833116143876347600403212543
    
    print([is_num(True), is_num(0), is_num(1.0), is_num('1')])  # Expected: [False, True, True, False]
    
    print([length(x) for x in (True, 0, 1.0, '1')])   # Expected: [1, 1, 1, 1]
    print([length(x) for x in ([], (), dict(), {} )]) # Expected: [0, 0, 0, 0]
    print(length([1, 2, 3]))                          # Expected: 3
    print(length(None))                               # Expected: None
    
    nvals = 3
    dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
    df = pd.DataFrame({'vals1': range(nvals), 'vals2': range(nvals)}, index = dates)
    print([isinstance(get_iterable(x), Iterable) for x in (None, 0, (0,1), df, df['vals1'])])  # Expected: [True, True, True, True, True]
    
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    print(slice_by_index(data, []))      # Expected: []
    print(slice_by_index(data, 0))       # Expected: [0]
    print(slice_by_index(data, [3, 5]))  # Expected: [3, 5]

if __name__ == '__main__':
    inline_testing()    
