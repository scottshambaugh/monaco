# test_order_statistics.py

import pytest
from monaco.order_statistics import (order_stat_TI_n, order_stat_TI_k,
                                     order_stat_TI_c, order_stat_TI_p,
                                     order_stat_P_n, order_stat_P_k,
                                     order_stat_P_c, order_stat_var_check)
from monaco.mc_enums import StatBound

'''
Reference:
Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A Guide for
    Practitioners." Germany, Wiley, 1991.
'''

# Order Statistic Tolerance Interval Functions
def test_order_stat_TI_n():
    assert order_stat_TI_n(k=2, p=0.99, c=0.90, bound=StatBound.TWOSIDED) == 667

def test_order_stat_TI_p():
    assert order_stat_TI_p(n=667, k=2, c=0.90, bound=StatBound.TWOSIDED) == pytest.approx(0.9900114)

def test_order_stat_TI_c():
    assert order_stat_TI_c(n=667, k=2, p=0.99, bound=StatBound.TWOSIDED) == pytest.approx(0.9004787)

def test_order_stat_TI_k():
    assert order_stat_TI_k(n=667, p=0.99, c=0.90, bound=StatBound.TWOSIDED) == 2

def test_order_stat_TI_k_warning():
    with pytest.raises(ValueError, match='is too small'):
        _ = order_stat_TI_k(n=20, p=0.99, c=0.90, bound=StatBound.TWOSIDED)

# Order Statistic Percentile Functions
@pytest.mark.parametrize("n, k, P, bound, ans", [
    (1000,  3, 0.01, StatBound.TWOSIDED      , 0.7366882),  # Ref. Table A.15a
    (1000, 11, 0.95, StatBound.ONESIDED_UPPER, 0.9566518),  # Ref. Table A.16, 39+11=50
    (1000, 11, 0.05, StatBound.ONESIDED_LOWER, 0.9566518),  # Ref. Table A.16, 39+11=50
])
def test_order_stat_P_c(n, k, P, bound, ans):
    assert order_stat_P_c(n=n, k=k, P=P, bound=bound) == pytest.approx(ans)


@pytest.mark.parametrize("n, c, P, bound, ans", [
    (100, 0.95 , 0.50, StatBound.TWOSIDED      , 10),  # Ref. Table A.15g
    (100, 0.95 , 0.90, StatBound.ONESIDED_UPPER,  5),  # Ref. Table A.16
    (100, 0.95 , 0.10, StatBound.ONESIDED_LOWER,  5),  # Ref. Table A.16
])
def test_order_stat_P_k(n, c, P, bound, ans):
    assert order_stat_P_k(n=n, c=c, P=P, bound=bound) == ans

def test_order_stat_P_k_warning():
    with pytest.raises(ValueError, match='is too small'):
        order_stat_P_k(n=10, c=0.999, P=0.05, bound=StatBound.ONESIDED_LOWER)

@pytest.mark.parametrize("k, c, P, bound, ans", [
    (10, 0.95  , 0.50, StatBound.TWOSIDED      ,  108),  # Ref. Table A.15g (conservative)
    (11, 0.9566, 0.95, StatBound.ONESIDED_UPPER, 1018),  # Ref. Table A.16 (conservative)
    (11, 0.9566, 0.05, StatBound.ONESIDED_LOWER, 1018),  # Ref. Table A.16 (conservative)
])
def test_order_stat_P_n(k, c, P, bound, ans):
    assert order_stat_P_n(k=k, c=c, P=P, bound=bound) == pytest.approx(ans)


@pytest.mark.parametrize("kwargs, match", [
    ({'n': 0}, 'n=0 must be >= 1'),
    ({'l': -1}, 'l=-1 must be >= 0'),
    ({'u': 13, 'n': 10}, 'u=13 must be >= 11'),
    ({'u': 5, 'l': 6}, 'u=5 must be >= l=6'),
    ({'p': 0}, 'p=0 must be in the range 0 < p < 1'),
    ({'p': 1}, 'p=1 must be in the range 0 < p < 1'),
    ({'k': 0}, 'k=0 must be >= 1'),
    ({'c': 0}, 'c=0 must be in the range 0 < c < 1'),
    ({'c': 1}, 'c=1 must be in the range 0 < c < 1'),
    ({'nmax': 0}, 'nmax=0 must be >= 1'),
])
def test_order_stat_var_check_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        order_stat_var_check(**kwargs)
