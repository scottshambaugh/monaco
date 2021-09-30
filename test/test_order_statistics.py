# test_order_statistics.py

import pytest
from Monaco.order_statistics import order_stat_TI_n, order_stat_TI_k, order_stat_TI_c, order_stat_TI_p
from Monaco.order_statistics import order_stat_P_n, order_stat_P_k, order_stat_P_c
from Monaco.MCEnums import StatBound

'''
Reference:
Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A Guide for 
    Practitioners." Germany, Wiley, 1991.
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
    with pytest.warns(UserWarning, match='is too small'):
        orderstat = order_stat_TI_k(n=20, p=0.99, c=0.90, bound=StatBound.TWOSIDED)
    assert orderstat is None

# Order Statistic Percentile Functions
@pytest.mark.parametrize("n,k,P,bound,ans", [
    (1000,  3, 0.01, StatBound.TWOSIDED      , 0.7366882), # Ref. Table A.15a
    (1000, 11, 0.95, StatBound.ONESIDED_UPPER, 0.9566518), # Ref. Table A.16, 39+11=50
    (1000, 11, 0.05, StatBound.ONESIDED_LOWER, 0.9566518), # Ref. Table A.16, 39+11=50
])
def test_order_stat_P_c(n,k,P,bound,ans):
    assert order_stat_P_c(n=n, k=k, P=P, bound=bound) == pytest.approx(ans)


@pytest.mark.parametrize("n,c,P,bound,ans", [
    (100, 0.95 , 0.50, StatBound.TWOSIDED      , 10), # Ref. Table A.15g
    (100, 0.95 , 0.90, StatBound.ONESIDED_UPPER,  5), # Ref. Table A.16
    (100, 0.95 , 0.10, StatBound.ONESIDED_LOWER,  5), # Ref. Table A.16
])
def test_order_stat_P_k(n,c,P,bound,ans):
    assert order_stat_P_k(n=n, c=c, P=P, bound=bound) == ans

def test_order_stat_P_k_warning():
    with pytest.warns(UserWarning, match='is too small'):
        orderstat = order_stat_P_k(n=10, c=0.999, P=0.05, bound=StatBound.ONESIDED_LOWER)
    assert orderstat is None

@pytest.mark.parametrize("k,c,P,bound,ans", [
    (10, 0.95  , 0.50, StatBound.TWOSIDED      ,  108), # Ref. Table A.15g (conservative)
    (11, 0.9566, 0.95, StatBound.ONESIDED_UPPER, 1018), # Ref. Table A.16 (conservative)
    (11, 0.9566, 0.05, StatBound.ONESIDED_LOWER, 1018), # Ref. Table A.16 (conservative)
])
def test_order_stat_P_n(k,c,P,bound,ans):
    assert order_stat_P_n(k=k, c=c, P=P, bound=bound) == pytest.approx(ans)


### Inline Testing ###
# Can run here or copy into bottom of main file
def inline_testing():
    print('TI Functions:')
    print(order_stat_TI_n(k=2,   p=0.99, c=0.90, bound=StatBound.TWOSIDED)) # expected: 667
    print(order_stat_TI_p(n=667, k=2,    c=0.90, bound=StatBound.TWOSIDED)) # expected: 0.99001
    print(order_stat_TI_c(n=667, k=2,    p=0.99, bound=StatBound.TWOSIDED)) # expected: 0.90047
    print(order_stat_TI_k(n=667, p=0.99, c=0.90, bound=StatBound.TWOSIDED)) # expected: 2
    print(order_stat_TI_k(n=20,  p=0.99, c=0.90, bound=StatBound.TWOSIDED)) # expected: Warning message, None
    print('P Functions:')
    print(order_stat_P_c(n=1000, k=3,      P=0.01, bound=StatBound.TWOSIDED))       # expected: 0.7367, Table A.15a
    print(order_stat_P_c(n=1000, k=11,     P=0.95, bound=StatBound.ONESIDED_UPPER)) # expected: 0.9566, Table A.16, 39+11=50
    print(order_stat_P_c(n=1000, k=11,     P=0.05, bound=StatBound.ONESIDED_LOWER)) # expected: 0.9566, Table A.16, 39+11=50
    print(order_stat_P_k(n=100,  c=0.95,   P=0.50, bound=StatBound.TWOSIDED))       # expected: 10, Table A.15g
    print(order_stat_P_k(n=100,  c=0.95,   P=0.90, bound=StatBound.ONESIDED_UPPER)) # expected: 5, Table A.16
    print(order_stat_P_k(n=100,  c=0.95,   P=0.10, bound=StatBound.ONESIDED_LOWER)) # expected: 5, Table A.16
    print(order_stat_P_k(n=10,   c=0.999,  P=0.05, bound=StatBound.ONESIDED_LOWER)) # expected: Warning message, None
    print(order_stat_P_n(k=10,   c=0.950,  P=0.50, bound=StatBound.TWOSIDED))       # expected: 108, Table A.15g (conservative)
    print(order_stat_P_n(k=11,   c=0.9566, P=0.95, bound=StatBound.ONESIDED_UPPER)) # expected: 1018, Table A.16 (conservative)
    print(order_stat_P_n(k=11,   c=0.9566, P=0.05, bound=StatBound.ONESIDED_LOWER)) # expected: 1018, Table A.16 (conservative)


if __name__ == '__main__':
    inline_testing()
