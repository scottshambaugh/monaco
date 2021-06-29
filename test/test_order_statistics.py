# test_order_statistics.py

import pytest
import Monaco.order_statistics as os

'''
Reference:
Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A Guide for 
    Practitioners." Germany, Wiley, 1991.
'''

# Order Statistic Tolerance Interval Functions
def test_order_stat_TI_n():
    assert os.order_stat_TI_n(k=2, p=0.99, c=0.90, bound='2-sided') == 667

def test_order_stat_TI_p():
    assert os.order_stat_TI_p(n=667, k=2, c=0.90, bound='2-sided') == pytest.approx(0.9900114)

def test_order_stat_TI_c():
    assert os.order_stat_TI_c(n=667, k=2, p=0.99, bound='2-sided') == pytest.approx(0.9004787)

def test_order_stat_TI_k():
    assert os.order_stat_TI_k(n=667, p=0.99, c=0.90, bound='2-sided') == 2

def test_order_stat_TI_k_warning():
    with pytest.warns(UserWarning, match='is too small'):
        orderstat = os.order_stat_TI_k(n=20, p=0.99, c=0.90, bound='2-sided')
    assert orderstat is None

# Order Statistic Percentile Functions
@pytest.mark.parametrize("n,k,P,bound,ans", [
    (1000,  3, 0.01,       '2-sided', 0.7366882), # Ref. Table A.15a
    (1000, 11, 0.95, '1-sided upper', 0.9566518), # Ref. Table A.16, 39+11=50
    (1000, 11, 0.05, '1-sided lower', 0.9566518), # Ref. Table A.16, 39+11=50
])
def test_order_stat_P_c(n,k,P,bound,ans):
    assert os.order_stat_P_c(n=n, k=k, P=P, bound=bound) == pytest.approx(ans)


@pytest.mark.parametrize("n,c,P,bound,ans", [
    (100, 0.95 , 0.50,       '2-sided', 10), # Ref. Table A.15g
    (100, 0.95 , 0.90, '1-sided upper',  5), # Ref. Table A.16
    (100, 0.95 , 0.10, '1-sided lower',  5), # Ref. Table A.16
])
def test_order_stat_P_k(n,c,P,bound,ans):
    assert os.order_stat_P_k(n=n, c=c, P=P, bound=bound) == ans

def test_order_stat_P_k_warning():
    with pytest.warns(UserWarning, match='is too small'):
        orderstat = os.order_stat_P_k(n=10, c=0.999, P=0.05, bound='1-sided lower')
    assert orderstat is None

@pytest.mark.parametrize("k,c,P,bound,ans", [
    (10, 0.95  , 0.50,       '2-sided',  108), # Ref. Table A.15g (conservative)
    (11, 0.9566, 0.95, '1-sided upper', 1018), # Ref. Table A.16 (conservative)
    (11, 0.9566, 0.05, '1-sided lower', 1018), # Ref. Table A.16 (conservative)
])
def test_order_stat_P_n(k,c,P,bound,ans):
    assert os.order_stat_P_n(k=k, c=c, P=P, bound=bound) == pytest.approx(ans)


### Inline Testing ###
'''
if __name__ == '__main__':
    print(sig2pct(-3, bound='2-sided'), sig2pct(3, bound='1-sided')) # expected: -0.99730, 0.99865
    print(pct2sig(0.9973002, bound='2-sided'), pct2sig(0.0013499, bound='1-sided')) # expected: 3, -3
    print(conf_ellipsoid_sig2pct(3, df=1), conf_ellipsoid_sig2pct(3, df=2)) # expected: 0.99730, 0.98889
    print(conf_ellipsoid_pct2sig(0.9973002, df=1), conf_ellipsoid_pct2sig(0.988891, df=2)) # expected: 3.0, 3.0
    print('TI Functions:')
    print(order_stat_TI_n(k=2, p=0.99, c=0.90, bound='2-sided'))   # expected: 667
    print(order_stat_TI_p(n=667, k=2, c=0.90, bound='2-sided'))    # expected: 0.99001
    print(order_stat_TI_c(n=667, k=2, p=0.99, bound='2-sided'))    # expected: 0.90047
    print(order_stat_TI_k(n=667, p=0.99, c=0.90, bound='2-sided')) # expected: 2
    print(order_stat_TI_k(n=20, p=0.99, c=0.90, bound='2-sided'))  # expected: Warning message, None
    print('P Functions:')
    print(order_stat_P_c(n=1000, k=3, P=0.01, bound='2-sided'))          # expected: 0.7367, Table A.15a
    print(order_stat_P_c(n=1000, k=11, P=0.95, bound='1-sided upper'))   # expected: 0.9566, Table A.16, 39+11=50
    print(order_stat_P_c(n=1000, k=11, P=0.05, bound='1-sided lower'))   # expected: 0.9566, Table A.16, 39+11=50
    print(order_stat_P_k(n=100, c=0.95, P=0.50, bound='2-sided'))        # expected: 10, Table A.15g
    print(order_stat_P_k(n=100, c=0.95, P=0.90, bound='1-sided upper'))  # expected: 5, Table A.16
    print(order_stat_P_k(n=100, c=0.95, P=0.10, bound='1-sided lower'))  # expected: 5, Table A.16
    print(order_stat_P_k(n=10, c=0.999, P=0.05, bound='1-sided lower'))  # expected: Warning message, None
    print(order_stat_P_n(k=10, c=0.950, P=0.50, bound='2-sided'))        # expected: 108, Table A.15g (conservative)
    print(order_stat_P_n(k=11, c=0.9566, P=0.95, bound='1-sided upper')) # expected: 1018, Table A.16 (conservative)
    print(order_stat_P_n(k=11, c=0.9566, P=0.05, bound='1-sided lower')) # expected: 1018, Table A.16 (conservative)
#'''