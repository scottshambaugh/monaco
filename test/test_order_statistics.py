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
    assert os.order_stat_TI_k(n=20, p=0.99, c=0.90, bound='2-sided') is None


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
    ( 10, 0.999, 0.05, '1-sided lower',  None),
])
def test_order_stat_P_k(n,c,P,bound,ans):
    assert os.order_stat_P_k(n=n, c=c, P=P, bound=bound) == ans


@pytest.mark.parametrize("k,c,P,bound,ans", [
    (10, 0.95  , 0.50,       '2-sided',  108), # Ref. Table A.15g
    (11, 0.9566, 0.95, '1-sided upper', 1018), # Ref. Table A.16
    (11, 0.9566, 0.05, '1-sided lower', 1018), # Ref. Table A.16
])
def test_order_stat_P_n(k,c,P,bound,ans):
    assert os.order_stat_P_n(k=k, c=c, P=P, bound=bound) == pytest.approx(ans)

