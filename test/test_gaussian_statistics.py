# test_gaussian_statistics.py

import pytest
from gaussian_statistics import pct2sig, sig2pct, conf_ellipsoid_pct2sig, conf_ellipsoid_sig2pct

def test_sig2pct():
    assert sig2pct(sig= 3, bound='1-sided') == pytest.approx( 0.9986501)
    assert sig2pct(sig=-3, bound='2-sided') == pytest.approx(-0.9973002)

def test_pct2sig():
    assert pct2sig(p=0.0013499, bound='1-sided') == pytest.approx(-3)
    assert pct2sig(p=0.9973002, bound='2-sided') == pytest.approx( 3)

def test_conf_ellipsoid_sig2pct():
    assert conf_ellipsoid_sig2pct(sig=3, df=1) == pytest.approx(0.9973002)
    assert conf_ellipsoid_sig2pct(sig=3, df=2) == pytest.approx(0.9888910)

def test_conf_ellipsoid_pct2sig():
    assert conf_ellipsoid_pct2sig(p=0.9973002, df=1) == pytest.approx(3)
    assert conf_ellipsoid_pct2sig(p=0.9888910, df=2) == pytest.approx(3)
