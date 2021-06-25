# test_gaussian_statistics.py

import pytest
import Monaco.gaussian_statistics as gs

def test_sig2pct():
    assert gs.sig2pct(sig= 3, bound='1-sided') == pytest.approx( 0.9986501)
    assert gs.sig2pct(sig=-3, bound='2-sided') == pytest.approx(-0.9973002)

def test_pct2sig():
    assert gs.pct2sig(p=0.0013499, bound='1-sided') == pytest.approx(-3)
    assert gs.pct2sig(p=0.9973002, bound='2-sided') == pytest.approx( 3)

def test_conf_ellipsoid_sig2pct():
    assert gs.conf_ellipsoid_sig2pct(sig=3, df=1) == pytest.approx(0.9973002)
    assert gs.conf_ellipsoid_sig2pct(sig=3, df=2) == pytest.approx(0.9888910)

def test_conf_ellipsoid_pct2sig():
    assert gs.conf_ellipsoid_pct2sig(p=0.9973002, df=1) == pytest.approx(3)
    assert gs.conf_ellipsoid_pct2sig(p=0.9888910, df=2) == pytest.approx(3)
