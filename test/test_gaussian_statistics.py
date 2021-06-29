# test_gaussian_statistics.py

import pytest
from Monaco.gaussian_statistics import sig2pct, pct2sig, conf_ellipsoid_sig2pct, conf_ellipsoid_pct2sig

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


### Inline Testing ###
# Can run here or copy into bottom of main file
#'''
if __name__ == '__main__':
    print(sig2pct(-3, bound='2-sided'), sig2pct(3, bound='1-sided')) # expected: -0.99730, 0.99865
    print(pct2sig(0.9973002, bound='2-sided'), pct2sig(0.0013499, bound='1-sided')) # expected: 3, -3
    print(conf_ellipsoid_sig2pct(3, df=1), conf_ellipsoid_sig2pct(3, df=2)) # expected: 0.99730, 0.98889
    print(conf_ellipsoid_pct2sig(0.9973002, df=1), conf_ellipsoid_pct2sig(0.988891, df=2)) # expected: 3.0, 3.0
#'''
