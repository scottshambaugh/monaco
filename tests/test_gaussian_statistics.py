# test_gaussian_statistics.py

import pytest
from monaco.gaussian_statistics import (sig2pct, pct2sig, conf_ellipsoid_sig2pct,
                                        conf_ellipsoid_pct2sig)
from monaco.mc_enums import StatBound


def test_pct2sig():
    assert pct2sig(p=0.0013499, bound=StatBound.ONESIDED) == pytest.approx(-3)
    assert pct2sig(p=0.9973002, bound=StatBound.TWOSIDED) == pytest.approx( 3)
    assert pct2sig(p=0.0026998, bound=StatBound.TWOSIDED) == pytest.approx(-3)
    with pytest.raises(ValueError):
        pct2sig(p=0, bound=StatBound.TWOSIDED)
    with pytest.raises(ValueError):
        pct2sig(p=1, bound=StatBound.TWOSIDED)
    with pytest.raises(ValueError):
        pct2sig(p=0.5, bound=None)

def test_sig2pct():
    assert sig2pct(sig=3,  bound=StatBound.ONESIDED) == pytest.approx( 0.9986501)
    assert sig2pct(sig=-3, bound=StatBound.TWOSIDED) == pytest.approx(-0.9973002)
    with pytest.raises(ValueError):
        sig2pct(sig=3, bound=None)

def test_conf_ellipsoid_pct2sig():
    assert conf_ellipsoid_pct2sig(p=0.9973002, df=1) == pytest.approx(3)
    assert conf_ellipsoid_pct2sig(p=0.9888910, df=2) == pytest.approx(3)
    with pytest.raises(ValueError):
        conf_ellipsoid_pct2sig(p=0, df=1)
    with pytest.raises(ValueError):
        conf_ellipsoid_pct2sig(p=1, df=1)

def test_conf_ellipsoid_sig2pct():
    assert conf_ellipsoid_sig2pct(sig=3, df=1) == pytest.approx(0.9973002)
    assert conf_ellipsoid_sig2pct(sig=3, df=2) == pytest.approx(0.9888910)
    with pytest.raises(ValueError):
        conf_ellipsoid_sig2pct(sig=0, df=1)
