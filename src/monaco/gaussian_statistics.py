# gaussian_statistics.py
from __future__ import annotations

import scipy.stats
import numpy as np
from monaco.mc_enums import StatBound


def pct2sig(p     : float,
            bound : StatBound = StatBound.TWOSIDED,
            ) -> float:
    """
    Converts a percentile to a gaussian sigma value (1-sided), or to the
    sigma value for which the range (-sigma, +sigma) bounds that percent of
    the normal distribution (2-sided).

    Parameters
    ----------
    p : float
        The percentile to convert, 0 < p < 1.
    bound : monaco.mc_enums.StatBound
        The statistical bound, either '1-sided' or '2-sided'.

    Returns
    -------
    sig : float
        The gaussian sigma value for the input percentile.
    """
    sig = None
    if p <= 0 or p >= 1:
        raise ValueError(f'{p=} must be 0 < p < 1')

    if bound == StatBound.TWOSIDED:
        if p >= 0.5:
            sig = scipy.stats.norm.ppf(1-(1-p)/2)
        else:
            sig = -scipy.stats.norm.ppf(p/2)
    elif bound == StatBound.ONESIDED:
        sig = scipy.stats.norm.ppf(p)
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    return sig



def sig2pct(sig   : float,
            bound : StatBound = StatBound.TWOSIDED,
            ) -> float:
    """
    Converts a gaussian sigma value to a percentile (1-sided), or to the
    percent of the normal distribution bounded by (-sigma, +sigma) (2-sided).

    Parameters
    ----------
    sig : float
        The gaussian sigma value to convert.
    bound : monaco.mc_enums.StatBound
        The statistical bound, either '1-sided' or '2-sided'.

    Returns
    -------
    p : float
        The corresponding percentile or percent, 0 < p < 1.
    """
    p = None
    if bound == StatBound.TWOSIDED:
        p = 1-(1-scipy.stats.norm.cdf(sig))*2
    elif bound == StatBound.ONESIDED:
        p = scipy.stats.norm.cdf(sig)
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    return p



def conf_ellipsoid_pct2sig(p  : float,
                           df : int,
                           ) -> float:
    """
    Converts a percentile to a sigma value which bounds a df-dimensional
    gaussian distribution, used in generating confidence ellipsoids. Note
    that in the 1-D case, np.sqrt(scipy.stats.chi2.ppf(p, df=1)) is
    equivalent to pct2sig(p >= 0.5, bound = '2-sided') ==
    scipy.stats.norm.ppf(1-(1-p)/2)

    Parameters
    ----------
    p : float
        The percentile to convert, 0 < p < 1.
    df : int
        The degrees of freedom, df > 0.

    Returns
    -------
    sig : float
        The gaussian sigma value for the input percentile.
    """
    sig = None

    if p <= 0 or p >= 1:
        raise ValueError(f'{p=} must be 0 < p < 1')
    elif df <= 0:
        raise ValueError(f'{df=} must be > 0')
    else:
        sig = np.sqrt(scipy.stats.chi2.ppf(p, df=df))

    return sig



def conf_ellipsoid_sig2pct(sig : float,
                           df  : int,
                           ) -> float:
    """
    Converts a sigma value which bounds a df-dimensional gaussian distribution,
    to a percentil used in generating confidence ellipsoids. Note that in the
    1-D case, scipy.stats.chi2.cdf(sig**2, df=1) is equivalent to
    sig2pct(sig > 0, bound='2-sided) == 1-(1-scipy.stats.norm.cdf(sig))*2

    Parameters
    ----------
    sig : float
        The gaussian sigma value to convert, sig > 0.
    df : int
        The degrees of freedom, df > 0.

    Returns
    -------
    p : float
        The corresponding percentile, 0 < p < 1.
    """

    if sig <= 0:
        raise ValueError(f'{sig=} must be > 0')
    elif df <= 0:
        raise ValueError(f'{df=} must be > 0')

    p = scipy.stats.chi2.cdf(sig**2, df=df)
    return p
