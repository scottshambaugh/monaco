# order_statistics.py
from __future__ import annotations

import scipy.stats
import numpy as np
from monaco.mc_enums import StatBound


def order_stat_TI_n(k     : int,
                    p     : float,
                    c     : float,
                    nmax  : int       = int(1e7),
                    bound : StatBound = StatBound.TWOSIDED,
                    ) -> int:
    """
    For an Order Statistic Tolerance Interval, find the minimum n from k, p,
    and c

    Notes
    -----
    This function returns the number of cases n necessary to say that the true
    result of a measurement x will be bounded by the k'th order statistic with
    a probability p and confidence c. Variables l and u below indicate lower
    and upper indices of the order statistic.

    For example, if I want to use my 2nd highest measurement as a bound on 99%
    of all future samples with 90% confidence:

    .. code-block::

        n = order_stat_TI_n(k=2, p=0.99, c=0.90, bound='1-sided') = 389

    The 388th value of x when sorted from low to high, or sorted(x)[-2], will
    bound the upper end of the measurement with P99/90.

    '2-sided' gives the result for the measurement lying between the k'th lowest
    and k'th highest measurements. If we run the above function with
    bound='2-sided', then n = 668, and we can say that the true measurement lies
    between sorted(x)[1] and sorted(x)[-2] with P99/90.

    See chapter 5 of Reference [1]_ for statistical background.

    Parameters
    ----------
    k : int
        The k'th order statistic.
    p : float (0 < p < 1)
        The percent covered by the tolerance interval.
    c : float (0 < c < 1)
        The confidence of the interval bound.
    nmax : int, default: 1e7
        The maximum number of draws. Hard limit of 2**1000.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, either '1-sided' or '2-sided'.

    Returns
    -------
    n : int
        The number of samples necessary to meet the constraints.

    References
    ----------
    .. [1] Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A
       Guide for Practitioners." Germany, Wiley, 1991.
    """
    order_stat_var_check(p=p, k=k, c=c, nmax=nmax)

    if bound == StatBound.TWOSIDED:
        l = k  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    # use bisection to get minimum n (secant method is unstable due to flat portions of curve)
    n = [1, nmax]
    maxsteps = 100  # nmax hard limit of 2^100
    u = n[1] + 1 - k
    if EPTI(n[1], l, u, p) < c:
        raise ValueError(f'n exceeded {nmax=} for P{100*p}/{c*100}. ' +
                          'Increase nmax or loosen constraints.')

    for i in range(maxsteps):
        step = (n[1]-n[0])/2
        ntemp = n[0] + np.ceil(step)
        if step < 1:
            return int(n[1])
        else:
            u = ntemp + 1 - k
            if EPTI(ntemp, l, u, p) <= c:
                n[0] = ntemp
            else:
                n[1] = ntemp
    raise ValueError(f'With {n=}, could not converge in {maxsteps=} steps. ' +
                    f'Is {nmax=} > 2^{maxsteps}?')



def order_stat_TI_p(n     : int,
                    k     : int,
                    c     : float,
                    ptol  : float     = 1e-9,
                    bound : StatBound = StatBound.TWOSIDED,
                    ) -> float:
    """
    For an Order Statistic Tolerance Interval, find the maximum p from n, k,
    and c.

    Parameters
    ----------
    n : int
        The number of samples.
    k : int
        The k'th order statistic.
    c : float (0 < c < 1)
        The confidence of the interval bound.
    ptol : float, default: 1e-9
        The absolute tolerance on determining p.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, either '1-sided' or '2-sided'.

    Returns
    -------
    p : float (0 < p < 1)
        The percent which the tolerance interval covers corresponding to the
        input constraints.
    """
    order_stat_var_check(n=n, k=k, c=c)

    if bound == StatBound.TWOSIDED:
        l = k  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")
    u = n + 1 - k

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    p = [0.0, 1.0]
    maxsteps = 100  # p hard tolerance of 2^-100
    for i in range(maxsteps):
        step = (p[1]-p[0])/2
        ptemp = p[0] + step
        if step <= ptol:
            return p[1]
        else:
            if EPTI(n, l, u, ptemp) >= c:
                p[0] = ptemp
            else:
                p[1] = ptemp
    raise ValueError(f'With {p=}, could not converge under {ptol=} in {maxsteps} steps.')



def order_stat_TI_k(n     : int,
                    p     : float,
                    c     : float,
                    bound : StatBound = StatBound.TWOSIDED,
                    ) -> int:
    """
    For an Order Statistic Tolerance Interval, find the maximum k from n, p,
    and c.

    Parameters
    ----------
    n : int
        The number of samples.
    p : float (0 < p < 1)
        The percent covered by the tolerance interval.
    c : float (0 < c < 1)
        The confidence of the interval bound.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, either '1-sided' or '2-sided'.

    Returns
    -------
    k : int
        The k'th order statistic.
    """
    order_stat_var_check(n=n, p=p, c=c)

    if bound == StatBound.TWOSIDED:
        l = 1  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    if EPTI(n, l, n, p) < c:
        raise ValueError(f'{n=} is too small to meet {p=} at {c=} for {bound} ' +
                          'tolerance interval at any order statistic')

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    k = [1, np.ceil(n/2)]
    maxsteps = 100  # nmax hard limit of 2^100
    for _ in range(maxsteps):
        step = (k[1]-k[0])/2
        ktemp = k[0] + np.ceil(step)
        if step < 1:
            return int(k[1])-1
        else:
            if bound == StatBound.TWOSIDED:
                l = ktemp  # we won't be using assymmetrical order stats
            elif bound == StatBound.ONESIDED:
                l = 0
            u = n + 1 - ktemp

            if EPTI(n, l, u, p) > c:
                k[0] = ktemp
            else:
                k[1] = ktemp
    raise ValueError(f'With {n=}, could not converge in {maxsteps} steps. Is n > 2^{maxsteps}?')



def order_stat_TI_c(n     : int,
                    k     : int,
                    p     : float,
                    bound : StatBound = StatBound.TWOSIDED,
                    ) -> float:
    """
    For an Order Statistic Tolerance Interval, find the maximum c from n, k,
    and p.

    Parameters
    ----------
    n : int
        The number of samples.
    k : int
        The k'th order statistic.
    p : float (0 < p < 1)
        The percent covered by the tolerance interval.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, either '1-sided' or '2-sided'.

    Returns
    -------
    c : float (0 < c < 1)
        The confidence of the interval bound.
    """
    order_stat_var_check(n=n, p=p, k=k)

    if bound == StatBound.TWOSIDED:
        l = k  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    u = n + 1 - k

    c = EPTI(n, l, u, p)
    return c



def order_stat_P_n(k     : int,
                   P     : float,
                   c     : float,
                   nmax  : int = int(1e7),
                   bound : StatBound = StatBound.TWOSIDED,
                   ) -> int:
    """
    Order Statistic Percentile, find minimum n from k, P, and c.

    Notes
    -----
    This function returns the number of cases n necessary to say that the true
    Pth percentile located at or between indices iPl and iPu of a measurement x
    will be bounded by the k'th order statistic with confidence c.

    For example, if I want to use my 5th nearest measurement as a bound on the
    50th Percentile with 90% confidence:

    .. code-block::

        n = order_stat_P_n(k=5, P=0.50, c=0.90, bound='2-sided') = 38
        iPl = np.floor(P*(n + 1)) = 19
        iPu = np.ceil(P*(n + 1)) = 20

    The 19-5 = 14th and 20+5= 25th values of x when sorted from low to high, or
    [sorted(x)[13], sorted(x)[24]] will bound the 50th percentile with 90%
    confidence.

    '2-sided' gives the upper and lower bounds. '1-sided lower' and
    '1-sided upper' give the respective lower or upper bound of the Pth
    percentile over the entire rest of the distribution.

    See chapter 5 of Reference [2]_ for statistical background.

    Parameters
    ----------
    k : int
        The k'th order statistic.
    P : float (0 < P < 1)
        The target percentile.
    c : float (0 < c < 1)
        The confidence of the interval bound.
    nmax : int, default: 1e7
        The maximum number of draws. Hard limit of 2**1000.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, '1-sided upper', '1-sided lower', or '2-sided'.

    Returns
    -------
    n : int
        The number of samples necessary to meet the constraints.

    References
    ----------
    .. [2] Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A
       Guide for Practitioners." Germany, Wiley, 1991.
    """
    order_stat_var_check(p=P, k=k, c=c, nmax=nmax)

    # use bisection to get minimum n (secant method is unstable due to flat portions of curve)
    nmin = np.ceil(max(k/P - 1, k/(1-P) - 1))
    ntemp = nmin
    n = [nmin, nmax]
    maxsteps = 100  # nmax hard limit of 2^100

    (iPl, iP, iPu) = get_iP(n[0], P)
    if bound == StatBound.TWOSIDED:
        l = iPl - k + 1  # we won't be using assymmetrical order stats
        u = iPu + k - 1
        if l <= 0 or u >= n[1] + 1 or EPYP(n[0], l, u, P) < c:
            raise ValueError(f'n ouside bounds of {nmin=}:{nmax=} for {P=} with {k=} ' +
                             f'at {c=}. Increase nmax, raise k, or loosen constraints.')
    elif bound == StatBound.ONESIDED_UPPER:
        l = 0
        u = iPu + k - 1
        if u >= n[1] + 1 or EPYP(n[0], l, u, P) < c:
            raise ValueError(f'n ouside bounds of {nmin=}:{nmax=} for {P=} with {k=} ' +
                             f'at {c=}. Increase nmax, raise k, or loosen constraints.')
    elif bound == StatBound.ONESIDED_LOWER:
        l = iPl - k + 1
        u = n[0] + 1
        if l <= 0 or EPYP(n[0], l, u, P) < c:
            raise ValueError(f'n ouside bounds of {nmin=}:{nmax=} for {P=} with {k=} ' +
                             f'at {c=}. Increase nmax, raise k, or loosen constraints.')
    else:
        raise ValueError(f'{bound=} must be {StatBound.ONESIDED_UPPER}, ' +
                         f'{StatBound.ONESIDED_LOWER}, or {StatBound.TWOSIDED}')

    for i in range(maxsteps):
        step = (n[1]-n[0])/2
        if step < 1:
            return int(n[0])
        else:
            ntemp = n[0] + np.ceil(step)
            (iPl, iP, iPu) = get_iP(ntemp, P)
            if bound == StatBound.TWOSIDED:
                l = iPl - k  # we won't be using assymmetrical order stats
                u = iPu + k
            elif bound == StatBound.ONESIDED_UPPER:
                l = 0
                u = iPu + k
            elif bound == StatBound.ONESIDED_LOWER:
                l = iPl - k
                u = ntemp + 1
            if EPYP(ntemp, l, u, P) > c:
                n[0] = ntemp
            else:
                n[1] = ntemp
        # print(ntemp, ':', EPYP(ntemp, l, u, P), l, iP, u, n, step)
    raise ValueError(f'With {n=}, could not converge in {maxsteps=} steps. ' +
                     f'Is {nmax=} > 2^{maxsteps}?')



def order_stat_P_k(n     : int,
                   P     : float,
                   c     : float,
                   bound : StatBound = StatBound.TWOSIDED,
                   ) -> int:
    """
    For an Order Statistic Percentile, find the maximum p from n, k, and c.

    Parameters
    ----------
    n : int
        The number of samples.
    P : float (0 < P < 1)
        The target percentile.
    c : float (0 < c < 1)
        The confidence of the interval bound.
    ptol : float, default: 1e-9
        The absolute tolerance on determining p.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, '1-sided upper', '1-sided lower', or '2-sided'.

    Returns
    -------
    k : int
        The k'th order statistic meeting the input constraints.
    """
    order_stat_var_check(n=n, p=P, c=c)

    (iPl, iP, iPu) = get_iP(n, P)
    if bound == StatBound.TWOSIDED:
        k = [1, min(iPl, n + 1 - iPu)]
        l = iPl - k[1] + 1  # we won't be using assymmetrical order stats
        u = iPu + k[1] - 1
        if l <= 0 or u >= n+1 or EPYP(n, l, u, P) < c:
            raise ValueError(f'{n=} is too small to meet {P=} at {c=} for {bound} percentile ' +
                              'confidence interval at any order statistic. Use order_stat_P_n ' +
                              'to find the minimum n.')

    elif bound == StatBound.ONESIDED_UPPER:
        k = [1, n + 1 - iPu]
        l = 0
        u = iPu + k[1] - 1
        if u >= n + 1 or EPYP(n, l, u, P) < c:
            raise ValueError(f'{n=} is too small to meet {P=} at {c=} for {bound} percentile ' +
                              'confidence interval at any order statistic. Use order_stat_P_n ' +
                              'to find the minimum n.')

    elif bound == StatBound.ONESIDED_LOWER:
        k = [1, iPl]
        l = iPl - k[1] + 1
        u = n + 1
        if EPYP(n, l, u, P) < c:
            raise ValueError(f'{n=} is too small to meet {P=} at {c=} for {bound} percentile ' +
                              'confidence interval at any order statistic. Use order_stat_P_n ' +
                              'to find the minimum n.')
    else:
        raise ValueError(f'{bound=} must be {StatBound.ONESIDED_UPPER}, ' +
                         f'{StatBound.ONESIDED_LOWER}, or {StatBound.TWOSIDED}')

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    maxsteps = 100  # nmax hard limit of 2^100
    for _ in range(maxsteps):
        step = (k[1]-k[0])/2
        ktemp = k[0] + np.ceil(step)

        if step < 1:
            return int(k[1])

        else:
            if bound == StatBound.TWOSIDED:
                l = iPl - ktemp
                u = iPu + ktemp
            elif bound == StatBound.ONESIDED_UPPER:
                l = 0
                u = iPu + ktemp
            elif bound == StatBound.ONESIDED_LOWER:
                l = iPl - ktemp
                u = n + 1

            if EPYP(n, l, u, P) > c:
                k[1] = ktemp
            else:
                k[0] = ktemp

    raise ValueError(f'With {n=}, could not converge in {maxsteps} steps. Is n > 2^{maxsteps}?')



def order_stat_P_c(n     : int,
                   k     : int,
                   P     : float,
                   bound : StatBound = StatBound.TWOSIDED,
                   ) -> float:
    """
    For an Order Statistic percentile, find the maximum c from n, k, and P.

    Parameters
    ----------
    n : int
        The number of samples.
    k : int
        The k'th order statistic.
    P : float (0 < P < 1)
        The target percentile.
    bound : monaco.mc_enums.StatBound, default: '2-sided'
        The statistical bound, '1-sided upper', '1-sided lower', or '2-sided'.

    Returns
    -------
    c : float (0 < c < 1)
        The confidence of the interval bound.
    """
    order_stat_var_check(n=n, p=P, k=k)

    (iPl, iP, iPu) = get_iP(n, P)
    if bound == StatBound.TWOSIDED:
        l = iPl - k  # we won't be using assymmetrical order stats
        u = iPu + k
    elif bound == StatBound.ONESIDED_UPPER:
        l = 0
        u = iPu + k
    elif bound == StatBound.ONESIDED_LOWER:
        l = iPl - k
        u = n + 1
    else:
        raise ValueError(f'{bound=} must be {StatBound.ONESIDED_UPPER}, ' +
                         f'{StatBound.ONESIDED_LOWER}, or {StatBound.TWOSIDED}')

    if l < 0 or u > n+1:
        raise ValueError(f'{l=} or {u=} are outside the valid bounds of (0, {n+1}) ' +
                         f'(check: {iP=}, {k=})')

    c = EPYP(n, l, u, P)
    return c



def EPYP(n : int,
         l : int,
         u : int,
         p : float,
         ) -> float:
    """
    Estimated Probability for the Y'th Percentile, see Chp. 5.2 of Reference.

    Parameters
    ----------
    n : int
        TODO Description
    l : int
        TODO Description
    u : int
        TODO Description
    p : float (0 < p < 1)
        TODO Description

    Returns
    -------
    c : float (0 < c < 1)
        TODO Description
    """
    order_stat_var_check(n=n, l=l, u=u, p=p)
    c = scipy.stats.binom.cdf(u-1, n, p) - scipy.stats.binom.cdf(l-1, n, p)
    return c



def EPTI(n : int,
         l : int,
         u : int,
         p : float,
         ) -> float:
    """
    Estimated Probability for a Tolerance Interval, see Chp. 5.3 of Reference

    Parameters
    ----------
    n : int
        TODO Description
    l : int
        TODO Description
    u : int
        TODO Description
    p : float (0 < p < 1)
        TODO Description

    Returns
    -------
    c : float (0 < c < 1)
        TODO Description
    """
    order_stat_var_check(n=n, l=l, u=u, p=p)
    c = scipy.stats.binom.cdf(u-l-1, n, p)
    return c



def get_iP(n : int,
           P : float,
           ) -> tuple[int, int, int]:
    """
    Get the index of Percentile (1-based indexing)

    Parameters
    ----------
    n : int
        Number of samples
    P : float (0 < P < 1)
        Target percentile

    Returns
    -------
    (iPl, iP, iPu) : (int, int, int)
        Lower, closest, and upper index of the percentile.
    """
    iP = P*(n + 1)
    iPl = int(np.floor(iP))
    iPu = int(np.ceil(iP))
    iP = int(np.round(iP))
    return (iPl, iP, iPu)



def order_stat_var_check(n    : int | None   = None,
                         l    : int | None   = None,
                         u    : int | None   = None,
                         p    : float | None = None,
                         k    : int | None   = None,
                         c    : float | None = None,
                         nmax : int | None   = None
                         ) -> None:
    """
    Check the validity of the inputs to the order statistic functions.
    """
    if n is not None and n < 1:
        raise ValueError(f'{n=} must be >= 1')
    if l is not None and l < 0:
        raise ValueError(f'{l=} must be >= 0')
    if u is not None and n is not None and u > n+1:
        raise ValueError(f'{u=} must be >= {n+1}')
    if u is not None and l is not None and u < l:
        raise ValueError(f'{u=} must be >= {l=}')
    if p is not None and (p <= 0 or p >= 1):
        raise ValueError(f'{p=} must be in the range 0 < p < 1')
    if k is not None and k < 1:
        raise ValueError(f'{k=} must be >= 1')
    if c is not None and (c <= 0 or c >= 1):
        raise ValueError(f'{c=} must be in the range 0 < c < 1')
    if nmax is not None and nmax < 1:
        raise ValueError(f'{nmax=} must be >= 1')
