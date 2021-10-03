# order_statistics.py

import scipy.stats
import numpy as np
from warnings import warn
from monaco.MCEnums import StatBound
'''
Reference:
Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A Guide for 
    Practitioners." Germany, Wiley, 1991.
'''


def order_stat_TI_n(k     : int, 
                    p     : float, # 0 < p < 1 
                    c     : float, # 0 < c < 1
                    nmax  : int       = int(1e7), 
                    bound : StatBound = StatBound.TWOSIDED, # '1-sided' or '2-sided'
                    ) -> int:
    '''
    Order Statistic Tolerance Interval, find n
    This function returns the number of cases n necessary to say that the true 
    result of a measurement x will be bounded by the k'th order statistic with 
    a probability p and confidence c. Variables l and u below indicate lower 
    and upper indices of the order statistic.
    
    For example, if I want to use my 2nd highest measurement as a bound on 99% 
    of all future samples with 90% confidence:
        n = order_stat_TI_n(k=2, p=0.99, c=0.90, bound='1-sided') = 389
    The 388th value of x when sorted from low to high, or sorted(x)[-2], will 
    bound the upper end of the measurement with P99/90. 
    
    '2-sided' gives the result for the measurement lying between the k'th lowest 
    and k'th highest measurements. If we run the above function with 
    bound='2-sided', then n = 668, and we can say that the true measurement lies 
    between sorted(x)[1] and sorted(x)[-2] with P99/90.
    
    See chapter 5 of Reference at the top of this file for statistical 
    background.
    '''
    order_stat_var_check(p=p, k=k, c=c, nmax=nmax)
    
    if bound == StatBound.TWOSIDED:
        l = k  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    # use bisection to get minimum n (secant method is unstable due to flat portions of curve)
    n = [1,nmax]
    maxsteps = 1000 # nmax hard limit of 2^1000
    u = n[1] + 1 - k
    if EPTI(n[1], l, u, p) < c:
        print(f'n exceeded {nmax=} for P{100*p}/{c*100}. Increase nmax or loosen constraints.')
        return None

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
    raise ValueError(f'With {n=}, could not converge in {maxsteps=} steps. Is {nmax=} > 2^{maxsteps}?')        



def order_stat_TI_p(n     : int, 
                    k     : int, 
                    c     : float, # 0 < c < 1 
                    ptol  : float     = 1e-9, 
                    bound : StatBound = StatBound.TWOSIDED, # '1-sided' or '2-sided'
                    ) -> float:
    # Order Statistic Tolerance Interval, find p
    order_stat_var_check(n=n, k=k, c=c)
    
    if bound == StatBound.TWOSIDED:
        l = k  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")
    u = n + 1 - k

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    p = [0,1]
    maxsteps = 1000 # p hard tolerance of 2^-1000
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
    raise ValueError(f'With {p=}, could not converge under {ptol=} in {maxsteps=} steps.')        



def order_stat_TI_k(n     : int, 
                    p     : float, # 0 < p < 1 
                    c     : float, # 0 < p < 1
                    bound : StatBound = StatBound.TWOSIDED, # '1-sided' or '2-sided'
                    ) -> int:
    # Order Statistic Tolerance Interval, find maximum k
    order_stat_var_check(n=n, p=p, c=c)
    
    if bound == StatBound.TWOSIDED:
        l = 1  # we won't be using assymmetrical order stats
    elif bound == StatBound.ONESIDED:
        l = 0
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED} or {StatBound.TWOSIDED}")

    if EPTI(n, l, n, p) < c:
        warn(f'{n=} is too small to meet {p=} at {c=} for {bound} tolerance interval at any order statistic')
        return None

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    k = [1,np.ceil(n/2)]
    maxsteps = 1000 # nmax hard limit of 2^1000
    for i in range(maxsteps):
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
    raise ValueError(f'With {n=}, could not converge in {maxsteps=} steps. Is n > 2^{maxsteps}?') 
       


def order_stat_TI_c(n     : int, 
                    k     : int,
                    p     : float, # 0 < p < 1 
                    bound : StatBound = StatBound.TWOSIDED, # '1-sided' or '2-sided'
                    ) -> float:
    # Order Statistic Tolerance Interval, find c
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
                   P     : float,           # 0 < P < 1 
                   c     : float,           # 0 < c < 1
                   nmax  : int = int(1e7), 
                   bound : StatBound = StatBound.TWOSIDED, # '1-sided' or '2-sided'
                   ) -> int:
    '''
    Order Statistic Percentile, find n
    This function returns the number of cases n necessary to say that the true 
    Pth percentile located at or between indices iPl and iPu of a measurement x
    will be bounded by the k'th order statistic with confidence c. 
    
    For example, if I want to use my 5th nearest measurement as a bound on the 
    50th Percentile with 90% confidence:
        n = order_stat_P_n(k=5, P=0.50, c=0.90, bound='2-sided') = 38
        iPl = np.floor(P*(n + 1)) = 19
        iPu = np.ceil(P*(n + 1)) = 20
    The 19-5 = 14th and 20+5= 25th values of x when sorted from low to high, 
    or [sorted(x)[13], sorted(x)[24]] will bound the 50th percentile with 90%
    confidence. 
    
    '2-sided' gives the upper and lower bounds. '1-sided lower' and 
    '1-sided upper' give the respective lower or upper bound of the Pth 
    percentile over the entire rest of the distribution. 
    
    See chapter 5 of Reference at the top of this file for statistical 
    background.
    '''
    order_stat_var_check(p=P, k=k, c=c, nmax=nmax)
    
    # use bisection to get minimum n (secant method is unstable due to flat portions of curve)
    nmin = np.ceil(max(k/P - 1, k/(1-P) - 1))
    ntemp = nmin
    n = [nmin,nmax]
    maxsteps = 1000 # nmax hard limit of 2^1000
    
    (iPl, iP, iPu) = get_iP(n[0], P)
    if bound == StatBound.TWOSIDED:
        l = iPl - k + 1 # we won't be using assymmetrical order stats
        u = iPu + k - 1
        if l <= 0 or u >= n[1] + 1 or EPYP(n[0], l, u, P) < c:
            print(f'n ouside bounds of {nmin=}:{nmax=} for {P=} with {k=} at {c=}. Increase nmax, raise k, or loosen constraints.')
            return None
    elif bound == StatBound.ONESIDED_UPPER:
        l = 0
        u = iPu + k -1
        if u >= n[1] + 1 or EPYP(n[0], l, u, P) < c:
            print(f'n ouside bounds of {nmin=}:{nmax=} for {P=} with {k=} at {c=}. Increase nmax, raise k, or loosen constraints.')
            return None
    elif bound == StatBound.ONESIDED_LOWER:
        l = iPl - k + 1
        u = n[0] + 1
        if l <= 0 or EPYP(n[0], l, u, P) < c:
            print(f'n ouside bounds of {nmin=}:{nmax=} for {P=} with {k=} at {c=}. Increase nmax, raise k, or loosen constraints.')
            return None
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED_UPPER}, {StatBound.ONESIDED_LOWER}, or {StatBound.TWOSIDED}")


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
        #print(ntemp, ':', EPYP(ntemp, l, u, P), l, iP, u, n, step)
    raise ValueError(f'With {n=}, could not converge in {maxsteps=} steps. Is {nmax=} > 2^{maxsteps}?')    



def order_stat_P_k(n     : int, 
                   P     : float,           # 0 < P < 1 
                   c     : float,           # 0 < c < 1
                   bound : StatBound = StatBound.TWOSIDED, # '1-sided upper', '1-sided lower', or '2-sided'
                   ) -> int:
    # Order Statistic Percentile, find maximum k
    order_stat_var_check(n=n, p=P, c=c)
    
    (iPl, iP, iPu) = get_iP(n, P)    
    if bound == StatBound.TWOSIDED:
        k = [1, min(iPl, n + 1 - iPu)]
        l = iPl - k[1] + 1 # we won't be using assymmetrical order stats
        u = iPu + k[1] - 1
        if l <= 0 or u >= n+1 or EPYP(n, l, u, P) < c:
            warn(f'{n=} is too small to meet {P=} at {c=} for {bound} percentile confidence interval at any order statistic')
            return None

    elif bound == StatBound.ONESIDED_UPPER:
        k = [1, n + 1 - iPu]
        l = 0
        u = iPu + k[1] - 1
        if u >= n + 1 or EPYP(n, l, u, P) < c:
            warn(f'{n=} is too small to meet {P=} at {c=} for {bound} percentile confidence interval at any order statistic')
            return None

    elif bound == StatBound.ONESIDED_LOWER:
        k = [1, iPl]
        l = iPl - k[1] + 1
        u = n + 1
        if EPYP(n, l, u, P) < c:
            warn(f'{n=} is too small to meet {P=} at {c=} for {bound} percentile confidence interval at any order statistic')
            return None
    else:
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED_UPPER}, {StatBound.ONESIDED_LOWER}, or {StatBound.TWOSIDED}")

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    maxsteps = 1000 # nmax hard limit of 2^1000
    for i in range(maxsteps):
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
                
    raise ValueError(f'With {n=}, could not converge in {maxsteps=} steps. Is n > 2^{maxsteps}?')



def order_stat_P_c(n     : int, 
                   k     : int, 
                   P     : float,           # 0 < P < 1
                   bound : StatBound = StatBound.TWOSIDED, # '1-sided upper', '1-sided lower', or '2-sided'
                   ) -> float:
    # Order Statistic Percentile, find c
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
        raise ValueError(f"{bound=} must be {StatBound.ONESIDED_UPPER}, {StatBound.ONESIDED_LOWER}, or {StatBound.TWOSIDED}")
        
    if l < 0 or u > n+1:
        raise ValueError(f'{l=} or {u=} are outside the valid bounds of (0, {n+1}) (check: {iP=}, {k=})') 
    
    c = EPYP(n, l, u, P)
    return c



def EPYP(n : int, 
         l : int, 
         u : int, 
         p : float, # 0 < p < 1
         ) -> float:
    # Estimated Probabiliity for the Y'th Percentile, see Chp. 5.2 of Reference
    order_stat_var_check(n=n, l=l, u=u, p=p)
    c = scipy.stats.binom.cdf(u-1, n, p) - scipy.stats.binom.cdf(l-1, n, p)
    return c



def EPTI(n : int, 
         l : int, 
         u : int, 
         p : float, # 0 < p < 1
         ) -> float:
    # Estimated Probabiliity for a Tolerance Interval, see Chp. 5.3 of Reference
    order_stat_var_check(n=n, l=l, u=u, p=p)
    c = scipy.stats.binom.cdf(u-l-1, n, p)
    return c


def get_iP(n : int, 
           P : float, # 0 < P < 1
           ) -> tuple[int, int, int]:
    # Index of Percentile (1-based indexing)
    iP = P*(n + 1) 
    iPl = int(np.floor(iP))
    iPu = int(np.ceil(iP))
    iP = int(np.round(iP))
    return (iPl, iP, iPu)



def order_stat_var_check(n=None, l=None, u=None, p=None, k=None, c=None, nmax=None):
    if n is not None and n < 1:
        raise ValueError(f'{n=} must be >= 1')
    if l is not None and l < 0:
        raise ValueError(f'{l=} must be >= 0')
    if u is not None and n is not None and u > n+1:
        raise ValueError(f'{u=} must be >= {n+1}')
    if u is not None and l is not None and u < l:
        raise ValueError(f'{u=} must be >= {l=}')
    if p is not None and (p <= 0 or p >=1):
        raise ValueError(f'{p=} must be in the range 0 < p < 1')
    if k is not None and k < 1:
        raise ValueError(f'{k=} must be >= 1')
    if c is not None and (c <= 0 or c >=1):
        raise ValueError(f'{c=} must be in the range 0 < c < 1')
    if nmax is not None and nmax < 1:
        raise ValueError(f'{nmax=} must be >= 1')    

