import scipy.stats
import numpy as np
'''
Reference:
Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A Guide for 
    Practitioners." Germany, Wiley, 1991.
'''


def pct2sig(p, bound='2sided'):
    # Converts a to a percentile to a gaussian sigma value (1sided), or to the 
    # sigma value for which the range (-sigma, +sigma) bounds that percent of 
    # the normal distribution (2sided)
    if p <= 0 or p >= 1:
        raise ValueError(f'{p=} must be 0 < p < 1')            
        return None
    if bound == '2sided':
        return scipy.stats.norm.ppf(1-(1-p)/2)
    if bound == '1sided':
        return scipy.stats.norm.ppf(p)



def sig2pct(sig, bound='2sided'):
    # Converts a gaussian sigma value to a percentile (1sided), or to the percent
    # of the normal distribution bounded by (-sigma, +sigma) (2sided)
    if bound == '2sided':
        if sig < 0:
            raise ValueError(f'{sig=} must be >= 0 for a 2-sided bound')            
        return 1-(1-scipy.stats.norm.cdf(sig))*2
    elif bound == '1sided':
        return scipy.stats.norm.cdf(sig)



def order_stat_TI_n(v, p, c, nmax=int(1e7), bound='2sided'):
    '''
    Order Statistic Tolerance Interval, find n
    This function returns the number of cases n necessary to say that the true 
    result of a measurement x will be bounded by the v'th order statistic with 
    a probability p and confidence c. Variables l and u below indicate lower 
    and upper indices of the order statistic.
    
    For example, if I want to use my 2nd highest measurement as a bound on 99% 
    of all samples with 90% confidence:
        n = order_stat(2, 0.99, 0.90, bound='1sided') = 389
    The 388th value of x when sorted from low to high, or sorted(x)[-2], will 
    bound the upper end of the measurement with P99/90. 
    
    '2sided' gives the result for the measurement lying between the v'th lowest 
    and v'th highest measurements. If we run the above function with 
    bound='2sided', then n = 668, and we can say that the true measurement lies 
    between sorted(x)[1] and sorted(x)[-2] with P99/90.
    
    See chapter 5 of Reference at the top of this file for statistical 
    background.
    '''
    order_stat_var_check(p=p, v=v, c=c, nmax=nmax)
    
    if bound == '2sided':
        l = v  # we won't be using assymmetrical order stats
    elif bound == '1sided':
        l = 0

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    n = [1,nmax]
    maxsteps = 1000 # nmax hard limit of 2^1000
    u = n[1]+1-v
    if EPTI(n[1], l, u, p) < c:
        print(f'n exceeded {nmax=} for P{100*p}/{c*100}. Increase nmax or loosen constraints.')
        return None

    for i in range(maxsteps):
        step = np.ceil((n[1]-n[0])/2)
        if step <= 1:
            return int(n[1])
        else:
            u = n[0]+step+1-v
            if EPTI(n[0]+step, l, u, p) <= c:
                n[0] = n[0] + step
            else:
                n[1] = n[0] + step



def order_stat_TI_p(n, v, c, ptol=1e-9, bound='2sided'):
    # Order Statistic Tolerance Interval, find p
    order_stat_var_check(n=n, v=v, c=c)
    
    if bound == '2sided':
        l = v  # we won't be using assymmetrical order stats
    elif bound == '1sided':
        l = 0
    u = n+1-v

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    p = [0,1]
    maxsteps = 1000 # nmax hard tolerance of 2^-1000
    for i in range(maxsteps):
        step = (p[1]-p[0])/2
        if step <= ptol:
            return p[1]
        else:
            if EPTI(n, l, u, p[0]+step) >= c:
                p[0] = p[0] + step
            else:
                p[1] = p[0] + step



def order_stat_TI_c(n, v, p, bound='2sided'):
    # Order Statistic Tolerance Interval, find c
    order_stat_var_check(n=n, p=p, v=v)
    
    if bound == '2sided':
        l = v  # we won't be using assymmetrical order stats
    elif bound == '1sided':
        l = 0
    u = n+1-v
    
    c = EPTI(n, l, u, p)
    return c



def EPYP(n, l, u, p):
    # Estimated Probabiliity for the Y'th Percentile, see Chp. 5.2 of Reference
    order_stat_var_check(n=n, l=l, u=u, p=p)
    c = scipy.stats.binom.cdf(u-1, n, p) - scipy.stats.binom.cdf(l-1, n, p)
    return c



def EPTI(n, l, u, p):
    # Estimated Probabiliity for a Tolerance Interval, see Chp. 5.3 of Reference
    order_stat_var_check(n=n, l=l, u=u, p=p)
    c = scipy.stats.binom.cdf(u-l-1, n, p)
    return c



def order_stat_var_check(n=None, l=None, u=None, p=None, v=None, c=None, nmax=None):
    if n!= None and n < 1:
        raise ValueError(f'{n=} must be >= 1')
    if l != None and l <= 0:
        raise ValueError(f'{l=} must be greater than 0')
    if u != None and n != None and u > n+1:
        raise ValueError(f'{u=} must be less than n+1')
    if u != None and l != None and u <= l:
        raise ValueError(f'{u=} must be greater than {l=}')
    if p != None and (p <= 0 or p >=1):
        raise ValueError(f'{p=} must be in the range 0 < p < 1')
    if v!= None and v < 1:
        raise ValueError(f'{v=} must be >= 1')
    if c != None and (c <= 0 or c >=1):
        raise ValueError(f'{c=} must be in the range 0 < c < 1')
    if nmax != None and nmax < 1:
        raise ValueError(f'{nmax=} must be >= 1')



'''
### Test ###
print(sig2pct(3, bound='2sided'), sig2pct(3, bound='1sided')) # 0.99730, 0.99865
print(pct2sig(0.9973002, bound='2sided'), pct2sig(0.0013499, bound='1sided')) # 3, -3
print(order_stat_TI_n(v=2, p=0.99, c=0.90, bound='2sided')) # 668
print(order_stat_TI_p(n=668, v=2, c=0.90, bound='2sided')) # 0.99003
print(order_stat_TI_c(n=668, v=2, p=0.99, bound='2sided')) # 0.90110
#'''