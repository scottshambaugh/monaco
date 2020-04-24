import scipy.stats
import numpy as np
'''
Reference:
Hahn, Gerald J., and Meeker, William Q. "Statistical Intervals: A Guide for 
    Practitioners." Germany, Wiley, 1991.
'''


def p2sig(p, bound='2sided'):
    if p <= 0 or p >= 1:
        raise ValueError(f'{p=} must be 0 < p < 1')            
        return None
    if bound == '2sided':
        return scipy.stats.norm.ppf(1-(1-p)/2)
    if bound == '1sided':
        return scipy.stats.norm.ppf(p)

def sig2p(sig, bound='2sided'):
    if bound == '2sided':
        if sig < 0:
            raise ValueError(f'{sig=} must be >= 0 for a 2-sided bound')            
        return 1-(1-scipy.stats.norm.cdf(sig))*2
    elif bound == '1sided':
        return scipy.stats.norm.cdf(sig)


def order_stat(v, p, c, nmax=int(1e7), bound='2sided'):
    '''
    This function returns the number of cases n necessary to say that the true 
    result of a measurement x will be bounded by the v'th order statistic with 
    a probability p and confidence c. 
    
    For example, if I want to use my 2nd highest measurement as a bound on 99% 
    of all samples with 90% confidence:
        n = order_stat(2, 0.99, 0.90, bound='1sided') = 389
    The 388th value of x when sorted from low to high, or sorted(x)[-2], will 
    bound the upper end of the measurement with P99/90. 
    
    '2sided' gives the result for the measurement lying between the v'th lowest 
    and v'th highest measurements. If we run the above function with 
    bound='2sided', then n = 668, and we can say that the true measurement lies 
    between x[1] and x[-2] with P99/90.
    
    See chapter 5 of Reference at the top of this file for statistical 
    background.
    '''
    if v < 1:
        raise ValueError(f'{v=} must be >= 1')
    if nmax < 1:
        raise ValueError(f'{nmax=} must be >= 1')
    if c <= 0 or c >=1:
        raise ValueError(f'{c=} must be in the range 0 < c < 1')
    if p <= 0 or p >=1:
        raise ValueError(f'{p=} must be in the range 0 < p < 1')
    
    if bound == '2sided':
        l = v  # we won't be using assymmetrical order stats
    elif bound == '1sided':
        l = 0

    # use bisection to get n (secant method is unstable due to flat portions of curve)
    n = [1,nmax]
    maxsteps = 1000 # nmax hard limit of 2^1000
    u = n[1]+1-v
    if scipy.stats.binom.cdf(u-l-1, n[1], p) < c:
        print(f'n exceeded {nmax=} for P{100*p}/{c*100}. Increase nmax or loosen constraints.')
        return None

    for i in range(maxsteps):
        step = np.ceil((n[1]-n[0])/2)
        if step <= 1:
            return int(n[1])
        else:
            u = n[0]+step+1-v
            if scipy.stats.binom.cdf(u-l-1, n[0] + step, p) <= c:
                n[0] = n[0] + step
            else:
                n[1] = n[0] + step


'''
### Test ###
print(sig2p(3, bound='2sided'), sig2p(-3, bound='1sided'))
print(p2sig(0.9973, bound='2sided'), p2sig(0.00135, bound='1sided'))
print(order_stat(v=2, p=0.99, c=0.90, bound='2sided'))
#'''