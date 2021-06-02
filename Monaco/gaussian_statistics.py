import scipy.stats
import numpy as np


def pct2sig(p     : float,           # 0 < p < 1
            bound : str = '2-sided', # '1-sided' or '2-sided'
            ) -> float:
    # Converts a percentile to a gaussian sigma value (1-sided), or to the 
    # sigma value for which the range (-sigma, +sigma) bounds that percent of 
    # the normal distribution (2-sided)
    sig = None
    if p <= 0 or p >= 1:
        raise ValueError(f'{p=} must be 0 < p < 1')
          
    if bound == '2-sided':
        if p >= 0.5:
            sig = scipy.stats.norm.ppf(1-(1-p)/2)
        else:
            sig = scipy.stats.norm.ppf(p/2)
    elif bound == '1-sided':
        sig = scipy.stats.norm.ppf(p)
    else:
        raise ValueError(f"{bound=} must be '1-sided' or '2-sided'")
    
    return sig



def sig2pct(sig   : float, 
            bound : str = '2-sided', # '1-sided' or '2-sided'
            ) -> float:
    # Converts a gaussian sigma value to a percentile (1-sided), or to the percent
    # of the normal distribution bounded by (-sigma, +sigma) (2-sided)
    p = None
    if bound == '2-sided':
        p = 1-(1-scipy.stats.norm.cdf(sig))*2
    elif bound == '1-sided':
        p = scipy.stats.norm.cdf(sig)
    else:
        raise ValueError(f"{bound=} must be '1-sided' or '2-sided'")   
    
    return p



def conf_ellipsoid_pct2sig(p  : float,  # 0 < p < 1
                           df : int,    # degrees of freedom
                           ) -> float:
    # Converts a percentile to a sigma value which bounds a df-dimensional 
    # gaussian distribution, used in generating confidence ellipsoids. Note
    # that in the 1-D case, np.sqrt(scipy.stats.chi2.ppf(p, df=1)) is 
    # equivalent to pct2sig(p >= 0.5, bound = '2-sided') 
    #                = scipy.stats.norm.ppf(1-(1-p)/2)
    sig = None
    
    if p <= 0 or p >= 1:
        raise ValueError(f'{p=} must be 0 < p < 1')            
    else:
        sig = np.sqrt(scipy.stats.chi2.ppf(p, df=df))
        
    return sig



def conf_ellipsoid_sig2pct(sig : float,  # sig > 0
                           df  : int,    # degrees of freedom
                           ) -> float:
    # Converts a percentile to a sigma value which bounds a df-dimensional 
    # gaussian distribution, used in generating confidence ellipsoids. Note
    # that in the 1-D case, scipy.stats.chi2.cdf(sig**2, df=1) is 
    # equivalent to sig2pct(sig > 0, bound='2-sided)
    #                = 1-(1-scipy.stats.norm.cdf(sig))*2
    if sig <= 0:
        raise ValueError(f'{sig=} must be sig > 0')            

    p = scipy.stats.chi2.cdf(sig**2, df=df)
    return p


'''
### Test ###
if __name__ == '__main__':
    print(sig2pct(-3, bound='2-sided'), sig2pct(3, bound='1-sided')) # expected: -0.99730, 0.99865
    print(pct2sig(0.9973002, bound='2-sided'), pct2sig(0.0013499, bound='1-sided')) # expected: 3, -3
    print(conf_ellipsoid_sig2pct(3, df=1), conf_ellipsoid_sig2pct(3, df=2)) # expected: 0.99730, 0.98889
    print(conf_ellipsoid_pct2sig(0.9973002, df=1), conf_ellipsoid_pct2sig(0.988891, df=2)) # expected: 3.0, 3.0
#'''
