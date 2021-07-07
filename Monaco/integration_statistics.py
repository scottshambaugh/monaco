# integration_statistics.py

import numpy as np
from typing import Union
from gaussian_statistics import pct2sig

def integration_error(isUnderCurve : Union[list[int], list[bool]],  # List of 0 and 1, or False and True values
                      volume       : float     = 1,  # By default, returns an unscaled error
                      runningError : bool      = False,
                      conf         : float     = 0.95,
                      ) -> Union[float, list[float]]:
    
    isUnderCurve = np.array([int(x) if x in [True, False] else x for x in isUnderCurve]) # Convert True and False to 1 and 0
    if set(isUnderCurve) - {0, 1} != set():
        raise ValueError('isUnderCurve must be a list of either all True/False, or 1/0')
    
    n = isUnderCurve.size
    if n == 1:
        error1sig = volume*1
    
    if not runningError:
        stdev = np.std(isUnderCurve, ddof=1)
        error1sig = volume*stdev/np.sqrt(n)
    
    else:
        cummean = np.cumsum(isUnderCurve)/np.arange(1 ,n+1)
        variances = (cummean - cummean**2)*n/(n-1) # Sample varaince eqn. which holds true only for values of only 0 and 1, where x == x**2
        stdevs = np.sqrt(variances)
        stdevs[stdevs == 0] = pct2sig(0.5) # Leading zeros will throw off plots, fill with reasonable dummy data
        error1sig = volume*stdevs/np.sqrt(np.arange(1, n+1))
    
    error = error1sig*pct2sig(conf)
    return error

'''
if __name__ == '__main__':
    generator = np.random.RandomState(seed = 24565213)
    generator.randint(0, 1, size=1000)

    #print(integration_error([0, 2, 3, 1, 2]))
    print(integration_error([True, False, True]))
    x = generator.randint(0, 2, size=1000)
    print(np.mean(x), integration_error(x, volume=1, conf=0.95))
    #print(integration_error(x, volume=1, runningError=True, conf=0.95))
    
    n = len(x)
    cummean = np.cumsum(x)/np.arange(1 ,n+1)
    err = integration_error(x, volume=1, runningError=True, conf=0.99)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hlines(0.5, 0., 1000, 'k')
    plt.plot(cummean)
    plt.plot(cummean+err)
    plt.plot(cummean-err)
    plt.figure()
    plt.loglog(err)
    plt.plot(np.abs(cummean - 0.5))
    plt.plot()
    
#'''