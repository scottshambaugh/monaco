# n_sphere_sampling_testing.py
from __future__ import annotations

import numpy as np
import scipy
import matplotlib.pyplot as plt
from monaco.mc_sampling import sampling
from monaco.integration_statistics import  integration_args_check
from monaco.gaussian_statistics import pct2sig
from monaco.mc_enums import SampleMethod

n = int(2**19)
conf = 0.95
seed = 251060543
d = 2
r = 1
volume_scale = 1.1
volume = (2*r*volume_scale)**d

def unit_d_sphere_vol(d):
    return np.pi**(d/2) / scipy.special.gamma(d/2 + 1)*(r**d)


trueans = unit_d_sphere_vol(d)


def integration_error(nums         : list[float],
                      dimension    : int          = None,
                      volume       : float        = 1,
                      conf         : float        = 0.95,
                      samplemethod : SampleMethod = SampleMethod.RANDOM,
                      runningerror : bool         = False,
                      ) -> float | list[float]:

    integration_args_check(error=None, volume=volume, stdev=None, conf=conf,
                           samplemethod=samplemethod, dimension=dimension)

    n = len(nums)
    if n == 1:
        error1sig = volume

    elif not runningerror:
        stdev = np.std(nums, ddof=1)
        if samplemethod == SampleMethod.RANDOM:
            error1sig = volume*stdev/np.sqrt(n)
        elif samplemethod == SampleMethod.SOBOL:
            error1sig = volume*stdev*np.log(n)**dimension/n

    else:
        # Use Welford's algorithm to calculate the running variance
        M = np.zeros(n)  # Running mean
        S = np.zeros(n)  # Sum of variances
        M[0] = nums[0]
        for i in range(1, n):
            M[i] = M[i-1] + (nums[i]-M[i-1])/(i+1)
            S[i] = S[i-1] + (nums[i]-M[i-1])*(nums[i]-M[i])
        variances = np.zeros(n)
        variances[1:] = S[1:]/np.arange(1, n)
        stdevs = np.sqrt(variances)

        error1sig_random = volume*np.sqrt((2**(-1*dimension) - 3**(-1*dimension))/np.arange(1, n+1))
        if samplemethod == SampleMethod.RANDOM:
            error1sig = error1sig_random
        elif samplemethod == SampleMethod.SOBOL:
            error1sig_sobol = volume*stdevs*np.log(np.arange(1, n+1))**dimension/np.arange(1, n+1)
            error1sig = error1sig_sobol

        # Leading zeros will throw off plots, fill with reasonable dummy data
        error1sig[error1sig == 0] = max(error1sig)

    error = error1sig*pct2sig(conf)
    return error


samplepointsrandom = []
samplepointssobol = []
for i in range(d):
    samplepointsrandom.append(sampling(ndraws=n, method=SampleMethod.RANDOM,
                              ninvar=i+1, seed=seed+i))  # Need different seed for random draws
    samplepointssobol.append( sampling(ndraws=n, method=SampleMethod.SOBOL,
                              ninvar=i+1, seed=None, ninvar_max=d))

# Discrepancy calculations take too long
'''
samplepointsrandom = np.array(samplepointsrandom).transpose()
samplepointsrandom = np.reshape(samplepointsrandom, (n, d))
errrandom = scipy.stats.qmc.discrepancy(samplepointsrandom)
samplepointssobol = np.array(samplepointssobol).transpose()
samplepointssobol = np.reshape(samplepointssobol, (n, d))
errsobol = scipy.stats.qmc.discrepancy(samplepointssobol)
'''

distancerandom = [np.sqrt(np.sum([samplepointsrandom[i][j]**2
                  for i in range(d)])) for j in range(n)]
distancesobol  = [np.sqrt(np.sum([ samplepointssobol[i][j]**2
                  for i in range(d)])) for j in range(n)]

insphererandom = np.array([int(x*r*volume_scale < r) for x in distancerandom])
inspheresobol  = np.array([int(x*r*volume_scale < r) for x in distancesobol])

cummeanrandom = volume*np.cumsum(insphererandom) / np.arange(1, n+1)
cummeansobol  = volume*np.cumsum(inspheresobol)  / np.arange(1, n+1)

errrandom = integration_error(distancerandom, dimension=d, volume=volume,
                              conf=conf, samplemethod=SampleMethod.RANDOM,
                              runningerror=True)
errsobol  = integration_error(distancesobol,  dimension=d, volume=volume,
                              conf=conf, samplemethod=SampleMethod.SOBOL,
                              runningerror=True)

# '''
alpha = 0.85
plt.figure()
plt.hlines(trueans, 0, n, 'k')
h1, = plt.plot(cummeanrandom, 'lightcoral', alpha=alpha)
h3, = plt.plot(cummeanrandom + errrandom, 'turquoise', alpha=alpha)
plt.plot(cummeanrandom - errrandom, 'turquoise', alpha=alpha)
h2, = plt.plot(cummeansobol , 'darkred', alpha=alpha)
h4, = plt.plot(cummeansobol + errsobol, 'darkblue', alpha=alpha)
plt.plot(cummeansobol - errsobol, 'darkblue', alpha=alpha)
plt.ylim((trueans*0.99, trueans*1.01))
plt.ylabel(f'{round(conf*100, 2)}% Confidence Integration Bounds')
plt.xlabel('Sample #')
plt.legend([h3, h1, h4, h2],
           ['Random Error Bound', 'Random True Error', 'Sobol Error Bound', 'Sobol True Error'])
plt.title(f'Monte Carlo Integration of {d}D Unit Sphere')

plt.figure()
h1, = plt.plot(np.abs(cummeanrandom - trueans), 'lightcoral', alpha=alpha)
h2, = plt.plot(np.abs(cummeansobol  - trueans), 'darkred', alpha=alpha)
h3, = plt.loglog(errrandom, 'turquoise', alpha=alpha)
h4, = plt.loglog(errsobol, 'darkblue', alpha=alpha)
plt.ylabel(f'{round(conf*100, 2)}% Confidence Absolute Error')
plt.xlabel('Sample #')
plt.legend([h3, h1, h4, h2],
           ['Random Error Bound', 'Random True Error', 'Sobol Error Bound', 'Sobol True Error'])
plt.title(f'Monte Carlo Integration of {d}D Unit Sphere')
# '''
