# test_integration_statistics.py

import pytest
from Monaco.integration_statistics import integration_error, integration_n_from_err, max_variance, max_stdev

def test_integration_error():
    assert integration_error([1, 0, 2], runningError=False, conf=0.95) == pytest.approx(1.1315857)
    assert integration_error([1, 0, 2], runningError=True , conf=0.95) == pytest.approx([1.9599639, 0.9799819, 1.1315857])


def test_integration_n_from_err():
    assert integration_n_from_err(error=0.01, volume=1, stdev=1, conf=0.95) == 38415


def test_max_variance():
    assert max_variance(low=0, high=1) == pytest.approx(0.25)


def test_max_stdev():
    assert max_stdev(low=0, high=1) == pytest.approx(0.5)


### Inline Testing ###
# Can run here or copy into bottom of main file
#'''
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    print(integration_error([1, 0, 2])) # Expected: 1.1315857
    print(integration_error([1, 0, 2], runningError=True, conf=0.95))       # Expected: [1.9599639, 0.9799819, 1.1315857]
    print(integration_n_from_err(error=0.01, volume=1, stdev=1, conf=0.95)) # Expected: 38415
    print(max_variance(low=0, high=1))  # Expected: 0.25
    print(max_stdev(low=0, high=1))     # Expected: 0.5
    
    n = int(1e4)
    conf = 0.95
    generator = np.random.RandomState(seed=25113604)
    midpoint = 5
    x = generator.randint(0, 2*midpoint+1, size=n)
    
    cummean = np.cumsum(x)/np.arange(1 ,n+1)
    err = integration_error(x, volume=1, runningError=True, conf=0.95)
    plt.figure()
    plt.hlines(midpoint, 0, n, 'k')
    plt.plot(cummean, 'r')
    plt.plot(cummean+err, 'b')
    plt.plot(cummean-err, 'b')
    plt.ylim((midpoint*0.75, midpoint*1.25))
    plt.ylabel(f'{round(conf*100, 2)}% Confidence Integration Bounds')
    
    plt.figure()
    plt.loglog(err, 'r')
    plt.plot(np.abs(cummean - midpoint), 'b')
    plt.ylabel(f'{round(conf*100, 2)}% Confidence Absolute Error')
    
#'''