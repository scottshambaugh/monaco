# test_integration_statistics.py

import pytest
from Monaco.integration_statistics import integration_error, integration_n_from_err

def test_integration_error():
    assert integration_error([True, False, True])          == pytest.approx(                       0.6533213)
    assert integration_error([1, 0, 1], runningError=True) == pytest.approx([1.3219756, 0.8486893, 0.6533213])


def test_integration_n_from_err():
    assert integration_n_from_err(error=0.01, volume=1, conf=0.95) == 9025


### Inline Testing ###
# Can run here or copy into bottom of main file
#'''
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    #print(integration_error([0, 2, 3, 1, 2]))
    print(integration_error([True, False, True]))  # Expected: 0.6533213
    print(integration_error([1, 0, 1], runningError=True))  # Expected: [1.3219756, 0.8486893, 0.6533213]
    
    print(integration_n_from_err(error=0.01, volume=1, conf=0.95))  # Expected: 9025
    
    n = int(1e4)
    conf = 0.95
    generator = np.random.RandomState(seed=2516214)
    x = generator.randint(0, 2, size=n)
    
    cummean = np.cumsum(x)/np.arange(1 ,n+1)
    err = integration_error(x, volume=2, runningError=True, conf=0.95)
    plt.figure()
    plt.hlines(0.5, 0, n, 'k')
    plt.plot(cummean, 'r')
    plt.plot(cummean+err, 'b')
    plt.plot(cummean-err, 'b')
    plt.ylim((0, 1))
    plt.ylabel(f'{round(conf*100, 2)}% Confidence Integration Bounds')
    
    plt.figure()
    plt.loglog(err, 'r')
    plt.plot(np.abs(cummean - 0.5), 'b')
    plt.ylabel(f'{round(conf*100, 2)}% Confidence Absolute Error')
    
#'''