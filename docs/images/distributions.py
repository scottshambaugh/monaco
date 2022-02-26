from monaco import InVar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (uniform, norm, lognorm, expon,
                         randint, rv_discrete, bernoulli, binom, geom, poisson)

n = int(2e4)
method = 'random'

dist = InVar('Uniform', ndraws=n, dist=uniform, distkwargs={'loc': 1, 'scale': 4},
             samplemethod=method)
fig, _ = dist.plot(title="dist=uniform, distkwargs={'loc': 1, 'scale': 4}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/uniform.png', dpi=100)

dist = InVar('Normal', ndraws=n, dist=norm, distkwargs={'loc': 5, 'scale': 2},
             samplemethod=method)
fig, _ = dist.plot(title="dist=norm, distkwargs={'loc': 5, 'scale': 2}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/norm.png', dpi=100)

dist = InVar('Log-Normal', ndraws=n, dist=lognorm, distkwargs={'s': 0.5, 'scale': np.exp(1)},
             samplemethod=method)
fig, _ = dist.plot(title="dist=lognorm, distkwargs={'s': 0.5, 'scale': np.exp(1)}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/lognorm.png', dpi=100)

dist = InVar('Exponential', ndraws=n, dist=expon, distkwargs={'scale': 1/1},
             samplemethod=method)
fig, _ = dist.plot(title="dist=expon, distkwargs={'scale': 1/1}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/expon.png', dpi=100)

dist = InVar('Random Integers in Range', ndraws=n, dist=randint, distkwargs={'low': 1, 'high': 6+1},
             samplemethod=method)
fig, _ = dist.plot(title="dist=randint, distkwargs={'low': 1, 'high': 6+1}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/randint.png', dpi=100)

custom_dist = rv_discrete(values=([1, 2, 3], [0.1, 0.3, 0.6]))
dist = InVar('Random Integers with Custom Weights', ndraws=n, dist=custom_dist,
             distkwargs=dict(), samplemethod=method)
fig, _ = dist.plot(title="dist=rv_discrete(values=([1, 2, 3], [0.1, 0.3, 0.6]))")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/rv_discrete.png', dpi=100)

dist = InVar('Bernoulli', ndraws=n, dist=bernoulli, distkwargs={'p': 0.2},
             samplemethod=method)
fig, _ = dist.plot(title="dist=bernoulli, distkwargs={'p': 0.2}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/bernoulli.png', dpi=100)

dist = InVar('Binomial', ndraws=n, dist=binom, distkwargs={'n': 10, 'p': 0.2},
             samplemethod=method)
fig, _ = dist.plot(title="dist=binom, distkwargs={'n': 10, 'p': 0.2}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/binom.png', dpi=100)

dist = InVar('Geometric', ndraws=n, dist=geom, distkwargs={'p': 0.2},
             samplemethod=method)
fig, _ = dist.plot(title="dist=geom, distkwargs={'p': 0.2}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/geom.png', dpi=100)

dist = InVar('Poisson', ndraws=n, dist=poisson, distkwargs={'mu': 2},
             samplemethod=method)
fig, _ = dist.plot(title="dist=poisson, distkwargs={'mu': 2}")
fig.set_size_inches(6.0, 2.5)
plt.savefig('docs/images/poisson.png', dpi=100)
