from scipy.stats._distn_infrastructure import rv_sample

class const(rv_sample):
    def __init__(self, val, *args, **kwds):
        super(const, self).__init__(values=(val, 1), *args, **kwds)


'''
### Test ###
dist = const(5)
print(dist.expect())
from scipy.stats import poisson
import inspect
print('\nThese should both be rv_discrete:')
print('1: ', inspect.getmro(poisson.__class__))
print('2: ', inspect.getmro(const.__class__))
print('\nThese should both be rv_frozen:')
print('1: ', inspect.getmro(poisson(5).__class__))
print('2: ', inspect.getmro(const(5).__class__))
#'''