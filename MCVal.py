
class MCVal():
    def __init__(self, name, ndraw, val, dist, isnom = False):
        self.name = name    # name is a string
        self.ndraw = ndraw  # ncdraw is an integer
        self.val = val      # val is a number
        self.dist = dist    # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous
        self.isnom = isnom  # isnom is a boolean

'''
## Test ##
from scipy.stats import *
a = MCVal(1, 'Test', 0, norm)
print(a.val)
#'''
