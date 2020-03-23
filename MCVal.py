
class MCVal():
    def __init__(self, val, name, ndraw, dist, isnom = False):
        self.val = val
        self.name = name
        self.ndraw = ndraw
        self.dist = dist
        self.isnom = isnom

'''
## Test ##
from scipy.stats import *
a = MCVal(1, 'Test', 0, norm)
print(a.val)
#'''
