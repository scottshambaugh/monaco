
class MCVal():
    def __init__(self, name, ncase, val, isnom = False):
        self.name = name    # name is a string
        self.ncase = ncase  # ncase is an integer
        self.isnom = isnom  # isnom is a boolean
        self.val = val

class MCInVal(MCVal):
    def __init__(self, name, ncase, val, dist, isnom = False):
        super().__init__(name, ncase, val, isnom)
        self.dist = dist    # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous

class MCOutVal(MCVal):
    # No additional methods or variables
    pass


'''
### Test ###
from scipy.stats import *
a = MCInVal('Test', 1, 0, norm)
print(a.val)
b = MCOutVal('Test', 1, 2)
print(b.val)
#'''
