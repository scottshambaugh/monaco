import numpy as np

### MCVal Base Class ###
class MCVal():
    def __init__(self, name, ncase, val, isnom):
        self.name = name    # name is a string
        self.ncase = ncase  # ncase is an integer
        self.val = val      # val can be anything
        self.isnom = isnom  # isnom is a boolean
        
        self.size = None



### MCInVal Class ###
class MCInVal(MCVal):
    def __init__(self, name, ncase, val, dist, isnom = False):
        super().__init__(name, ncase, val, isnom)
        self.dist = dist    # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous
        
        self.size = (1, 1)



### MCOutVal Class ###
class MCOutVal(MCVal):
    def __init__(self, name, ncase, val, isnom = False):
        super().__init__(name, ncase, val, isnom)
        
        if isinstance(val,(list, tuple, np.ndarray)):
            if isinstance(val[0],(list, tuple, np.ndarray)):
                self.size = (len(val), len(val[0]))
            else:
                self.size = (1, len(val))
        else:
            self.size = (1, 1)

    def split(self):
        mcvals = dict()
        if self.size[0] > 1:
            for i in range(self.size[0]):
                name = self.name + f' [{i}]'
                mcvals[name] = MCOutVal(name=name, ncase=self.ncase, val=self.val[i], isnom=self.isnom)
        return mcvals


'''
### Test ###
from scipy.stats import norm
a = MCInVal('Test', 1, 0, norm, True)
print(a.val)
b = MCOutVal('Test', 1, [[0,0],[0,0],[0,0]], True)
print(b.size)
print(b.val)
bsplit = b.split()
print(bsplit['Test [0]'].val)
#'''
