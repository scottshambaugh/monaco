#import datetime
#from MCVar import MCVar
#from MCVal import MCVal

class MCCase():
    def __init__(self, ncase, mcvars):
        self.ncase = ncase    # ncase is an integer
        self.mcvars = mcvars  # mcvars is a single, list, or tuple of multiple MCVar objects
        self.starttime = None
        self.endtime = None
        self.runtime = None
        
        # Package length 1 into an interable tuple
        if not isinstance(self.mcvars, list) and not isinstance(self.mcvars, tuple):
            self.mcvars = (self.mcvars,)
        
        self.mcvals = self.getMCVals()
        
    def getMCVals(self):
        mcvals = []
        for mcvar in self.mcvars:
            mcvals.append(mcvar.getVal(self.ncase))
        return mcvals

'''
## Test ##
import numpy as np
from scipy.stats import *
from MCVar import MCVar
np.random.seed(74494861)
var = MCVar('Test', norm, (10, 4), 10)
case = MCCase(0, (var))
print(case.mcvals[0].val)
#'''
