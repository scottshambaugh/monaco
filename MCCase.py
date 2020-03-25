#from datetime import datetime

class MCCase():
    def __init__(self, ncase, mcvars):
        self.ncase = ncase    # ncase is an integer
        self.mcvars = mcvars  # mcvars is a dict of MCVar objects
        self.starttime = None
        self.endtime = None
        self.runtime = None
        
        self.mcvals = self.getMCVals()


    def getMCVals(self):
        mcvals = dict()
        for mcvar in self.mcvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals


'''
## Test ##
import numpy as np
from scipy.stats import *
from MCVar import MCVar
np.random.seed(74494861)
var = {'Test':MCVar('Test', norm, (10, 4), 10)}
case = MCCase(0, var)
print(case.mcvals['Test'].val)
#'''
