#from datetime import datetime

class MCCase():
    def __init__(self, ncase, mcinvars):
        self.ncase = ncase    # ncase is an integer
        self.mcinvars = mcinvars  # mcvars is a dict of MCVar objects
        self.mcoutvars = dict()
        self.starttime = None
        self.endtime = None
        self.runtime = None
        
        self.mcinvals = self.getMCInVals()
        self.mcoutvals = dict()


    def getMCInVals(self):
        mcvals = dict()
        for mcvar in self.mcinvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals

    def getMCOutVals(self):
        mcvals = dict()
        for mcvar in self.mcoutvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals
    

'''
### Test ###
import numpy as np
from scipy.stats import *
from MCVar import *
np.random.seed(74494861)
var = {'Test':MCInVar('Test', norm, (10, 4), 10)}
case = MCCase(0, var)
print(case.mcvals['Test'].val)
#'''
