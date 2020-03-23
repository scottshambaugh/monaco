import numpy as np
from datetime import datetime
from MCCase import MCCase
from MCVar import MCVar

class MCSim:
    def __init__(self, name, ncases, mcvars=[], seed=np.random.get_state()[1][0]):
        self.name = name          # name is a string
        self.ncases = ncases      # ndraws is an integer
        self.seed = seed          # seed is a number between 0 and 2^32-1
        np.random.seed(seed)
        
        self.starttime = datetime.now()
        self.endtime = None
        self.runtime = None
        
        self.mccases = []        
        self.mcvars = mcvars  # mcvar is a single or list of MCVar objects
        # Package length 1 into an interable list
        if not isinstance(self.mcvars, list):
            self.mcvars = [self.mcvars,]
        
        self.setNCases(self.ncases)
        
        
    def addVar(self, name, dist, distargs):  # mcvar is a single MCVar object 
        self.mcvars.append(MCVar(name, dist, distargs, self.ncases))
      
        
    def setNCases(self, ncases):
        self.ncases = ncases
        np.random.seed(self.seed)
        if (self.mcvars != []):
            for mcvar in self.mcvars:
                mcvar.setNDraws(ncases)
            if self.mccases != []:
                self.genCases()
        else:
            self.clearCases()
    

    def genCases(self):
        self.mccases = []
        for ncase in range(self.ncases):
            self.mccases.append(MCCase(ncase, self.mcvars))
            
            
    def clearCases(self):
        self.mccases = []


'''
## Test ##
from scipy.stats import *
np.random.seed(74494861)
var1 = MCVar('Var1', randint, (1, 5), 20)
sim = MCSim('Sim', 10, var1)
sim.addVar('Var2', norm, (10, 4))
sim.genCases()
print(sim.mcvars[0].name)
print(sim.mccases[0].mcvals[0].val)
print(sim.mcvars[1].name)
print(sim.mccases[0].mcvals[1].val)
#'''
