import numpy as np
from datetime import datetime
from MCCase import MCCase
from MCVar import MCVar

class MCSim:
    def __init__(self, name, ncases, seed=np.random.get_state()[1][0]):
        self.name = name          # name is a string
        self.ncases = ncases      # ndraws is an integer
        self.seed = seed          # seed is a number between 0 and 2^32-1
        np.random.seed(seed)
        
        self.starttime = datetime.now()
        self.endtime = None
        self.runtime = None
        
        self.mcvars = dict()     
        self.mccases = []        
        self.setNCases(self.ncases)


    def addVar(self, name, dist, distargs):  
        # name is a string
        # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous 
        # distargs is a tuple of the arguments to the above distribution
        self.mcvars[name] = MCVar(name, dist, distargs, self.ncases)


    def setNCases(self, ncases):  # ncases is an integer
        self.ncases = ncases
        np.random.seed(self.seed)
        if self.mcvars == dict():
            self.clearCases()
        else:
            for mcvar in self.mcvars.values():
                mcvar.setNDraws(ncases)
            if self.mccases != []:
                self.genCases()


    def genCases(self):
        self.clearCases()
        for ncase in range(self.ncases):
            self.mccases.append(MCCase(ncase, self.mcvars))


    def clearCases(self):
        self.mccases = []


    def clearVars(self):
        self.mcvars = dict()
        self.setNCases(self.ncases)


'''
## Test ##
from scipy.stats import *
np.random.seed(74494861)
sim = MCSim('Sim', 10)
sim.addVar('Var1', randint, (1, 5))
sim.addVar('Var2', norm, (10, 4))
sim.genCases()
print(sim.mcvars['Var1'].name)
print(sim.mccases[0].mcvals['Var1'].val)
print(sim.mcvars['Var2'].name)
print(sim.mccases[0].mcvals['Var2'].val)
#'''
