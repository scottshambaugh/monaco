import numpy as np
from scipy.stats import rv_continuous, rv_discrete, describe
from PyMonteCarlo.MCVal import MCInVal, MCOutVal
from PyMonteCarlo.MCVarStat import MCVarStat
from copy import copy
from helper_functions import get_iterable

### MCVar Base Class ###
class MCVar:
    def __init__(self, name, ndraws, firstcaseisnom):
        self.name = name                      # name is a string
        self.ndraws = ndraws                  # ndraws is an integer
        self.firstcaseisnom = firstcaseisnom  # firstcaseisnom is a boolean
        
        self.ncases = ndraws + 1
        self.setFirstCaseNom(firstcaseisnom)
        self.vals = []
        self.valmap = None
        self.nums = []
        self.nummap = None
        self.size = None
        self.isscalar = None
        self.mcvarstats = []
        

    def setFirstCaseNom(self, firstcaseisnom):  # firstdrawisnom is a boolean
        if firstcaseisnom:
           self.firstcaseisnom = True
           self.ncases = self.ndraws + 1
        else:
           self.firstcaseisnom = False
           self.ncases = self.ndraws


    def stats(self):
        stats = describe(self.nums)
        return stats
    
    
    def addVarStat(self, stattype, statkwargs={}, name=None):
        self.mcvarstats.append(MCVarStat(mcvar=self, stattype=stattype, statkwargs=statkwargs, name=name))


    def clearVarStats(self):
        self.mcvarstats = []


    def getVal(self, ncase):  # ncase is an integer
        raise NotImplementedError() # abstract method

    def getNom(self):
        raise NotImplementedError() # abstract method



### MCInVar Class ###
class MCInVar(MCVar):
    def __init__(self, name, dist, distargs, ndraws, nummap=None, seed=np.random.get_state()[1][0], firstcaseisnom=True):
        super().__init__(name=name, ndraws=ndraws, firstcaseisnom=firstcaseisnom)
        self.dist = dist                        # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous 
        self.distargs = get_iterable(distargs)  # distargs is a tuple of the arguments to the above distribution
        self.seed = seed                        # seed is a number between 0 and 2^32-1
        self.nummap = nummap                    # nummap is a dict
        
        self.isscalar = True
        self.size = (1, 1)
        
        self.genValMap()
        self.draw()


    def mapNums(self):
        self.vals = copy(self.nums)
        if self.nummap != None:
            for i in range(self.ncases):
                self.vals[i] = self.nummap[self.nums[i]]
            
            
    def genNumMap(self):
        if self.nummap == None:
            self.nummap = {val:num for num, val in self.nummap.items()}


    def genValMap(self):
        if self.nummap == None:
            self.valmap = None
        else:
            self.valmap = {val:num for num, val in self.nummap.items()}


    def setNDraws(self, ndraws):  # ndraws is an integer
        self.ndraws = ndraws
        self.setFirstCaseNom(self.firstcaseisnom)
        self.draw()
        
        
    def draw(self):
        self.nums = []
        dist = self.dist(*self.distargs)

        if self.firstcaseisnom:
            self.ncases = self.ndraws + 1
            self.nums.append(self.getNom())
  
        np.random.seed(self.seed)
        self.nums.extend(dist.rvs(size=self.ndraws).tolist())
        self.mapNums()


    def getNom(self):
        dist = self.dist(*self.distargs)
        ev = dist.expect()
        
        if isinstance(self.dist, rv_continuous):
            return ev

        # For a discrete distribution, we take the nearest discrete value closest to the expected value
        elif isinstance(self.dist, rv_discrete):
            eps = np.finfo(float).eps
            p = dist.cdf(ev)
            ev_candidates = dist.ppf([p - eps, p, p + eps])
            ev_candidates_dist = abs(ev_candidates - ev)
            ev_closest = ev_candidates[np.argmin(ev_candidates_dist)]
            return ev_closest
        
        else:
            return None


    def getVal(self, ncase):  # ncase is an integer
        isnom = False
        if (ncase == 0) and self.firstcaseisnom:
            isnom = True
            
        val = MCInVal(name=self.name, ncase=ncase, num=self.nums[ncase], dist=self.dist, nummap=self.nummap, isnom=isnom)
        return val



### MCOutVar Class ###
class MCOutVar(MCVar):
    def __init__(self, name, vals, valmap=None, ndraws=None, firstcaseisnom=True):
        if ndraws == None:
            ndraws = len(vals)
            if firstcaseisnom:
                ndraws = ndraws - 1
        
        super().__init__(name=name, ndraws=ndraws, firstcaseisnom=firstcaseisnom)
        self.vals = vals      # vals is a list
        self.valmap = valmap
        if valmap == None:
            self.extractValMap()
        self.genSize()
        self.genNumMap()
        self.mapVals()


    def genSize(self):
        if isinstance(self.vals[0],(list, tuple, np.ndarray)):
            self.isscalar = False
            if isinstance(self.vals[0][0],(list, tuple, np.ndarray)):
                self.size = (len(self.vals[0]), len(self.vals[0][0]))
            else:
                self.size = (1, len(self.vals[0]))
        else:
            self.isscalar = True
            self.size = (1, 1)
            
    
    def extractValMap(self):
        Val0 = self.getVal(0)
        if Val0.valmap != None:
            if Val0.valmapsource == 'auto':
                uniquevals = set()
                for i in range(self.ncases):
                    uniquevals.update(self.getVal(i).valmap.keys())
                self.valmap = dict()
                for i, val in enumerate(uniquevals):
                    self.valmap[val] = i
            else:
                self.valmap = Val0.valmap


    def genNumMap(self):
        if self.valmap == None:
            self.nummap = None
        else:
            self.nummap = {num:val for val, num in self.valmap.items()}


    def mapVals(self):
        self.nums = copy(self.vals)
        for i in range(self.ncases):
            self.nums[i] = self.getVal(i).num  


    def getVal(self, ncase):  # ncase is an integer
        isnom = False
        if (ncase == 0) and self.firstcaseisnom:
            isnom = True
            
        val = MCOutVal(name=self.name, ncase=ncase, val=self.vals[ncase], valmap=self.valmap, isnom=isnom)
        return val
        
    
    def getNom(self):
        val = None
        if self.firstcaseisnom:
            val = self.vals[0]            
        return val
    
    
    def split(self):
        mcvars = dict()
        if self.size[0] > 1:
            for i in range(self.size[0]):
                name = self.name + f' [{i}]'
                vals = []
                for j in range(self.ncases):
                    vals.append(self.vals[j][i])
                mcvars[name] = MCOutVar(name=name, vals=vals, ndraws=self.ndraws, \
                                        valmap=self.valmap, firstcaseisnom=self.firstcaseisnom)
                for mcvarstat in self.mcvarstats:
                    mcvars[name].addVarStat(p=mcvarstat.p, c=mcvarstat.c, bound=mcvarstat.bound, name=mcvarstat.name)
        return mcvars



'''
### Test ###
from scipy.stats import norm, randint
generator = np.random.RandomState(74494861)
invarseeds = generator.randint(0, 2**31-1, size=10)

mcinvars = dict()
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000, seed=invarseeds[0])
print(mcinvars['randint'].stats())
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000, seed=invarseeds[1])
print(mcinvars['norm'].stats())
mcinvars['norm'].addVarStat(p=0.75, c=0.95, bound='2-sided')
print(mcinvars['norm'].mcvarstats[0].vals)
xk = np.array([1, 5, 6])
pk = np.ones(len(xk))/len(xk)
custom = rv_discrete(name='custom', values=(xk, pk))
mcinvars['custom'] = MCInVar('custom', custom, (), 1000, seed=invarseeds[2])
print(mcinvars['custom'].stats())
print(mcinvars['custom'].vals[1:10])
print(mcinvars['custom'].getVal(0).val)
mcinvars['map'] = MCInVar('map', custom, (), 10, nummap={1:'a',5:'e',6:'f'}, seed=invarseeds[3])
print(mcinvars['map'].vals)
print(mcinvars['map'].stats())

mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)
print(mcoutvars['test'].getVal(1).val)
print(mcoutvars['test'].stats())

v = np.array([[1,1],[2,2],[3,3]])
mcoutvars['test2'] = MCOutVar('test2', [v, v, v, v, v])
mcoutvars['test2'].addVarStat(p=0.25, c=0.50, bound='1-sided')
mcoutvars.update(mcoutvars['test2'].split())
print(mcoutvars['test2 [0]'].nums)
print(mcoutvars['test2 [0]'].mcvarstats[0].vals)
#'''
