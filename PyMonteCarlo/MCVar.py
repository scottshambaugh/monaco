import numpy as np
from scipy.stats import rv_continuous, rv_discrete, describe
from PyMonteCarlo.MCVal import MCInVal, MCOutVal

### MCVar Base Class ###
class MCVar:
    def __init__(self, name, ndraws, firstcaseisnom):
        self.name = name                      # name is a string
        self.ndraws = ndraws                  # ndraws is an integer
        self.firstcaseisnom = firstcaseisnom  # firstcaseisnom is a boolean
        
        self.ncases = ndraws + 1
        self.setFirstCaseNom(firstcaseisnom)
        self.vals = []
        self.size = None
        self.isscalar = None
        

    def setFirstCaseNom(self, firstcaseisnom):  # firstdrawisnom is a boolean
        if firstcaseisnom:
           self.firstcaseisnom = True
           self.ncases = self.ndraws + 1
        else:
           self.firstcaseisnom = False
           self.ncases = self.ndraws


    def stats(self):
        stats = describe(self.vals)
        return stats


    def getVal(self, ncase):  # ncase is an integer
        raise NotImplementedError() # abstract method

    def getNom(self):
        raise NotImplementedError() # abstract method



### MCInVar Class ###
class MCInVar(MCVar):
    def __init__(self, name, dist, distargs, ndraws, seed=np.random.get_state()[1][0], firstcaseisnom=True):
        super().__init__(name=name, ndraws=ndraws, firstcaseisnom=firstcaseisnom)
        self.dist = dist          # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous 
        self.distargs = distargs  # distargs is a tuple of the arguments to the above distribution
        self.seed = seed          # seed is a number between 0 and 2^32-1
        
        self.size = (1, 1)
        self.isscalar = True

        if not isinstance(self.distargs, tuple):
            self.distargs = (self.distargs,)
        
        self.draw()


    def setNDraws(self, ndraws):  # ndraws is an integer
        self.ndraws = ndraws
        self.setFirstCaseNom(self.firstcaseisnom)
        self.draw()
        
        
    def draw(self):
        self.vals = []
        dist = self.dist(*self.distargs)

        if self.firstcaseisnom:
            self.ncases = self.ndraws + 1
            self.vals.append(self.getNom())
  
        np.random.seed(self.seed)
        self.vals.extend(dist.rvs(size=self.ndraws).tolist())


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
            return np.NaN

    def getVal(self, ncase):  # ncase is an integer
        isnom = False
        if (ncase == 0) and self.firstcaseisnom:
            isnom = True
            
        val = MCInVal(name=self.name, ncase=ncase, val=self.vals[ncase], dist=self.dist, isnom=isnom)
        return val



### MCOutVar Class ###
class MCOutVar(MCVar):
    def __init__(self, name, vals, ndraws=None, firstcaseisnom=True):
        if ndraws == None:
            ndraws = len(vals)
            if firstcaseisnom:
                ndraws = ndraws - 1
        
        super().__init__(name=name, ndraws=ndraws, firstcaseisnom=firstcaseisnom)
        self.vals = vals  # vals is a list
        
        if isinstance(vals[0],(list, tuple, np.ndarray)):
            if isinstance(vals[0][0],(list, tuple, np.ndarray)):
                self.size = (len(vals[0]), len(vals[0][0]))
            else:
                self.size = (1, len(vals[0]))
                self.isscalar = False
        else:
            self.size = (1, 1)
            self.isscalar = True


    def getVal(self, ncase):  # ncase is an integer
        isnom = False
        if (ncase == 0) and self.firstcaseisnom:
            isnom = True
            
        val = MCOutVal(name=self.name, ncase=ncase, val=self.vals[ncase], isnom=isnom)
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
                mcvars[name] = MCOutVar(name=name, vals=vals, ndraws=self.ndraws, firstcaseisnom=self.firstcaseisnom)
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
xk = np.array([1, 5, 6])
pk = np.ones(len(xk))/len(xk)
custom = rv_discrete(name='custom', values=(xk, pk))
mcinvars['custom'] = MCInVar('custom', custom, (), 1000, seed=invarseeds[2])
print(mcinvars['custom'].stats())
print(mcinvars['custom'].getVal(0).val)

mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)
print(mcoutvars['test'].getVal(1).val)
print(mcoutvars['test'].stats())

v = np.array([[1,1],[2,2],[3,3]])
mcoutvars['test2'] = MCOutVar('test2', [v, v, v, v, v])
mcoutvars.update(mcoutvars['test2'].split())
print(mcoutvars['test2 [0]'].vals)
#'''
