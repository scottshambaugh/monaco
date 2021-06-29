# MCVar.py

import numpy as np
from scipy.stats import rv_continuous, rv_discrete, describe
from Monaco.MCVal import MCInVal, MCOutVal
from Monaco.MCVarStat import MCVarStat
from Monaco.mc_sampling import mc_sampling
from copy import copy
from typing import Union, Any
from warnings import warn


### MCVar Base Class ###
class MCVar:
    def __init__(self, 
                 name           : str, 
                 ndraws         : int, 
                 firstcaseisnom : bool,
                 ):
        self.name = name 
        self.ndraws = ndraws 
        self.firstcaseisnom = firstcaseisnom
        
        self.ncases = ndraws + 1
        self.setFirstCaseNom(firstcaseisnom)
        self.vals = []
        self.valmap = None
        self.nums = []
        self.nummap = None
        self.pcts = []
        self.size = None
        self.isscalar = None
        self.mcvarstats = []
        

    def setFirstCaseNom(self, 
                        firstcaseisnom : bool,
                        ):
        if firstcaseisnom:
           self.firstcaseisnom = True
           self.ncases = self.ndraws + 1
        else:
           self.firstcaseisnom = False
           self.ncases = self.ndraws


    def stats(self):
        stats = describe(self.nums)
        return stats
    
    
    def addVarStat(self, 
                   stattype   : str, 
                   statkwargs : dict[str, Any]   = dict(), 
                   name       : Union[None, str] = None,
                   ):
        self.mcvarstats.append(MCVarStat(mcvar=self, stattype=stattype, statkwargs=statkwargs, name=name))


    def clearVarStats(self):
        self.mcvarstats = []


    def getVal(self, 
               ncase : int,
               ):
        raise NotImplementedError() # abstract method

    def getNom(self):
        raise NotImplementedError() # abstract method



### MCInVar Class ###
class MCInVar(MCVar):
    def __init__(self, 
                 name           : str, 
                 ndraws         : int, 
                 dist           : Union[rv_discrete, rv_continuous], 
                 distkwargs     : dict                        = dict(),
                 nummap         : Union[None, dict[int, Any]] = None,
                 samplemethod   : str                         = 'sobol_random',
                 ninvar         : int                         = None,
                 seed           : int                         = np.random.get_state()[1][0], 
                 firstcaseisnom : bool                        = True,
                 autodraw       : bool                        = True,
                 ):
        super().__init__(name=name, ndraws=ndraws, firstcaseisnom=firstcaseisnom)
        
        self.dist = dist  
        self.distkwargs = distkwargs
        self.samplemethod = samplemethod
        self.ninvar = ninvar 
        self.seed = seed 
        self.nummap = nummap 
        
        self.isscalar = True
        self.size = (1, 1)
        
        self.genValMap()
        if autodraw:
            self.draw(ninvar_max=None)


    def mapNums(self):
        self.vals = copy(self.nums)
        if self.nummap is not None:
            for i in range(self.ncases):
                self.vals[i] = self.nummap[self.nums[i]]
            
            
    def genNumMap(self):
        if self.nummap is None:
            self.nummap = {val:num for num, val in self.nummap.items()}


    def genValMap(self):
        if self.nummap is None:
            self.valmap = None
        else:
            self.valmap = {val:num for num, val in self.nummap.items()}


    def setNDraws(self, 
                  ndraws : int,
                  ):
        self.ndraws = ndraws
        self.setFirstCaseNom(self.firstcaseisnom)
        
        
    def draw(self, 
             ninvar_max : int = None,
             ):
        self.pcts = []
        self.nums = []
        dist = self.dist(**self.distkwargs)

        if self.firstcaseisnom:
            self.ncases = self.ndraws + 1
            nom_num = self.getNom()
            self.nums.append(nom_num)
            self.pcts.append(dist.cdf(nom_num))
            
        pcts = mc_sampling(ndraws=self.ndraws, method=self.samplemethod, ninvar=self.ninvar, ninvar_max=ninvar_max, seed=self.seed)
        self.pcts.extend(pcts)
        self.nums.extend(dist.ppf(pcts).tolist())
        
        if any(np.isinf(num) for num in self.nums):
            warn(f'Infinite value drawn. Check distribution and parameters: {self.dist=}, {self.distkwargs=}')
            if self.samplemethod in ('sobol', 'halton'):    
                warn(f"Infinite value draw may happen with {self.dist=} for the first point of the {self.samplemethod} sampling method. Consider using 'sobol_random' instead.")

        if any(np.isnan(num) for num in self.nums):
            raise ValueError(f'Invalid draw. Check distribution and parameters: {self.dist=}, {self.distkwargs=}')

        self.mapNums()


    def getNom(self):
        dist = self.dist(**self.distkwargs)
        ev = dist.expect()
        
        if isinstance(self.dist, rv_continuous):
            return ev

        # For a discrete distribution, we take the nearest discrete value closest to the expected value
        elif isinstance(self.dist, rv_discrete):
            eps = np.finfo(float).eps
            p = dist.cdf(ev)
            ev_candidates = dist.ppf([p - eps, p, p + eps])
            ev_candidates_dist = abs(ev_candidates - ev)
            ev_closest = ev_candidates[np.nanargmin(ev_candidates_dist)]
            return ev_closest
        
        else:
            return None


    def getVal(self, 
               ncase : int,
               ):
        isnom = False
        if (ncase == 0) and self.firstcaseisnom:
            isnom = True
            
        val = MCInVal(name=self.name, ncase=ncase, pct=self.pcts[ncase], num=self.nums[ncase], dist=self.dist, nummap=self.nummap, isnom=isnom)
        return val



### MCOutVar Class ###
class MCOutVar(MCVar):
    def __init__(self, 
                 name           : str, 
                 vals           : list[Any], 
                 valmap         : Union[None, dict[Any, int]] = None, 
                 ndraws         : Union[None, int]            = None, 
                 firstcaseisnom : bool                        = True,
                 ):
        if ndraws is None:
            ndraws = len(vals)
            if firstcaseisnom:
                ndraws = ndraws - 1
        
        super().__init__(name=name, ndraws=ndraws, firstcaseisnom=firstcaseisnom)
        self.vals = vals
        self.valmap = valmap
        if valmap is None:
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
        if Val0.valmap is not None:
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
        if self.valmap is None:
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
                    mcvars[name].addVarStat(stattype=mcvarstat.stattype, statkwargs=mcvarstat.statkwargs, name=mcvarstat.name)
        return mcvars

