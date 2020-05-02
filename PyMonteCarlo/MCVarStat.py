import numpy as np
from copy import copy
from PyMonteCarlo.order_statistics import order_stat_TI_k

class MCVarStat:
    def __init__(self, mcvar, p, c=0.95, bound='2-sided', name=None):
        self.mcvar = mcvar
        self.p = p
        self.c = c
        self.bound = bound
        self.name = name
        if self.name == None:
            self.name = f'{bound} P{p*100}/{c*100}% Confidence Interval'
        
        self.k = order_stat_TI_k(n=self.mcvar.ncases, p=self.p, c=self.c, bound=self.bound)
        self.nums = None
        self.vals = None
                
        self.genStats()
        
        
    def genStats(self):
        if self.mcvar.isscalar:
            sortednums = sorted(self.mcvar.nums)
            self.nums = np.array([sortednums[self.k-1], sortednums[-self.k]])
            self.vals = copy(self.nums)
            if self.mcvar.nummap != None:
                self.vals = np.array([self.mcvar.nummap[self.nums[0]], self.mcvar.nummap[self.nums[1]]])
                
        elif self.mcvar.size[0] == 1:
            self.nums = np.empty((self.mcvar.size[1], 2))
            for i in range(self.mcvar.size[1]):
                sortednums = sorted([x[i] for x in self.mcvar.nums])
                self.nums[i,:] = [sortednums[self.k-1], sortednums[-self.k]]
            self.vals = copy(self.nums)
            if self.mcvar.nummap != None:
                self.vals = np.array([[self.mcvar.nummap[x] for x in y] for y in self.nums])
                
        else:
            print('Warning: MCVarStat only available for scalar or 1-D data')
    


'''
### Test ###
if __name__ == '__main__':
    from scipy.stats import norm
    from PyMonteCarlo.MCVar import MCInVar, MCOutVar
    from PyMonteCarlo.order_statistics import sig2pct
    seed = 74494861
    
    mcinvar = MCInVar('norm', norm, (0, 1), 100000, seed=seed)
    bound='2-sided'
    mcinvarstat = MCVarStat(mcinvar, sig2pct(3, bound=bound), c=0.50, bound=bound)
    print(mcinvarstat.k)
    print(mcinvarstat.vals)
    
    v = np.array([-2, -1, 2, 3, 4, 5])
    var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
    mcoutvarstat = MCVarStat(var2, p=.6, c=0.50, bound=bound)
    print(mcoutvarstat.name)
    print(mcoutvarstat.vals)

#'''
