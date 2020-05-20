import numpy as np
from copy import copy
from scipy.stats import mode
from scipy.stats.mstats import gmean
from PyMonteCarlo.order_statistics import order_stat_TI_k

class MCVarStat:
    def __init__(self, mcvar, stattype, statkwargs = {}, name=None):
        self.mcvar = mcvar
        self.stattype = stattype
        self.statkwargs = statkwargs
        self.name = name

        self.nums = None
        self.vals = None
        
        if stattype == 'max':
            self.genStatsMax()
        elif stattype == 'min':
            self.genStatsMin()
        elif stattype == 'median':
            self.genStatsMedian()
        elif stattype == 'mean':
            self.genStatsMean()
        elif stattype == 'geomean':
            self.genStatsGeoMean()
        elif stattype == 'mode':
            self.genStatsMode()
        elif stattype == 'orderstatTI':
            self.genStatsOrderStatTI()


    def genStatsMax(self):
        self.name = 'Max'
        self.genStatsFunction(fcn=np.max)


    def genStatsMin(self):
        self.name = 'Min'
        self.genStatsFunction(fcn=np.min)


    def genStatsMedian(self):
        self.name = 'Median'
        self.genStatsFunction(fcn=np.median)


    def genStatsMean(self):
        self.name = 'Mean'
        self.genStatsFunction(fcn=np.mean)


    def genStatsGeoMean(self):
        self.name = 'Geometric Mean'
        self.genStatsFunction(fcn=gmean)


    def genStatsMode(self):
        self.name = 'Mode'
        self.genStatsFunction(fcn=mode)



    def genStatsFunction(self, fcn):
        if self.mcvar.isscalar:
            self.nums = fcn(self.mcvar.nums, **self.statkwargs)
            self.vals = copy(self.nums)
            if self.mcvar.nummap != None:
                self.vals = self.mcvar.nummap[self.nums]
                
        elif self.mcvar.size[0] == 1:
            self.nums = np.empty(self.mcvar.size[1])
            for i in range(self.mcvar.size[1]):
                self.nums[i] = fcn([x[i] for x in self.mcvar.nums])
            self.vals = copy(self.nums)
            if self.mcvar.nummap != None:
                self.vals = np.array([[self.mcvar.nummap[x] for x in y] for y in self.nums])
                
        else:
            # Suppress warning since this will become valid when MCVar is split
            #print('Warning: MCVarStat only available for scalar or 1-D data')
            pass

        
        
    def genStatsOrderStatTI(self):
        if 'p' not in self.statkwargs:
            raise ValueError(f'{self.stattype} requires the kwarg ''p''')
        else:
            self.p = self.statkwargs['p']
        if 'c' not in self.statkwargs:
            self.c = 0.95
        else:
            self.c = self.statkwargs['c']
        if 'bound' not in self.statkwargs:
            self.bound = '2-sided'
        else:
            self.bound = self.statkwargs['bound']
        
        if self.name == None:
            self.name = f'{self.bound} P{self.p*100}/{self.c*100}% Confidence Interval'

        self.k = order_stat_TI_k(n=self.mcvar.ncases, p=self.p, c=self.c, bound=self.bound)

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
            # Suppress warning since this will become valid when MCVar is split
            #print('Warning: MCVarStat only available for scalar or 1-D data')
            pass
    


'''
### Test ###
if __name__ == '__main__':
    from scipy.stats import norm
    from PyMonteCarlo.MCVar import MCInVar, MCOutVar
    from PyMonteCarlo.order_statistics import sig2pct
    seed = 74494861
    
    mcinvar = MCInVar('norm', norm, (0, 1), 100000, seed=seed)
    bound='1-sided'
    mcinvarstat1 = MCVarStat(mcinvar, stattype='orderstatTI', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':bound})
    mcinvarstat2 = MCVarStat(mcinvar, stattype='mean')
    print(mcinvarstat1.k)
    print(mcinvarstat1.vals)
    print(mcinvarstat2.vals)
    
    v = np.array([-2, -1, 2, 3, 4, 5])
    var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
    mcoutvarstat1 = MCVarStat(var2, stattype='orderstatTI', statkwargs={'p':0.6, 'c':0.50, 'bound':bound})
    mcoutvarstat2 = MCVarStat(var2, stattype='max')
    print(mcoutvarstat1.name)
    print(mcoutvarstat1.vals)
    print(mcoutvarstat2.vals)

#'''
