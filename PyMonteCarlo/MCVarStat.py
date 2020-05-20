import numpy as np
from copy import copy
from statistics import mode
from scipy.stats.mstats import gmean
from PyMonteCarlo.order_statistics import order_stat_TI_k, pct2sig, sig2pct

class MCVarStat:
    def __init__(self, mcvar, stattype, statkwargs = {}, name=None):
        '''
        valid stattypes with corresponding statkwargs:
            max
            min
            median
            mean
            geomean
            mode
            sigmaP
                sig     -inf < sig < inf
                bound   '1-sided', '2-sided'    (default 2-sided)
            gaussianP
                p       0 < p < 1
                bound   '1-sided', '2-sided'    (default 2-sided)
            orderstatTI
                p       0 <= p <= 1
                c       0 < c < 1               (default 0.95)
                bound   '1-sided', '2-sided'    (default 2-sided)
        '''

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
        elif stattype == 'sigmaP':
            self.genStatsSigmaP()
        elif stattype == 'gaussianP':
            self.genStatsGaussianP()
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


    def genStatsSigmaP(self):
        if 'sig' not in self.statkwargs:
            raise ValueError(f'{self.stattype} requires the kwarg ''sig''')
        if 'bound' not in self.statkwargs:
            self.bound = '2-sided'
        else:
            self.bound = self.statkwargs['bound']

        self.sig = self.statkwargs['sig']
        self.p = sig2pct(self.sig, bound=self.bound)
        self.name = f'{self.sig} Sigma'
        self.genStatsFunction(self.sigmaP)


    def genStatsGaussianP(self):
        if 'p' not in self.statkwargs:
            raise ValueError(f'{self.stattype} requires the kwarg ''p''')
        if 'bound' not in self.statkwargs:
            self.bound = '2-sided'
        else:
            self.bound = self.statkwargs['bound']

        self.p = self.statkwargs['p']
        self.sig = pct2sig(self.p, bound=self.bound)
        self.name = f'Guassian {self.p*100}%'
        self.genStatsFunction(self.sigmaP)


    def sigmaP(self, x):
        std = np.std(x)
        return np.mean(x) + self.sig*std


    def genStatsFunction(self, fcn):
        if self.mcvar.isscalar:
            self.nums = fcn(self.mcvar.nums)
            self.vals = copy(self.nums)
            if self.mcvar.nummap != None:
                self.vals = self.mcvar.nummap[self.nums]
                
        elif self.mcvar.size[0] == 1:
            self.nums = np.empty(self.mcvar.size[1])
            for i in range(self.mcvar.size[1]):
                numsatidx = [x[i] for x in self.mcvar.nums]
                self.nums[i] = fcn(numsatidx)
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
      
        
        if self.bound == '1-sided' and self.p >= 0.5:
            self.side = 'high'
        elif self.bound == '1-sided':
            self.side = 'low'
        else:
            self.side = 'both'

        
        if self.name == None:
            self.name = f'{self.bound} P{self.p*100}/{self.c*100}% Confidence Interval'

        self.k = order_stat_TI_k(n=self.mcvar.ncases, p=self.p, c=self.c, bound=self.bound)

        if self.mcvar.isscalar:
            sortednums = sorted(self.mcvar.nums)
            if self.side == 'low':
                sortednums.reverse()
            if self.side in ('high', 'low'):
                self.nums = sortednums[-self.k]
                self.vals = copy(self.nums)
                if self.mcvar.nummap != None:
                    self.vals = self.mcvar.nummap[self.nums]
            elif self.side == 'both':
                self.nums = np.array([sortednums[self.k-1], sortednums[-self.k]])
                self.vals = copy(self.nums)
                if self.mcvar.nummap != None:
                    self.vals = np.array([self.mcvar.nummap[self.nums[0]], self.mcvar.nummap[self.nums[1]]])
                
        elif self.mcvar.size[0] == 1:
            self.nums = np.empty(self.mcvar.size[1])
            if self.side == 'both':
                self.nums = np.empty((self.mcvar.size[1], 2))
            for i in range(self.mcvar.size[1]):
                numsatidx = [x[i] for x in self.mcvar.nums]
                sortednums = sorted(numsatidx)
                if self.side == 'low':
                    sortednums.reverse()
                if self.side in ('high', 'low'):
                    self.nums[i] = sortednums[-self.k]
                elif self.side == 'both':
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
    seed = 74494861
    
    mcinvar = MCInVar('norm', norm, (0, 1), 100000, seed=seed)
    bound='1-sided'
    mcinvarstat1 = MCVarStat(mcinvar, stattype='orderstatTI', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':bound})
    mcinvarstat2 = MCVarStat(mcinvar, stattype='gaussianP', statkwargs={'p':sig2pct(-3, bound=bound), 'bound':bound})
    mcinvarstat3 = MCVarStat(mcinvar, stattype='sigmaP', statkwargs={'sig':3, 'bound':bound})
    mcinvarstat4 = MCVarStat(mcinvar, stattype='mean')
    print(mcinvarstat1.k)
    print(mcinvarstat1.vals)
    print(mcinvarstat2.vals)
    print(mcinvarstat3.vals)
    print(mcinvarstat4.vals)
    
    v = np.array([-2, -1, 2, 3, 4, 5])
    var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
    mcoutvarstat1 = MCVarStat(var2, stattype='orderstatTI', statkwargs={'p':0.6, 'c':0.50, 'bound':'2-sided'})
    mcoutvarstat2 = MCVarStat(var2, stattype='min')
    print(mcoutvarstat1.name)
    print(mcoutvarstat1.vals)
    print(mcoutvarstat2.vals)

#'''
