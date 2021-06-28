# MCVarStat.py

# Somewhat hacky type checking to avoid circular imports:
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Monaco.MCVar import MCVar
    
import numpy as np
from copy import copy
from statistics import mode
from scipy.stats.mstats import gmean
from Monaco.gaussian_statistics import pct2sig, sig2pct
from Monaco.order_statistics import order_stat_P_k, order_stat_TI_k, get_iP
from typing import Union, Any, Callable


class MCVarStat:
    def __init__(self, 
                 mcvar      : MCVar,  
                 stattype   : str, 
                 statkwargs : dict[str, Any]   = dict(), 
                 name       : Union[None, str] = None,
                 ):
        '''
        valid stattypes with corresponding statkwargs:
            max
            min
            median
            mean
            geomean
            mode
            sigmaP(sig, bound)
                sig     -inf < sig < inf        Sigma Value
                bound   '1-sided', '2-sided'    Bound (default 2-sided)
            gaussianP(p, bound)
                p       0 < p < 1               Percentile
                bound   '1-sided', '2-sided'    Bound (default 2-sided)
            orderstatTI(p, c, bound)
                p       0 <= p <= 1             Percentage
                c       0 < c < 1               Confidence (default 0.95)
                bound   '1-sided', '2-sided',   Bound (default 2-sided)
                        'all'
            orderstatP(p, c, bound)
                p       0 <= p <= 1             Percentile
                c       0 < c < 1               Confidence (default 0.95)
                bound   '1-sided low', 'all',   Bound (default 2-sided)
                        '1-sided high', 
                        '2-sided', 'nearest', 
        '''

        self.mcvar = mcvar
        self.stattype = stattype
        self.statkwargs = statkwargs

        self.nums = None
        self.vals = None
        self.name = name
        
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
        elif stattype == 'orderstatP':
            self.genStatsOrderStatP()
        else:
            raise ValueError(f"{self.stattype=} must be one of the following: ",
                             "'max', 'min', 'median', 'mean', 'geomean', 'mode', 'sigmaP', 'gaussianP', 'orderstatTI', 'orderstatP'")


    def genStatsMax(self):
        self.setName('Max')
        self.genStatsFunction(fcn=np.max)


    def genStatsMin(self):
        self.setName('Min')
        self.genStatsFunction(fcn=np.min)


    def genStatsMedian(self):
        self.setName('Median')
        self.genStatsFunction(fcn=np.median)


    def genStatsMean(self):
        self.setName('Mean')
        self.genStatsFunction(fcn=np.mean)


    def genStatsGeoMean(self):
        self.setName('Geometric Mean')
        self.genStatsFunction(fcn=gmean)


    def genStatsMode(self):
        self.setName('Mode')
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
        self.setName(f'{self.sig} Sigma')
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
        self.setName(f'Guassian {self.p*100}%')
        self.genStatsFunction(self.sigmaP)


    def sigmaP(self, 
               x, # TODO: explicit typing here
               ):
        std = np.std(x)
        return np.mean(x) + self.sig*std


    def genStatsFunction(self, 
                         fcn : Callable,
                         ):
        if self.mcvar.isscalar:
            self.nums = fcn(self.mcvar.nums)
            self.vals = copy(self.nums)
            if self.mcvar.nummap is not None:
                self.vals = self.mcvar.nummap[self.nums]
                
        elif self.mcvar.size[0] == 1:
            npoints = max(len(x) for x in self.mcvar.nums)
            self.nums = np.empty(npoints)
            for i in range(npoints):
                numsatidx = [x[i] for x in self.mcvar.nums if len(x)>i]
                self.nums[i] = fcn(numsatidx)
            self.vals = copy(self.nums)
            if self.mcvar.nummap is not None:
                self.vals = np.array([[self.mcvar.nummap[x] for x in y] for y in self.nums])
                
        else:
            # Suppress warning since this will become valid when MCVar is split
            #warn('MCVarStat only available for scalar or 1-D data')
            pass


    def genStatsOrderStatTI(self):
        self.checkOrderStatsKWArgs()
      
        if self.bound == '1-sided' and self.p >= 0.5:
            self.side = 'high'
        elif self.bound == '1-sided':
            self.side = 'low'
        elif self.bound == '2-sided':
            self.side = 'both'
        elif self.bound == 'all':
            self.bound = '2-sided'
            self.side = 'all'
        else:
            raise ValueError(f'{self.bound} is not a valid bound for genStatsOrderStatTI')

        self.setName(f'{self.bound} P{self.p*100}/{self.c*100}% Confidence Interval')

        self.k = order_stat_TI_k(n=self.mcvar.ncases, p=self.p, c=self.c, bound=self.bound)

        if self.mcvar.isscalar:
            sortednums = sorted(self.mcvar.nums)
            if self.side == 'low':
                sortednums.reverse()
            if self.side in ('high', 'low'):
                self.nums = sortednums[-self.k]
                if self.mcvar.nummap is not None:
                    self.vals = self.mcvar.nummap[self.nums]
            elif self.side == 'both':
                self.nums = np.array([sortednums[self.k-1], sortednums[-self.k]])
                if self.mcvar.nummap is not None:
                    self.vals = np.array([self.mcvar.nummap[self.nums[0]], self.mcvar.nummap[self.nums[1]]])
            elif self.side == 'all':
                self.nums = np.array([sortednums[self.k-1], np.median(sortednums), sortednums[-self.k]])
                if self.mcvar.nummap is not None:
                    self.vals = np.array([self.mcvar.nummap[self.nums[0]], self.mcvar.nummap[self.nums[1]], self.mcvar.nummap[self.nums[2]]])
            if self.mcvar.nummap is None:
                self.vals = copy(self.nums)
                
        elif self.mcvar.size[0] == 1:
            npoints = max(len(x) for x in self.mcvar.nums)
            self.nums = np.empty(npoints)
            if self.side == 'both':
                self.nums = np.empty((npoints, 2))
            elif self.side == 'all':
                self.nums = np.empty((npoints, 3))
            for i in range(npoints):
                numsatidx = [x[i] for x in self.mcvar.nums if len(x)>i]
                sortednums = sorted(numsatidx)
                if self.side == 'low':
                    sortednums.reverse()
                if self.side in ('high', 'low'):
                    self.nums[i] = sortednums[-self.k]
                elif self.side == 'both':
                    self.nums[i,:] = [sortednums[self.k-1], sortednums[-self.k]]
                elif self.side == 'all':
                    self.nums[i,:] = [sortednums[self.k-1], sortednums[int(np.round(len(sortednums)/2)-1)], sortednums[-self.k]]
            if self.mcvar.nummap is not None:
                self.vals = np.array([[self.mcvar.nummap[x] for x in y] for y in self.nums])
            else:
                self.vals = copy(self.nums)
                
        else:
            # Suppress warning since this will become valid when MCVar is split
            #warn('MCVarStat only available for scalar or 1-D data')
            pass


    def genStatsOrderStatP(self):
        self.checkOrderStatsKWArgs()
      
        bound = self.bound
        if self.bound not in ('1-sided high', '1-sided low', '2-sided', 'nearest', 'all'):
            raise ValueError(f'{self.bound} is not a valid bound for genStatsOrderStatP')
        elif self.bound in ('nearest', 'all'):
            bound = '2-sided'
        
        self.setName(f'{self.bound} {self.c*100}% Confidence Bound around {self.p*100}th Percentile')

        self.k = order_stat_P_k(n=self.mcvar.ncases, P=self.p, c=self.c, bound=bound)

        (iPl, iP, iPu) = get_iP(n=self.mcvar.ncases, P=self.p) 
        if self.mcvar.isscalar:
            sortednums = sorted(self.mcvar.nums)
            if self.bound == '1-sided low':
                self.nums = sortednums[iPl - self.k]
            elif self.bound == '1-sided high':
                self.nums = sortednums[iPu + self.k]
            elif self.bound == 'nearest':
                self.nums = sortednums[iP]
            if self.bound in ('1-sided low', '1-sided high', 'nearest'):
                if self.mcvar.nummap is not None:
                    self.vals = self.mcvar.nummap[self.nums]
            elif self.bound == '2-sided':
                self.nums = np.array([sortednums[iPl - self.k], sortednums[iPu + self.k]])
                if self.mcvar.nummap is not None:
                    self.vals = np.array([self.mcvar.nummap[self.nums[0]], self.mcvar.nummap[self.nums[1]]])
            elif self.bound == 'all':
                self.nums = np.array([sortednums[iPl - self.k], sortednums[iP], sortednums[iPu + self.k]])
                if self.mcvar.nummap is not None:
                    self.vals = np.array([self.mcvar.nummap[self.nums[0]], self.mcvar.nummap[self.nums[1]], self.mcvar.nummap[self.nums[2]]])
            if self.mcvar.nummap is None:
                self.vals = copy(self.nums)

        elif self.mcvar.size[0] == 1:
            npoints = max(len(x) for x in self.mcvar.nums)
            self.nums = np.empty(npoints)
            if self.bound == '2-sided':
                self.nums = np.empty((npoints, 2))
            elif self.bound == 'all':
                self.nums = np.empty((npoints, 3))
            for i in range(npoints):
                numsatidx = [x[i] for x in self.mcvar.nums if len(x)>i]
                sortednums = sorted(numsatidx)
                if self.bound == '1-sided low':
                    self.nums[i] = sortednums[iPl - self.k]
                elif self.bound == '1-sided high':
                    self.nums[i] = sortednums[iPu + self.k]
                elif self.bound == 'nearest':
                    self.nums[i] = sortednums[iP]
                elif self.bound == '2-sided':
                    self.nums[i,:] = [sortednums[iPl - self.k], sortednums[iPu + self.k]]
                elif self.bound == 'all':
                    self.nums[i,:] = [sortednums[iPl - self.k], sortednums[iP], sortednums[iPu + self.k]]
            if self.mcvar.nummap is not None:
                self.vals = np.array([[self.mcvar.nummap[x] for x in y] for y in self.nums])
            else:
                self.vals = copy(self.nums)
                
        else:
            # Suppress warning since this will become valid when MCVar is split
            #warn('MCVarStat only available for scalar or 1-D data')
            pass


    def checkOrderStatsKWArgs(self):
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


    def setName(self, 
                name : str,
                ):
        if self.name is None:
            self.name = name



'''
### Test ###
if __name__ == '__main__':
    from scipy.stats import norm
    from Monaco.MCVar import MCInVar, MCOutVar
    seed = 74494861

    mcinvar = MCInVar('norm', ndraws=100000, dist=norm, distkwargs={'loc':0, 'scale':1}, seed=seed)
    bound='1-sided'
    mcinvarstat1 = MCVarStat(mcinvar, stattype='orderstatTI', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':bound})
    mcinvarstat2 = MCVarStat(mcinvar, stattype='orderstatP', statkwargs={'p':sig2pct(3, bound=bound), 'c':0.50, 'bound':'all'})
    mcinvarstat3 = MCVarStat(mcinvar, stattype='gaussianP', statkwargs={'p':sig2pct(-3, bound=bound), 'bound':bound})
    mcinvarstat4 = MCVarStat(mcinvar, stattype='sigmaP', statkwargs={'sig':3, 'bound':bound})
    mcinvarstat5 = MCVarStat(mcinvar, stattype='mean')
    print(mcinvarstat1.k)    # expected: 135
    print(mcinvarstat1.vals) # expected: 3.024354335001775
    print(mcinvarstat2.k)    # expected: 8
    print(mcinvarstat2.vals) # expected: [3.01323888 3.02664409 3.03566906]
    print(mcinvarstat3.vals) # expected: -2.998215730189507
    print(mcinvarstat4.vals) # expected: 2.995372635006932
    print(mcinvarstat5.vals) # expected: -0.0014215475912877358
    
    v = np.array([-2, -1, 2, 3, 4, 5])
    var2 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, -2*v], firstcaseisnom=True)
    mcoutvarstat1 = MCVarStat(var2, stattype='orderstatTI', statkwargs={'p':0.6, 'c':0.50, 'bound':'all'})
    mcoutvarstat2 = MCVarStat(var2, stattype='min')
    var3 = MCOutVar('testy', [1*v, 2*v, 0*v, -1*v, [0,0]], firstcaseisnom=True)
    mcoutvarstat3 = MCVarStat(var3, stattype='min')
    print(mcoutvarstat1.name) # expected: 2-sided P60.0/50.0% Confidence Interval
    print(mcoutvarstat1.vals) # expected: [[ -4.  -2.   4.] [ -2.  -1.   2.] [ -4.  -2.   4.] [ -6.  -3.   6.] [ -8.  -4.   8.] [-10.  -5.  10.]]
    print(mcoutvarstat2.vals) # expected: [ -4.  -2.  -4.  -6.  -8. -10.]
    print(mcoutvarstat3.vals) # expected: [-4. -2. -2. -3. -4. -5.]
#'''
