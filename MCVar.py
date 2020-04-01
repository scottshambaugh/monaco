import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, rv_discrete, mode
from MCVal import *

### MCVar Base Class ###
class MCVar:
    def __init__(self, name, ndraws=0, firstdrawisnom=True):
        self.name = name          # name is a string
        self.ndraws = ndraws      # ndraws is an integer

        self.firstdrawisnom = firstdrawisnom
        self.nvals = ndraws + 1
        self.setFirstDrawNom(firstdrawisnom)
        
        self.vals = np.array([])


    def setFirstDrawNom(self, firstdrawisnom):  # firstdrawisnom is a boolean
        if firstdrawisnom:
           self.firstdrawisnom = True
           self.nvals = self.ndraws + 1
        else:
           self.firstdrawisnom = False
           self.nvals = self.ndraws


    def setNDraws(self, ndraws):  # ndraws is an integer
        raise NotImplementedError() # abstract method

    def getVal(self, ndraw):  # ndraw is an integer
        raise NotImplementedError() # abstract method

    def getNom(self):
        raise NotImplementedError() # abstract method

    def hist(self):
        raise NotImplementedError() # abstract method



### MCInVar Class ###
class MCInVar(MCVar):
    def __init__(self, name, dist, distargs, ndraws=0, seed=np.random.get_state()[1][0], firstdrawisnom=True):
        super().__init__(name, ndraws, firstdrawisnom)
        self.dist = dist          # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous 
        self.distargs = distargs  # distargs is a tuple of the arguments to the above distribution
        self.seed = seed          # seed is a number between 0 and 2^32-1
        
        if not isinstance(self.distargs, tuple):
            self.distargs = (self.distargs,)
        
        self.draw()


    def setNDraws(self, ndraws):  # ndraws is an integer
        self.ndraws = ndraws
        self.setFirstDrawNom(self.firstdrawisnom)
        self.seed = np.random.get_state()[1][0]
        self.draw()
        
        
    def draw(self):
        self.vals = np.array([])
        dist = self.dist(*self.distargs)

        if self.firstdrawisnom:
            self.nvals = self.ndraws + 1
            self.vals = np.append(self.vals, self.getNom())
  
        self.vals = np.append(self.vals, dist.rvs(size=self.ndraws))


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

    def getVal(self, ndraw):  # ndraw is an integer
        isnom = False
        if (ndraw == 0) and self.firstdrawisnom:
            isnom = True
            
        val = MCInVal(self.name, ndraw, self.vals[ndraw], self.dist, isnom)
        return(val)


    def hist(self):
        # TODO: take in an axis as an argument
        fig, ax = plt.subplots(1, 1)
        
        # Histogram generation
        counts, bins = np.histogram(self.vals, bins='auto')
        binwidth = mode(np.diff(bins))[0]
        bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
        counts, bins = np.histogram(self.vals, bins=bins)

        # Continuous distribution
        if isinstance(self.dist, rv_continuous):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=True, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/100)
            dist = self.dist(*self.distargs)
            plt.plot(x, dist.pdf(x), color='k', alpha=0.9)
        
        # Discrete distribution
        elif isinstance(self.dist, rv_discrete):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.concatenate(([xlim[0]], bins, [xlim[1]]))
            dist = self.dist(*self.distargs)
            pdf = np.diff(dist.cdf(x))
            plt.step(x[1:], pdf, color='k', alpha=0.9)

        plt.xlabel(self.name)
        plt.ylabel('Probability Density')



### MCOutVar Class ###
class MCOutVar(MCVar):
    def getVal(self, ndraw):  # ndraw is an integer
        isnom = False
        if (ndraw == 0) and self.firstdrawisnom:
            isnom = True
            
        val = MCOutVal(self.name, ndraw, self.vals[ndraw], isnom)
        return(val)



'''
### Test ###
np.random.seed(74494861)
from scipy.stats import *
mcvars = dict()
mcvars['randint'] = MCInVar('randint', randint, (1, 5), 1000)
mcvars['randint'].hist()
mcvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
mcvars['norm'].hist()
xk = (1, 5, 6)
pk = np.ones(len(xk))/len(xk)
custom = rv_discrete(name='custom', values=(xk, pk))
mcvars['custom'] = MCInVar('custom', custom, (), 1000)
mcvars['custom'].hist()
print(var.getVal(0).val)
#'''
