import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, rv_discrete, mode
from MCVal import MCInVal, MCOutVal

### MCVar Base Class ###
class MCVar:
    def __init__(self, name, ndraws, firstcaseisnom):
        self.name = name          # name is a string
        self.ndraws = ndraws      # ndraws is an integer
        self.firstcaseisnom = firstcaseisnom
        self.ncases = ndraws + 1
        self.setFirstCaseNom(firstcaseisnom)
        


    def setFirstCaseNom(self, firstcaseisnom):  # firstdrawisnom is a boolean
        if firstcaseisnom:
           self.firstcaseisnom = True
           self.ncases = self.ndraws + 1
        else:
           self.firstcaseisnom = False
           self.ncases = self.ndraws

    def getVal(self, ncase):  # ncase is an integer
        raise NotImplementedError() # abstract method

    def getNom(self):
        raise NotImplementedError() # abstract method

    def hist(self):
        raise NotImplementedError() # abstract method



### MCInVar Class ###
class MCInVar(MCVar):
    def __init__(self, name, dist, distargs, ndraws,  seed=np.random.get_state()[1][0], firstcaseisnom=True):
        super().__init__(name, ndraws, firstcaseisnom)
        self.dist = dist          # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous 
        self.distargs = distargs  # distargs is a tuple of the arguments to the above distribution
        self.seed = seed          # seed is a number between 0 and 2^32-1
        self.vals = []

        if not isinstance(self.distargs, tuple):
            self.distargs = (self.distargs,)
        
        self.draw()


    def setNDraws(self, ndraws):  # ndraws is an integer
        self.ndraws = ndraws
        self.setFirstCaseNom(self.firstcaseisnom)
        self.seed = np.random.get_state()[1][0]
        self.draw()
        
        
    def draw(self):
        self.vals = []
        dist = self.dist(*self.distargs)

        if self.firstcaseisnom:
            self.ncases = self.ndraws + 1
            self.vals.append(self.getNom())
  
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
            
        val = MCInVal(self.name, ncase, self.vals[ncase], self.dist, isnom)
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
    def __init__(self, name, vals, ndraws=None, firstcaseisnom=True):
        if ndraws == None:
            ndraws = len(vals)
            if firstcaseisnom:
                ndraws = ndraws - 1
        
        super().__init__(name, ndraws, firstcaseisnom)
        self.vals = vals  # vals is a list


    def getVal(self, ncase):  # ncase is an integer
        isnom = False
        if (ncase == 0) and self.firstcaseisnom:
            isnom = True
            
        val = MCOutVal(self.name, ncase, self.vals[ncase], isnom)
        return(val)
        
    
    def getNom(self):
        val = None
        if self.firstcaseisnom:
            val = self.vals[0]            
        return(val)


    def hist(self):
        # TODO: take in an axis as an argument
        fig, ax = plt.subplots(1, 1)
        
        # Histogram generation
        counts, bins = np.histogram(self.vals, bins='auto')
        binwidth = mode(np.diff(bins))[0]
        bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
        counts, bins = np.histogram(self.vals, bins=bins)

        plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, histtype='bar', facecolor='k', alpha=0.5)
        if self.firstcaseisnom:
            plt.plot([self.getNom(), self.getNom()], ax.get_ylim(), 'k-')

        plt.xlabel(self.name)
        plt.ylabel('Probability Density')



'''
### Test ###
np.random.seed(74494861)
from scipy.stats import *
mcinvars = dict()
mcinvars['randint'] = MCInVar('randint', randint, (1, 5), 1000)
mcinvars['randint'].hist()
mcinvars['norm'] = MCInVar('norm', norm, (10, 4), 1000)
mcinvars['norm'].hist()
xk = np.array([1, 5, 6])
pk = np.ones(len(xk))/len(xk)
custom = rv_discrete(name='custom', values=(xk, pk))
mcinvars['custom'] = MCInVar('custom', custom, (), 1000)
mcinvars['custom'].hist()
print(mcinvars['custom'].getVal(0).val)

mcoutvars = dict()
mcoutvars['test'] = MCOutVar('test', [1, 0, 2, 2], firstcaseisnom=True)
print(mcoutvars['test'].getVal(1).val)
mcoutvars['test'].hist()

#'''
