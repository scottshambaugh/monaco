import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, rv_discrete, mode
from MCVal import MCVal
#from scipy.stats import *

class MCVar:
    def __init__(self, name, dist, distvars, ndraws, seed=np.random.get_state()[1][0]):
        self.seed = seed
        self.name = name
        self.dist = dist
        self.distvars = distvars
        self.ndraws = ndraws
        self.nvals = ndraws + 1
        self.firstdrawisev = True
        self.vals = np.array([])
        
        self.draw()
        
        
    def setFirstDrawEV(self, truefalse):
        if truefalse == True:
           self.firstdrawisev = True
           self.nvals = self.ndraws + 1
        else:
           self.firstdrawisev = False
           self.nvals = self.ndraws
         
    
    def draw(self):
        self.vals = np.array([])
        dist = self.dist(*self.distvars)

        if self.firstdrawisev:
            self.nvals = self.ndraws + 1
            self.vals = np.append(self.vals, dist.expect())
  
        self.vals = np.append(self.vals, dist.rvs(size=self.ndraws))
        
    
    def getVal(self, ndraw):
        isev = False
        if (ndraw == 0) and self.firstdrawisev:
            isev = True
            
        val = MCVal(self.vals[ndraw], self.name, ndraw, self.dist, isev)
        return(val)
        
        
    def hist(self, ax = np.NaN):
        if np.isnan(ax):
            fig, ax = plt.subplots(1, 1)
        
        # histogram
        counts, bins = np.histogram(self.vals, bins='auto')
        binwidth = mode(np.diff(bins))[0]
        bins = np.concatenate((bins - binwidth/2, bins[-1] + binwidth/2))
        counts, bins = np.histogram(self.vals, bins=bins)

        # continuous
        if isinstance(self.dist, rv_continuous):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=True, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/100)
            dist = self.dist(*self.distvars)
            plt.plot(x, dist.pdf(x), color='k', alpha=0.9)
            
        elif isinstance(self.dist, rv_discrete):
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), density=False, histtype='bar', facecolor='k', alpha=0.5)
            xlim = ax.get_xlim()
            x = np.concatenate(([xlim[0]], bins, [xlim[1]]))
            dist = self.dist(*self.distvars)
            pdf = np.diff(dist.cdf(x))
            plt.step(x[1:], pdf, color='k', alpha=0.9)

        plt.xlabel(self.name)
        plt.ylabel('Probability Density')


'''
## Test ##
np.random.seed(74494861)
from scipy.stats import *
var = MCVar('Test', randint, (1, 5), 1000)
var.hist()
var = MCVar('Test', norm, (10, 4), 1000)
var.hist()
xk = (1, 4, 5)
pk = np.ones(len(xk))/len(xk)
custom = rv_discrete(name='custom', values=(xk, pk))
var = MCVar('Test', custom, (), 1000)
var.hist()
#'''
