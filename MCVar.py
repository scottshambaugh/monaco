import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import *

class MCVar:
    def __init__(self, name, dist, distvars, ndraws, seed = np.random.get_state()[1][0]):
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
        
        
    def hist(self, ax = np.NaN):
        if np.isnan(ax):
            fig, ax = plt.subplots(1, 1)
        
        # histogram
        plt.hist(self.vals, density=True, histtype='stepfilled', facecolor='k', alpha=0.5)
        plt.xlabel(self.name)
        plt.ylabel('Probability Density')

        # pdf
        xlim = ax.get_xlim()
        x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/100)
        rv = self.dist(*self.distvars)
        plt.plot(x, rv.pdf(x), color='k', alpha=0.9)

