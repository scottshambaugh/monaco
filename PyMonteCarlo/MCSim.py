import numpy as np
from datetime import datetime
from PyMonteCarlo.MCCase import MCCase
from PyMonteCarlo.MCVar import MCInVar, MCOutVar
from psutil import cpu_count
from pathos.pools import ThreadPool as Pool
from tqdm import tqdm
from helper_functions import get_iterable, vprint



class MCSim:
    def __init__(self, name, ndraws, fcns, firstcaseisnom=True, seed=np.random.get_state()[1][0], cores=cpu_count(logical=False), verbose=True):
        self.name = name                      # name is a string
        self.verbose = verbose                # verbose is a boolean
        self.ndraws = ndraws                  # ndraws is an integer
        self.fcns = fcns                      # fcns is a dict with keys 'preprocess', 'run', 'postprocess' for those functions
        self.firstcaseisnom = firstcaseisnom  # firstcaseisnom is a boolean
        self.seed = seed                      # seed is a number between 0 and 2^32-1
        self.cores = cores                    # cores is an integer

        self.invarseeds = []
        self.caseseeds = []
        
        self.inittime = datetime.now()
        self.starttime = None
        self.endtime = None
        self.runtime = None      
        self.casesrun = None
        
        self.mcinvars = dict()     
        self.mcoutvars = dict()     
        self.mccases = []
        self.ninvars = 0
        
        self.corrcoeffs = None
        self.covs = None
        self.covvarlist = None

        self.ncases = ndraws + 1
        self.setFirstCaseNom(firstcaseisnom)
        self.setNDraws(self.ndraws)
        
        self.pbar = None


    def setFirstCaseNom(self, firstcaseisnom):  # firstdrawisnom is a boolean
        if firstcaseisnom:
           self.firstcaseisnom = True
           self.ncases = self.ndraws + 1
        else:
           self.firstcaseisnom = False
           self.ncases = self.ndraws
        if self.mcinvars != dict():
            for mcvar in self.mcinvars.values():
                mcvar.setFirstCaseNom(firstcaseisnom)


    def addInVar(self, name, dist, distargs, nummap=None):  
        # name is a string
        # dist is a scipy.stats.rv_discrete or scipy.stats.rv_continuous 
        # distargs is a tuple of the arguments to the above distribution
        # nummap is a dict mapping integers to values
        self.ninvars += 1
        invarseed = (self.seed + hash(name)) % 2**32  # make seed dependent on var name and not order added
        self.invarseeds.append(invarseed)
        self.mcinvars[name] = MCInVar(name=name, dist=dist, distargs=distargs, ndraws=self.ndraws, nummap=nummap, \
                                      seed=invarseed, firstcaseisnom=self.firstcaseisnom)


    def setNDraws(self, ndraws):  # ncases is an integer
        self.ndraws = ndraws
        self.setFirstCaseNom(self.firstcaseisnom)
        for mcinvar in self.mcinvars.values():
            mcinvar.setNDraws(ndraws)
        if self.mcinvars != dict():
            self.genCases()


    def genCases(self):
        self.clearResults()
        generator = np.random.RandomState(self.seed)
        self.caseseeds = generator.randint(0, 2**31-1, size=self.ncases)
        for i in range(self.ncases):
            isnom = False
            if self.firstcaseisnom and i == 0:
                isnom = True
            self.mccases.append(MCCase(ncase=i, mcinvars=self.mcinvars, isnom=isnom, seed=self.caseseeds[i]))
            self.mccases[i].siminput = self.fcns['preprocess'](self.mccases[i])
        #self.genCovarianceMatrix()


    def genOutVars(self):
        for varname in self.mccases[0].mcoutvals.keys():
            vals = []
            for i in range(self.ncases):
                vals.append(self.mccases[i].mcoutvals[varname].val)

            if self.mccases[0].mcoutvals[varname].valmapsource == 'auto':
                uniquevals = set()
                valmap = None
                for i in range(self.ncases):
                    if self.mccases[i].mcoutvals[varname].valmap == None:
                        uniquevals = None
                    else:
                        uniquevals.update(self.mccases[i].mcoutvals[varname].valmap.keys())
                if uniquevals != None:
                    valmap = dict()
                    for i, val in enumerate(uniquevals):
                        valmap[val] = i
            else:
                valmap = self.mccases[0].mcoutvals[varname].valmap

            self.mcoutvars[varname] = MCOutVar(name=varname, vals=vals, valmap=valmap, ndraws=self.ndraws, firstcaseisnom=self.firstcaseisnom)
            for i in range(self.ncases):
                self.mccases[i].mcoutvars[varname] = self.mcoutvars[varname]


    def genCovarianceMatrix(self):
        self.covvarlist = []
        allnums = []
        j = 0
        for var in self.mcinvars.keys():
            if self.mcinvars[var].isscalar:
                allnums.append(self.mcinvars[var].nums)
                self.covvarlist.append(self.mcinvars[var].name)
                j = j+1
        for var in self.mcoutvars.keys():
            if self.mcoutvars[var].isscalar:
                allnums.append(self.mcoutvars[var].nums)
                self.covvarlist.append(self.mcoutvars[var].name)
                j = j+1
        self.covs = np.cov(np.array(allnums))
        self.corrcoeffs = np.corrcoef(np.array(allnums))


    def corr(self):
        self.genCovarianceMatrix()
        return self.corrcoeffs, self.covvarlist


    def cov(self):
        self.genCovarianceMatrix()
        return self.covs, self.covvarlist


    def clearResults(self):
        self.mccases = []
        self.mcoutvars = dict()
        self.casesrun = None
        self.corrcoeff = None
        self.covcoeff = None
        self.covvarlist = None


    def reset(self):
        self.clearResults()
        self.mcinvars = dict()
        self.ninvars = 0
        self.setNDraws(self.ndraws)
        self.invarseeds = []
        self.caseseeds = []
        self.starttime = None
        self.endtime = None
        self.runtime = None
        self.pbar = None
        
                    
    def runSim(self):
        vprint(self.verbose, f'Running "{self.name}" Monte Carlo simulation with {self.ncases} cases: ', flush=True)
        self.starttime = datetime.now()
        
        self.genCases()

        if self.verbose:
            self.pbar = tqdm(total=self.ncases, unit=' cases', position=0, leave=True)

        if self.cores == 1:
            for i in range(self.ncases):
                self.runCase(mccase=self.mccases[i])
        else:
            p = Pool(self.cores)
            casesrun = p.imap(self.runCase, self.mccases)
            self.casesrun = list(casesrun)
            p.terminate()
            p.restart()
        
        if self.verbose:
            self.pbar.refresh()
        self.pbar = None
            
        self.genOutVars()

        self.endtime = datetime.now()
        self.runtime = self.endtime - self.starttime
        vprint(self.verbose, f'\nRuntime: {self.runtime}', flush=True)


    def runCase(self, mccase):
        mccase.starttime = datetime.now()
        sim_raw_output = self.fcns['run'](*get_iterable(mccase.siminput))
        self.fcns['postprocess'](mccase, *get_iterable(sim_raw_output))
        mccase.endtime = datetime.now()
        mccase.runtime = mccase.endtime - mccase.starttime
    
        if not (self.pbar is None):
            self.pbar.update(1)
            
        return(mccase.ncase)


'''
### Test ###
def dummyfcn(*args):
    return 1
from scipy.stats import norm, randint
np.random.seed(74494861)
sim = MCSim('Sim', 10, {'preprocess':dummyfcn, 'run':dummyfcn, 'postprocess':dummyfcn})
sim.addInVar('Var1', randint, (1, 5))
sim.addInVar('Var2', norm, (10, 4))
sim.genCases()
print(sim.mcinvars['Var1'].name)
print(sim.mccases[0].mcinvals['Var1'].val)
print(sim.mcinvars['Var2'].name)
print(sim.mccases[0].mcinvals['Var2'].val)
print(sim.corr())
print(sim.cov())
#'''
