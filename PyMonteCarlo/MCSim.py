import os
import numpy as np
import dill
import pathlib
from datetime import datetime
from PyMonteCarlo.MCCase import MCCase
from PyMonteCarlo.MCVar import MCInVar, MCOutVar
from psutil import cpu_count
from pathos.pools import ThreadPool as Pool
from tqdm import tqdm
from helper_functions import get_iterable, vprint, vwrite


class MCSim:
    def __init__(self, name, ndraws, fcns, 
                 firstcaseisnom = True, 
                 seed           = np.random.get_state()[1][0], 
                 cores          = cpu_count(logical=False), 
                 verbose        = True,
                 savesimdata    = True,
                 savecasedata   = True,
                 resultsdir     = None):
        
        self.name = name                      # name is a string
        self.verbose = verbose                # verbose is a boolean
        self.ndraws = ndraws                  # ndraws is an integer
        self.fcns = fcns                      # fcns is a dict with keys 'preprocess', 'run', 'postprocess' for those functions
        self.firstcaseisnom = firstcaseisnom  # firstcaseisnom is a boolean
        self.seed = seed                      # seed is a number between 0 and 2^32-1
        self.cores = cores                    # cores is an integer
        self.savesimdata = savesimdata        # savesimdata is a boolean
        self.savecasedata = savecasedata      # savecasedata is a boolean

        self.rootdir = pathlib.Path.cwd()
        if resultsdir is pathlib.Path:        # resultsdir is a pathlib, string, or None
            self.resultsdir = resultsdir
        elif resultsdir is str:                 
            self.resultsdir = self.rootdir / resultsdir
        else:
            self.resultsdir = self.rootdir / f'{self.name}_results'
        self.filepath = self.resultsdir / f'{self.name}.mcsim'

        self.invarseeds = []
        self.caseseeds = []
        
        self.inittime = datetime.now()
        self.starttime = None
        self.endtime = None
        self.runtime = None
        self.casesrun = []
        
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


    def __getstate__(self):
        state = self.__dict__.copy()
        state['mccases'] = []  # don't save mccase data when pickling self
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.savecasedata:
            self.loadCases()
                

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
        generator = np.random.RandomState(self.seed)
        self.caseseeds = generator.randint(0, 2**31-1, size=self.ncases)
        mccases = []
        for i in range(self.ncases):
            isnom = False
            if self.firstcaseisnom and i == 0:
                isnom = True
            mccases.append(MCCase(ncase=i, mcinvars=self.mcinvars, isnom=isnom, seed=self.caseseeds[i]))
            mccases[i].siminput = self.fcns['preprocess'](mccases[i])
        self.mccases = mccases
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
        self.casesrun = []
        self.corrcoeff = None
        self.covcoeff = None
        self.covvarlist = None
        self.endtime = None
        self.runtime = None


    def reset(self):
        self.clearResults()
        self.mcinvars = dict()
        self.ninvars = 0
        self.setNDraws(self.ndraws)
        self.invarseeds = []
        self.caseseeds = []
        self.starttime = None
        
                    
    def runSim(self):
        vprint(self.verbose, f"Running '{self.name}' Monte Carlo simulation with {self.ncases} cases: ", flush=True)
        self.starttime = datetime.now()
        self.clearResults()

        if self.savesimdata or self.savecasedata:
            if not os.path.exists(self.resultsdir):
                os.makedirs(self.resultsdir)
            if self.savesimdata:
                self.pickleSelf()
            
        self.genCases()

        if self.verbose:
            self.pbar = tqdm(total=self.ncases, unit=' cases', position=0)

        if self.cores == 1:
            self.casesrun = []
            for i in range(self.ncases):
                self.runCase(mccase=self.mccases[i])
                self.casesrun.append(i)
        else:
            p = Pool(self.cores)
            casesrun = p.imap(self.runCase, self.mccases)
            casesrun = list(casesrun) # dummy function to ensure we wait for imap to finish
            p.terminate()
            p.restart()

        if self.verbose:
            self.pbar.refresh()
        self.pbar.close()
        self.pbar = None
            
        self.genOutVars()

        self.endtime = datetime.now()
        self.runtime = self.endtime - self.starttime
        
        vprint(self.verbose, f'\nRuntime: {self.runtime}', flush=True)
        
        if self.savesimdata:
            self.pickleSelf()
            vprint(self.verbose, f"Results saved in '{self.resultsdir}'")


    def runCase(self, mccase):
        try:
            mccase.starttime = datetime.now()
            sim_raw_output = self.fcns['run'](*get_iterable(mccase.siminput))
            self.fcns['postprocess'](mccase, *get_iterable(sim_raw_output))
            mccase.endtime = datetime.now()
            mccase.runtime = mccase.endtime - mccase.starttime
            mccase.hasrun = True
            
            if self.savecasedata:
                filepath = self.resultsdir / f'{self.name}_{mccase.ncase}.mccase'
                mccase.filepath = filepath
                filepath.unlink(missing_ok = True)
                with open(filepath,'wb') as file:
                    dill.dump(mccase, file, protocol=dill.HIGHEST_PROTOCOL)
    
            self.casesrun.append(mccase.ncase)
            
        except:
            vwrite(self.verbose, f'\nCase {mccase.ncase} failed')
        
        if not (self.pbar is None):
            self.pbar.update(1)



    def pickleSelf(self):
        if self.savesimdata:
            self.filepath.unlink(missing_ok = True)
            self.filepath.touch()
            with open(self.filepath,'wb') as file:
                dill.dump(self, file, protocol=dill.HIGHEST_PROTOCOL)


    def loadCases(self):
        vprint(self.verbose, f"{self.filepath.name} indicates {len(self.casesrun)}/{self.ncases} cases were run, attempting to load raw case data from disk...", flush=True)
        self.mccases = []
        casesloaded = []
        pbar = tqdm(total=len(self.casesrun), unit=' cases', position=0)
        
        for i in self.casesrun:
            filepath = self.resultsdir / f'{self.name}_{i}.mccase'
            try:
                with open(filepath,'rb') as file:
                    try:
                        mccase = dill.load(file)
                        if mccase.runtime is None:  # only load mccase if it completed running
                            vwrite(self.verbose, f'Warning: {filepath.name} did not finish running, not loaded')
                        else:
                            self.mccases.append(mccase)
                            casesloaded.append(i)
                    except: 
                        vwrite(f'\nWarning: Unknown error loading {filepath.name}', end='')
            except FileNotFoundError:
                vwrite(self.verbose, f'\nWarning: {filepath.name} expected but not found', end='')
            pbar.update(1)
        
        self.casesrun = casesloaded
        pbar.refresh()
        vwrite(self.verbose, f"\nData for {len(casesloaded)}/{self.ncases} cases loaded from disk", end='')


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
