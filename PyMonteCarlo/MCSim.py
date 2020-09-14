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
from helper_functions import get_iterable, slice_by_index, vprint, vwrite


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
        self.casesrun = set()
        self.casespostprocessed = set()
        
        self.mcinvars = dict()
        self.mcoutvars = dict()
        self.mccases = []
        self.ninvars = 0
        
        self.corrcoeffs = None
        self.covs = None
        self.covvarlist = None

        self.runsimid = self.genID()

        self.ncases = ndraws + 1
        self.setFirstCaseNom(firstcaseisnom)
        self.setNDraws(self.ndraws)
                

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
            mccases.append(MCCase(ncase=i, mcinvars=self.mcinvars, isnom=isnom, seed=int(self.caseseeds[i])))
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
        self.casesrun = set()
        self.casespostprocessed = set()
        self.corrcoeff = None
        self.covcoeff = None
        self.covvarlist = None
        self.endtime = None
        self.runtime = None
        self.runsimid = self.genID()


    def reset(self):
        self.clearResults()
        self.mcinvars = dict()
        self.ninvars = 0
        self.setNDraws(self.ndraws)
        self.invarseeds = []
        self.caseseeds = []
        self.starttime = None
        

    def genRunSimID(self):
        self.runsimid = self.genID()


    def genID(self):
        uniqueid = (self.seed + hash(self.name) + hash(datetime.now())) % 2**32
        return uniqueid


    def runIncompleteSim(self):
        casestorun = set(range(self.ncases)) - self.casesrun
        casestopostprocess = set(range(self.ncases)) - self.casesrun

        vprint(self.verbose, f"Resuming incomplete '{self.name}' Monte Carlo simulation with {len(casestorun)}/{self.ncases} cases remaining to run, " + \
                             f"and {len(casestopostprocess)}/{self.ncases} cases remaining to post process...", end='', flush=True)
        self.runSimWorker(casestorun=casestorun, casestopostprocess=casestopostprocess)


    def runSim(self, cases=None):
        casestorun = self.downselectCases(cases=cases)
        casestopostprocess = self.downselectCases(cases=cases)

        vprint(self.verbose, f"Running '{self.name}' Monte Carlo simulation with {len(casestorun)}/{self.ncases} cases...", end='', flush=True)
        self.runSimWorker(casestorun=casestorun, casestopostprocess=casestopostprocess)


    def runSimWorker(self, casestorun, casestopostprocess):            
        self.starttime = datetime.now()

        if casestorun == set(range(self.ncases)):
            self.clearResults() # only clear results if we are rerunning all cases
        else:
            self.runsimid = self.genID()

        if self.savesimdata or self.savecasedata:
            if not os.path.exists(self.resultsdir):
                os.makedirs(self.resultsdir)
            if self.savesimdata:
                self.saveSimToFile()
        
        self.genCases()
        self.runCases(cases=casestorun, calledfromrunsim=True)
        self.postProcessCases(cases=casestopostprocess)
        self.genOutVars()

        self.endtime = datetime.now()
        self.runtime = self.endtime - self.starttime
        
        vprint(self.verbose, f'\nRuntime: {self.runtime}', flush=True)
        
        if self.savesimdata:
            self.saveSimToFile()
            vprint(self.verbose, f"Sim results saved in '{self.filepath}'", flush=True)


    def downselectCases(self, cases=None):
        if cases == None:
            cases = set(range(self.ncases))
        else:
            cases = set(get_iterable(cases))
        return cases


    def runCases(self, cases, calledfromrunsim=False):
        cases = self.downselectCases(cases=cases)
        
        if not calledfromrunsim:
            self.runsimid = self.genID()
            
        if self.verbose:
            self.pbar1 = tqdm(total=len(cases), desc='Running cases', unit=' cases', position=0)

        if self.cores == 1:
            for case in cases:
                self.runCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            casesrun = p.imap(self.runCase, slice_by_index(self.mccases, cases))
            casesrun = list(casesrun) # dummy function to ensure we wait for imap to finish
            p.terminate()
            p.restart()

        if self.verbose:
            self.pbar1.refresh()
            self.pbar1.close()
            self.pbar1 = None

        if self.savecasedata:
            vprint(self.verbose, f"\nCase results saved in '{self.resultsdir}'", end='', flush=True)


    def postProcessCases(self, cases):
        cases = self.downselectCases(cases=cases)
        
        if self.verbose:
            self.pbar2 = tqdm(total=len(cases), desc='Post processing cases', unit=' cases', position=0)

        if self.cores == 1:
            for case in cases:
                self.postProcessCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            casespostprocessed = p.imap(self.postProcessCase, slice_by_index(self.mccases, cases))
            casespostprocessed = list(casespostprocessed) # dummy function to ensure we wait for imap to finish
            p.terminate()
            p.restart()

        if self.verbose:
            self.pbar2.refresh()
            self.pbar2.close()
            self.pbar2 = None


    def runCase(self, mccase):
        try:
            mccase.starttime = datetime.now()
            mccase.simrawoutput = self.fcns['run'](*get_iterable(mccase.siminput))
            mccase.endtime = datetime.now()
            mccase.runtime = mccase.endtime - mccase.starttime
            mccase.runsimid = self.runsimid
            mccase.hasrun = True
            
            if self.savecasedata:
                filepath = self.resultsdir / f'{self.name}_{mccase.ncase}.mccase'
                mccase.filepath = filepath
                filepath.unlink(missing_ok = True)
                with open(filepath,'wb') as file:
                    dill.dump(mccase, file, protocol=dill.HIGHEST_PROTOCOL)
    
            self.casesrun.add(mccase.ncase)
            
        except:
            vwrite(self.verbose, f'\nRunning case {mccase.ncase} failed')
        
        if not (self.pbar1 is None):
            self.pbar1.update(1)


    def postProcessCase(self, mccase):
        try:
            self.fcns['postprocess'](mccase, *get_iterable(mccase.simrawoutput))
            self.casespostprocessed.add(mccase.ncase)
            mccase.haspostprocessed = True
            
        except:
            vwrite(self.verbose, f'\nPostprocessing case {mccase.ncase} failed')
            
        if not (self.pbar2 is None):
            self.pbar2.update(1)


    def saveSimToFile(self):
        if self.savesimdata:
            self.filepath.unlink(missing_ok = True)
            self.filepath.touch()
            with open(self.filepath,'wb') as file:
                dill.dump(self, file, protocol=dill.HIGHEST_PROTOCOL)


    def loadCases(self):
        vprint(self.verbose, f"{self.filepath} indicates {len(self.casesrun)}/{self.ncases} cases were run, attempting to load raw case data from disk...", end='', flush=True)
        self.mccases = []
        casesloaded = set()
        casesstale = set()
        casesnotloaded = set(range(self.ncases))
        casesnotpostprocessed = set(range(self.ncases))
        
        pbar = tqdm(total=len(self.casesrun), unit=' cases', desc='Loading', position=0)
        
        for case in self.casesrun:
            filepath = self.resultsdir / f'{self.name}_{case}.mccase'
            try:
                with open(filepath,'rb') as file:
                    try:
                        mccase = dill.load(file)
                        if mccase.runtime is None:  # only load mccase if it completed running
                            vwrite(self.verbose, f'\nWarning: {filepath.name} did not finish running, not loaded', end='')
                        else:
                            self.mccases.append(mccase)
                            
                            if mccase.runsimid != self.runsimid:
                                vwrite(self.verbose, f'\nWarning: {filepath.name} is not from the most recent run and may be stale', end='')
                                casesstale.add(case)
                            casesloaded.add(case)
                            casesnotloaded.remove(case)
                            if case in self.casespostprocessed:
                                casesnotpostprocessed.remove(case)
                    except: 
                        vwrite(f'\nWarning: Unknown error loading {filepath.name}', end='')
            except FileNotFoundError:
                vwrite(self.verbose, f'\nWarning: {filepath.name} expected but not found', end='')
            pbar.update(1)
        
        self.casesrun = set(casesloaded)
        pbar.refresh()
        pbar.close()
        vprint(self.verbose, f'\nData for {len(casesloaded)}/{self.ncases} cases loaded from disk', flush=True)
        
        if casesnotloaded != set():
            vprint(self.verbose, 'Warning: The following cases were not loaded: [' + ', '.join([str(i) for i in casesnotloaded]) + ']')
        if casesnotpostprocessed != set():
            vprint(self.verbose, 'Warning: The following cases have not been postprocessed: [' + ', '.join([str(i) for i in casesnotpostprocessed]) + ']')
        if casesstale != set():
            vprint(self.verbose, 'Warning: The following cases were loaded but may be stale: [' + ', '.join([str(i) for i in casesstale]) + ']')
        
        extrafiles = self.findExtraResultsFiles()
        if extrafiles != set():
            vprint(self.verbose, "Warning: The following extra .mcsim and .mccase files were found in the results directory, run removeExtraResultsFiles() to clean them up: ['" + \
                                 "', '".join([file for file in extrafiles]) + "']")
        

    def findExtraResultsFiles(self):
        files = set(self.resultsdir.glob('**/*.mcsim')) | set(self.resultsdir.glob('**/*.mccase'))
        filenames = set(file.name for file in files)
        try:
            filenames.remove(f'{self.name}.mcsim')
        except:
            pass
        for case in range(self.ncases):
            try:
                filenames.remove(f'{self.name}_{case}.mccase')
            except:
                pass
        return filenames


    def removeExtraResultsFiles(self):
        extrafiles = self.findExtraResultsFiles()
        for file in extrafiles:
            filepath = self.resultsdir / file
            filepath.unlink()


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
