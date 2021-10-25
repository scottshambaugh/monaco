# MCSim.py

import os
import numpy as np
import dill
import pathlib
from datetime import datetime, timedelta
from monaco.MCCase import MCCase
from monaco.MCVar import MCInVar, MCOutVar
from monaco.MCEnums import MCFunctions, SampleMethod
from monaco.helper_functions import get_sequence, slice_by_index, vprint, vwarn, vwrite, hash_str_repeatable
from psutil import cpu_count
from pathos.pools import ThreadPool as Pool
from tqdm import tqdm
from typing import Callable, Union, Any
from scipy.stats import rv_continuous, rv_discrete

class MCSim:
    def __init__(self, 
                 name            : str, 
                 ndraws          : int, 
                 fcns            : dict[MCFunctions, Callable], # fcns is a dict with keys MCFunctions.PREPROCESS, RUN, and POSTPROCESS
                 firstcaseismean : bool = False, 
                 samplemethod    : SampleMethod = SampleMethod.SOBOL_RANDOM,
                 seed            : int  = np.random.get_state(legacy=False)['state']['key'][0], 
                 cores           : int  = cpu_count(logical=False), 
                 verbose         : bool = True,
                 debug           : bool = False,
                 savesimdata     : bool = True,
                 savecasedata    : bool = True,
                 resultsdir      : Union[str, pathlib.Path] = None,
                 ):
        
        self.checkFcnsInput(fcns)
        
        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.ndraws = ndraws
        self.fcns = fcns
        self.firstcaseismean = firstcaseismean
        self.samplemethod = samplemethod
        self.seed = seed
        self.cores = cores
        self.savesimdata = savesimdata 
        self.savecasedata = savecasedata

        self.rootdir = pathlib.Path.cwd()
        if isinstance(resultsdir, str):
            resultsdir = pathlib.Path(resultsdir)
        if isinstance(resultsdir, pathlib.Path):
            self.resultsdir = resultsdir
        else:
            self.resultsdir = self.rootdir / f'{self.name}_results'
        self.filepath = self.resultsdir / f'{self.name}.mcsim'

        self.invarseeds : list[int] = []
        self.caseseeds  : list[int] = []
        
        self.inittime  : datetime = datetime.now()
        self.starttime : datetime = None
        self.endtime   : datetime = None
        self.runtime   : timedelta = None

        self.casespreprocessed  : set[int] = set()
        self.casesrun           : set[int] = set()
        self.casespostprocessed : set[int] = set()
        
        self.mcinvars  : dict[str, MCInVar] = dict()
        self.mcoutvars : dict[str, MCOutVar] = dict()
        self.constvals : dict[str, Any] = dict()
        self.mccases : list[MCCase] = []
        self.ninvars : int = 0
        
        self.corrcoeffs : np.ndarray = None
        self.covs       : np.ndarray = None
        self.covvarlist : list[str] = None
        
        self.pbar0 : tqdm = None
        self.pbar1 : tqdm = None
        self.pbar2 : tqdm = None

        self.runsimid : int = None

        self.ncases : int = ndraws + 1
        self.setFirstCaseMean(firstcaseismean)
        self.setNDraws(self.ndraws) # Will regen runsimid
                

    def __getstate__(self):
        state = self.__dict__.copy()
        state['mccases'] = []  # don't save mccase data when pickling self
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.savecasedata:
            self.loadCases()


    def checkFcnsInput(self,
                       fcns: dict,
                       ) -> None:
        if set(fcns.keys()) != {MCFunctions.PREPROCESS, MCFunctions.RUN, MCFunctions.POSTPROCESS}:
            raise ValueError(f"MCSim argument {fcns=} must have keys {MCFunctions.PREPROCESS}, {MCFunctions.RUN}, and {MCFunctions.POSTPROCESS}")
        if any(not callable(f) for f in fcns.values()):
            raise ValueError(f"MCSim argument {fcns=} must contain functions as values")
                

    def setFirstCaseMean(self, 
                         firstcaseismean : bool,
                         ) -> None:
        if firstcaseismean:
           self.firstcaseismean = True
           self.ncases = self.ndraws + 1
        else:
           self.firstcaseismean = False
           self.ncases = self.ndraws
        if self.mcinvars != dict():
            for mcvar in self.mcinvars.values():
                mcvar.setFirstCaseMean(firstcaseismean)


    def addInVar(self, 
                 name       : str, 
                 dist       : Union[rv_discrete, rv_continuous],
                 distkwargs : dict[str, Any], 
                 nummap     : dict[int, Any] = None,
                 seed       : int = None,
                 ) -> None:  
        self.ninvars += 1
        if seed is None:
            seed = (self.seed + self.ninvars) % 2**32  # seed is dependent order added
        self.invarseeds.append(seed)
        self.mcinvars[name] = MCInVar(name=name, dist=dist, distkwargs=distkwargs, ndraws=self.ndraws, nummap=nummap, \
                                      samplemethod=self.samplemethod, ninvar=self.ninvars, seed=seed, firstcaseismean=self.firstcaseismean, autodraw=False)


    def addConstVal(self, 
                    name : str, 
                    val  : Any,
                    ) -> None:  
        self.constvals[name] = val


    def setNDraws(self, 
                  ndraws: int,
                  ) -> None:
        self.clearResults()
        self.ndraws = ndraws
        self.setFirstCaseMean(self.firstcaseismean)
        for mcinvar in self.mcinvars.values():
            mcinvar.setNDraws(ndraws)
        if self.mcinvars != dict():
            self.drawVars()


    def drawVars(self):
        if self.ninvars > 0:
            vprint(self.verbose, f"Drawing random samples for {self.ninvars} input variables via the '{self.samplemethod}' method...", flush=True)
        for mcinvar in self.mcinvars.values():
            mcinvar.draw(ninvar_max=self.ninvars)


    def runSim(self, 
               cases : Union[None, int, list[int], set[int]] = None,
               ) -> None:
        cases = self.downselectCases(cases=cases)
        vprint(self.verbose, f"Running '{self.name}' Monte Carlo simulation with {len(cases)}/{self.ncases} cases...", flush=True)
        self.runSimWorker(casestogenerate=cases, casestopreprocess=cases, casestorun=cases, casestopostprocess=cases)


    def runIncompleteSim(self) -> None:
        casestopreprocess  = self.allCases() - self.casespreprocessed
        casestorun         = self.allCases() - self.casesrun           | casestopreprocess
        casestopostprocess = self.allCases() - self.casespostprocessed | casestopreprocess | casestorun
        casestogenerate    = casestopreprocess

        vprint(self.verbose, f"Resuming incomplete '{self.name}' Monte Carlo simulation with " + \
                             f"{len(casestopostprocess)}/{self.ncases} cases remaining to preprocess, " + \
                             f"{len(casestorun)}/{self.ncases} cases remaining to run, " + \
                             f"and {len(casestopostprocess)}/{self.ncases} cases remaining to postprocess...", flush=True)
        self.runSimWorker(casestogenerate=casestogenerate, casestopreprocess=casestopreprocess, casestorun=casestorun, casestopostprocess=casestopostprocess)


    def runSimWorker(self, 
                     casestogenerate    : Union[None, int, list[int], set[int]],
                     casestopreprocess  : Union[None, int, list[int], set[int]],
                     casestorun         : Union[None, int, list[int], set[int]],
                     casestopostprocess : Union[None, int, list[int], set[int]],
                     ) -> None:            
        self.starttime = datetime.now()

        if casestorun in (None, self.allCases()):
            self.clearResults() # only clear results if we are rerunning all cases

        self.runsimid = self.genID()

        if self.savesimdata or self.savecasedata:
            if not os.path.exists(self.resultsdir):
                os.makedirs(self.resultsdir)
            if self.savesimdata:
                self.saveSimToFile()
        
        self.drawVars()
        self.genCases(cases=casestogenerate)
        self.preProcessCases(cases=casestopreprocess)
        self.runCases(cases=casestorun, calledfromrunsim=True)
        self.postProcessCases(cases=casestopostprocess)
        self.genOutVars()

        self.endtime = datetime.now()
        self.runtime = self.endtime - self.starttime
        
        vprint(self.verbose, f'Simulation complete! Runtime: {self.runtime}', flush=True)
        
        if self.savesimdata:
            vprint(self.verbose, 'Saving sim results to file...', flush=True)
            self.saveSimToFile()
            vprint(self.verbose, f"Sim results saved in '{self.filepath}'", flush=True)


    def genRunSimID(self) -> None:
        self.runsimid = self.genID()


    def genID(self) -> int:
        uniqueid = (self.seed + hash(self.name) + hash(datetime.now())) % 2**32
        return uniqueid


    def genCases(self,
                 cases : Union[None, int, list[int], set[int]] = None,
                 ) -> None:
        vprint(self.verbose, 'Generating cases...', flush=True)
        self.genCaseSeeds()
        
        if cases is None:
            self.mccases = []    
            
        cases = self.downselectCases(cases)
        for case in cases:
            ismean = False
            if self.firstcaseismean and case == 0:
                ismean = True
            self.mccases.append(MCCase(ncase=case, ismean=ismean, mcinvars=self.mcinvars, constvals=self.constvals, seed=int(self.caseseeds[case])))
        self.mccases.sort(key=lambda mccase: mccase.ncase)

    
    def genCaseSeeds(self) -> None:
        generator = np.random.RandomState(self.seed)
        self.caseseeds = list(generator.randint(0, 2**31-1, size=self.ncases))


    def preProcessCases(self, 
                        cases : Union[None, int, list[int], set[int]],
                        ) -> None:
        cases = self.downselectCases(cases=cases)
        
        if self.verbose:
            self.pbar0 = tqdm(total=len(cases), desc='Preprocessing cases', unit=' cases', position=0)

        if self.cores == 1:
            for case in cases:
                 self.preProcessCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            try:
                mccases = p.imap(self.preProcessCase, slice_by_index(self.mccases, cases))
                mccases = list(mccases)
                p.terminate()
                p.restart()
            except KeyboardInterrupt:
                p.terminate()
                p.restart()
                raise
                
        if self.verbose:
            self.pbar0.refresh()
            self.pbar0.close()
            self.pbar0 = None


    def preProcessCase(self, 
                       mccase : MCCase,
                       ) -> MCCase:
        try:
            mccase.siminput = self.fcns[MCFunctions.PREPROCESS](mccase)
            self.casespreprocessed.add(mccase.ncase)
            mccase.haspreprocessed = True
            
        except Exception:
            if self.debug:
                raise
            else:
                vwarn(self.verbose, f'\nPreprocessing case {mccase.ncase} failed')
            
        if self.pbar0 is not None:
            self.pbar0.update(1)
            
        return mccase


    def runCases(self, 
                 cases            : Union[None, int, list[int], set[int]],
                 calledfromrunsim : bool = False,
                 ) -> None:
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
            try:
                casesrun = p.imap(self.runCase, slice_by_index(self.mccases, cases))
                casesrun = list(casesrun) # dummy function to ensure we wait for imap to finish
                p.terminate()
                p.restart()
            except KeyboardInterrupt:
                p.terminate()
                p.restart()
                raise

        if self.verbose:
            self.pbar1.refresh()
            self.pbar1.close()
            self.pbar1 = None

        if self.savecasedata:
            vprint(self.verbose, f"\nRaw case results saved in '{self.resultsdir}'", end='', flush=True)


    def runCase(self, 
                mccase : MCCase,
                ) -> None:
        try:
            mccase.starttime = datetime.now()
            mccase.simrawoutput = self.fcns[MCFunctions.RUN](*get_sequence(mccase.siminput))
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
            
        except Exception:
            if self.debug:
                raise
            vwrite(self.verbose, f'\nRunning case {mccase.ncase} failed')
        
        if self.pbar1 is not None:
            self.pbar1.update(1)


    def postProcessCases(self, 
                         cases : Union[None, int, list[int], set[int]],
                         ) -> None:
        cases = self.downselectCases(cases=cases)
        
        if self.verbose:
            self.pbar2 = tqdm(total=len(cases), desc='Postprocessing cases', unit=' cases', position=0)

        if self.cores == 1:
            for case in cases:
                self.postProcessCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            try:
                casespostprocessed = p.imap(self.postProcessCase, slice_by_index(self.mccases, cases))
                casespostprocessed = list(casespostprocessed) # dummy function to ensure we wait for imap to finish
                p.terminate()
                p.restart()
            except KeyboardInterrupt:
                p.terminate()
                p.restart()
                raise
            
        if self.verbose:
            self.pbar2.refresh()
            self.pbar2.close()
            self.pbar2 = None


    def postProcessCase(self, 
                        mccase : MCCase,
                        ) -> None:
        try:
            self.fcns[MCFunctions.POSTPROCESS](mccase, *get_sequence(mccase.simrawoutput))
            self.casespostprocessed.add(mccase.ncase)
            mccase.haspostprocessed = True
            
        except Exception:
            if self.debug:
                raise
            else:
                vwrite(self.verbose, f'\nPostprocessing case {mccase.ncase} failed')
            
        if self.pbar2 is not None:
            self.pbar2.update(1)
            
            
    def genOutVars(self) -> None:
        for varname in self.mccases[0].mcoutvals.keys():
            vals = []
            for i in range(self.ncases):
                vals.append(self.mccases[i].mcoutvals[varname].val)

            if self.mccases[0].mcoutvals[varname].valmapsource == 'auto':
                uniquevals : set[Any] = set()
                valmap = None
                for i in range(self.ncases):
                    if self.mccases[i].mcoutvals[varname].valmap is None:
                        uniquevals = None
                    else:
                        uniquevals.update(self.mccases[i].mcoutvals[varname].valmap.keys())
                if uniquevals is not None:
                    valmap = dict()
                    for i, val in enumerate(uniquevals):
                        valmap[val] = i
            else:
                valmap = self.mccases[0].mcoutvals[varname].valmap

            self.mcoutvars[varname] = MCOutVar(name=varname, vals=vals, valmap=valmap, ndraws=self.ndraws, firstcaseismean=self.firstcaseismean)
            for i in range(self.ncases):
                self.mccases[i].mcoutvars[varname] = self.mcoutvars[varname]


    def genCovarianceMatrix(self) -> None:
        self.covvarlist = []
        allnums = []
        for var in self.mcinvars.keys():
            if self.mcinvars[var].isscalar:
                allnums.append(self.mcinvars[var].nums)
                self.covvarlist.append(self.mcinvars[var].name)
        for var in self.mcoutvars.keys():
            if self.mcoutvars[var].isscalar:
                allnums.append(self.mcoutvars[var].nums)
                self.covvarlist.append(self.mcoutvars[var].name)
        self.covs = np.cov(np.array(allnums))
        self.corrcoeffs = np.corrcoef(np.array(allnums))
        
        for i, coeff in enumerate(self.corrcoeffs[0]):
            if np.isnan(coeff):
                vwarn(self.verbose, f"Unable to generate correlation coefficient for '{self.covvarlist[i]}'. " + \
                                     "This may happen if this variable does not vary, or if an infinite value was drawn.")


    def corr(self) -> tuple[np.ndarray, list[str]]:
        self.genCovarianceMatrix()
        return self.corrcoeffs, self.covvarlist


    def cov(self) -> tuple[np.ndarray, list[str]]:
        self.genCovarianceMatrix()
        return self.covs, self.covvarlist


    def clearResults(self) -> None:
        self.mccases = []
        self.mcoutvars = dict()
        self.casespreprocessed = set()
        self.casesrun = set()
        self.casespostprocessed = set()
        self.corrcoeff = None
        self.covcoeff = None
        self.covvarlist = None
        self.endtime = None
        self.runtime = None
        self.runsimid = self.genID()


    def reset(self) -> None:
        self.clearResults()
        self.mcinvars = dict()
        self.constvals = dict()
        self.ninvars = 0
        self.invarseeds = []
        self.caseseeds = []
        self.starttime = None


    def downselectCases(self, 
                        cases : Union[None, int, list[int], set[int]] = None,
                        ) -> set[int]:
        if cases is None:
            cases_downselect = self.allCases()
        else:
            cases_downselect = set(get_sequence(cases))
        return cases_downselect


    def allCases(self) -> set[int]:
        allCases = set(range(self.ncases))
        return allCases


    def saveSimToFile(self) -> None:
        if self.savesimdata:
            self.filepath.unlink(missing_ok = True)
            self.filepath.touch()
            with open(self.filepath,'wb') as file:
                dill.dump(self, file, protocol=dill.HIGHEST_PROTOCOL)


    def loadCases(self) -> None:
        vprint(self.verbose, f"{self.filepath} indicates {len(self.casesrun)}/{self.ncases} cases were run, attempting to load raw case data from disk...", end='\n', flush=True)
        self.mccases = []
        casesloaded = set()
        casesstale  = set()
        casesnotloaded        = self.allCases()
        casesnotpostprocessed = self.allCases()
        
        pbar = tqdm(total=len(self.casesrun), unit=' cases', desc='Loading', position=0)
        
        for case in self.casesrun:
            filepath = self.resultsdir / f'{self.name}_{case}.mccase'
            try:
                with open(filepath,'rb') as file:
                    try:
                        mccase = dill.load(file)
                        if (not mccase.haspreprocessed) or (not mccase.hasrun) or (mccase.runtime is None):  # only load mccase if it completed running
                            self.mccases.append(None)
                            vwarn(self.verbose, f'{filepath.name} did not finish running, not loaded')
                        else:
                            self.mccases.append(mccase)
                            
                            if mccase.runsimid != self.runsimid:
                                vwarn(self.verbose, f'{filepath.name} is not from the most recent run and may be stale')
                                casesstale.add(case)
                            casesloaded.add(case)
                            casesnotloaded.remove(case)
                            if case in self.casespostprocessed:
                                casesnotpostprocessed.remove(case)
                    except Exception: 
                        vwarn(self.verbose, f'Unknown error loading {filepath.name}')
            except FileNotFoundError:
                vwarn(self.verbose, f'{filepath.name} expected but not found')
            pbar.update(1)
        
        self.casespreprocessed  = set(casesloaded)
        self.casesrun           = set(casesloaded)
        self.casespostprocessed = set(casesloaded) - casesnotpostprocessed
        pbar.refresh()
        pbar.close()
        vprint(self.verbose, f'\nData for {len(casesloaded)}/{self.ncases} cases loaded from disk', flush=True)
        
        if casesnotloaded != set():
            vwarn(self.verbose, 'The following cases were not loaded: ['              + ', '.join([str(i) for i in sorted(casesnotloaded)]) + ']')
        if casesnotpostprocessed != set():
            vwarn(self.verbose, 'The following cases have not been postprocessed: ['  + ', '.join([str(i) for i in sorted(casesnotpostprocessed)]) + ']')
        if casesstale != set():
            vwarn(self.verbose, 'The following cases were loaded but may be stale: [' + ', '.join([str(i) for i in sorted(casesstale)]) + ']')
        
        extrafiles = self.findExtraResultsFiles()
        if extrafiles != set():
            vwarn(self.verbose, "The following extra .mcsim and .mccase files were found in the results directory, run removeExtraResultsFiles() to clean them up: ['" + \
                                "', '".join([file for file in sorted(extrafiles)]) + "']")
        

    def findExtraResultsFiles(self):
        files = set(self.resultsdir.glob('**/*.mcsim')) | set(self.resultsdir.glob('**/*.mccase'))
        filenames = set(file.name for file in files)
        try:
            filenames.remove(f'{self.name}.mcsim')
        except Exception:
            pass
        for case in range(self.ncases):
            try:
                filenames.remove(f'{self.name}_{case}.mccase')
            except Exception:
                pass

        return filenames


    def removeExtraResultsFiles(self) -> None:
        extrafiles = self.findExtraResultsFiles()
        for file in extrafiles:
            filepath = self.resultsdir / file
            filepath.unlink()
