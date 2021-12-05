# MCSim.py
from __future__ import annotations

import os
import numpy as np
import dill
import pathlib
from datetime import datetime, timedelta
from monaco.mc_case import MCCase
from monaco.mc_var import MCInVar, MCOutVar
from monaco.mc_enums import MCFunctions, SampleMethod
from monaco.helper_functions import (get_tuple, slice_by_index, vprint, vwarn,
                                     vwrite, hash_str_repeatable)
from psutil import cpu_count
from pathos.pools import ThreadPool as Pool
from tqdm import tqdm
from typing import Callable, Any, Iterable
from scipy.stats import rv_continuous, rv_discrete


class MCSim:
    """
    The main Monte-Carlo Simulation object.

    Parameters
    ----------
    name : str
        The name for the simulation.
    ndraws : int
        The number of random draws to perform.
    fcns : dict[monaco.mc_enums.MCFunctions, Callable]
        fcns is a dict with keys MCFunctions.PREPROCESS, RUN, and POSTPROCESS.
        These point to user-defined functions with certain input and output
        structures, please see the documentation on how to construct these
        functions.
    firstcaseismedian : bool, default: False
        Whether the first case represents the median value.
    samplemethod : monaco.mc_enums.SampleMethod, default: 'sobol_random'
        The random sampling method to use.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random number to seed the simulation.
    cores : int, default: psutil.cpu_count(logical=False)
        The number of cores to use for running the simulation. Defaults to the
        number of physical cores on the machine.
    verbose : bool, default: True
        Whether to print out warning and status messages.
    debug : bool, default: False
        If False, cases that fail while running will be skipped over. If True,
        cases that fail will raise an exception.
    savesimdata : bool, default: True
        Whether to save the simulation data to disk as a .mcsim file.
    savecasedata : bool, default: True
        Whether to save the full output data for each case to disk as .mccase
        files.
    resultsdir : str | pathlib.Path
        The directory to save simulation and case data to. If None, then this
        defaults to a directory named `name`_results.

    Attributes
    ----------
    rootdir : pathlib.Path
        The directory the simulation was run in.
    filepath : pathlib.Path
        The filepath to the simulation .mcsim datafile.
    invarseeds : list[int]
        The random seeds for each of the input variables.
    caseseeds : list[int]
        The random seeds for each of the cases.
    inittime : datetime.datetime
        The timestamp when this simulation object was created.
    starttime : datetime.datetime
        The timestamp when the simulation began running.
    endtime : datetime.datetime
        The timestamp when the simulation stopped running.
    runtime : datetime.timedelta
        The length of time it took the simulation to run.
    casespreprocessed : set[int]
        The cases which were sucessfully preprocessed.
    casesrun : set[int]
        The cases which were sucessfully run.
    casespostprocessed : set[int]
        The cases which were sucessfully postprocessed.
    mcinvars : dict[str, monaco.mc_var.MCInVar]
        The Monte-Carlo Input Variables.
    mcoutvars : dict[str, monaco.mc_var.MCOutVar]
        The Monte-Carlo Output Variables.
    constvals : dict[str, Any]
        The constant values to pass to each of the cases.
    mccases : list[monaco.mc_case.MCCase]
        The Monte-Carlo Cases.
    ninvars : int
        The number of input variables.
    corrcoeffs : numpy.ndarray
        The correlation coefficients between all of the scalar variables.
    covs : numpy.ndarray
        The covariance matrix between all of the scalar variables.
    covvarlist : list[str]
        The names of all the scalar variables.
    pbar0 : tqdm.tqdm
        Handle for the preprocessing progress bar.
    pbar1 : tqdm.tqdm
        Handle for the running progress bar.
    pbar2 : tqdm.tqdm
        Handle for the postprocessing progress bar.
    runsimid : int
        The unique ID for a particular run of this simulation.
    ncases : int
        The number of cases.
    """
    def __init__(self,
                 name              : str,
                 ndraws            : int,
                 fcns              : dict[MCFunctions, Callable],
                 firstcaseismedian : bool = False,
                 samplemethod      : SampleMethod = SampleMethod.SOBOL_RANDOM,
                 seed              : int  = np.random.get_state(legacy=False)['state']['key'][0],
                 cores             : int  = cpu_count(logical=False),
                 verbose           : bool = True,
                 debug             : bool = False,
                 savesimdata       : bool = True,
                 savecasedata      : bool = True,
                 resultsdir        : str | pathlib.Path = None,
                 ):

        self.checkFcnsInput(fcns)

        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.ndraws = ndraws
        self.fcns = fcns
        self.firstcaseismedian = firstcaseismedian
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
        if self.savesimdata:
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
        self.setFirstCaseMedian(firstcaseismedian)
        self.setNDraws(self.ndraws)  # will regen runsimid


    def __getstate__(self):
        """Function for pickling self to save to file."""
        state = self.__dict__.copy()
        state['mccases'] = []  # don't save mccase data when pickling self
        return state


    def __setstate__(self, state):
        """Function to unpickle self when loading from file."""
        self.__dict__.update(state)
        if self.savecasedata:
            self.loadCases()


    def checkFcnsInput(self,
                       fcns: dict,
                       ) -> None:
        """
        Check the `fcns` input dictionary for correctness.

        Parameters
        ----------
        fcns : dict[monaco.mc_enums.MCFunctions, Callable]
            fcns must be a dict with keys MCFunctions.PREPROCESS, RUN, and
            POSTPROCESS, which point to special user-defined functions.
        """
        if set(fcns.keys()) != {MCFunctions.PREPROCESS, MCFunctions.RUN, MCFunctions.POSTPROCESS}:
            raise ValueError(f'MCSim argument fcns={fcns} must have keys ' +
                             f'{MCFunctions.PREPROCESS}, {MCFunctions.RUN}, ' +
                             f'and {MCFunctions.POSTPROCESS}')
        if any(not callable(f) for f in fcns.values()):
            raise ValueError(f'MCSim argument fcns={fcns} must contain functions as values')


    def setFirstCaseMedian(self,
                           firstcaseismedian : bool,
                           ) -> None:
        """
        Make the first case represent the median expected case or not.

        Parameters
        ----------
        firstcaseismedian : bool
            Whether to make the first case the median case.
        """
        if firstcaseismedian:
            self.firstcaseismedian = True
            self.ncases = self.ndraws + 1
        else:
            self.firstcaseismedian = False
            self.ncases = self.ndraws
        if self.mcinvars != dict():
            for invar in self.mcinvars.values():
                invar.setFirstCaseMedian(firstcaseismedian)


    def addInVar(self,
                 name       : str,
                 dist       : rv_discrete | rv_continuous,
                 distkwargs : dict[str, Any],
                 nummap     : dict[int, Any] = None,
                 seed       : int = None,
                 ) -> None:
        """
        Add an input variable to the simulation.

        Parameters
        ----------
        name : str
            The name of this variable.
        dist : scipy.stats.rv_discrete | scipy.stats.rv_continuous
            The statistical distribution to draw from.
        distkwargs : dict
            The keyword argument pairs for the statistical distribution function.
        nummap : dict[int, Any]
            A dictionary mapping numbers to nonnumeric values.
        seed : int
            The random seed for this variable. If None, a seed will be assigned
            based on the order added.
        """
        self.ninvars += 1
        if seed is None:
            seed = (self.seed + self.ninvars) % 2**32  # seed is dependent on the order added
        self.invarseeds.append(seed)
        invar = MCInVar(name=name, dist=dist, distkwargs=distkwargs, ndraws=self.ndraws,
                        nummap=nummap, samplemethod=self.samplemethod, ninvar=self.ninvars,
                        seed=seed, firstcaseismedian=self.firstcaseismedian, autodraw=False)
        self.mcinvars[name] = invar


    def addConstVal(self,
                    name : str,
                    val  : Any,
                    ) -> None:
        """
        Add a constant value for all the cases to use.

        Parameters
        ----------
        name : str
            Name for this value.
        val : Any
            The constant value.
        """
        self.constvals[name] = val


    def setNDraws(self,
                  ndraws: int,
                  ) -> None:
        """
        Set the number of random draws to perform. Will clear the results.

        Parameters
        ----------
        ndraws : int
            The number of random draws to perform.
        """
        self.clearResults()
        self.ndraws = ndraws
        self.setFirstCaseMedian(self.firstcaseismedian)
        for invar in self.mcinvars.values():
            invar.setNDraws(ndraws)
        if self.mcinvars != dict():
            self.drawVars()


    def drawVars(self):
        """Draw the random values for all the input variables."""
        if self.ninvars > 0:
            vprint(self.verbose, f"Drawing random samples for {self.ninvars} input variables " +
                                 f"via the '{self.samplemethod}' method...", flush=True)
        for invar in self.mcinvars.values():
            invar.draw(ninvar_max=self.ninvars)


    def runSim(self,
               cases : None | int | Iterable[int] = None,
               ) -> None:
        """
        Run the full simulation.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to run. If None, then all cases are run.
        """
        cases_downselect = self.downselectCases(cases=cases)
        vprint(self.verbose, f"Running '{self.name}' Monte Carlo simulation with " +
                             f"{len(cases_downselect)}/{self.ncases} cases...", flush=True)
        self.runSimWorker(casestogenerate=cases_downselect, casestopreprocess=cases_downselect,
                          casestorun=cases_downselect, casestopostprocess=cases_downselect)


    def runIncompleteSim(self) -> None:
        """
        Run the full sim, but only the cases which previously failed to
        preprocess, run, or postprocess.
        """
        allcases = self.allCases()
        casestopreprocess  = allcases - self.casespreprocessed
        casestorun         = allcases - self.casesrun           | casestopreprocess
        casestopostprocess = allcases - self.casespostprocessed | casestopreprocess | casestorun
        casestogenerate    = casestopreprocess

        vprint(self.verbose, f"Resuming incomplete '{self.name}' Monte Carlo simulation with " +
                             f"{len(casestopostprocess)}/{self.ncases} " +
                              "cases remaining to preprocess, " +
                             f"{len(casestorun)}/{self.ncases} " +
                              "cases remaining to run, and " +
                             f"{len(casestopostprocess)}/{self.ncases} " +
                              "cases remaining to postprocess...", flush=True)

        self.runSimWorker(casestogenerate=casestogenerate,
                          casestopreprocess=casestopreprocess,
                          casestorun=casestorun,
                          casestopostprocess=casestopostprocess)


    def runSimWorker(self,
                     casestogenerate    : None | int | Iterable[int],
                     casestopreprocess  : None | int | Iterable[int],
                     casestorun         : None | int | Iterable[int],
                     casestopostprocess : None | int | Iterable[int],
                     ) -> None:
        """
        The worker function to run the full sim.

        Parameters
        ----------
        casestogenerate : None | int | Iterable[int]
            The cases to generate. If None, then all cases are generated.
        casestopreprocess : None | int | Iterable[int]
            The cases to preprocess. If None, then all cases are preprocessed.
        casestorun : None | int | Iterable[int]
            The cases to run. If None, then all cases are run.
        casestopostprocess : None | int | Iterable[int]
            The cases to postprocess. If None, then all cases are
            postprocessed.
        """
        self.starttime = datetime.now()

        if casestorun in (None, self.allCases()):
            self.clearResults()  # only clear results if we are rerunning all cases

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
        """Regenerate the unique ID for this simulation run."""
        self.runsimid = self.genID()


    def genID(self) -> int:
        """
        Generate a unique ID based on the simulation seed, name, and current
        timestamp.

        Returns
        -------
        uniqueid : int
            A unique ID.
        """
        uniqueid = (self.seed + hash_str_repeatable(self.name) + hash(datetime.now())) % 2**32
        return uniqueid


    def genCases(self,
                 cases : None | int | Iterable[int] = None,
                 ) -> None:
        """
        Generate all the Monte-Carlo case objects.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to generate. If None, then all cases are generated.
        """
        vprint(self.verbose, 'Generating cases...', flush=True)
        self.genCaseSeeds()

        if cases is None:
            self.mccases = []

        cases_downselect = self.downselectCases(cases)
        for case in cases_downselect:
            ismedian = False
            if self.firstcaseismedian and case == 0:
                ismedian = True
            self.mccases.append(MCCase(ncase=case, ismedian=ismedian, mcinvars=self.mcinvars,
                                       constvals=self.constvals, seed=int(self.caseseeds[case])))
        self.mccases.sort(key=lambda mccase: mccase.ncase)


    def genCaseSeeds(self) -> None:
        """Generate the random seeds for each of the random cases."""
        generator = np.random.RandomState(self.seed)
        self.caseseeds = list(generator.randint(0, 2**31-1, size=self.ncases))


    def preProcessCases(self,
                        cases : None | int | Iterable[int],
                        ) -> None:
        """
        Preprocess all the Monte-Carlo cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to preprocess. If None, then all cases are preprocessed.
        """
        cases_downselect = self.downselectCases(cases=cases)

        if self.verbose:
            self.pbar0 = tqdm(total=len(cases_downselect), desc='Preprocessing cases',
                              unit=' cases', position=0)

        if self.cores == 1:
            for case in cases_downselect:
                self.preProcessCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            try:
                mccases_downselect = slice_by_index(self.mccases, cases_downselect)
                mccases = p.imap(self.preProcessCase, mccases_downselect)
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
        """
        Preprocess a single Monte-Carlo case.

        Parameters
        ----------
        mccase : monaco.mc_case.MCCase
            The case to preprocess.

        Returns
        -------
        mccase : monaco.mc_case.MCCase
            The same case, preprocessed.
        """
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
                 cases            : None | int | Iterable[int],
                 calledfromrunsim : bool = False,
                 ) -> None:
        """
        Run all the Monte-Carlo cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to run. If None, then all cases are run.
        calledfromrunsim : bool, default: False
            Whether this was called from self.runSim(). If False, a new ID for
            this simulation run is generated.
        """
        cases_downselect = self.downselectCases(cases=cases)

        if not calledfromrunsim:
            self.runsimid = self.genID()

        if self.verbose:
            self.pbar1 = tqdm(total=len(cases_downselect), desc='Running cases',
                              unit=' cases', position=0)

        if self.cores == 1:
            for case in cases_downselect:
                self.runCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            try:
                mccases_downselect = slice_by_index(self.mccases, cases_downselect)
                casesrun = p.imap(self.runCase, mccases_downselect)
                # below is a dummy function to ensure we wait for imap to finish
                casesrun = list(casesrun)
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
            vprint(self.verbose, f"\nRaw case results saved in '{self.resultsdir}'",
                   end='', flush=True)


    def runCase(self,
                mccase : MCCase,
                ) -> None:
        """
        Run a single Monte-Carlo case.

        Parameters
        ----------
        mccase : monaco.mc_case.MCCase
            The case to run.

        Returns
        -------
        mccase : monaco.mc_case.MCCase
            The same case, ran.
        """
        try:
            mccase.starttime = datetime.now()
            mccase.simrawoutput = self.fcns[MCFunctions.RUN](*get_tuple(mccase.siminput))
            mccase.endtime = datetime.now()
            mccase.runtime = mccase.endtime - mccase.starttime
            mccase.runsimid = self.runsimid
            mccase.hasrun = True

            if self.savecasedata:
                filepath = self.resultsdir / f'{self.name}_{mccase.ncase}.mccase'
                mccase.filepath = filepath
                filepath.unlink(missing_ok=True)
                with open(filepath, 'wb') as file:
                    dill.dump(mccase, file, protocol=dill.HIGHEST_PROTOCOL)

            self.casesrun.add(mccase.ncase)

        except Exception:
            if self.debug:
                raise
            vwrite(self.verbose, f'\nRunning case {mccase.ncase} failed')

        if self.pbar1 is not None:
            self.pbar1.update(1)


    def postProcessCases(self,
                         cases : None | int | Iterable[int],
                         ) -> None:
        """
        Postprocess all the Monte-Carlo cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to postprocess. If None, then all cases are
            postprocessed.
        """
        cases_downselect = self.downselectCases(cases=cases)

        if self.verbose:
            self.pbar2 = tqdm(total=len(cases_downselect), desc='Postprocessing cases',
                              unit=' cases', position=0)

        if self.cores == 1:
            for case in cases_downselect:
                self.postProcessCase(mccase=self.mccases[case])

        else:
            p = Pool(self.cores)
            try:
                mccases_downselect = slice_by_index(self.mccases, cases_downselect)
                casespostprocessed = p.imap(self.postProcessCase, mccases_downselect)
                # below is a dummy function to ensure we wait for imap to finish
                casespostprocessed = list(casespostprocessed)
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
        """
        Postprocess a single Monte-Carlo case.

        Parameters
        ----------
        mccase : monaco.mc_case.MCCase
            The case to postprocess.

        Returns
        -------
        mccase : monaco.mc_case.MCCase
            The same case, postprocessed.
        """
        try:
            self.fcns[MCFunctions.POSTPROCESS](mccase, *get_tuple(mccase.simrawoutput))
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
        """Generate the output variables."""
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

            self.mcoutvars[varname] = MCOutVar(name=varname, vals=vals, valmap=valmap,
                                               ndraws=self.ndraws,
                                               firstcaseismedian=self.firstcaseismedian)
            for i in range(self.ncases):
                self.mccases[i].mcoutvars[varname] = self.mcoutvars[varname]


    def genCovarianceMatrix(self) -> None:
        """
        Generate the covariance matrix and correlation coefficients between all
        the scalar variables.
        """
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
                vwarn(self.verbose, "Unable to generate correlation coefficient for " +
                                   f"'{self.covvarlist[i]}'. This may happen if this variable " +
                                    "does not vary, or if an infinite value was drawn.")


    def corr(self) -> tuple[np.ndarray, list[str]]:
        """
        Generate a correlation matrix between all the scalar variables.

        Returns
        -------
        (corcoeffs, covvarlist) : (numpy.ndarray, list[str])
            corrcoeffs is a correlation matrix between all the scalar input
            and output variables.
            covvarlist is a list of all the scalar input and output variables.
        """
        self.genCovarianceMatrix()
        return self.corrcoeffs, self.covvarlist


    def cov(self) -> tuple[np.ndarray, list[str]]:
        """
        Generate a covariance matrix between all the scalar variables.

        Returns
        -------
        (covs, covvarlist) : (numpy.ndarray, list[str])
            covs is a covariance matrix between all the scalar input and
            output variables.
            covvarlist is a list of all the scalar input and output variables.
        """
        self.genCovarianceMatrix()
        return self.covs, self.covvarlist


    def clearResults(self) -> None:
        """Clear all the simulation results."""
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
        """Completely reset the simulation to the default object state."""
        self.clearResults()
        self.mcinvars = dict()
        self.constvals = dict()
        self.ninvars = 0
        self.invarseeds = []
        self.caseseeds = []
        self.starttime = None


    def downselectCases(self,
                        cases : None | int | Iterable[int] = None,
                        ) -> set[int]:
        """
        Convert the `cases` input to a set of all the target cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to downselect. If None, returns all cases.

        Returns
        -------
        cases_downselect : set[int]
            A set of all the cases to use.
        """
        if cases is None:
            cases_downselect = self.allCases()
        else:
            cases_downselect = set(get_tuple(cases))
        return cases_downselect


    def allCases(self) -> set[int]:
        """
        Get a set of the indices for all the cases.

        Returns
        -------
        allCases : set[int]
            A set of all the case numbers.
        """
        allCases = set(range(self.ncases))
        return allCases


    def saveSimToFile(self) -> None:
        """Save the simulation to a .mcsim file"""
        if self.savesimdata:
            self.filepath.unlink(missing_ok=True)
            self.filepath.touch()
            with open(self.filepath, 'wb') as file:
                dill.dump(self, file, protocol=dill.HIGHEST_PROTOCOL)


    def loadCases(self) -> None:
        """Load the data for cases from file."""
        vprint(self.verbose, f'{self.filepath} indicates {len(self.casesrun)}/{self.ncases} ' +
                              'cases were run, attempting to load raw case data from disk...',
                             end='\n', flush=True)
        self.mccases = []
        casesloaded = set()
        casesstale  = set()
        casesnotloaded        = self.allCases()
        casesnotpostprocessed = self.allCases()

        pbar = tqdm(total=len(self.casesrun), unit=' cases', desc='Loading', position=0)

        for case in self.casesrun:
            filepath = self.resultsdir / f'{self.name}_{case}.mccase'
            try:
                with open(filepath, 'rb') as file:
                    try:
                        mccase = dill.load(file)
                        if (not mccase.haspreprocessed) \
                            or (not mccase.hasrun) \
                            or (mccase.runtime is None):  # only load mccase if it completed running
                            self.mccases.append(None)
                            vwarn(self.verbose, f'{filepath.name} did not finish running, ' +
                                                 'not loaded')
                        else:
                            self.mccases.append(mccase)

                            if mccase.runsimid != self.runsimid:
                                vwarn(self.verbose, f'{filepath.name} is not from the most ' +
                                                     'recent run and may be stale')
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
        vprint(self.verbose, f'\nData for {len(casesloaded)}/{self.ncases} cases loaded from disk',
               flush=True)

        if casesnotloaded != set():
            vwarn(self.verbose, 'The following cases were not loaded: ' +
                               f'[{", ".join([str(i) for i in sorted(casesnotloaded)])}]')
        if casesnotpostprocessed != set():
            vwarn(self.verbose, 'The following cases have not been postprocessed: ' +
                               f'[{", ".join([str(i) for i in sorted(casesnotpostprocessed)])}]')
        if casesstale != set():
            vwarn(self.verbose, 'The following cases were loaded but may be stale: ' +
                               f'[{", ".join([str(i) for i in sorted(casesstale)])}]')

        extrafiles = self.findExtraResultsFiles()
        if extrafiles != set():
            vwarn(self.verbose, 'The following extra .mcsim and .mccase files were found in the ' +
                                'results directory, run removeExtraResultsFiles() to clean them ' +
                               f'up: [{", ".join([str(i) for i in sorted(extrafiles)])}]')


    def findExtraResultsFiles(self):
        """
        Find .mcsim and .mccase files that we don't expect to see in the
        results directory.

        Returns
        -------
        filenames : set[pathlib.Path]
            The extra files.
        """
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
        """
        Delete all unexpected .mcsim and .mccase files in the results
        directory.
        """
        extrafiles = self.findExtraResultsFiles()
        for file in extrafiles:
            filepath = self.resultsdir / file
            filepath.unlink()
