# mc_sim.py
from __future__ import annotations

import os
import numpy as np
import dask
from dask.distributed import Client
import cloudpickle
import pathlib
from datetime import datetime, timedelta
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm
from typing import Callable, Any, Iterable, Optional
from scipy.stats import rv_continuous, rv_discrete
from monaco.mc_case import Case
from monaco.mc_var import InVar, OutVar, InVarSpace
from monaco.mc_enums import SimFunctions, SampleMethod
from monaco.helper_functions import (get_list, vprint, vwarn, empty_list,
                                     hash_str_repeatable)
from monaco.case_runners import preprocess_case, run_case, postprocess_case
from monaco.tqdm_dask_distributed import tqdm_dask
from monaco.dvars_sensitivity import calc_sensitivities
from monaco.mc_multi_plot import multi_plot


class Sim:
    """
    The main Monte Carlo Simulation object.

    Parameters
    ----------
    name : str
        The name for the simulation.
    ndraws : int
        The number of random draws to perform.
    fcns : dict[monaco.mc_enums.SimFunctions, Callable]
        fcns is a dict with keys SimFunctions.PREPROCESS, RUN, and POSTPROCESS.
        These point to user-defined functions with certain input and output
        structures, please see the documentation on how to construct these
        functions.
    firstcaseismedian : bool, default: False
        Whether the first case represents the median value.
    samplemethod : monaco.mc_enums.SampleMethod, default: 'sobol_random'
        The random sampling method to use.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random number to seed the simulation.
    singlethreaded : bool, default: False
        Whether to run single threaded rather than using dask.
    daskkwargs : dict, default: dict()
        Kwargs to pass to the dask Client constructor, see:
        https://distributed.dask.org/en/stable/api.html#client
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
        defaults to a directory named {name}_results.

    Attributes
    ----------
    rootdir : pathlib.Path
        The directory the simulation was run in.
    filepath : pathlib.Path
        The filepath to the simulation .mcsim datafile.
    invarseeds : list[int]
        The random seeds for each of the input variables.
    outvarseeds : list[int]
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
        The case numbers which were sucessfully preprocessed.
    casesrun : set[int]
        The case numbers which were sucessfully run.
    casespostprocessed : set[int]
        The case numbers which were sucessfully postprocessed.
    invars : dict[str, monaco.mc_var.InVar]
        The Monte Carlo Input Variables.
    outvars : dict[str, monaco.mc_var.OutVar]
        The Monte Carlo Output Variables.
    constvals : dict[str, Any]
        The constant values to pass to each of the cases.
    cases : list[monaco.mc_case.Case]
        The Monte Carlo Cases.
    ninvars : int
        The number of input variables.
    noutvars : int
        The number of output variables.
    corrcoeffs : numpy.ndarray
        The correlation coefficients between all of the scalar variables.
    covs : numpy.ndarray
        The covariance matrix between all of the scalar variables.
    covvarlist : list[str]
        The names of all the scalar variables.
    runsimid : int
        The unique ID for a particular run of this simulation.
    ncases : int
        The number of cases.
    """
    def __init__(self,
                 name              : str,
                 ndraws            : int,
                 fcns              : dict[SimFunctions, Callable],
                 firstcaseismedian : bool = False,
                 samplemethod      : SampleMethod = SampleMethod.SOBOL_RANDOM,
                 seed              : int  = np.random.get_state(legacy=False)['state']['key'][0],
                 singlethreaded    : bool = False,
                 daskkwargs        : dict = dict(),
                 verbose           : bool = True,
                 debug             : bool = False,
                 savesimdata       : bool = True,
                 savecasedata      : bool = True,
                 resultsdir        : str | pathlib.Path = None,
                 ) -> None:

        self.checkFcnsInput(fcns)

        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.ndraws = ndraws
        self.fcns = fcns
        self.firstcaseismedian = firstcaseismedian
        self.samplemethod = samplemethod
        self.seed = seed
        self.singlethreaded = singlethreaded
        self.daskkwargs = daskkwargs
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

        self.invarseeds  : list[int] = []
        self.outvarseeds : list[int] = []
        self.caseseeds   : list[int] = []

        self.inittime  : datetime = datetime.now()
        self.starttime : datetime = None
        self.endtime   : datetime = None
        self.runtime   : timedelta = None

        self.casespreprocessed  : set[int] = set()
        self.casesrun           : set[int] = set()
        self.casespostprocessed : set[int] = set()

        self.invars  : dict[str, InVar] = dict()
        self.outvars : dict[str, OutVar] = dict()
        self.constvals : dict[str, Any] = dict()
        self.cases : list[Case] = []
        self.ninvars  : int = 0
        self.noutvars : int = 0

        self.corrcoeffs : np.ndarray = None
        self.covs       : np.ndarray = None
        self.covvarlist : list[str] = None

        self.runsimid : int = None

        self.ncases : int = ndraws + 1
        self.setFirstCaseMedian(firstcaseismedian)
        self.setNDraws(self.ndraws)  # will regen runsimid

        self.client = None
        self.cluster = None
        if not self.singlethreaded:
            self.initDaskClient()


    def __del__(self) -> None:
        if self.client is not None:
            self.client.close()


    def __getstate__(self) -> dict:
        """Function for pickling self to save to file."""
        state = self.__dict__.copy()
        state['client'] = None  # don't save cluster to file
        state['cluster'] = None  # don't save cluster to file
        state['cases'] = []  # don't save case data when pickling self
        return state


    def __setstate__(self,
                     state: dict,
                     ) -> None:
        """Function to unpickle self when loading from file."""
        self.__dict__.update(state)
        if self.savecasedata:
            self.loadCases()
        if not self.singlethreaded:
            self.initDaskClient()


    def checkFcnsInput(self,
                       fcns: dict[SimFunctions, Callable],
                       ) -> None:
        """
        Check the `fcns` input dictionary for correctness.

        Parameters
        ----------
        fcns : dict[monaco.mc_enums.SimFunctions, Callable]
            fcns must be a dict with keys SimFunctions.PREPROCESS, RUN, and
            POSTPROCESS, which point to special user-defined functions.
        """
        if set(fcns.keys()) \
           != {SimFunctions.PREPROCESS, SimFunctions.RUN, SimFunctions.POSTPROCESS}:
            raise ValueError(f'Sim argument {fcns=} must have keys ' +
                             f'{SimFunctions.PREPROCESS}, {SimFunctions.RUN}, ' +
                             f'and {SimFunctions.POSTPROCESS}')
        if any(not callable(f) for f in fcns.values()):
            raise ValueError(f'Sim argument {fcns=} must contain functions as values')


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
        if self.invars != dict():
            for invar in self.invars.values():
                invar.setFirstCaseMedian(firstcaseismedian)


    def initDaskClient(self):
        """
        Initialize the dask distributed client.
        """
        if not self.singlethreaded:
            self.client = Client(**self.daskkwargs)
            self.cluster = self.client.cluster

            nworkers = len(self.cluster.workers)
            nthreads = nworkers * self.cluster.worker_spec[0]['options']['nthreads']
            memory = nworkers * self.cluster.worker_spec[0]['options']['memory_limit']
            vprint(self.verbose,
                   f'Dask cluster initiated with {nworkers} workers,' +
                   f'{nthreads} threads, {memory/2**30:0.2f} GiB memory.')
            vprint(self.verbose, f'Dask dashboard link: {self.cluster.dashboard_link}')


    def addInVar(self,
                 name       : str,
                 dist       : rv_discrete | rv_continuous,
                 distkwargs : dict[str, Any],
                 nummap     : dict[float, Any] = None,
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
        nummap : dict[float, Any], default: None
            A dictionary mapping numbers to nonnumeric values.
        seed : int
            The random seed for this variable. If None, a seed will be assigned
            based on the order added.
        """
        if name in self.invars.keys():
            raise ValueError(f"'{name}' is already an InVar")

        self.ninvars += 1
        if seed is None:
            # seed is dependent on the order added
            seed = (self.seed + self.ninvars) % 2**32
        self.invarseeds.append(seed)
        invar = InVar(name=name, dist=dist, distkwargs=distkwargs, ndraws=self.ndraws,
                      nummap=nummap, samplemethod=self.samplemethod, ninvar=self.ninvars,
                      seed=seed, firstcaseismedian=self.firstcaseismedian, autodraw=False)
        self.invars[name] = invar


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
        for invar in self.invars.values():
            invar.setNDraws(ndraws)
        if self.invars != dict():
            self.drawVars()


    def drawVars(self) -> None:
        """Draw the random values for all the input variables."""
        if self.ninvars > 0:
            vprint(self.verbose, f"Drawing random samples for {self.ninvars} input variables " +
                                 f"via the '{self.samplemethod}' method...", flush=True)
        for invar in self.invars.values():
            invar.draw(ninvar_max=self.ninvars)


    def runSim(self,
               cases : None | int | Iterable[int] = None,
               ) -> None:
        """
        Run the full simulation.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The case numbers to run. If None, then all cases are run.
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
            The case numbers to generate. If None, then all cases are
            generated.
        casestopreprocess : None | int | Iterable[int]
            The case numbers to preprocess. If None, then all cases are
            preprocessed.
        casestorun : None | int | Iterable[int]
            The case numbers to run. If None, then all cases are run.
        casestopostprocess : None | int | Iterable[int]
            The case numbers to postprocess. If None, then all cases are
            postprocessed.
        """
        self.starttime = datetime.now()

        if set(casestorun) in (None, self.allCases()):
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

        if self.savecasedata:
            self.saveCasesToFile()

        if self.savesimdata:
            self.saveSimToFile()


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
        Generate all the Monte Carlo case objects.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The case numbers to generate. If None, then all cases are
            generated.
        """
        vprint(self.verbose, 'Generating cases...', flush=True)
        self.genCaseSeeds()

        # If we are rerunning partial cases we don't want to reset this
        if cases is None:
            self.cases = []

        cases_downselect = self.downselectCases(cases)
        for ncase in cases_downselect:
            ismedian = False
            if self.firstcaseismedian and ncase == 0:
                ismedian = True
            self.cases.append(Case(ncase=ncase, ismedian=ismedian, invars=self.invars,
                                   constvals=self.constvals, seed=int(self.caseseeds[ncase])))
        self.cases.sort(key=lambda case: case.ncase)


    def genCaseSeeds(self) -> None:
        """Generate the random seeds for each of the random cases."""
        generator = np.random.RandomState(self.seed)
        self.caseseeds = list(generator.randint(0, 2**31-1, size=self.ncases))


    def preProcessCases(self,
                        cases : None | int | Iterable[int],
                        ) -> None:
        """
        Preprocess all the Monte Carlo cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The case numbers to preprocess. If None, then all cases are
            preprocessed.
        """
        cases_downselect = self.downselectCases(cases=cases)
        preprocessedcases = []

        # Single-threaded for loop
        if self.singlethreaded:
            if self.verbose:
                pbar = tqdm(total=len(cases_downselect), desc='Preprocessing cases',
                            unit=' cases', position=0)
            for case in self.cases:
                if case.ncase in cases_downselect:
                    case.haspreprocessed = False
                    case = preprocess_case(self.fcns[SimFunctions.PREPROCESS],
                                            case, self.debug, self.verbose)
                    preprocessedcases.append(case)
                    if self.verbose:
                        pbar.update(1)
            if self.verbose:
                pbar.refresh()
                pbar.close()

        # Dask parallel processing
        else:
            try:
                for case in self.cases:
                    if case.ncase in cases_downselect:
                        case.haspreprocessed = False
                        case_delayed = dask.delayed(preprocess_case)(
                            self.fcns[SimFunctions.PREPROCESS], case,
                            self.debug, self.verbose)
                        preprocessedcases.append(case_delayed)

                if self.verbose:
                    x = dask.persist(preprocessedcases)
                    tqdm_dask(x, total=len(cases_downselect),
                              desc='Preprocessing cases',
                              unit=' cases', position=0)
                    preprocessedcases = dask.compute(*x)[0]
                else:
                    preprocessedcases = dask.compute(*preprocessedcases)

            except KeyboardInterrupt:
                raise

        # Save out results
        for case in preprocessedcases:
            if case.haspreprocessed:
                self.cases[case.ncase] = case
                self.casespreprocessed.add(case.ncase)


    def runCases(self,
                 cases            : None | int | Iterable[int],
                 calledfromrunsim : bool = False,
                 ) -> None:
        """
        Run all the Monte Carlo cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The case numbers to run. If None, then all cases are run.
        calledfromrunsim : bool, default: False
            Whether this was called from self.runSim(). If False, a new ID for
            this simulation run is generated.
        """
        cases_downselect = self.downselectCases(cases=cases)
        runcases = []

        if not calledfromrunsim:
            self.runsimid = self.genID()

        # Single-threaded for loop
        if self.singlethreaded:
            if self.verbose:
                pbar = tqdm(total=len(cases_downselect), desc='Running cases',
                            unit=' cases', position=0)
            for case in self.cases:
                if case.ncase in cases_downselect:
                    case.hasrun = False
                    case = run_case(self.fcns[SimFunctions.RUN], case,
                                    self.debug, self.verbose, self.runsimid)
                    runcases.append(case)
                    if self.verbose:
                        pbar.update(1)
            if self.verbose:
                pbar.refresh()
                pbar.close()

        # Dask parallel processing
        else:
            try:
                for case in self.cases:
                    if case.ncase in cases_downselect:
                        case.hasrun = False
                        case_delayed = dask.delayed(run_case)(
                            self.fcns[SimFunctions.RUN], case,
                            self.debug, self.verbose, self.runsimid)
                        runcases.append(case_delayed)

                if self.verbose:
                    x = dask.persist(runcases)
                    tqdm_dask(x, total=len(cases_downselect),
                              desc='Running cases',
                              unit=' cases', position=0)
                    runcases = dask.compute(*x)[0]
                else:
                    runcases = dask.compute(*runcases)

            except KeyboardInterrupt:
                raise

        # Save out results
        for case in runcases:
            if case.hasrun:
                self.cases[case.ncase] = case
                self.casesrun.add(case.ncase)


    def postProcessCases(self,
                         cases : None | int | Iterable[int],
                         ) -> None:
        """
        Postprocess all the Monte Carlo cases.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The case numbers to postprocess. If None, then all cases are
            postprocessed.
        """
        cases_downselect = self.downselectCases(cases=cases)
        postprocessedcases = []

        # Single-threaded for loop
        if self.singlethreaded:
            if self.verbose:
                pbar = tqdm(total=len(cases_downselect), desc='Preprocessing cases',
                            unit=' cases', position=0)
            for case in self.cases:
                if case.ncase in cases_downselect:
                    case.haspostprocessed = False
                    case = postprocess_case(self.fcns[SimFunctions.POSTPROCESS],
                                            case, self.debug, self.verbose)
                    postprocessedcases.append(case)
                    if self.verbose:
                        pbar.update(1)
            if self.verbose:
                pbar.refresh()
                pbar.close()

        # Dask parallel processing
        else:
            try:
                for case in self.cases:
                    if case.ncase in cases_downselect:
                        case.haspostprocessed = False
                        case_delayed = dask.delayed(postprocess_case)(
                            self.fcns[SimFunctions.POSTPROCESS], case,
                            self.debug, self.verbose)
                        postprocessedcases.append(case_delayed)

                if self.verbose:
                    x = dask.persist(postprocessedcases)
                    tqdm_dask(x, total=len(cases_downselect),
                              desc='Postprocessing cases',
                              unit=' cases', position=0)
                    postprocessedcases = dask.compute(*x)[0]
                else:
                    postprocessedcases = dask.compute(*postprocessedcases)

            except KeyboardInterrupt:
                raise

        # Save out results
        for case in postprocessedcases:
            if case.haspostprocessed:
                self.cases[case.ncase] = case
                self.casespostprocessed.add(case.ncase)


    def genOutVars(self) -> None:
        """Generate the output variables."""
        for i_var, varname in enumerate(self.cases[0].outvals.keys()):
            # seed is dependent on the order added
            seed = (self.seed - 1 - i_var) % 2**32
            self.outvarseeds.append(seed)

            vals = []
            for i in range(self.ncases):
                vals.append(self.cases[i].outvals[varname].val)

            if self.cases[0].outvals[varname].valmapsource == 'auto':
                uniquevals : set[Any] = set()
                valmap : dict[Any, float] = None
                for i in range(self.ncases):
                    if self.cases[i].outvals[varname].valmap is None:
                        uniquevals = None
                    else:
                        uniquevals.update(self.cases[i].outvals[varname].valmap.keys())
                if uniquevals is not None:
                    valmap = dict()
                    for i, val in enumerate(uniquevals):
                        valmap[val] = i
            else:
                valmap = self.cases[0].outvals[varname].valmap

            self.outvars[varname] = OutVar(name=varname, vals=vals, valmap=valmap,
                                           ndraws=self.ndraws, seed=seed,
                                           firstcaseismedian=self.firstcaseismedian)
            for i in range(self.ncases):
                self.cases[i].outvars[varname] = self.outvars[varname]

        self.noutvars = len(self.outvars)


    def scalarOutVars(self) -> dict[str, OutVar]:
        """
        Return a dict of just the scalar output variables.
        """
        scalaroutvars = dict()
        if self.outvars != dict():
            for name, outvar in self.outvars.items():
                if outvar.isscalar:
                    scalaroutvars[name] = outvar

        return scalaroutvars


    def calcSensitivities(self,
                          outvarnames: None | str | Iterable[str] = None,
                          ) -> None:
        """
        Calculate the sensitivity indices for the specified outvars.

        Parameters
        ----------
        outvarnames : None | str | Iterable[str] (default: None)
            The outvar names to calculate sensitivity indices for. If None,
            then calculates sensitivities for all scalar outvars.
        """
        if outvarnames is None:
            outvarname = list(self.scalarOutVars.keys())
        outvarnames = get_list(outvarnames)

        for outvarname in outvarnames:
            if not self.outvars[outvarname].isscalar:
                vwarn(self.verbose, f"Output variable '{outvarname}' is not scalar," +
                                     'skipping sensitivity calculations.')
            else:
                vprint(self.verbose, f"Calculating sensitivity indices for '{outvarname}'.")
                sensitivities, ratios = calc_sensitivities(self, outvarname)

                sensitivities_dict = dict()
                ratios_dict = dict()
                for i, name in enumerate(self.invars.keys()):
                    sensitivities_dict[name] = sensitivities[i]
                    ratios_dict[name] = ratios[i]

                self.outvars[outvarname].sensitivity_indices = sensitivities_dict
                self.outvars[outvarname].sensitivity_ratios = ratios_dict


    def genCovarianceMatrix(self) -> None:
        """
        Generate the covariance matrix and correlation coefficients between all
        the scalar variables.
        """
        self.covvarlist = []
        allnums = []
        for var in self.invars.keys():
            if self.invars[var].isscalar:
                allnums.append(self.invars[var].nums)
                self.covvarlist.append(self.invars[var].name)
        for var in self.outvars.keys():
            if self.outvars[var].isscalar:
                allnums.append(self.outvars[var].nums)
                self.covvarlist.append(self.outvars[var].name)
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


    def plot(self,
             scalarvars  : Optional[list[InVar | OutVar]] = None,
             cases           : None | int | Iterable[int] = None,
             highlight_cases : None | int | Iterable[int] = empty_list(),
             rug_plot    : bool   = True,
             cov_plot    : bool   = True,
             cov_p       : None | float | Iterable[float] = None,
             invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
             fig         : Figure = None,
             title       : str    = '',
             plotkwargs  : dict   = dict(),
             ) -> tuple[Figure, tuple[Axes, ...]]:
        """
        Plot all the scalar variables against each other in a grid.

        Parameters
        ----------
        scalarvars : list[monaco.mc_var.InVar | monaco.mc_var.OutVar]
            The variables to plot. If None, then grabs all the input variables
            and scalar output variables.
        cases : None | int | Iterable[int], default: None
            The cases to plot. If None, then all cases are plotted.
        highlight_cases : None | int | Iterable[int], default: []
            The cases to highlight. If [], then no cases are highlighted.
        rug_plot : bool, default: True
            Whether to plot rug marks.
        cov_plot : bool, default: False
            Whether to plot a covariance ellipse at a certain gaussian percentile
            level.
        cov_p : None | float | Iterable[float], default: None
            The gaussian percentiles for the covariance plot.
        invar_space : monaco.InVarSpace | Iterable[InVarSpace], default: 'nums'
            The space to plot invars in, either 'nums' or 'pcts'. If an iterable,
            specifies this individually for each of varx, vary, and varz.
        fig : matplotlib.figure.Figure, default: None
            The figure handle to plot in. If None, a new figure is created.
        title : str, default: ''
            The figure title.

        Returns
        -------
        (fig, axes) : (matplotlib.figure.Figure, (matplotlib.axes.Axes, ...))
            fig is the figure handle for the plot.
            axes is a tuple of the axes handles for the plots.
        """
        if scalarvars is None:
            scalarvars = []
            for invar in self.invars.values():
                scalarvars.append(invar)
            for scalaroutvar in self.scalarOutVars().values():
                scalarvars.append(scalaroutvar)

        fig, axs = multi_plot(vars=scalarvars,
                              cases=cases, highlight_cases=highlight_cases,
                              rug_plot=rug_plot,
                              cov_plot=cov_plot, cov_p=cov_p,
                              invar_space=invar_space,
                              fig=fig, title=title, plotkwargs=plotkwargs)
        return fig, axs


    def clearResults(self) -> None:
        """Clear all the simulation results."""
        self.cases = []
        self.outvars = dict()
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
        self.invars = dict()
        self.constvals = dict()
        self.ninvars = 0
        self.noutvars = 0
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
            The case numbers to downselect. If None, returns all cases.

        Returns
        -------
        cases_downselect : set[int]
            A set of all the case numbers to use.
        """
        if cases is None:
            cases_downselect = self.allCases()
        else:
            cases_downselect = set(get_list(cases))
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
            vprint(self.verbose, 'Saving sim results to file...', flush=True)

            try:
                self.filepath.unlink()
            except FileNotFoundError:
                pass
            self.filepath.touch()
            with open(self.filepath, 'wb') as file:
                cloudpickle.dump(self, file)

            vprint(self.verbose, f"Sim results saved in '{self.filepath}'", flush=True)


    def saveCasesToFile(self,
                        cases : None | int | Iterable[int] = None,
                        ) -> None:
        """
        Save the specified cases to .mccase files.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The cases to save. If None, save all cases.
        """
        cases_downselect = self.downselectCases(cases=cases)

        if self.savecasedata:
            vprint(self.verbose, 'Saving cases to file...', flush=True)

            for ncase in cases_downselect:
                filepath = self.resultsdir / f'{self.name}_{ncase}.mccase'
                self.cases[ncase].filepath = filepath
                try:
                    filepath.unlink()
                except FileNotFoundError:
                    pass
                with open(filepath, 'wb') as file:
                    cloudpickle.dump(self.cases[ncase], file)

            vprint(self.verbose, f"Raw case results saved in '{self.resultsdir}'",
                   flush=True)


    def loadCases(self) -> None:
        """Load the data for cases from file."""
        vprint(self.verbose, f'{self.filepath} indicates {len(self.casesrun)}/{self.ncases} ' +
                              'cases were run, attempting to load raw case data from disk...',
                             end='\n', flush=True)
        self.cases = []
        casesloaded = set()
        casesstale  = set()
        casesnotloaded        = self.allCases()
        casesnotpostprocessed = self.allCases()

        # pbar = tqdm(total=len(self.casesrun), unit=' cases', desc='Loading', position=0)

        for ncase in self.casesrun:
            filepath = self.resultsdir / f'{self.name}_{ncase}.mccase'
            try:
                with open(filepath, 'rb') as file:
                    try:
                        case = cloudpickle.load(file)
                        if (not case.haspreprocessed) \
                            or (not case.hasrun) \
                            or (case.runtime is None):  # only load case if it completed running
                            self.cases.append(None)
                            vwarn(self.verbose, f'{filepath.name} did not finish running, ' +
                                                 'not loaded')
                        else:
                            self.cases.append(case)

                            if case.runsimid != self.runsimid:
                                vwarn(self.verbose, f'{filepath.name} is not from the most ' +
                                                     'recent run and may be stale')
                                casesstale.add(ncase)
                            casesloaded.add(ncase)
                            casesnotloaded.remove(ncase)
                            if ncase in self.casespostprocessed:
                                casesnotpostprocessed.remove(ncase)

                    except Exception:
                        vwarn(self.verbose, f'Unknown error loading {filepath.name}')

            except FileNotFoundError:
                vwarn(self.verbose, f'{filepath.name} expected but not found')
            # pbar.update(1)

        self.casespreprocessed  = set(casesloaded)
        self.casesrun           = set(casesloaded)
        self.casespostprocessed = set(casesloaded) - casesnotpostprocessed
        # pbar.refresh()
        # pbar.close()
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


    def findExtraResultsFiles(self) -> set[str]:
        """
        Find .mcsim and .mccase files that we don't expect to see in the
        results directory.

        Returns
        -------
        filenames : set[str]
            The extra files.
        """
        files = set(self.resultsdir.glob('**/*.mcsim')) | set(self.resultsdir.glob('**/*.mccase'))
        filenames = set(file.name for file in files)
        try:
            filenames.remove(f'{self.name}.mcsim')
        except Exception:
            pass
        for ncase in range(self.ncases):
            try:
                filenames.remove(f'{self.name}_{ncase}.mccase')
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
