# mc_sim.py
from __future__ import annotations

import os
import numpy as np
import csv
import json
import cloudpickle
import pickle
import pathlib
import multiprocessing
import concurrent.futures
import warnings

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
from monaco.case_runners import preprocess_case, run_case, postprocess_case, execute_full_case
from monaco.globals import _worker_init, register_global_vars
from monaco.dvars_sensitivity import calc_sensitivities
from monaco.mc_multi_plot import multi_plot_grid_rect

try:
    import dask
    from dask.distributed import Client, progress, as_completed
    from monaco.globals import GlobalsPlugin
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class Sim:
    """
    The main Monte Carlo Simulation object.

    `Case`s can be accessed by case number, eg `sim[0]`, and `InVar`s or
    `OutVar`s can be accessed by name, eg `sim['Var1']`.

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
        Whether to run single threaded. This takes precedence over usedask.
    usedask : bool, default: False
        Whether to use dask for parallelization. If False and not
        singlethreaded, will use multiprocessing.
    ncores : int, default: None
        The number of cores to use for multiprocessing. If None, will use all
        available cores. This will override the n_workers kwarg in daskkwargs
        if provided.
    multiprocessing_method : str, default: None
        The method to use for multiprocessing. If None, will use the method
        specified by the multiprocessing.get_start_method() function.
        Valid options are: 'fork', 'spawn', 'forkserver'.
    daskkwargs : dict, default: dict()
        Keyword arguments (kwargs) to pass to the dask Client constructor, see:
        https://distributed.dask.org/en/stable/api.html#client
    verbose : bool, default: True
        Whether to print out warning and status messages.
    debug : bool, default: False
        If False, cases that fail while running will be skipped over. If True,
        cases that fail will raise an exception.
    keepsiminput : bool, default: True
        Whether to keep the siminput for each case after running.
    keepsimrawoutput : bool, default: True
        Whether to keep the simrawoutput for each case after postprocessing.
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
    pool : multiprocessing.Pool | None
        The multiprocessing pool object.
    cluster : dask.distributed.LocalCluster | None
        The dask cluster object. Override this to use a remote cluster.
    client : dask.distributed.Client | None
        The dask client object. Override this to use a remote cluster.
    invarseeds : list[int]
        The random seeds for each of the input variables.
    outvarseeds : list[int]
        The random seeds for each of the output variables.
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
        The case numbers which were successfully preprocessed.
    casesrun : set[int]
        The case numbers which were successfully run.
    casespostprocessed : set[int]
        The case numbers which were successfully postprocessed.
    vars : dict[str, monaco.mc_var.InVar]
        All Variables.
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
                 singlethreaded    : bool = True,
                 usedask           : bool = False,
                 ncores            : int | None = None,
                 multiprocessing_method : str | None = None,
                 daskkwargs        : dict = dict(),
                 verbose           : bool = True,
                 debug             : bool = False,
                 keepsiminput      : bool = True,
                 keepsimrawoutput  : bool = True,
                 savesimdata       : bool = False,
                 savecasedata      : bool = False,
                 resultsdir        : str | pathlib.Path | None = None,
                 ) -> None:

        self.checkFcnsInput(fcns)

        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.ndraws = ndraws
        self.fcns = fcns
        self.firstcaseismedian = firstcaseismedian
        self.samplemethod = samplemethod
        self.seed = int(seed)
        self.singlethreaded = singlethreaded
        self.usedask = usedask
        self.ncores = ncores
        self.multiprocessing_method = multiprocessing_method
        self.daskkwargs = daskkwargs
        self.keepsiminput = keepsiminput
        self.keepsimrawoutput = keepsimrawoutput
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
        self.starttime : datetime | None = None
        self.endtime   : datetime | None = None
        self.runtime   : timedelta | None = None

        self.casespreprocessed  : set[int] = set()
        self.casesrun           : set[int] = set()
        self.casespostprocessed : set[int] = set()

        self.vars    : dict[str, InVar | OutVar] = dict()
        self.invars  : dict[str, InVar] = dict()
        self.outvars : dict[str, OutVar] = dict()
        self.constvals : dict[str, Any] = dict()
        self.cases : list[Case] = []
        self.ninvars  : int = 0
        self.noutvars : int = 0

        self.corrcoeffs : np.ndarray | None = None
        self.covs       : np.ndarray | None = None
        self.covvarlist : list[str] | None = None

        self.runsimid : int | None = None

        self.ncases : int = ndraws + 1
        self.setFirstCaseMedian(firstcaseismedian)
        self.setNDraws(self.ndraws)  # will regen runsimid

        self.pool = None
        self.client = None
        self.cluster = None
        if not self.singlethreaded and self.usedask and not HAS_DASK:
            vwarn(self.verbose, "Dask is not installed, falling back to multiprocessing")
            self.usedask = False

        self.invals_cache = []


    def __del__(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources explicitly."""
        if self.client is not None:
            try:
                result = self.client.close()
                if hasattr(result, '__await__'):
                    warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
            except Exception:
                pass
            finally:
                self.client = None

        if self.pool is not None:
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            finally:
                self.pool = None


    def __getstate__(self) -> dict:
        """Function for pickling self to save to file."""
        state = self.__dict__.copy()
        state['client'] = None  # don't save dask client to file
        state['cluster'] = None  # don't save dask cluster to file
        state['pool'] = None  # don't save multiprocessing pool to file
        state['cases'] = []  # don't save case data when pickling self
        return state


    def __setstate__(self,
                     state: dict,
                     ) -> None:
        """Function to unpickle self when loading from file."""
        self.__dict__.update(state)
        if self.savecasedata:
            self.loadCases()


    def __getitem__(self,
                    ncase_or_varname : int | str,
                    ) -> Case | InVar | OutVar:
        """Get a case, InVar, or OutVar from the simulation.

        Parameters
        ----------
        ncase_or_varname : int | str
            The case number or variable name to get.

        Returns
        -------
        var : Case | InVar | OutVar
            The case, InVar, or OutVar requested.
        """
        if isinstance(ncase_or_varname, int):
            return self.cases[ncase_or_varname]
        elif isinstance(ncase_or_varname, str):
            return self.vars[ncase_or_varname]
        else:
            raise ValueError(f'Sim argument {ncase_or_varname=} must be an int ' +
                             '(case number) or str (variable name)')


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
        if self.singlethreaded or not self.usedask:
            return

        if not HAS_DASK:
            vwarn(self.verbose, "Dask is not installed, skipping dask client initialization")
            return

        plugin = GlobalsPlugin(*self.pickleLargeData())

        if self.client is not None:
            self.client.register_worker_plugin(plugin, name="sim_globals")
            return

        vprint(self.verbose, "Initializing dask client...")
        if self.ncores is not None:
            if 'n_workers' in self.daskkwargs:
                vwarn(self.verbose, "Dask argument n_workers is being overridden " +
                                    "by Sim argument ncores")
            self.daskkwargs['n_workers'] = self.ncores
        self.client = Client(**self.daskkwargs)
        self.cluster = self.client.cluster

        # Initialize the global variables in each worker
        self.client.register_worker_plugin(plugin, name="sim_globals")

        nworkers = len(self.cluster.workers)
        nthreads = nworkers * self.cluster.worker_spec[0]['options']['nthreads']
        memory = nworkers * self.cluster.worker_spec[0]['options']['memory_limit']
        vprint(self.verbose,
               f'Dask cluster initiated with {nworkers} workers, ' +
               f'{nthreads} threads, {memory/2**30:0.2f} GiB memory.')
        vprint(self.verbose, f'Dask client scheduler address: {self.client.scheduler.address}')
        vprint(self.verbose, f'Dask dashboard link: {self.cluster.dashboard_link}')


    def initMultiprocessingPool(self):
        """
        Initialize the multiprocessing pool.
        """
        if self.singlethreaded or self.usedask:
            vwarn(self.verbose, "Multiprocessing pool is not needed in singlethreaded or dask mode")
            return
        elif self.pool is not None:
            return

        start_method = self.multiprocessing_method
        if start_method not in multiprocessing.get_all_start_methods():
            start_method = multiprocessing.get_start_method()
            if self.multiprocessing_method is not None:
                vwarn(self.verbose, "Invalid multiprocessing method " +
                                    f"{self.multiprocessing_method}, using " +
                                    f"{start_method} instead")

        vprint(self.verbose, "Initializing multiprocessing pool...")
        if self.ncores is None:
            self.ncores = multiprocessing.cpu_count()
        ctx = multiprocessing.get_context(start_method)
        if start_method == "fork":
            # For fork, the global variables don't need to be pickled
            initializer = register_global_vars
            data = (self.invars, self.outvars, self.constvals)
        else:
            initializer = _worker_init
            data = self.pickleLargeData()
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.ncores,
                                                           mp_context=ctx,
                                                           initializer=initializer,
                                                           initargs=data)
        vprint(self.verbose,
              f'Multiprocessing pool initiated with {self.ncores} workers ' +
              f'and "{start_method}" start method.')


    def addInVar(self,
                 name       : str,
                 dist       : rv_discrete | rv_continuous | None = None,
                 distkwargs : dict[str, Any]     = None,
                 nummap     : dict[float, Any]   = None,
                 vals       : list[Any] | None   = None,
                 pcts       : list[float] | None = None,
                 seed       : int | None         = None,
                 datasource : Optional[str]      = None,
                 ) -> None:
        """
        Add an input variable to the simulation.

        If `dist` is provided, then the variable will be drawn from a
        statistical distribution. If `vals` is provided, then the variable will
        be set to the provided values. Must provide one or the other.

        Examples
        --------
        >>> sim.addInVar(name='Var1', dist=scipy.stats.norm, distkwargs={'loc': 0, 'scale': 1})
        >>> sim.addInVar(name='Var2', vals=[1, 2, 3])
        >>> sim.addInVar(name='Var3', vals=['a', 'b', 'c'], nummap={0: 'a', 1: 'b', 2: 'c'})

        Parameters
        ----------
        name : str
            The name of this variable.
        dist : scipy.stats.rv_discrete | scipy.stats.rv_continuous | None
            The statistical distribution to draw from. If None, must provide vals.
        distkwargs : dict[str, Any] | None
            The keyword argument pairs for the statistical distribution
            function.
        nummap : dict[float, Any], default: None
            A dictionary mapping numbers to nonnumeric values.
        vals : list[Any] | None, default: None
            Custom values to use instead of drawing from a distribution.
            Length must match ncases.
        pcts : list[float] | None, default: None
            Custom percentiles between 0 and 1 to use instead of random draws.
            Length must match ncases, and the first percentile must be 0.5 if
            firstcaseismedian is True.
        seed : int | None
            The random seed for this variable. If None, a seed will be assigned
            based on the order added.
        datasource : str, default: None
            If the invals were imported from a file, this is the filepath. If
            generated through monaco, then None.
        """
        if name in self.vars.keys():
            raise ValueError(f"'{name}' is already a Variable")

        self.ninvars += 1
        if seed is None:
            # seed is dependent on the order added
            seed = (self.seed + self.ninvars) % 2**32
        self.invarseeds.append(seed)

        invar = InVar(name=name, dist=dist, distkwargs=distkwargs, ndraws=self.ndraws,
                      nummap=nummap, vals=vals, pcts=pcts, samplemethod=self.samplemethod,
                      ninvar=self.ninvars, seed=seed,
                      firstcaseismedian=self.firstcaseismedian, autodraw=False,
                      datasource=datasource)
        self.invars[name] = invar
        self.vars[name] = invar


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
                                 f"via the '{self.samplemethod}' method...", end=' ', flush=True)
            for invar in self.invars.values():
                if invar.datasource is None:
                    invar.draw(ninvar_max=self.ninvars)
            vprint(self.verbose, 'Done', flush=True)


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
        self.executeAllFcns(casestopreprocess=casestopreprocess,
                            casestorun=casestorun,
                            casestopostprocess=casestopostprocess,
                            calledfromrunsim=True)
        self.genOutVars()

        self.endtime = datetime.now()
        self.runtime = self.endtime - self.starttime

        vprint(self.verbose, end='', flush=True)
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
        vprint(self.verbose, 'Generating cases...', end=' ', flush=True)
        self.genCaseSeeds()

        # If we are rerunning partial cases we don't want to reset this
        if cases is None:
            self.cases = []

        cases_downselect = self.downselectCases(cases)
        if self.verbose:
            pbar = tqdm(total=len(cases_downselect), desc='Generating cases',
                        unit="case", position=0)
        for ncase in cases_downselect:
            ismedian = False
            if self.firstcaseismedian and ncase == 0:
                ismedian = True
            self.cases.append(Case(ncase=ncase, ismedian=ismedian, invars=self.invars,
                                   constvals=self.constvals, keepsiminput=self.keepsiminput,
                                   keepsimrawoutput=self.keepsimrawoutput,
                                   seed=int(self.caseseeds[ncase])))
            if self.verbose:
                pbar.update(1)
        if self.verbose:
            pbar.refresh()
            pbar.close()
        self.cases.sort(key=lambda case: case.ncase)

        self.invals_cache = [case.invals for case in self.cases]


    def genCaseSeeds(self) -> None:
        """Generate the random seeds for each of the random cases."""
        generator = np.random.RandomState(self.seed)
        self.caseseeds = list(generator.randint(0, 2**31-1, size=self.ncases))


    def restorePickledCases(self, cases: list[Case]) -> None:
        """Restore the pickled cases to their original state."""
        for case in cases:
            case.invars = self.invars
            case.invals = self.invals_cache[case.ncase]
            case.vals = {**case.invals, **case.outvals}
            case.outvars = self.outvars
            case.vars = self.vars
            case.constvals = self.constvals


    def executeAllFcns(self,
                       casestopreprocess : None | int | Iterable[int] = None,
                       casestorun : None | int | Iterable[int] = None,
                       casestopostprocess : None | int | Iterable[int] = None,
                       calledfromrunsim : bool = False
                       ):
        """
        Preprocess, run, and postprocess all the Monte Carlo cases.

        Parameters
        ----------
        casestopreprocess : None | int | Iterable[int]
            The case numbers to preprocess. If None, then all cases are
            preprocessed.
        casestorun : None | int | Iterable[int]
            The case numbers to run. If None, then all cases are run.
        casestopostprocess : None | int | Iterable[int]
            The case numbers to postprocess. If None, then all cases are
            postprocessed.
        calledfromrunsim : bool, default: False
            Whether this was called from self.runSim(). If False, a new ID for
            this simulation run is generated.
        """
        if self.singlethreaded or not self.usedask:
            same_cases = (casestopreprocess == casestorun == casestopostprocess)
            if same_cases and not self.singlethreaded:
                # This reduces data transfer for multiprocessing by keeping the case in the worker
                self.executeFullPipeline(cases=casestopreprocess,
                                         calledfromrunsim=calledfromrunsim)

            else:
                self.preProcessCases(cases=casestopreprocess)
                self.runCases(cases=casestorun, calledfromrunsim=calledfromrunsim)
                self.postProcessCases(cases=casestopostprocess)

        # Dask has its own path because it can chain delayed functions
        else:
            self.executeFullPipelineDask(casestopreprocess=casestopreprocess,
                                         casestorun=casestorun,
                                         casestopostprocess=casestopostprocess)


    def executeFullPipeline(self,
                            cases: None | int | Iterable[int] = None,
                            calledfromrunsim: bool = False) -> None:
        """
        Execute the full preprocess, run, and postprocess pipeline with minimal
        data transfer, for multiprocessing speed up.

        Parameters
        ----------
        cases : None | int | Iterable[int]
            The case numbers to execute. If None, then all cases are executed.
        calledfromrunsim : bool, default: False
            Whether this was called from self.runSim(). If False, a new ID for
            this simulation run is generated.
        """
        cases_downselect = self.downselectCases(cases=cases)
        fullyexecutedcases = []

        if not calledfromrunsim:
            self.runsimid = self.genID()

        # Multiprocessing with chained operations
        self.initMultiprocessingPool()
        futures = []

        for i in cases_downselect:
            case = self.cases[i]
            case.haspreprocessed = False
            case.hasrun = False
            case.haspostprocessed = False

            inputs = (
                self.fcns[SimFunctions.PREPROCESS],
                self.fcns[SimFunctions.RUN],
                self.fcns[SimFunctions.POSTPROCESS],
                case, self.debug, self.verbose, self.runsimid
            )
            futures.append(self.pool.submit(execute_full_case, *inputs))

        if self.verbose:
            pbar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Preprocessing, running, and postprocessing cases",
                unit="case",
                position=0,
                leave=True
            )
            for future in pbar:
                case = future.result()
                fullyexecutedcases.append(case)
            pbar.refresh()
            pbar.close()
        else:
            for future in concurrent.futures.as_completed(futures):
                case = future.result()
                fullyexecutedcases.append(case)

        # Restore pickled cases and save results
        self.restorePickledCases(fullyexecutedcases)

        for case in fullyexecutedcases:
            self.cases[case.ncase] = case
            if case.haspreprocessed:
                self.casespreprocessed.add(case.ncase)
            if case.hasrun:
                self.casesrun.add(case.ncase)
            if case.haspostprocessed:
                self.casespostprocessed.add(case.ncase)


    def executeFullPipelineDask(self,
                                casestopreprocess : None | int | Iterable[int] = None,
                                casestorun : None | int | Iterable[int] = None,
                                casestopostprocess : None | int | Iterable[int] = None
                                ) -> None:
        """
        Execute the full preprocessing, run, and postprocessing pipeline for all the cases.
        """
        self.initDaskClient()

        casestopreprocess_downselect = self.downselectCases(cases=casestopreprocess)
        casestorun_downselect = self.downselectCases(cases=casestorun)
        casestopostprocess_downselect = self.downselectCases(cases=casestopostprocess)

        preprocessedcases = dict()
        runcases = dict()
        postprocessedcases = dict()
        try:
            for case in self.cases:
                if case.ncase in casestopreprocess_downselect:
                    case.haspreprocessed = False
                if case.ncase in casestorun_downselect:
                    case.hasrun = False
                if case.ncase in casestopostprocess_downselect:
                    case.haspostprocessed = False

                if case.ncase in casestopreprocess_downselect:
                    casepreprocessed_delayed = dask.delayed(preprocess_case)(
                        self.fcns[SimFunctions.PREPROCESS], case,
                        self.debug, self.verbose)
                    preprocessedcases[case.ncase] = casepreprocessed_delayed

                if case.ncase in casestorun_downselect:
                    if case.haspreprocessed:
                        case_to_delay = case
                    else:
                        case_to_delay = preprocessedcases[case.ncase]
                    caserun_delayed = dask.delayed(run_case)(
                        self.fcns[SimFunctions.RUN], case_to_delay,
                        self.debug, self.verbose, self.runsimid)
                    runcases[case.ncase] = caserun_delayed

                if case.ncase in casestopostprocess_downselect:
                    if case.hasrun:
                        case_to_delay = case
                    else:
                        case_to_delay = runcases[case.ncase]
                    casepostprocessed_delayed = dask.delayed(postprocess_case)(
                        self.fcns[SimFunctions.POSTPROCESS], case_to_delay,
                        self.debug, self.verbose)
                    postprocessedcases[case.ncase] = casepostprocessed_delayed

            vprint(self.verbose, f'Preprocessing {len(casestopreprocess_downselect)}, ' +
                                    f'running {len(casestorun_downselect)}, and ' +
                                    f'postprocessing {len(casestopostprocess_downselect)} ' +
                                    'cases...', end='\n', flush=True)
            futures = self.client.compute(list(postprocessedcases.values()),
                                          optimize_graph=False)
            if self.verbose:
                progress(futures, multi=True)
            fullyexecutedcases = []
            for future, case in as_completed(futures, with_results=True):
                fullyexecutedcases.append(case)
                future.release()

        except KeyboardInterrupt:
            raise

        self.restorePickledCases(fullyexecutedcases)

        # Save out results
        for case in fullyexecutedcases:
            if any([case.haspreprocessed, case.hasrun, case.haspostprocessed]):
                self.cases[case.ncase] = case
            if case.haspreprocessed:
                self.casespreprocessed.add(case.ncase)
            if case.hasrun:
                self.casesrun.add(case.ncase)
            if case.haspostprocessed:
                self.casespostprocessed.add(case.ncase)


    def preProcessCases(self,
                        cases : None | int | Iterable[int] = None,
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
                            unit="case", position=0)
            for i in cases_downselect:
                case = self.cases[i]
                case.haspreprocessed = False
                case = preprocess_case(self.fcns[SimFunctions.PREPROCESS],
                                       case, self.debug, self.verbose)
                preprocessedcases.append(case)
                if self.verbose:
                    pbar.update(1)
            if self.verbose:
                pbar.refresh()
                pbar.close()

        # Multiprocessing
        elif not self.usedask:
            self.initMultiprocessingPool()
            futures = []
            for i in cases_downselect:
                case = self.cases[i]
                case.haspreprocessed = False
                inputs = (self.fcns[SimFunctions.PREPROCESS], case, self.debug, self.verbose)
                futures.append(self.pool.submit(preprocess_case, *inputs))

            if self.verbose:
                pbar = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Preprocessing cases",
                    unit="case",
                    position=0,
                    leave=True
                )
                for future in pbar:
                    case = future.result()
                    preprocessedcases.append(case)
                pbar.refresh()
                pbar.close()
            else:
                for future in concurrent.futures.as_completed(futures):
                    case = future.result()
                    preprocessedcases.append(case)

        # Dask parallel processing
        else:
            self.initDaskClient()
            try:
                for i in cases_downselect:
                    case = self.cases[i]
                    case.haspreprocessed = False
                    case_delayed = dask.delayed(preprocess_case)(
                        self.fcns[SimFunctions.PREPROCESS], case,
                        self.debug, self.verbose)
                    preprocessedcases.append(case_delayed)

                vprint(self.verbose, 'Preprocessing ' +
                                     f'{len(cases_downselect)} cases...', flush=True)
                futures = self.client.compute(preprocessedcases, optimize_graph=False)
                if self.verbose:
                    progress(futures, multi=True)
                preprocessedcases = []
                for future, case in as_completed(futures, with_results=True):
                    preprocessedcases.append(case)
                    future.release()

            except KeyboardInterrupt:
                raise

        # We stripped data from the cases during pickling, so we need to re-attach it
        if not self.singlethreaded:
            self.restorePickledCases(preprocessedcases)

        # Save out results
        for case in preprocessedcases:
            if case.haspreprocessed:
                self.cases[case.ncase] = case
                self.casespreprocessed.add(case.ncase)


    def runCases(self,
                 cases            : None | int | Iterable[int] = None,
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
                            unit="case", position=0)
            for i in cases_downselect:
                case = self.cases[i]
                case.hasrun = False
                case = run_case(self.fcns[SimFunctions.RUN], case,
                                self.debug, self.verbose, self.runsimid)
                runcases.append(case)
                if self.verbose:
                    pbar.update(1)
            if self.verbose:
                pbar.refresh()
                pbar.close()

        # Multiprocessing
        elif not self.usedask:
            self.initMultiprocessingPool()
            futures = []
            for i in cases_downselect:
                case = self.cases[i]
                case.hasrun = False
                inputs = (self.fcns[SimFunctions.RUN], case,
                          self.debug, self.verbose, self.runsimid)
                futures.append(self.pool.submit(run_case, *inputs))

            if self.verbose:
                pbar = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Running cases",
                    unit="case",
                    position=0,
                    leave=True
                )
                for future in pbar:
                    case = future.result()
                    runcases.append(case)
                pbar.refresh()
                pbar.close()
            else:
                for future in concurrent.futures.as_completed(futures):
                    case = future.result()
                    runcases.append(case)

        # Dask parallel processing
        else:
            self.initDaskClient()
            try:
                for i in cases_downselect:
                    case = self.cases[i]
                    case.hasrun = False
                    case_delayed = dask.delayed(run_case)(
                        self.fcns[SimFunctions.RUN], case,
                        self.debug, self.verbose, self.runsimid)
                    runcases.append(case_delayed)

                vprint(self.verbose, 'Running ' +
                                     f'{len(cases_downselect)} cases...', flush=True)
                futures = self.client.compute(runcases, optimize_graph=False)
                if self.verbose:
                    progress(futures, multi=True)
                runcases = []
                for future, case in as_completed(futures, with_results=True):
                    runcases.append(case)
                    future.release()

            except KeyboardInterrupt:
                raise

        # We stripped data from the cases during pickling, so we need to re-attach it
        if not self.singlethreaded:
            self.restorePickledCases(runcases)

        # Save out results
        for case in runcases:
            if case.hasrun:
                self.cases[case.ncase] = case
                self.casesrun.add(case.ncase)


    def postProcessCases(self,
                         cases : None | int | Iterable[int] = None,
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
                pbar = tqdm(total=len(cases_downselect), desc='Postprocessing cases',
                            unit="case", position=0)
            for i in cases_downselect:
                case = self.cases[i]
                case.haspostprocessed = False
                case = postprocess_case(self.fcns[SimFunctions.POSTPROCESS],
                                        case, self.debug, self.verbose)
                postprocessedcases.append(case)
                if self.verbose:
                    pbar.update(1)
            if self.verbose:
                pbar.refresh()
                pbar.close()

        # Multiprocessing
        elif not self.usedask:
            self.initMultiprocessingPool()
            futures = []
            for i in cases_downselect:
                case = self.cases[i]
                case.haspostprocessed = False
                inputs = (self.fcns[SimFunctions.POSTPROCESS], case, self.debug, self.verbose)
                futures.append(self.pool.submit(postprocess_case, *inputs))

            if self.verbose:
                pbar = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Postprocessing cases",
                    unit="case",
                    position=0,
                    leave=True
                )
                for future in pbar:
                    case = future.result()
                    postprocessedcases.append(case)
                pbar.refresh()
                pbar.close()
            else:
                for future in concurrent.futures.as_completed(futures):
                    case = future.result()
                    postprocessedcases.append(case)

        # Dask parallel processing
        else:
            self.initDaskClient()
            try:
                for i in cases_downselect:
                    case = self.cases[i]
                    case.haspostprocessed = False
                    case_delayed = dask.delayed(postprocess_case)(
                        self.fcns[SimFunctions.POSTPROCESS], case,
                        self.debug, self.verbose)
                    postprocessedcases.append(case_delayed)

                vprint(self.verbose, 'Postprocessing ' +
                                     f'{len(cases_downselect)} cases...', flush=True)
                futures = self.client.compute(postprocessedcases, optimize_graph=False)
                if self.verbose:
                    progress(futures, multi=True)
                postprocessedcases = []
                for future, case in as_completed(futures, with_results=True):
                    postprocessedcases.append(case)
                    future.release()

            except KeyboardInterrupt:
                raise

        # We stripped data from the cases during pickling, so we need to re-attach it
        if not self.singlethreaded:
            self.restorePickledCases(postprocessedcases)

        # Save out results
        for case in postprocessedcases:
            if case.haspostprocessed:
                self.cases[case.ncase] = case
                self.casespostprocessed.add(case.ncase)


    def genOutVars(self,
                   datasource : Optional[str] = None,
                   ) -> None:
        """
        Generate the output variables.

        Parameters
        ----------
        datasource : str, default: None
            If the outvals were imported from a file, this is the filepath. If
            generated through monaco, then None.
        """
        outval_names = list(self.cases[0].outvals.keys())
        if self.verbose:
            pbar = tqdm(total=len(outval_names), desc='Generating output variables',
                        unit='outvar', position=0)
        for i_var, varname in enumerate(outval_names):
            if varname in self.invars.keys():
                raise ValueError(f"'{varname}' is already a Variable")

            # seed is dependent on the order added, used for bootstrapping
            seed = (self.seed - 1 - i_var) % 2**32
            self.outvarseeds.append(seed)

            # Generate valmap
            if self.cases[0].outvals[varname].valmapsource == 'auto':
                uniquevals : set[Any] = set()
                valmap : dict[Any, float] = None
                for case in self.cases:
                    if case.outvals[varname].valmap is None:
                        uniquevals = None
                        break
                    else:
                        uniquevals.update(case.outvals[varname].valmap.keys())
                if uniquevals is not None:
                    valmap = dict()
                    for i, val in enumerate(uniquevals):
                        valmap[val] = i
            else:
                valmap = self.cases[0].outvals[varname].valmap

            # Generate vals
            vals = [case.outvals[varname].val for case in self.cases]
            outvar = OutVar(name=varname, vals=vals, valmap=valmap,
                            ndraws=self.ndraws, seed=seed,
                            firstcaseismedian=self.firstcaseismedian,
                            datasource=datasource)
            self.outvars[varname] = outvar
            self.vars[varname] = outvar
            for case in self.cases:
                case.addOutVar(outvar)

            if self.verbose:
                pbar.update(1)

        if self.verbose:
            pbar.refresh()
            pbar.close()

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


    def extendOutVars(self,
                      outvars : None | str | Iterable[str] = None,
                      ) -> None:
        """
        Extend the non-scalar output variables with their last value, so that
        they all have the same shape. Can be useful for plotting or calculating
        variable statistics.

        Parameters
        ----------
        outvars : dict[str, OutVar], default: None
            The output variables to extend. If None, then extends all the
            output variables.
        """
        outvars_to_extend = []
        if outvars is None:
            outvars_to_extend = [outvar for outvar in self.outvars.values()
                                 if not outvar.isscalar]
        else:
            outvars = get_list(outvars)
            for name in outvars:
                if not self.outvars[name].isscalar:
                    outvars_to_extend.append(self.outvars[name])
                else:
                    vwarn(self.verbose, f"Output variable '{name}' is scalar," +
                                         'skipping extension.')

        outvars_extended = []
        for outvar in outvars_to_extend:
            n_points = max(len(outvar.vals[i]) for i in range(self.ncases))
            for i in range(self.ncases):
                vals = outvar.vals[i]
                deficit = n_points - len(vals)
                if deficit < 0:
                    continue

                if isinstance(vals, list):
                    outvar.vals[i] = vals + [vals[-1]] * deficit
                elif isinstance(vals, np.ndarray):
                    rep = np.repeat(vals[-1][np.newaxis, ...], deficit, axis=0)
                    outvar.vals[i] = np.concatenate([vals, rep], axis=0)
                elif HAS_PANDAS and isinstance(vals, pd.Series):
                    rep = pd.Series([vals.iloc[-1]] * deficit, index=range(len(vals), n_points))
                    outvar.vals[i] = pd.concat([vals, rep])
                elif HAS_PANDAS and isinstance(vals, pd.Index):
                    rep = pd.Index([vals.iloc[-1]] * deficit, dtype=vals.dtype)
                    outvar.vals[i] = pd.Index(np.concatenate([vals.to_numpy(), rep.to_numpy()]),
                                              dtype=vals.dtype)
                else:
                    vwarn(self.verbose, f"Outvar '{outvar.name}' has an unsupported type" +
                                        f"for extension: {type(vals)}. Skipping extension.")
                    break

            outvar_extended = OutVar(name=outvar.name, vals=outvar.vals, valmap=outvar.valmap,
                                     ndraws=outvar.ndraws, seed=outvar.seed,
                                     firstcaseismedian=outvar.firstcaseismedian,
                                     datasource=outvar.datasource)
            outvars_extended.append(outvar_extended)

        for outvar in outvars_extended:
            self.outvars[outvar.name] = outvar
            self.vars[outvar.name] = outvar
            for i in range(self.ncases):
                self.cases[i].addOutVar(outvar)


    def calcSensitivities(self,
                          outvarnames : None | str | Iterable[str] = None,
                          cases       : None | int | Iterable[int] = None,
                          tol         : float = 1e-6,
                          verbose     : bool = False,
                          ) -> None:
        """
        Calculate the sensitivity indices for the specified outvars.

        Parameters
        ----------
        outvarnames : None | str | Iterable[str] (default: None)
            The outvar names to calculate sensitivity indices for. If None,
            then calculates sensitivities for all scalar outvars.
        cases : None | int | Iterable[int], default: None
            The cases to calculate sensitivities for. If None, then all cases
            are used.
        tol : float, default 1e-6
            The convergence tolerance for scipy's minimize function acting on the
            negative log likelihood function.
        verbose : bool, default False
            Whether to print diagnostic information.
        """
        if outvarnames is None:
            outvarnames = list(self.scalarOutVars().keys())
        outvarnames = get_list(outvarnames)

        for outvarname in outvarnames:
            if not self.outvars[outvarname].isscalar:
                vwarn(self.verbose, f"Output variable '{outvarname}' is not scalar," +
                                     'skipping sensitivity calculations.')
            else:
                vprint(self.verbose, f"Calculating sensitivity indices for '{outvarname}'...")
                sensitivities, ratios = calc_sensitivities(self, outvarname,
                                                           cases=cases,
                                                           tol=tol, verbose=verbose)

                sensitivities_dict = dict()
                ratios_dict = dict()
                for i, name in enumerate(self.invars.keys()):
                    sensitivities_dict[name] = sensitivities[i]
                    ratios_dict[name] = ratios[i]

                self.outvars[outvarname].sensitivity_indices = sensitivities_dict
                self.outvars[outvarname].sensitivity_ratios = ratios_dict
                vprint(self.verbose, "Done calculating sensitivity indices.")


    def genCovarianceMatrix(self) -> None:
        """
        Generate the covariance matrix and correlation coefficients between all
        the scalar variables.
        """
        self.covvarlist = []
        allnums = []
        for var in self.vars.keys():
            if self.vars[var].isscalar:
                allnums.append(self.vars[var].nums)
                self.covvarlist.append(self.vars[var].name)
        self.covs = np.cov(np.asarray(allnums))
        self.corrcoeffs = np.corrcoef(np.asarray(allnums))

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
             scalarvars  : Optional[list[InVar | OutVar | str]] = None,
             cases           : None | int | Iterable[int] = None,
             highlight_cases : None | int | Iterable[int] = empty_list(),
             rug_plot    : bool   = False,
             cov_plot    : bool   = False,
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
        scalarvars : list[monaco.mc_var.InVar | monaco.mc_var.OutVar | str]
            The variables to plot. If None, then grabs all the input variables
            and scalar output variables.
        cases : None | int | Iterable[int], default: None
            The cases to plot. If None, then all cases are plotted.
        highlight_cases : None | int | Iterable[int], default: []
            The cases to highlight. If [], then no cases are highlighted.
        rug_plot : bool, default: False
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

        scalarvars = get_list(scalarvars)

        for i, scalarvar in enumerate(scalarvars):
            if isinstance(scalarvar, str):
                scalarvars[i] = self.vars[scalarvar]

        invars = []
        scalaroutvars = []
        for scalarvar in scalarvars:
            if isinstance(scalarvar, InVar):
                invars.append(scalarvar)
            elif isinstance(scalarvar, OutVar) and scalarvar.isscalar:
                scalaroutvars.append(scalarvar)

        fig, axs = multi_plot_grid_rect(varsx=invars,
                                        varsy=scalaroutvars,
                                        cases=cases, highlight_cases=highlight_cases,
                                        rug_plot=rug_plot,
                                        cov_plot=cov_plot, cov_p=cov_p,
                                        invar_space=invar_space,
                                        fig=fig, title=title, plotkwargs=plotkwargs)
        return fig, axs


    def clearResults(self) -> None:
        """Clear all the simulation results."""
        for varname in self.outvars.keys():
            if varname in self.vars.keys():
                del self.vars[varname]
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
        self.vars = dict()
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


    def exportVars(self,
                   vars : dict[str, InVar | OutVar],
                   filename : Optional[str | pathlib.Path],
                   ) -> pathlib.Path:
        """
        Export the nums for the selected to file for use externally.

        The format for csv will be:

        .. code-block:: text

            "vars[0].name", "vars[1].name", "vars[2].name", ...
            vars[0].nums[0], vars[1].nums[0], vars[2].nums[0], ...
            vars[0].nums[1], vars[1].nums[1], vars[2].nums[1], ...
            ...

        The format for json will be:

        .. code-block:: text

            {
                "vars[0].name": [vars[0].nums],
                "vars[1].name": [vars[1].nums],
                ...
            }

        Parameters
        ----------
        vars : dict[str, InVar | OutVar]
            The vars to save to file. Will export the nums, not the vals.
        filename : Optional[str | pathlib.Path]
            The file to save to. Must be a csv or json.
            If a str, then will save in the resultsdir.

        Returns
        -------
        filepath : pathlib.Path
            The filepath the vars were saved to.
        """
        if vars == []:
            raise ValueError('No vars to save to file.')

        if isinstance(filename, str):
            filepath = self.resultsdir / filename
        elif isinstance(filename, pathlib.Path):
            filepath = filename

        if filepath.suffix.lower() not in ('.csv', '.json'):
            raise ValueError(f"'{filename}' must be a .csv or .json file.")
        if filepath.exists():
            vwarn(self.verbose, f'{filepath.name} already exists, overwriting.')

        if filepath.suffix.lower() == '.csv':
            varnames = list(vars.keys())
            for i, var in enumerate(vars.values()):
                if i == 0:
                    data_csv = np.asarray(var.nums)
                else:
                    data_csv = np.vstack([data_csv, np.asarray(var.nums)])

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(varnames)
                for i in range(self.ncases):
                    if len(data_csv.shape) == 1:
                        writer.writerow((data_csv[i], ))
                    else:
                        writer.writerow(data_csv[:, i])

        elif filepath.suffix.lower() == '.json':
            data_json = dict()
            for varname, var in vars.items():
                data_json[varname] = np.asarray(var.nums).tolist()

            with open(filepath, 'w', newline='') as f:
                json.dump(data_json, f, indent=0)

        return filepath


    def exportInVars(self,
                     filename : Optional[str | pathlib.Path] = None,
                     ) -> None:
        """
        Export the drawn nums for all the invars to file for use externally.
        See `monaco.Sim.exportVars` docstring for csv and json formatting.

        Parameters
        ----------
        filename : Optional[str | pathlib.Path]
            The file to save to. Must be a csv or json.
            If a str, then will save in the resultsdir.
            If None, then will save to '{self.name}_invarnums.json'.
        """
        vprint(self.verbose, 'Exporting InVar draws to file...', flush=True)

        if filename is None:
            filename = self.resultsdir / f'{self.name}_invarnums.json'

        filepath = self.exportVars(self.invars, filename)

        vprint(self.verbose, f"InVar nums saved in '{filepath.name}'", flush=True)


    def exportOutVars(self,
                      filename : Optional[str | pathlib.Path] = None,
                      ) -> None:
        """
        Export the nums for all the outvars to file for use externally.
        See `monaco.Sim.exportVars` docstring for csv and json formatting.

        Parameters
        ----------
        filename : Optional[str | pathlib.Path]
            The file to save to. Must be a csv or json.
            If a str, then will save in the resultsdir.
            If None, then will save to '{self.name}_outvarnums.json'.
        """
        vprint(self.verbose, 'Exporting InVar draws to file...', flush=True)

        if filename is None:
            filename = self.resultsdir / f'{self.name}_outvarnums.json'

        filepath = self.exportVars(self.outvars, filename)

        vprint(self.verbose, f"OutVar nums saved in '{filepath.name}'", flush=True)


    def importVars(self,
                   filepath : str | pathlib.Path,
                   ) -> tuple[dict[str, list[Any]], pathlib.Path]:
        """
        Import values from an external file.

        The format for csv must be:

        .. code-block:: text

            "vars[0].name", "vars[1].name", "vars[2].name", ...
            vars[0].nums[0], vars[1].nums[0], vars[2].nums[0], ...
            vars[0].nums[1], vars[1].nums[1], vars[2].nums[1], ...
            ...

        The format for json must be:

        .. code-block:: text

            {
                "vars[0].name": [vars[0].nums],
                "vars[1].name": [vars[1].nums],
                ...
            }

        Parameters
        ----------
        filepath : str | pathlib.Path
            The file to load from. Must be a csv or json.

        Returns
        -------
        data : dict[str, Any]
            A dictionary where the keys are the variable name and the values
            are a list of the vals.
        filepath : pathlib.Path
            The full filepath for the file which was loaded from.
        """
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)

        if filepath.suffix.lower() not in ('.csv', '.json'):
            raise ValueError(f"'{filepath.name}' must be a .csv or .json file.")

        if self.cases == []:
            self.genCases()

        data : dict[str, list] = dict()
        if filepath.suffix.lower() == '.csv':
            with open(filepath, 'r') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                headers = next(reader)
                for outvalname in headers:
                    data[outvalname] = []
                for row in reader:
                    for outvalname, vals in zip(headers, row):
                        data[outvalname].append(vals)

        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)

        for vals_list in data.values():
            if len(vals_list) != self.ncases:
                raise ValueError(f'Length of data ({len(vals_list)}) must match ' +
                                 f'the number of sim cases ({self.ncases}).')

        return data, filepath


    def importInVars(self,
                     filepath     : str | pathlib.Path,
                     dists        : Optional[list[rv_discrete | rv_continuous]] = None,
                     distskwargs  : Optional[list[dict[str, Any]]] = None,
                     nummaps      : Optional[list[dict[Any, float]]] = None,
                     ) -> None:
        """
        Import draws from an external file as InVals. For each of the keyword
        arguments, they must be the same length as the number of invars.
        See `monaco.Sim.importVars` docstring for csv and json formatting.

        Parameters
        ----------
        filepath : str | pathlib.Path
            The file to load from. Must be a csv or json.
        dists : list[rv_discrete | rv_continuous], default: None
            A list of the distribution that was used for the draws. Needed if
            it is desired to plot the analytical distribution or run a DVARS
            sensitivity analysis. Note that for discrete distributions, the
            calcuated percentiles will not likely match the original draws.
        distskwargs : list[dict[str, Any]], default: None
            A list of the distribution kwargs that were used for the draws.
            Must be matched with `dists` above.
        nummaps : list[dict[Any, float]], default: None
            A list of nummap dicts mapping numbers to nonnumeric values.
        """
        vprint(self.verbose, 'Importing InVals from file...', flush=True)

        data, filepath = self.importVars(filepath)

        if dists is not None and distskwargs is not None:
            dists = get_list(dists)
            distskwargs = get_list(distskwargs)
            if len(dists) != len(data):
                raise ValueError(f'Length of dists ({len(dists)}) ',
                                 f'must match the number of invars ({len(data)}).')
            if len(distskwargs) != len(data):
                raise ValueError(f'Length of distskwargs ({len(distskwargs)}) ',
                                 f'must match the number of invars ({len(data)}).')
        else:
            dists = [None for _ in range(len(data))]
            distskwargs = [None for _ in range(len(data))]

        if nummaps is not None:
            nummaps = get_list(nummaps)
            if len(nummaps) != len(data):
                raise ValueError(f'Length of nummaps ({len(nummaps)}) ',
                                 f'must match the number of invars ({len(data)}).')
        else:
            nummaps = [None for _ in range(len(data))]

        for i, (valname, nums) in enumerate(data.items()):

            if dists[i] is None or distskwargs[i] is None:
                pcts = [None for _ in range(self.ncases)]
                if distskwargs[i] is None:
                    distskwargs[i] = dict()
            else:
                dist = dists[i](**distskwargs[i])
                pcts = np.asarray(dist.cdf(nums))

            self.addInVar(name=valname, dist=dists[i], distkwargs=distskwargs[i], nummap=nummaps[i],
                          seed=None, datasource=str(filepath.resolve()))
            nums = np.asarray(nums).tolist()
            self.invars[valname].nums = nums
            self.invars[valname].pcts = pcts
            self.invars[valname].mapNums()

            for case in self.cases:
                case.invars[valname] = self.invars[valname]
                case.invals[valname] = self.invars[valname].getVal(case.ncase)


        vprint(self.verbose, f"InVals loaded from '{filepath.name}' and converted to variables",
               flush=True)


    def importOutVars(self,
                      filepath : str | pathlib.Path,
                      nummaps  : Optional[list[dict[float, Any]]] = None,
                      ) -> None:
        """
        Import results from an external file as OutVals, convert to OutVars.
        See `monaco.Sim.importVars` docstring for csv and json formatting.

        Parameters
        ----------
        filepath : str | pathlib.Path
            The file to load from. Must be a csv or json.
        nummaps : list[dict[float, Any]], default: None
            A list of nummap dicts mapping numbers to nonnumeric values. Note
            that this is reversed from providing valmaps to OutVals.
        """
        vprint(self.verbose, 'Importing OutVals from file...', flush=True)

        data, filepath = self.importVars(filepath)

        if nummaps is not None:
            nummaps = get_list(nummaps)
            if len(nummaps) != len(data):
                raise ValueError(f'Length of nummaps ({len(nummaps)}) ',
                                 f'must match the number of outvars ({len(data)}).')
            valmaps = []
            for nummap in nummaps:
                if nummap is not None:
                    valmaps.append({val: num for num, val in nummap.items()})
                else:
                    valmaps.append(None)
        else:
            nummaps = [None for _ in range(len(data))]
            valmaps = [None for _ in range(len(data))]

        for i, (valname, nums) in enumerate(data.items()):
            if nummaps[i] is None:
                vals = nums
            else:
                vals = [nummaps[i][num] for num in nums]
            for case in self.cases:
                case.addOutVal(valname, vals[case.ncase], valmap=valmaps[i])

        self.genOutVars(datasource=str(filepath.resolve()))

        vprint(self.verbose, f"OutVals loaded from '{filepath.name}' and converted to variables",
               flush=True)


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

        # pbar = tqdm(total=len(self.casesrun), unit="case", desc='Loading', position=0)

        for ncase in self.casesrun:
            filepath = self.resultsdir / f'{self.name}_{ncase}.mccase'
            try:
                with open(filepath, 'rb') as file:
                    try:
                        case = cloudpickle.load(file)
                        self.restorePickledCases([case])
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


    def pickleLargeData(self, protocol: int = 5) -> tuple[bytes, bytes, bytes]:
        """
        Pickle the large data objects for use in multiprocessing.

        Parameters
        ----------
        protocol : int, default: 5
            The pickle protocol to use.

        Returns
        -------
        invars_blob  : bytes
            The pickled invars.
        outvars_blob : bytes
            The pickled outvars.
        constvals_blob : bytes
            The pickled constvals.
        """
        invars_blob  = pickle.dumps(self.invars, protocol=protocol)
        outvars_blob = pickle.dumps(self.outvars, protocol=protocol)
        constvals_blob = pickle.dumps(self.constvals, protocol=protocol)
        return invars_blob, outvars_blob, constvals_blob
