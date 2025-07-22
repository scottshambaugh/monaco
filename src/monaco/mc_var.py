# mc_var.py
from __future__ import annotations

import numpy as np
from scipy.stats import rv_continuous, rv_discrete, describe
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from copy import copy
from typing import Any, Callable, Iterable, Optional, NamedTuple
from warnings import warn
from abc import ABC, abstractmethod
from monaco.mc_val import Val, InVal, OutVal
from monaco.mc_varstat import VarStat
from monaco.mc_enums import SampleMethod, Sensitivities, VarStatType, InVarSpace
from monaco.mc_sampling import sampling
from monaco.mc_plot import plot, plot_sensitivities
from monaco.helper_functions import empty_list, hashable_val, flatten, get_list, is_num


### Var Base Class ###
class Var(ABC):
    """
    Abstract base class to hold the data for a Monte Carlo variable.

    Parameters
    ----------
    name : str
        The name of this variable.
    ndraws : int
        The number of random draws.
    seed : int
        The random seed to use for bootstrapping.
    firstcaseismedian : bool
        Whether the first case represents the median case.
    datasource : str | None
        If the vals were imported from a file, this is the filepath. If
        generated through monaco, then None.

    Attributes
    ----------
    name : str
        The name of this variable.
    ndraws : int
        The number of random draws.
    seed : int
        The random seed to use for bootstrapping.
    firstcaseismedian : bool
        Whether the first case represents the median case.
    datasource : str | None
        If the vals were imported from a file, this is the filepath. If
        generated through monaco, then None.
    ncases : int
        The number of cases, which is `ndraws + 1` if `firstcaseismedian` and
        `ndraws` otherwise.
    vals : list[Any]
        The values corresponding to the randomly drawn numbers. If valmap is
        None, then `vals == nums.tolist()`.
    valmap : dict[Any, float]
        A dictionary mapping nonnumeric values to numbers (the inverse of
        `nummap`).
    nums : list[np.ndarray]
        The randomly drawn numbers obtained by feeding `pcts` into `dist`.
    nummap : dict[float, Any]
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    pcts : list[float]
        The randomly drawn percentiles.
    maxdim : int
        The maximum dimension of the values.
    isscalar : bool
        Whether this is a scalar variable.
    varstats : list[moncao.mc_varstat.VarStat]
        A list of all the variable statistics for this variable.
    """
    def __init__(self,
                 name              : str,
                 ndraws            : int,
                 seed              : int,
                 firstcaseismedian : bool,
                 datasource        : Optional[str],
                 ):
        self.name = name
        self.ndraws = ndraws
        self.seed = int(seed)
        self.firstcaseismedian = firstcaseismedian
        self.datasource = datasource

        self.ncases = ndraws + 1
        self.setFirstCaseMedian(firstcaseismedian)
        self.vals     : list[Any]
        self.valmap   : dict[Any, float]
        self.nums     : list[np.ndarray]
        self.nummap   : dict[float, Any]
        self.pcts     : list[float]
        self.maxdim   : int
        self.isscalar : bool
        self.varstats : list[VarStat] = empty_list()


    def __getitem__(self,
                    ncase : int,
                    ) -> Val:
        """Get a value from the variable.

        Parameters
        ----------
        ncase : int
            The number of the case to get the value for.

        Returns
        -------
        val : Val
            The value requested.
        """
        return self.getVal(ncase)


    def setFirstCaseMedian(self,
                           firstcaseismedian : bool,
                           ) -> None:
        """
        Generate `val` based on the drawn number and the nummap.

        Parameters
        ----------
        firstcaseismedian : bool
            Whether the first case represents a median case.
        """
        if firstcaseismedian:
            self.firstcaseismedian = True
            self.ncases = self.ndraws + 1
        else:
            self.firstcaseismedian = False
            self.ncases = self.ndraws


    def stats(self) -> NamedTuple:
        """
        Calculates statistics of the variable nums from scipy.stats.describe.

        Returns
        -------
        stats : NamedTuple
            A named tuple with descriptive statistics for the variable nums.
        """
        stats = describe(self.nums)
        return stats


    def addVarStat(self,
                   stat        : VarStatType | Callable,
                   statkwargs  : dict[str, Any] | None = None,
                   cases       : None | int | Iterable[int] = None,
                   bootstrap   : bool = False,
                   bootstrap_k : int = 10,
                   conf        : float = 0.95,
                   seed        : int | None = None,
                   name        : str | None = None,
                   ) -> None:
        """
        Add a variable statistic to this variable.

        Parameters
        ----------
        stat : monaco.mc_enums.VarStatType | Callable
            The type of variable statistic to add.
        statkwargs : dict[str, Any]
            Keyword arguments for the specified variable statistic.
        cases : None | int | Iterable[int]
            The cases to use to calculate the statistic. If None, then all cases
            are used.
        bootstrap : bool (default: True)
            Whether to use bootstrapping to generate confidence intervals for
            the statistic.
        bootstrap_k : int (default: 10)
            The k'th order statistic to determine the number of bootstrap draws
            for the given confidence level. Must be >= 1. Set higher for a
            smoother bootstrap distribution.
        conf : float (default: 0.95)
            The confidence level for the confidence interval.
        seed : int
            The random seed to use for bootstrapping.
        name : str
            The name of the variable statistic to add.

        Notes
        -----
        These are the valid stats with their statkwargs

        max()
            ``'max'``, no kwargs
        min()
            ``'min'``, no kwargs
        median()
            ``'median'``, no kwargs
        mean()
            ``'mean'``, no kwargs
        geomean()
            ``'geomean'``, no kwargs
        mode()
            ``'mode'``, no kwargs
        variance()
            ``'variance'``, no kwargs
        skewness()
            ``'skewness'``, no kwargs
        kurtosis()
            ``'kurtosis'``, no kwargs
        moment(n : int)
            ``'moment'``

            `n` is the n'th moment, `n > 0`.
        percentile(p : float)
            ``'percentile'``

            `p` is the percentile, `0 < p < 1`.
        sigma(sig : float, bound : monaco.mc_enums.StatBound)
            ``'sigma'``

            `sig` is the gaussian sigma value, `-inf < sig < inf`.

            `bound` is the statistical bound, ether `'1-sided'` or `'2-sided'`.
            Default is `'2-sided'`.
        gaussianP(p : float, bound : monaco.mc_enums.StatBound)
            ``'gaussianP'``

            `p` is the percentile, `0 < p < 1`.

            `bound` is the statistical bound, ether `'1-sided'` or `'2-sided'`.
            Default is `'2-sided'`.
        orderstatTI(p : float, c : float, bound : monaco.mc_enums.StatBound)
            ``'orderstatTI'``

            `p` is the percentage, `0 < p < 1`

            `c` is the confidence, `0 < c < 1`. Default is `0.95`.

            `bound` is the statistical bound, ether `'1-sided'`, `'2-sided'`, or
            `'all'`. Default is `'2-sided'`.
        orderstatP(p : float, c : float, bound : monaco.mc_enums.StatBound)
            ``'orderstatP'``

            `p` is the percentage, `0 < p < 1`

            `c` is the confidence, `0 < c < 1`. Default is `0.95`.

            `bound` is the statistical bound, ether `'1-sided lower'`,
            `'1-sided upper'`, `'2-sided'`, `'all'`, or '`nearest'`. Default is
            `'2-sided'`.
        """
        if statkwargs is None:
            statkwargs = dict()
        if seed is None:
            # seed is dependent on the order added
            seed = (self.seed + 1 + len(self.varstats)) % 2**32

        self.varstats.append(VarStat(var=self, stat=stat, statkwargs=statkwargs,
                                     cases=cases,
                                     bootstrap=bootstrap, bootstrap_k=bootstrap_k,
                                     conf=conf, seed=seed, name=name))


    def clearVarStats(self) -> None:
        """
        Remove all the variable statistics for this variable.
        """
        self.varstats = []


    def plot(self,
             vary   : InVar | OutVar | None = None,
             varz   : InVar | OutVar | None = None,
             cases           : None | int | Iterable[int] = None,
             highlight_cases : None | int | Iterable[int] = empty_list(),
             rug_plot    : bool           = False,
             cov_plot    : bool           = False,
             cov_p       : None | float | Iterable[float] = None,
             invar_space : InVarSpace | Iterable[InVarSpace] = InVarSpace.NUMS,
             ax          : Optional[Axes] = None,
             title       : str            = '',
             ) -> tuple[Figure, Axes]:
        """
        Plot this variable, against other variables if desired.
        See monaco.mc_plot.plot() for API details.
        """
        fig, ax = plot(varx=self, vary=vary, varz=varz,
                       cases=cases, highlight_cases=highlight_cases,
                       rug_plot=rug_plot, cov_plot=cov_plot, cov_p=cov_p,
                       invar_space=invar_space,
                       ax=ax, title=title)
        return fig, ax


    @abstractmethod
    def getVal(self,
               ncase : int,
               ) -> Val:
        pass



### InVar Class ###
class InVar(Var):
    """
    A Monte Carlo input variable.

    If `dist` is provided, then the variable will be drawn from a
    statistical distribution. If `vals` is provided, then the variable will
    be set to the provided values. Must provide one or the other.

    `InVal`s can be accessed by case number, eg `invar[0]`.

    Parameters
    ----------
    name : str
        The name of this variable.
    ndraws : int
        The number of random draws.
    dist : scipy.stats.rv_discrete | scipy.stats.rv_continuous
        The statistical distribution to draw from.
    distkwargs : dict
        The keyword argument pairs for the statistical distribution function.
    nummap : dict[float, Any] | None, default: None
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    vals : list[Any] | None, default: None
        Custom values to use instead of drawing from a distribution.
        Length must match ncases.
    pcts : list[float] | None, default: None
        Custom percentiles between 0 and 1 to use instead of random draws.
        Length must match ncases, and the first percentile must be 0.5 if
        firstcaseismedian is True.
    samplemethod : monaco.mc_enums.SampleMethod, default: 'sobol_random'
        The random sampling method to use.
    ninvar : int
        The number of the input variable this is.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random seed for drawing this variable and bootstrapping.
    firstcaseismedian : bool, default: False
        Whether the first case represents the median case.
    autodraw : bool, default: True
        Whether to draw the random values when this variable is created.
    datasource : str | None, default: None
        If the invals were imported from a file, this is the filepath. If
        generated through monaco, then None.

    Attributes
    ----------
    isscalar : bool
        Whether this is a scalar variable. Alway True for an input variable.
    maxdim : int
        The maximum dimensions of the values. Always 0 for an input variable.
    valmap : dict[Any, float]
        A dictionary mapping nonnumeric values to numbers (the inverse of
        `nummap`).
    pcts : list[float]
        The randomly drawn percentiles. If custom percentiles are provided, then
        these are the custom percentiles.
    nums : list[float]
        The randomly drawn numbers obtained by feeding `pcts` into `dist`.
    vals : list[Any]
        The values corresponding to the randomly drawn numbers. If valmap is
        None, then `vals == nums.tolist()`. If custom values are provided, then
        these are the custom values.
    varstats : list[moncao.mc_varstat.VarStat]
        A list of all the variable statistics for this variable.
    """
    def __init__(self,
                 name              : str,
                 ndraws            : int,
                 dist              : rv_discrete | rv_continuous | None = None,
                 distkwargs        : dict[str, Any] | None   = None,
                 nummap            : dict[float, Any] | None = None,
                 vals              : list[Any] | None        = None,
                 pcts              : list[float] | None      = None,
                 samplemethod      : SampleMethod  = SampleMethod.SOBOL_RANDOM,
                 ninvar            : int           = None,
                 seed              : int = np.random.get_state(legacy=False)['state']['key'][0],
                 firstcaseismedian : bool          = False,
                 autodraw          : bool          = True,
                 datasource        : Optional[str] = None,
                 ):
        super().__init__(name=name, ndraws=ndraws, seed=seed,
                         firstcaseismedian=firstcaseismedian,
                         datasource=datasource)

        # Validate inputs for custom values case
        if dist is None and vals is None:
            raise ValueError("Either 'dist' or 'vals' must be provided")
        if dist is not None and vals is not None:
            raise ValueError("Cannot provide both 'dist' and 'vals', choose one")
        if vals is not None:
            samplemethod = None
            vals = get_list(vals)
            if len(vals) != self.ncases:
                raise ValueError(f"Length of 'vals' ({len(vals)}) must match " +
                                 f"ncases ({self.ncases})")

        if pcts is not None:
            pcts = get_list(pcts)
            self.check_pcts(pcts)

        self.dist = dist
        self.custom_vals = vals
        self.custom_pcts = pcts
        if distkwargs is None:
            distkwargs = dict()
        self.distkwargs = distkwargs
        self.samplemethod = samplemethod
        self.ninvar = ninvar
        self.nummap = None
        if nummap is None and vals is not None:
            self.extractNumMap()
        else:
            self.nummap = nummap

        self.isscalar = True
        self.maxdim = 0

        self.genValMap()
        if autodraw:
            self.draw(ninvar_max=None)


    def __repr__(self):
        return (f"{self.__class__.__name__}('{self.name}', ndraws={self.ndraws}, " +
                f"dist={self.dist}, distkwargs={self.distkwargs}, " +
                f"pcts={self.pcts[0]:0.4f}..{self.pcts[-1]:0.4f}, " +
                f"nums={self.nums[0]}..{self.nums[-1]}, " +
                f"vals={self.vals[0]}..{self.vals[-1]}, " +
                f"samplemethod={self.samplemethod}, firstcaseismedian={self.firstcaseismedian})")

    def check_pcts(self,
                   pcts : list[float],
                   ) -> None:
        """
        Check that the percentiles are valid.
        """
        if len(pcts) != self.ncases:
            raise ValueError(f"Length of 'pcts' ({len(pcts)}) must match " +
                             f"ncases ({self.ncases})")
        pcts = np.asarray(pcts)
        if any(pcts <= 0) or any(pcts > 1):
            # Disallow 0 due to https://github.com/scipy/scipy/issues/23358
            raise ValueError("Percentiles must be between 0 and 1")
        if pcts[0] != 0.5 and self.firstcaseismedian:
            raise ValueError("If firstcaseismedian is True, then the first " +
                             "percentile must be 0.5")


    def mapNums(self) -> None:
        """
        Generate `vals` based on the drawn numbers and the nummap.
        """
        self.vals = copy(self.nums)
        if self.nummap is not None:
            for i in range(self.ncases):
                self.vals[i] = self.nummap[self.nums[i]]


    def extractNumMap(self) -> None:
        """
        Parse the input values and extract a nummap.
        """
        try:
            vals_flattened = np.asanyarray(self.custom_vals).flatten()
        except ValueError:
            vals_flattened = flatten([self.custom_vals])
        if all(isinstance(x, (bool, np.bool_)) for x in vals_flattened):
            self.nummap = {1: True, 0: False}
        elif any(not is_num(x) for x in vals_flattened):
            sorted_vals = sorted(set(hashable_val(x) for x in vals_flattened))
            self.nummap = {idx: val for idx, val in enumerate(sorted_vals)}


    def genValMap(self) -> None:
        """
        Generate `valmap` by inverting `nummap`.
        """
        if self.nummap is None:
            self.valmap = None
        else:
            self.valmap = {hashable_val(val): num for num, val in self.nummap.items()}


    def setNDraws(self,
                  ndraws : int,
                  ) -> None:
        """
        Set the number of random draws.

        Parameters
        ----------
        ndraws : int
            The number of random input draws.
        """
        self.ndraws = ndraws
        self.setFirstCaseMedian(self.firstcaseismedian)


    def draw(self,
             ninvar_max : int = None,
             ) -> None:
        """
        Perform the random draws based on the sampling method and the
        statistical distribution, and map those draws to values.
        Alternatively, use custom values if provided.

        Parameters
        ----------
        ninvar_max : int
            The total number of input variables for the simulation.
        """
        self.pcts = []
        self.nums = []

        # Custom values
        if self.custom_vals is not None:
            self.vals = copy(self.custom_vals)

            if self.valmap is not None:
                for val in self.vals:
                    self.nums.append(self.valmap[val])
            else:
                nums = np.asarray(self.vals)
                self.nums = nums.tolist()

            if self.custom_pcts is not None:
                self.pcts = self.custom_pcts
            else:
                # Generate uniform percentiles for custom values
                self.pcts = [i / (self.ndraws - 1) for i in range(self.ndraws)]
                # p = 0, 1 are marginally defined, so we deconflict with a small offset
                eps = np.finfo(float).eps
                self.pcts[0] = eps
                self.pcts[-1] = 1 - eps
                if self.firstcaseismedian:
                    # First gets median, and we deconflict 0.5 with a small offset
                    self.pcts = [0.5] + [p if p != 0.5 else p + eps for p in self.pcts]

            return

        # Distribution-based drawing logic
        dist = self.dist(**self.distkwargs)
        if self.custom_pcts is not None:
            self.pcts = self.custom_pcts
            self.nums = dist.ppf(self.pcts)
            if self.firstcaseismedian:
                # We have already checked that the first percentile is 0.5
                self.nums[0] = self.getDistMedian()
        else:
            if self.firstcaseismedian:
                self.pcts.append(0.5)
                self.nums.append(self.getDistMedian())
            pcts = sampling(ndraws=self.ndraws, method=self.samplemethod,
                            ninvar=self.ninvar, ninvar_max=ninvar_max,
                            seed=self.seed)
            self.pcts.extend(pcts)
            self.nums.extend(dist.ppf(pcts))

        nums = np.asarray(self.nums)
        self.nums = nums.tolist()
        if any(np.isinf(nums)):
            warn( 'Infinite value drawn. Check distribution and parameters: ' +
                 f'{self.dist=}, {self.distkwargs=}')
            if self.samplemethod in (SampleMethod.SOBOL, SampleMethod.HALTON):
                warn(f'Infinite value draw may happen with {self.dist=} for the ' +
                     f'first point of the {self.samplemethod} sampling method. ' +
                     f'Consider using {SampleMethod.SOBOL_RANDOM} instead.')

        if any(np.isnan(nums)):
            raise ValueError( 'Invalid draw. Check distribution and parameters: ' +
                             f'{self.dist=}, {self.distkwargs=}')

        self.mapNums()


    def getVal(self,
               ncase : int,
               ) -> InVal:
        """
        Get the input value for a specific case.

        Parameters
        ----------
        ncase : int
            The number of the case to get the value for.

        Returns
        -------
        val : monaco.mc_val.InVal
            The input value for that case.
        """
        ismedian = False
        if (ncase == 0) and self.firstcaseismedian:
            ismedian = True

        val = InVal(name=self.name, ncase=ncase,
                    pct=self.pcts[ncase], num=self.nums[ncase],
                    dist=self.dist, nummap=self.nummap, valmap=self.valmap,
                    ismedian=ismedian, datasource=self.datasource)
        return val


    def getDistMedian(self) -> float:
        """
        Get the median value for the statistical distribution.

        Returns
        -------
        median : float
            The median value of that distribution.
        """
        dist = self.dist(**self.distkwargs)
        median = dist.ppf(0.5)
        return median


    def getDistMean(self) -> float:
        """
        Get the mean value (also called the expected value) for the statistical
        distribution.

        Returns
        -------
        mean : float
            The mean value of that distribution.
        """
        dist = self.dist(**self.distkwargs)
        ev = dist.expect()

        if isinstance(self.dist, rv_continuous):
            return ev

        # For a discrete distribution, we take the nearest discrete value
        # closest to the expected value
        elif isinstance(self.dist, rv_discrete):
            eps = np.finfo(float).eps
            p = dist.cdf(ev)
            ev_candidates = dist.ppf([p - eps, p, p + eps])
            ev_candidates_dist = abs(ev_candidates - ev)
            ev_closest = ev_candidates[np.nanargmin(ev_candidates_dist)]
            return ev_closest

        else:
            return None



### OutVar Class ###
class OutVar(Var):
    """
    A Monte Carlo output variable.

    `OutVal`s can be accessed by case number, eg `outvar[0]`.

    Parameters
    ----------
    name : str
        The name of this value.
    vals : list[Any]
        The values returned from the simulation.
    valmap : dict, default: None
        A dictionary mapping nonnumeric values to numbers.
    ndraws : int
        The number of random draws.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random seed for bootstrapping.
    firstcaseismedian : bool, default: False
        Whether the first case represents the median case.
    datasource : str | None, default: None
        If the outvals were imported from a file, this is the filepath. If
        generated through monaco, then None.

    Attributes
    ----------
    isscalar : bool
        Whether this is a scalar variable. Alway True for an input variable.
    maxdim : int
        The maximum dimension of the values. Always 0 for an input variable.
    nummap : dict
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    nums : list[np.ndarry]
        The numbers corresponding to the output values. If valmap is None, then
        `nums == list[np.array(vals)]`.
    varstats : list[monaco.VarStat.VarStat]
        A list of all the variable statistics for this variable.
    """
    def __init__(self,
                 name              : str,
                 vals              : list[Any],
                 valmap            : dict = None,
                 ndraws            : int  = None,
                 seed              : int = np.random.get_state(legacy=False)['state']['key'][0],
                 firstcaseismedian : bool = False,
                 datasource        : Optional[str] = None,
                 ):
        if ndraws is None:
            ndraws = len(vals)
            if firstcaseismedian:
                ndraws = ndraws - 1

        super().__init__(name=name, ndraws=ndraws, seed=seed,
                         firstcaseismedian=firstcaseismedian,
                         datasource=datasource)
        self.vals = vals
        self.valmap = valmap
        if valmap is None:
            self.extractValMap()
        self.genNumMap()
        self.mapVals()
        self.genMaxDim()
        self.sensitivity_indices : None | dict = None
        self.sensitivity_ratios  : None | dict = None


    def __repr__(self):
        return (f"{self.__class__.__name__}('{self.name}', ndraws={self.ndraws}, " +
                f"vals={self.vals[0]}..{self.vals[-1]}, " +
                f"nums={self.nums[0]}..{self.nums[-1]}, " +
                f"firstcaseismedian={self.firstcaseismedian})")


    def genMaxDim(self) -> None:
        """
        Parse the output values to determine the maximum dimension of each of
        their shapes.
        """
        self.maxdim = 0
        for num in self.nums:
            self.maxdim = max(self.maxdim, len(num.shape))

        self.isscalar = False
        if self.maxdim == 0:
            self.isscalar = True


    def extractValMap(self) -> None:
        """
        Parse the output values and extract a valmap.
        """
        if self.getVal(0).valmapsource == 'assigned':
            self.valmap = self.getVal(0).valmap
            return

        if any([self.getVal(i).valmap is not None for i in range(self.ncases)]):
            uniquevals : set[Any] = set()
            for i in range(self.ncases):
                if self.getVal(i).valmap is not None:
                    uniquevals.update(self.getVal(i).valmap.keys())
                else:
                    uniquevals.update(set(flatten([self.getVal(i).val])))

            # Sort values, handling None values by putting them first
            sorted_vals = sorted([x for x in uniquevals if x is not None])
            if None in uniquevals:
                sorted_vals.insert(0, None)

            self.valmap = dict()
            for i, val in enumerate(sorted_vals):
                self.valmap[val] = i


    def genNumMap(self) -> None:
        """
        Generate the nummap by inverting `valmap`.
        """
        if self.valmap is None:
            self.nummap = None
        else:
            self.nummap = {num: val for val, num in self.valmap.items()}


    def mapVals(self) -> None:
        """
        Generate `nums` by mapping the values with their valmap.
        """
        self.nums = []
        for i in range(self.ncases):
            self.nums.append(np.asarray(self.getVal(i).num))


    def getVal(self, ncase : int) -> OutVal:
        """
        Get the variable value for a specific case.

        Parameters
        ----------
        ncase : int
            The number of the case to get the value for.

        Returns
        -------
        val : monaco.mc_val.OutVal
            The output value.
        """
        ismedian = False
        if (ncase == 0) and self.firstcaseismedian:
            ismedian = True

        val = OutVal(name=self.name, ncase=ncase, val=self.vals[ncase],
                     valmap=self.valmap, ismedian=ismedian,
                     datasource=self.datasource)
        return val


    def getMedianVal(self) -> OutVal:
        """
        Get the median value for this output variable if `firstcaseismedian`.

        Returns
        -------
        val : monaco.mc_val.OutVal
            The median output value.
        """
        val = None
        if self.firstcaseismedian:
            val = OutVal(name=self.name, ncase=0, val=self.vals[0],
                         valmap=self.valmap, ismedian=True)
        return val


    def split(self) -> dict[str, 'OutVar']:  # Quotes in typing to avoid import error
        """
        Split a multidimensional output variable along its outermost dimension,
        and generate individual OutVar objects for each index.

        Returns
        -------
        vars : dict[str : monaco.mc_var.OutVar]
        """
        vars : dict[str, 'OutVar'] = dict()
        if self.maxdim > 0:
            # First ensure that the vals have the same shape over all cases
            shape = self.nums[0].shape
            for num in self.nums:
                if num.shape[0] != shape[0]:
                    return vars

            for i in range(shape[0]):
                name = self.name + f' [{i}]'
                vals = []
                for j in range(self.ncases):
                    vals.append(self.vals[j][i])
                vars[name] = OutVar(name=name, vals=vals, ndraws=self.ndraws,
                                    valmap=self.valmap,
                                    firstcaseismedian=self.firstcaseismedian)
                for varstat in self.varstats:
                    vars[name].addVarStat(stat=varstat.stat,
                                          statkwargs=varstat.statkwargs,
                                          name=varstat.name)
        return vars


    def plotSensitivities(self,
                          sensitivities : Sensitivities = Sensitivities.RATIOS,
                          ax            : Optional[Axes] = None,
                          title         : str            = '',
                          ) -> tuple[Figure, Axes]:
        """
        Plot the sensitivity indices for this variable.
        See monaco.mc_plot.plot_sensitivities() for API details.
        """
        fig, ax = plot_sensitivities(outvar=self, sensitivities=sensitivities,
                                     ax=ax, title=title)
        return fig, ax
