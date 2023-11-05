# mc_varstat.py
from __future__ import annotations

# Somewhat hacky type checking to avoid circular imports:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from monaco.mc_var import Var

import numpy as np
from copy import copy
from statistics import mode
from scipy.stats import bootstrap, moment, skew, kurtosis
from scipy.stats.mstats import gmean
from monaco.helper_functions import get_list
from monaco.gaussian_statistics import pct2sig, sig2pct
from monaco.order_statistics import (order_stat_TI_n, order_stat_TI_k,
                                     order_stat_P_k, get_iP)
from monaco.mc_enums import StatBound, VarStatType, VarStatSide
from typing import Any, Callable


class VarStat:
    """
    A variable statistic for a Monte Carlo variable.

    Parameters
    ----------
    var : monaco.mc_var.Var
        The variable to generate statistics for.
    stat : monaco.mc_enums.VarStatType | Callable
        The statistic to generate. Can be custom. If custom, must be able to
        accept an "axis" kwarg for bootstrap vectorization.
    statkwargs : dict[str:Any]
        The keyword arguments for the variable statistic.
    bootstrap : bool (default: True)
        Whether to use bootstrapping to generate confidence intervals for the
        statistic.
    bootstrap_k : int (default: 10)
        The k'th order statistic to determine the number of bootstrap draws for
        the given confidence level. Must be >= 1. Set higher for a smoother
        bootstrap distribution.
    conf : float (default: 0.95)
        The confidence level for the confidence interval.
    seed : int (default: np.random.get_state(legacy=False)['state']['key'][0])
        The random seed to use for bootstrapping.
    name : str
        The name of the variable statistic.

    Attributes
    ----------
    nums : numpy.ndarray
        The output of the variable statistic function applied to `var.nums`
    confidence_interval_low_nums : numpy.ndarray
        The nums for the low side of the confidence interval (None if no CI).
    confidence_interval_high_nums : numpy.ndarray
        The nums for the high side of the confidence interval (None if no CI).
    vals : list[Any]
        The values for the `nums` as determined by `var.nummap`
    confidence_interval_low_vals : list[Any]
        The values for `confidence_interval_low_nums` via `var.nummap`
    confidence_interval_high_vals : list[Any]
        The values for `confidence_interval_high_nums` via `var.nummap`
    bootstrap_n : int
        The number of bootstrap samples.

    Notes
    -----
    These are the valid stats with their statkwargs

    max()
        'max', no kwargs
    min()
        'min', no kwargs
    median()
        'median', no kwargs
    mean()
        'mean', no kwargs
    geomean()
        'geomean', no kwargs
    mode()
        'mode', no kwargs
    variance()
        'variance', no kwargs
    skewness()
        'skewness', no kwargs
    kurtosis()
        'kurtosis', no kwargs
    moment(n : int)
        'moment'
        `n` is the n'th moment, `n > 0`.
    percentile(p : float)
        'percentile'
        `p` is the percentile, `0 < p < 1`.
    sigma(sig : float, bound : monaco.mc_enums.StatBound)
        'sigma'
        `sig` is the gaussian sigma value, `-inf < sig < inf`.

        `bound` is the statistical bound, ether `'1-sided'` or `'2-sided'`.
        Default is `'2-sided'`.
    gaussianP(p : float, bound : monaco.mc_enums.StatBound)
        'gaussianP'
        `p` is the percentile, `0 < p < 1`.

        `bound` is the statistical bound, ether `'1-sided'` or `'2-sided'`.
        Default is `'2-sided'`.
    orderstatTI(p : float, c : float, bound : monaco.mc_enums.StatBound)
        'orderstatTI'
        `p` is the percentage, `0 < p < 1`

        `c` is the confidence, `0 < c < 1`. Default is `0.95`.

        `bound` is the statistical bound, ether `'1-sided'`, `'2-sided'`, or
        `'all'`. Default is `'2-sided'`.
    orderstatP(p : float, c : float, bound : monaco.mc_enums.StatBound)
        'orderstatP'
        `p` is the percentage, `0 < p < 1`

        `c` is the confidence, `0 < c < 1`. Default is `0.95`.

        `bound` is the statistical bound, ether `'1-sided lower'`,
        `'1-sided upper'`, `'2-sided'`, `'all'`, or '`nearest'`. Default is
        `'2-sided'`.
    """
    def __init__(self,
                 var         : Var,
                 stat        : str | VarStatType | Callable,
                 statkwargs  : dict[str, Any] | None = None,
                 bootstrap   : bool = True,
                 bootstrap_k : int = 10,
                 conf        : float = 0.95,
                 seed        : int = np.random.get_state(legacy=False)['state']['key'][0],
                 name        : str | None = None,
                 ):

        self.var = var
        if isinstance(stat, str):
            stat = stat.lower()
        self.stat = stat
        if statkwargs is None:
            statkwargs = dict()
        self.statkwargs = statkwargs

        self.nums : np.ndarray = np.array([])
        self.vals : list[Any] | np.ndarray = []
        self.name = name

        self.bootstrap = bootstrap
        if bootstrap_k < 1:
            raise ValueError(f'bootstrap_k = {bootstrap_k} must be >= 1')
        self.bootstrap_k = bootstrap_k
        self.conf = conf
        self.confidence_interval_low_nums : list | np.ndarray = None
        self.confidence_interval_high_nums : list | np.ndarray = None
        self.confidence_interval_low_vals : list[Any] | np.ndarray = []
        self.confidence_interval_high_vals : list[Any] | np.ndarray = []
        self.bootstrap_n : int | None = None
        self.seed = seed

        if isinstance(stat, Callable):
            self.setName(f'{self.var.name} {str(stat)}')
            self.genStatsFunction(fcn=stat, fcnkwargs=statkwargs)
        elif stat == VarStatType.MAX:
            self.setName(f'{self.var.name} Max')
            self.genStatsFunction(fcn=np.max)
        elif stat == VarStatType.MIN:
            self.setName(f'{self.var.name} Min')
            self.genStatsFunction(fcn=np.min)
        elif stat == VarStatType.MEDIAN:
            self.setName(f'{self.var.name} Median')
            self.genStatsFunction(fcn=np.median)
        elif stat == VarStatType.MEAN:
            self.setName(f'{self.var.name} Mean')
            self.genStatsFunction(fcn=np.mean)
        elif stat == VarStatType.GEOMEAN:
            self.setName(f'{self.var.name} Geometric Mean')
            self.genStatsFunction(fcn=gmean)
        elif stat == VarStatType.MODE:
            self.setName(f'{self.var.name} Mode')
            self.genStatsFunction(fcn=mode)
        elif stat == VarStatType.VARIANCE:
            self.setName(f'{self.var.name} Variance')
            self.genStatsFunction(fcn=np.var)
        elif stat == VarStatType.SKEWNESS:
            self.setName(f'{self.var.name} Skewness')
            self.genStatsFunction(fcn=skew)
        elif stat == VarStatType.KURTOSIS:
            self.setName(f'{self.var.name} Kurtosis')
            self.genStatsFunction(fcn=kurtosis)
        elif stat == VarStatType.MOMENT:
            self.genStatsMoment()
        elif stat == VarStatType.PERCENTILE:
            self.genStatsPercentile()
        elif stat == VarStatType.SIGMA:
            self.genStatsSigma()
        elif stat == VarStatType.GAUSSIANP:
            self.genStatsGaussianP()
        elif stat == VarStatType.ORDERSTATTI:
            self.genStatsOrderStatTI()
        elif stat == VarStatType.ORDERSTATP:
            self.genStatsOrderStatP()
        else:
            raise ValueError(f'{stat=} must be callable, or one of the following: ' +
                             f'{VarStatType.MAX}, {VarStatType.MIN}, {VarStatType.MEDIAN}, ' +
                             f'{VarStatType.MEAN}, {VarStatType.GEOMEAN}, {VarStatType.MODE}, ' +
                             f'{VarStatType.PERCENTILE}, {VarStatType.SIGMA}, ' +
                             f'{VarStatType.GAUSSIANP}, {VarStatType.ORDERSTATTI}, ' +
                             f'{VarStatType.ORDERSTATP}')


    def genStatsMoment(self) -> None:
        """
        Get the n'th moment about the mean of the variable.
        """
        if 'n' not in self.statkwargs:
            raise ValueError(f'{self.stat} requires the kwarg ''n''')

        self.n = self.statkwargs['n']
        self.setName(f'{self.var.name} {self.n}''th Moment')
        self.genStatsFunction(moment, {'moment': self.n})


    def genStatsPercentile(self) -> None:
        """
        Get the value of the variable at the inputted percentile.
        """
        if 'p' not in self.statkwargs:
            raise ValueError(f'{self.stat} requires the kwarg ''p''')

        self.p = self.statkwargs['p']
        self.setName(f'{self.var.name} {self.p*100}% Percentile')
        self.genStatsFunction(np.quantile, {'q': self.p})


    def genStatsSigma(self) -> None:
        """
        Get the value of the variable at the inputted sigma value, assuming
        a gaussian distribution.
        """
        if 'sig' not in self.statkwargs:
            raise ValueError(f'{self.stat} requires the kwarg ''sig''')
        if 'bound' not in self.statkwargs:
            self.bound = StatBound.TWOSIDED
        else:
            self.bound = self.statkwargs['bound']

        self.sig = self.statkwargs['sig']
        self.p = sig2pct(self.sig, bound=self.bound)
        self.setName(f'{self.var.name} {self.sig} Sigma')
        self.genStatsFunction(self.sigma, {'sig': self.sig})


    def genStatsGaussianP(self) -> None:
        """
        Get the value of the variable at the inputted percentile value,
        assuming a gaussian distribution.
        """
        if 'p' not in self.statkwargs:
            raise ValueError(f'{self.stat} requires the kwarg ''p''')
        if 'bound' not in self.statkwargs:
            self.bound = StatBound.TWOSIDED
        else:
            self.bound = self.statkwargs['bound']

        self.p = self.statkwargs['p']
        self.sig = pct2sig(self.p, bound=self.bound)
        self.setName(f'{self.var.name} Guassian {self.p*100}%')
        self.genStatsFunction(self.sigma, {'sig': self.sig})


    def sigma(self,
              x,  # TODO: explicit typing here
              sig  : float,
              axis : int | None = None,
              ) -> float:
        """
        Calculate the sigma value of a normally distributed list of numbers.

        Parameters
        ----------
        x : TODO typing
            The numbers to calculate the sigma value for.
        sig : float
            The sigma value.
        axis : int (default: None)
            The axis of x to calculate along.
        """
        std = np.std(x, axis=axis)
        return np.mean(x, axis=axis) + sig*std


    def statsFunctionWrapper(self,
                             x : Any,
                             axis : int | None = None,  # Needed for bootstrap vectorization
                             ) -> Any:
        """
        A wrapper function to allow using a bootstrap function that uses kwargs.
        Relies on self.fcn and self.fcnkwargs already being set. Note that fcn
        must accept an `axis` kwarg if bootstrapping.

        Parameters
        ----------
        x : Any
            The input for the function.
        axis : int (default: None)
            The axis of x to calculate along.
        """
        return self.fcn(x, **self.fcnkwargs, axis=axis)


    def genStatsFunction(self,
                         fcn       : Callable,
                         fcnkwargs : dict[str, Any] = None,
                         ) -> None:
        """
        A wrapper function to generate statistics via a generic function.

        Parameters
        ----------
        fcn : Callable
            The function used to generate the desired statistics.
        fcnkwargs : dict[str, Any]
            The keyword arguments for the function.
        """
        self.fcn = fcn
        if fcnkwargs is None:
            fcnkwargs = dict()
        self.fcnkwargs = fcnkwargs
        if self.bootstrap:
            self.bootstrap_n = order_stat_TI_n(self.bootstrap_k, p=0.5, c=self.conf)

        # Scalar Variables
        if self.var.isscalar:
            # Calculate nums and confidence interval for each point in the sequence
            self.nums = self.statsFunctionWrapper(self.var.nums)
            if self.bootstrap:
                # Switch to method='Bca' once https://github.com/scipy/scipy/issues/15883 resolved
                res = bootstrap((np.array(self.var.nums),), self.statsFunctionWrapper,
                                confidence_level=self.conf,
                                n_resamples=self.bootstrap_n,
                                random_state=self.seed, method='basic')
                self.confidence_interval_low_nums = res.confidence_interval.low
                self.confidence_interval_high_nums = res.confidence_interval.high

            # Calculate the corresponding vals based on the nummap
            self.vals = copy(self.nums)
            if self.bootstrap:
                self.confidence_interval_low_vals = copy(self.confidence_interval_low_nums)
                self.confidence_interval_high_vals = copy(self.confidence_interval_high_nums)
            if self.var.nummap is not None:
                self.vals = [self.var.nummap[num] for num in self.nums]
                if self.bootstrap:
                    self.confidence_interval_low_vals = \
                        [self.var.nummap[num] for num in self.confidence_interval_low_nums]
                    self.confidence_interval_high_vals = \
                        [self.var.nummap[num] for num in self.confidence_interval_high_nums]

        # 1-D Variables
        elif self.var.maxdim == 1:
            nums_list = get_list(self.var.nums)
            npoints = max(len(x) for x in nums_list)
            if self.bootstrap:
                confidence_interval_low_nums = []
                confidence_interval_high_nums = []

            # Calculate nums and confidence interval for each point in the sequence
            nums = []
            for i in range(npoints):
                numsatidx = np.array([x[i] for x in nums_list if len(x) > i])
                nums.append(self.statsFunctionWrapper(numsatidx))
                if self.bootstrap:
                    # Switch to Bca once https://github.com/scipy/scipy/issues/15883 resolved
                    res = bootstrap((numsatidx,), self.statsFunctionWrapper,
                                    confidence_level=self.conf,
                                    n_resamples=self.bootstrap_n,
                                    random_state=self.seed, method='basic')
                    confidence_interval_low_nums.append(res.confidence_interval.low)
                    confidence_interval_high_nums.append(res.confidence_interval.high)
            self.nums = nums
            if self.bootstrap:
                self.confidence_interval_low_nums = confidence_interval_low_nums
                self.confidence_interval_high_nums = confidence_interval_high_nums

            # Calculate the corresponding vals based on the nummap
            self.vals = copy(self.nums)
            if self.bootstrap:
                self.confidence_interval_low_vals = copy(self.confidence_interval_low_nums)
                self.confidence_interval_high_vals = copy(self.confidence_interval_high_nums)
            if self.var.nummap is not None:
                self.vals = [[self.var.nummap[x] for x in y] for y in self.nums]
                if self.bootstrap:
                    self.confidence_interval_low_vals \
                        = [[self.var.nummap[x] for x in y]
                           for y in self.confidence_interval_low_nums]
                    self.confidence_interval_low_vals \
                        = [[self.var.nummap[x] for x in y]
                           for y in self.confidence_interval_high_nums]

        else:
            # Suppress warning since this will become valid when Var is split
            # warn('VarStat only available for scalar or 1-D data')
            pass


    def genStatsOrderStatTI(self) -> None:
        """Get the order statistic tolerance interval value of the variable."""
        self.checkOrderStatsKWArgs()

        if self.bound == StatBound.ONESIDED and self.p >= 0.5:
            self.side = VarStatSide.HIGH
        elif self.bound == StatBound.ONESIDED:
            self.side = VarStatSide.LOW
        elif self.bound == StatBound.TWOSIDED:
            self.side = VarStatSide.BOTH
        elif self.bound == StatBound.ALL:
            self.bound = StatBound.TWOSIDED
            self.side = VarStatSide.ALL
        else:
            raise ValueError(f'{self.bound} is not a valid bound for genStatsOrderStatTI')

        if isinstance(self.bound, StatBound):
            bound_str = self.bound.value
        else:
            bound_str = self.bound

        self.setName(f'{self.var.name} ' +
                     f'{bound_str} P{round(self.p*100,4)}/{round(self.c*100,4)}% ' +
                      'Confidence Interval')

        self.k = order_stat_TI_k(n=self.var.ncases, p=self.p, c=self.c, bound=self.bound)

        if self.var.isscalar:
            sortednums = sorted(self.var.nums)
            if self.side == VarStatSide.LOW:
                sortednums.reverse()
            if self.side in (VarStatSide.HIGH, VarStatSide.LOW):
                self.nums = np.array(sortednums[-self.k])
                if self.var.nummap is not None:
                    self.vals = self.var.nummap[self.nums.item()]
            elif self.side == VarStatSide.BOTH:
                self.nums = np.array([sortednums[self.k-1], sortednums[-self.k]])
                if self.var.nummap is not None:
                    self.vals = np.array([self.var.nummap[self.nums[0]],
                                          self.var.nummap[self.nums[1]]])
            elif self.side == VarStatSide.ALL:
                self.nums = np.array([sortednums[self.k-1],
                                      np.median(sortednums),
                                      sortednums[-self.k]])
                if self.var.nummap is not None:
                    self.vals = np.array([self.var.nummap[self.nums[0]],
                                          self.var.nummap[self.nums[1]],
                                          self.var.nummap[self.nums[2]]])
            if self.var.nummap is None:
                self.vals = copy(self.nums)

        elif self.var.maxdim == 1:
            npoints = max(x.shape[0] if len(x.shape) > 0 else 0 for x in self.var.nums)
            self.nums = np.empty(npoints)
            if self.side == VarStatSide.BOTH:
                self.nums = np.empty((npoints, 2))
            elif self.side == VarStatSide.ALL:
                self.nums = np.empty((npoints, 3))
            for i in range(npoints):
                numsatidx = [x[i] for x in self.var.nums
                             if (len(x.shape) > 0 and x.shape[0] > i)]
                sortednums = sorted(numsatidx)
                if self.side == VarStatSide.LOW:
                    sortednums.reverse()
                if self.side in (VarStatSide.HIGH, VarStatSide.LOW):
                    self.nums[i] = sortednums[-self.k]
                elif self.side == VarStatSide.BOTH:
                    self.nums[i, :] = [sortednums[self.k - 1], sortednums[-self.k]]
                elif self.side == VarStatSide.ALL:
                    self.nums[i, :] = [sortednums[self.k - 1],
                                       sortednums[int(np.round(len(sortednums)/2)-1)],
                                       sortednums[-self.k]]
            if self.var.nummap is not None:
                self.vals = np.array([[self.var.nummap[x] for x in y] for y in self.nums])
            else:
                self.vals = copy(self.nums)

        else:
            # Suppress warning since this will become valid when Var is split
            # warn('VarStat only available for scalar or 1-D data')
            pass


    def genStatsOrderStatP(self) -> None:
        """Get the order statistic percentile value of the variable."""
        self.checkOrderStatsKWArgs()

        bound = self.bound
        if self.bound not in (StatBound.ONESIDED_UPPER, StatBound.ONESIDED_LOWER,
                              StatBound.TWOSIDED, StatBound.NEAREST, StatBound.ALL):
            raise ValueError(f'{self.bound} is not a valid bound for genStatsOrderStatP')
        elif self.bound in (StatBound.NEAREST, StatBound.ALL):
            bound = StatBound.TWOSIDED

        self.setName(f'{self.var.name} ' +
                     f'{self.bound} {self.c*100}% Confidence Bound around ' +
                     f'{self.p*100}th Percentile')

        self.k = order_stat_P_k(n=self.var.ncases, P=self.p, c=self.c, bound=bound)

        (iPl, iP, iPu) = get_iP(n=self.var.ncases, P=self.p)
        if self.var.isscalar:
            sortednums = sorted(self.var.nums)
            if self.bound == StatBound.ONESIDED_LOWER:
                self.nums = np.array(sortednums[iPl - self.k])
            elif self.bound == StatBound.ONESIDED_UPPER:
                self.nums = np.array(sortednums[iPu + self.k])
            elif self.bound == StatBound.NEAREST:
                self.nums = np.array(sortednums[iP])
            if self.bound in (StatBound.ONESIDED_LOWER,
                              StatBound.ONESIDED_UPPER,
                              StatBound.NEAREST):
                if self.var.nummap is not None:
                    self.vals = self.var.nummap[self.nums.item()]
            elif self.bound == StatBound.TWOSIDED:
                self.nums = np.array([sortednums[iPl - self.k], sortednums[iPu + self.k]])
                if self.var.nummap is not None:
                    self.vals = np.array([self.var.nummap[self.nums[0]],
                                          self.var.nummap[self.nums[1]]])
            elif self.bound == StatBound.ALL:
                self.nums = np.array([sortednums[iPl - self.k],
                                      sortednums[iP],
                                      sortednums[iPu + self.k]])
                if self.var.nummap is not None:
                    self.vals = np.array([self.var.nummap[self.nums[0]],
                                          self.var.nummap[self.nums[1]],
                                          self.var.nummap[self.nums[2]]])
            if self.var.nummap is None:
                self.vals = copy(self.nums)

        elif self.var.maxdim == 1:
            npoints = max(len(get_list(x)) for x in self.var.nums)
            self.nums = np.empty(npoints)
            if self.bound == StatBound.TWOSIDED:
                self.nums = np.empty((npoints, 2))
            elif self.bound == StatBound.ALL:
                self.nums = np.empty((npoints, 3))
            for i in range(npoints):
                numsatidx = [get_list(x)[i] for x in self.var.nums if len(get_list(x)) > i]
                sortednums = sorted(numsatidx)
                if self.bound == StatBound.ONESIDED_LOWER:
                    self.nums[i] = sortednums[iPl - self.k]
                elif self.bound == StatBound.ONESIDED_UPPER:
                    self.nums[i] = sortednums[iPu + self.k]
                elif self.bound == StatBound.NEAREST:
                    self.nums[i] = sortednums[iP]
                elif self.bound == StatBound.TWOSIDED:
                    self.nums[i, :] = [sortednums[iPl - self.k], sortednums[iPu + self.k]]
                elif self.bound == StatBound.ALL:
                    self.nums[i, :] = [sortednums[iPl - self.k],
                                       sortednums[iP],
                                       sortednums[iPu + self.k]]
            if self.var.nummap is not None:
                self.vals = np.array([[self.var.nummap[x] for x in y] for y in self.nums])
            else:
                self.vals = copy(self.nums)

        else:
            # Suppress warning since this will become valid when Var is split
            # warn('VarStat only available for scalar or 1-D data')
            pass


    def checkOrderStatsKWArgs(self) -> None:
        """Check the order statistic keyword arguments."""
        if 'p' not in self.statkwargs:
            raise ValueError(f'{self.stat} requires the kwarg ''p''')
        else:
            self.p = self.statkwargs['p']
        if 'c' not in self.statkwargs:
            self.c = 0.95
        else:
            self.c = self.statkwargs['c']
        if 'bound' not in self.statkwargs:
            self.bound = StatBound.TWOSIDED
        else:
            self.bound = self.statkwargs['bound']


    def setName(self,
                name : str,
                ) -> None:
        """
        Set the name for this variable statistic.

        Parameters
        ----------
        name : str
            The new name.
        """
        if self.name is None:
            self.name = name
