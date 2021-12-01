# MCVar.py

import numpy as np
from scipy.stats import rv_continuous, rv_discrete, describe
from scipy.stats.stats import DescribeResult
from monaco.MCVal import MCVal, MCInVal, MCOutVal
from monaco.MCVarStat import MCVarStat
from monaco.MCEnums import SampleMethod, VarStat
from monaco.mc_sampling import mc_sampling
from monaco.helper_functions import empty_list
from copy import copy
from typing import Union, Any
from warnings import warn
from abc import ABC, abstractmethod


### MCVar Base Class ###
class MCVar(ABC):
    """
    Abstract base class to hold the data for a Monte-Carlo variable. 

    Parameters
    ----------
    name : str
        The name of this value.
    ndraws : int
        The number of random draws.
    firstcaseismedian : bool
        Whether the first case represents the median case.
    """
    def __init__(self,
                 name              : str, 
                 ndraws            : int, 
                 firstcaseismedian : bool,
                 ):
        self.name = name 
        self.ndraws = ndraws 
        self.firstcaseismedian = firstcaseismedian
        
        self.ncases = ndraws + 1
        self.setFirstCaseMedian(firstcaseismedian)
        self.vals       : list[Any]
        self.valmap     : dict
        self.nums       : list[float]
        self.nummap     : dict
        self.pcts       : list[float]
        self.size       : tuple
        self.isscalar   : bool
        self.mcvarstats : list[MCVarStat] = empty_list()
        

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


    def stats(self) -> DescribeResult:
        """
        Calculates statistics of the variable nums from scipy.stats.describe.

        Returns
        -------
        stats : DescribeResult
            A dict with descriptive statistics for the variable nums.
        """
        stats = describe(self.nums)
        return stats
    
    
    def addVarStat(self,
                   stattype   : VarStat, 
                   statkwargs : dict[str, Any] = None, 
                   name       : str = None,
                   ) -> None:
        """
        Add a variable statistic to this variable.

        Parameters
        ----------
        stattype : monaco.MCEnums.VarStat
            The type of variable statistic to add.
        statkwargs : dict[str, Any]
            Keyword arguments for the specified variable stastistic.
        name : str
            The name of the variable statistic to add.
        """
        if statkwargs is None:
            statkwargs = dict()
        self.mcvarstats.append(MCVarStat(mcvar=self, stattype=stattype, statkwargs=statkwargs, name=name))


    def clearVarStats(self) -> None:
        """
        Remove all the variable statistics for this variable.
        """
        self.mcvarstats = []


    @abstractmethod
    def getVal(self,
               ncase : int,
               ) -> MCVal:
        pass



### MCInVar Class ###
class MCInVar(MCVar):
    """
    A Monte-Carlo input variable. 

    Parameters
    ----------
    name : str
        The name of this variable.
    ndraws : int
        The number of random draws.
    dist : {scipy.stats.rv_discrete, scipy.stats.rv_continuous}
        The statistical distribution to draw from.
    distkwargs : dict
        The keyword argument pairs for the statistical distribution function.
    nummap : dict
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    samplemethod : monaco.MCEnums.SampleMethod, default: 'sobol_random'
        The random sampling method to use.
    ninvar : int
        The number of the input variable this is.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random seed for drawing this variable.
    firstcaseismedian : bool, default: False
        Whether the first case represents the median case.
    autodraw : bool, default: True
        Whether to draw the random values when this variable is created.

    Attributes
    ----------
    isscalar : bool
        Whether this is a scalar variable. Alway True for an input variable.
    size : tuple[int]
        The size of the values. Always (1,1) for an input variable.
    valmap : dict
        A dictionary mapping nonnumeric values to numbers (the inverse of
        `nummap`).
    pcts : list[float]
        The randomly drawn percentiles.
    nums : list[float]
        The randomly drawn numbers obtained by feeding `pcts` into `dist`.
    vals : list[Any]
        The values corresponding to the randomly drawn numbers. If valmap is
        None, then `vals == nums`
    mcvarstats : list[moncao.MCVarStat.MCVarStat]
        A list of all the variable statistics for this variable.
    """
    def __init__(self,
                 name              : str, 
                 ndraws            : int, 
                 dist              : Union[rv_discrete, rv_continuous], 
                 distkwargs        : dict         = None,
                 nummap            : dict         = None,
                 samplemethod      : SampleMethod = SampleMethod.SOBOL_RANDOM,
                 ninvar            : int          = None,
                 seed              : int          = np.random.get_state(legacy=False)['state']['key'][0], 
                 firstcaseismedian : bool         = False,
                 autodraw          : bool         = True,
                 ):
        super().__init__(name=name, ndraws=ndraws, firstcaseismedian=firstcaseismedian)
        
        self.dist = dist
        if distkwargs is None:
            distkwargs = dict()
        self.distkwargs = distkwargs
        self.samplemethod = samplemethod
        self.ninvar = ninvar 
        self.seed = seed 
        self.nummap = nummap 
        
        self.isscalar = True
        self.size = (1, 1)
        
        self.genValMap()
        if autodraw:
            self.draw(ninvar_max=None)


    def mapNums(self) -> None:
        """
        Generate `vals` based on the drawn numbers and the nummap.
        """
        self.vals = copy(self.nums)
        if self.nummap is not None:
            for i in range(self.ncases):
                self.vals[i] = self.nummap[self.nums[i]]


    def genValMap(self) -> None:
        """
        Generate `valmap` by inverting `nummap`.
        """
        if self.nummap is None:
            self.valmap = None
        else:
            self.valmap = {val:num for num, val in self.nummap.items()}


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

        Parameters
        ----------
        ninvar_max : int
            The total number of input variables for the simulation.
        """
        self.pcts = []
        self.nums = []
        dist = self.dist(**self.distkwargs)

        if self.firstcaseismedian:
            self.ncases = self.ndraws + 1
            self.pcts.append(0.5)
            self.nums.append(self.getDistMedian())
            
        pcts = mc_sampling(ndraws=self.ndraws, method=self.samplemethod, ninvar=self.ninvar, ninvar_max=ninvar_max, seed=self.seed)
        self.pcts.extend(pcts)
        self.nums.extend(dist.ppf(pcts).tolist())
        
        if any(np.isinf(num) for num in self.nums):
            warn(f'Infinite value drawn. Check distribution and parameters: {self.dist=}, {self.distkwargs=}')
            if self.samplemethod in (SampleMethod.SOBOL, SampleMethod.HALTON):
                warn(f"Infinite value draw may happen with {self.dist=} for the first point of the {self.samplemethod} sampling method. Consider using {SampleMethod.SOBOL_RANDOM} instead.")

        if any(np.isnan(num) for num in self.nums):
            raise ValueError(f'Invalid draw. Check distribution and parameters: {self.dist=}, {self.distkwargs=}')

        self.mapNums()


    def getVal(self,
               ncase : int,
               ) -> MCInVal:
        """
        Get the input value for a specific case.

        Parameters
        ----------
        ncase : int
            The number of the case to get the value for.
        
        Returns
        -------
        val : monaco.MCVal.MCInVal
            The input value for that case.
        """
        ismedian = False
        if (ncase == 0) and self.firstcaseismedian:
            ismedian = True

        val = MCInVal(name=self.name, ncase=ncase, pct=self.pcts[ncase], num=self.nums[ncase], dist=self.dist, nummap=self.nummap, ismedian=ismedian)
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

        # For a discrete distribution, we take the nearest discrete value closest to the expected value
        elif isinstance(self.dist, rv_discrete):
            eps = np.finfo(float).eps
            p = dist.cdf(ev)
            ev_candidates = dist.ppf([p - eps, p, p + eps])
            ev_candidates_dist = abs(ev_candidates - ev)
            ev_closest = ev_candidates[np.nanargmin(ev_candidates_dist)]
            return ev_closest
        
        else:
            return None



### MCOutVar Class ###
class MCOutVar(MCVar):
    """
    A Monte-Carlo output variable. 

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
    firstcaseismedian : bool, default: False
        Whether the first case represents the median case.

    Attributes
    ----------
    isscalar : bool
        Whether this is a scalar variable. Alway True for an input variable.
    size : tuple[int]
        The size of the values. Always (1,1) for an input variable.
    nummap : dict
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    nums : list[float]
        The numbers corresponding to the output values. If valmap is None, then
        `nums == vals`.
    mcvarstats : list[moncao.MCVarStat.MCVarStat]
        A list of all the variable statistics for this variable.
    """
    def __init__(self,
                 name              : str, 
                 vals              : list[Any], 
                 valmap            : dict = None, 
                 ndraws            : int  = None, 
                 firstcaseismedian : bool = False,
                 ):
        if ndraws is None:
            ndraws = len(vals)
            if firstcaseismedian:
                ndraws = ndraws - 1
        
        super().__init__(name=name, ndraws=ndraws, firstcaseismedian=firstcaseismedian)
        self.vals = vals
        self.valmap = valmap
        if valmap is None:
            self.extractValMap()
        self.genSize()
        self.genNumMap()
        self.mapVals()


    def genSize(self) -> None:
        """
        Parse the output value to determine the size of each value and whether
        it is scalar.
        """
        if isinstance(self.vals[0],(list, tuple, np.ndarray)):
            self.isscalar = False
            if isinstance(self.vals[0][0],(list, tuple, np.ndarray)):
                self.size = (len(self.vals[0]), len(self.vals[0][0]))
            else:
                self.size = (1, len(self.vals[0]))
        else:
            self.isscalar = True
            self.size = (1, 1)
            
    
    def extractValMap(self) -> None:
        """
        Parse the output values and extract a valmap.
        """
        Val0 = self.getVal(0)
        if Val0.valmap is not None:
            if Val0.valmapsource == 'auto':
                uniquevals : set[Any] = set()
                for i in range(self.ncases):
                    uniquevals.update(self.getVal(i).valmap.keys())
                self.valmap = dict()
                for i, val in enumerate(sorted(uniquevals)):
                    self.valmap[val] = i
            else:
                self.valmap = Val0.valmap


    def genNumMap(self) -> None:
        """
        Generate the nummap by inverting `valmap`.
        """
        if self.valmap is None:
            self.nummap = None
        else:
            self.nummap = {num:val for val, num in self.valmap.items()}


    def mapVals(self) -> None:
        """
        Generate `nums` by mapping the values with the valmap.
        """
        self.nums = copy(self.vals)
        for i in range(self.ncases):
            self.nums[i] = self.getVal(i).num  


    def getVal(self, ncase : int) -> MCOutVal:
        """
        Get the variable value for a specific case.

        Parameters
        ----------
        ncase : int
            The number of the case to get the value for.
        
        Returns
        -------
        val : monaco.MCVal.MCOutVal
            The output value.
        """
        ismedian = False
        if (ncase == 0) and self.firstcaseismedian:
            ismedian = True
            
        val = MCOutVal(name=self.name, ncase=ncase, val=self.vals[ncase], valmap=self.valmap, ismedian=ismedian)
        return val
        
    
    def getMedianVal(self) -> MCOutVal:
        """
        Get the median value for this output variable if `firstcaseismedian`.

        Returns
        -------
        val : monaco.MCVal.MCOutVal
            The median output value.
        """
        val = None
        if self.firstcaseismedian:
            val = MCOutVal(name=self.name, ncase=0, val=self.vals[0], valmap=self.valmap, ismedian=True)
        return val
    
    
    def split(self) -> dict[str, 'MCOutVar']:  # Quotes in typing to avoid import error
        """
        Split a multidimentional output variable along its outermost dimension,
        and generate individual MCOutVar objects for each index.
        
        Returns
        -------
        mcvars : dict[str : monaco.MCVar.MCOutVar]
        """
        mcvars = dict()
        if self.size[0] > 1:
            for i in range(self.size[0]):
                name = self.name + f' [{i}]'
                vals = []
                for j in range(self.ncases):
                    vals.append(self.vals[j][i])
                mcvars[name] = MCOutVar(name=name, vals=vals, ndraws=self.ndraws, \
                                        valmap=self.valmap, firstcaseismedian=self.firstcaseismedian)
                for mcvarstat in self.mcvarstats:
                    mcvars[name].addVarStat(stattype=mcvarstat.stattype, statkwargs=mcvarstat.statkwargs, name=mcvarstat.name)
        return mcvars
