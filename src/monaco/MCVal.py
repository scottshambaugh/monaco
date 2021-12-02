# MCVal.py

import numpy as np
from itertools import chain
from copy import copy, deepcopy
from monaco.helper_functions import is_num
from typing import Union, Any
from scipy.stats import rv_discrete, rv_continuous
from abc import ABC
try:
    import pandas as pd
except ImportError:
    pd = None 

### MCVal Base Class ###
class MCVal(ABC):
    """
    Abstract base class to hold the data for a Monte-Carlo value. 

    Parameters
    ----------
    name : str
        The name of this value.
    ncase : int
        The number of the case for this value.
    ismedian : bool
        Whether this case represents the median case.
    """
    def __init__(self,
                 name     : str, 
                 ncase    : int, 
                 ismedian : bool,
                 ):
        self.name = name
        self.ncase = ncase
        self.ismedian = ismedian

        self.val      : Any
        self.valmap   : dict
        self.num      : float
        self.nummap   : dict
        self.isscalar : bool
        self.size     : tuple



### MCInVal Class ###
class MCInVal(MCVal):
    """
    A Monte-Carlo input value. 

    Parameters
    ----------
    name : str
        The name of this value.
    ncase : int
        The number of the case for this value.
    pct : float
        The percentile of the value draw.
    num : float
        The number corresponding to the statistical percentile draw.
    dist : Union[scipy.stats.rv_discrete, scipy.stats.rv_continuous]
        The statistical distribution that `num` was drawn from.
    nummap : dict, default: None
        A dictionary mapping numbers to nonnumeric values.
    ismedian : bool, default: False
        Whether this case represents the median case,
    
    Attributes
    ----------
    val : Any
        The value corresponding to the drawn number. If `nummap` is None, then
        this is equal to `num`.
    isscalar : bool
        Whether the value is scalar.
    size : tuple[int]
        The size of the value.
    valmap : dict
        A dictionary mapping nonnumeric values to numbers (the inverse of
        `nummap`).
    """
    def __init__(self,
                 name     : str, 
                 ncase    : int, 
                 pct      : float,
                 num      : float,
                 dist     : Union[rv_discrete, rv_continuous], 
                 nummap   : dict = None, 
                 ismedian : bool = False,
                 ):
        
        super().__init__(name=name, ncase=ncase, ismedian=ismedian)
        self.dist = dist
        self.pct = pct
        self.num = num
        self.nummap = nummap
        self.isscalar = True
        self.size = (1, 1)
        
        self.mapNum()
        self.genValMap()


    def mapNum(self) -> None:
        """
        Generate `val` based on the drawn number and the nummap.
        """
        if self.nummap is None:
            self.val = self.num
        else:
            self.val = self.nummap[self.num]


    def genValMap(self) -> None:
        """
        Generate the valmap based on the nummap.
        """
        if self.nummap is None:
            self.valmap = None
        else:
            self.valmap = {val:num for num, val in self.nummap.items()}



### MCOutVal Class ###
class MCOutVal(MCVal):
    """
    A Monte-Carlo output value. 

    Parameters
    ----------
    name : str
        The name of this value.
    ncase : int
        The number of the case for this value.
    val : float
        The output value.
    valmap : dict, default: None
        A dictionary mapping nonnumeric values to numbers.
    ismedian : bool, default: False
        Whether this case represents the median case,
    
    Attributes
    ----------
    num : Any
        A number corresponding to the output value. If `valmap` is None and val
        is nonnumeric, then will be an integer as assigned by extractValMap().
    valmapsource : str
        Either 'assigned' or 'auto' based on whether a valmap was passed in.
    isscalar : bool
        Whether the value is scalar.
    size : tuple[int]
        The size of the value.
    nummap : dict
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    """
    def __init__(self,
                 name     : str, 
                 ncase    : int, 
                 val      : Any, 
                 valmap   : dict = None, 
                 ismedian : bool = False,
                 ):
        super().__init__(name=name, ncase=ncase, ismedian=ismedian)
        self.val = val
        self.valmap = valmap
        self.convertPandas()
        
        self.genSize()
        if valmap is None:
            self.valmapsource = 'auto'
            self.extractValMap()    
        else:
            self.valmapsource = 'assigned'
        self.mapVal()
        self.genNumMap()


    def convertPandas(self) -> None:
        """
        If the output value is a pandas dataseries or index, convert it to a
        format we understand.
        """
        if pd:
            if isinstance(self.val, pd.Series) or isinstance(self.val, pd.Index):
                self.val = self.val.values


    def genSize(self) -> None:
        """
        Calculate the size of the output value, and whether it is a scalar.
        """
        if isinstance(self.val,(list, tuple, np.ndarray)):
            self.isscalar = False
            if isinstance(self.val[0],(list, tuple, np.ndarray)):
                self.size = (len(self.val), len(self.val[0]))
            else:
                self.size = (1, len(self.val))
        else:
            self.isscalar = True
            self.size = (1, 1)


    def extractValMap(self) -> None:
        """
        Parse the output value and extract a valmap.
        """
        if self.isscalar:
            if isinstance(self.val, bool):
                self.valmap = {True:1, False:0}
            elif not is_num(self.val):
                self.valmap = {str(self.val):0}
        else:
            if self.size[0] == 1:
                if not all(is_num(x) for x in self.val):
                    self.valmap = {True:1, False:0}
                elif not all(is_num(x) for x in self.val):
                    self.valmap = {str(key):idx for idx, key in enumerate(sorted(set(self.val)))}
            else:
                if all(isinstance(x, bool) for x in chain(*self.val)):
                    self.valmap = {True:1, False:0}                  
                elif not all(is_num(x) for x in chain(*self.val)):
                    self.valmap = {str(key):idx for idx, key in enumerate(sorted(set(chain(*self.val))))}

                
    def mapVal(self) -> None:
        """
        Map the output value to a number or array of numbers.
        """
        if self.valmap is None:
            self.num = self.val
        elif self.isscalar:
            self.num = self.valmap[self.val]
        else:
            num = deepcopy(self.val)
            if self.size[0] == 1:
                for i in range(self.size[1]):
                    num[i] = self.valmap[self.val[i]]
            else:
                for i in range(self.size[0]):
                    for j in range(self.size[1]):
                        num[i][j] = self.valmap[self.val[i][j]]
            self.num = num


    def genNumMap(self) -> None:
        """
        Invert the valmap to get a nummap.
        """
        if self.valmap is None:
            self.nummap = None
        else:
            self.nummap = {num:val for val, num in self.valmap.items()}


    def split(self) -> dict[str, 'MCOutVal']:  # Quotes in typing to avoid import error
        """
        Split a multidimentional output value along its outermost dimension,
        and generate individual MCOutVal objects for each index.

        Returns
        -------
        mcvals : dict[str : monaco.MCVal.MCOutVal]
        """
        mcvals = dict()
        if self.size[0] > 1:
            for i in range(self.size[0]):
                name = self.name + f' [{i}]'
                mcvals[name] = MCOutVal(name=name, ncase=self.ncase, val=self.val[i], valmap=self.valmap, ismedian=self.ismedian)
        return mcvals
