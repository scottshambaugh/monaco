# mc_val.py
from __future__ import annotations

import numpy as np
from typing import Any
from scipy.stats import rv_discrete, rv_continuous
from abc import ABC
from monaco.helper_functions import is_num, hashable_val, flatten, hashable_val_vectorized

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


### Val Base Class ###
class Val(ABC):
    """
    Abstract base class to hold the data for a Monte Carlo value.

    Parameters
    ----------
    name : str
        The name of this value.
    ncase : int
        The number of the case for this value.
    ismedian : bool
        Whether this case represents the median case.
    datasource : str | None
        If the value was imported from a file, this is the filepath. If
        generated through monaco, then None.
    """
    def __init__(self,
                 name       : str,
                 ncase      : int,
                 ismedian   : bool,
                 datasource : str | None = None,
                 ):
        self.name = name
        self.ncase = ncase
        self.ismedian = ismedian
        self.datasource = datasource

        self.val      : Any
        self.valmap   : dict[Any, float]
        self.num      : np.float64 | np.ndarray
        self.nummap   : dict[float, Any]
        self.isscalar : bool
        self.shape    : tuple[int, ...]



### InVal Class ###
class InVal(Val):
    """
    A Monte Carlo input value.

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
    dist : scipy.stats.rv_discrete | scipy.stats.rv_continuous
        The statistical distribution that `num` was drawn from.
    nummap : dict[float, Any], default: None
        A dictionary mapping numbers to nonnumeric values.
    valmap : dict[Any, float], default: None
        A dictionary mapping nonnumeric values to numbers. If None, then
        will be generated from the nummap. Note that there is no checking
        that the valmap is consistent with the nummap.
    ismedian : bool, default: False
        Whether this case represents the median case.
    datasource : str | None, default: None
        If the value was imported from a file, this is the filepath. If
        generated through monaco, then None.

    Attributes
    ----------
    val : Any
        The value corresponding to the drawn number. If `nummap` is None, then
        this is equal to `num`.
    isscalar : bool
        Whether the value is scalar.
    shape : tuple[int]
        The shape of the value.
    valmap : dict[Any, float]
        A dictionary mapping nonnumeric values to numbers (the inverse of
        `nummap`).
    """
    __slots__ = ('name', 'ncase', 'ismedian', 'datasource', 'dist', 'pct', 'num',
                 'nummap', 'isscalar', 'shape', 'val', 'valmap')

    def __init__(self,
                 name       : str,
                 ncase      : int,
                 pct        : float,
                 num        : float,
                 dist       : rv_discrete | rv_continuous,
                 nummap     : dict[float, Any] | None = None,
                 valmap     : dict[Any, float] | None = None,
                 ismedian   : bool = False,
                 datasource : str | None = None,
                 ):

        super().__init__(name=name, ncase=ncase, ismedian=ismedian,
                         datasource=datasource)
        self.dist = dist
        self.pct = pct
        self.num = num
        self.nummap = nummap
        self.isscalar = True
        self.shape = ()

        self.mapNum()
        if valmap is None:
            self.genValMap()
        else:
            self.valmap = valmap


    def __repr__(self):
        if self.isscalar:
            return (f"{self.__class__.__name__}('{self.name}', ncase={self.ncase}, " +
                    f"val={self.val} ({self.num}), pct={self.pct:0.4f})")
        else:
            return (f"{self.__class__.__name__}('{self.name}', ncase={self.ncase}, " +
                    f"shape={self.shape}, val={self.val[0]}..{self.val[-1]} " +
                    f"({self.num[0]}..{self.num[-1]}), pct={self.pct:0.4f})")


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
            self.valmap = {hashable_val(val): num for num, val in self.nummap.items()}



### OutVal Class ###
class OutVal(Val):
    """
    A Monte Carlo output value.

    Parameters
    ----------
    name : str
        The name of this value.
    ncase : int
        The number of the case for this value.
    val : float
        The output value.
    valmap : dict[Any, float], default: None
        A dictionary mapping nonnumeric values to numbers.
    ismedian : bool, default: False
        Whether this case represents the median case,
    datasource : str | None, default: None
        If the value was imported from a file, this is the filepath. If
        generated through monaco, then None.

    Attributes
    ----------
    num : np.array
        A number corresponding to the output value. If `valmap` is None and val
        is nonnumeric, then will be an integer as assigned by extractValMap().
    valmapsource : str
        Either 'assigned' or 'auto' based on whether a valmap was passed in.
    isscalar : bool
        Whether the value is scalar.
    shape : tuple[int]
        The shape of the value.
    nummap : dict[float, Any]
        A dictionary mapping numbers to nonnumeric values (the inverse of
        `valmap`).
    """
    __slots__ = ('name', 'ncase', 'ismedian', 'datasource', 'val', 'valmap',
                 'num', 'nummap', 'isscalar', 'shape')

    def __init__(self,
                 name       : str,
                 ncase      : int,
                 val        : Any,
                 valmap     : dict[Any, float] | None = None,
                 ismedian   : bool = False,
                 datasource : str | None = None,
                 ):
        super().__init__(name=name, ncase=ncase, ismedian=ismedian,
                         datasource=datasource)
        self.val = val
        self.valmap = valmap
        self.convertPandas()

        self.genShape()
        if valmap is None:
            self.valmapsource = 'auto'
            self.extractValMap()
        else:
            self.valmapsource = 'assigned'
        self.mapVal()
        self.genNumMap()


    def __repr__(self):
        if self.isscalar:
            return (f"{self.__class__.__name__}('{self.name}', ncase={self.ncase}, " +
                    f"val={self.val} ({self.num}))")
        else:
            return (f"{self.__class__.__name__}('{self.name}', ncase={self.ncase}, " +
                    f"shape={self.shape}, val={self.val[0]}..{self.val[-1]} " +
                    f"({self.num[0]}..{self.num[-1]}))")


    def convertPandas(self) -> None:
        """
        If the output value is a pandas dataseries or index, convert it to a
        format we understand.
        """
        if pd:
            if isinstance(self.val, pd.Series) or isinstance(self.val, pd.Index):
                self.val = self.val.values


    def genShape(self) -> None:
        """
        Calculate the shape of the output value, and whether it is a scalar.
        """
        vals_array = np.asanyarray(self.val)
        self.shape = vals_array.shape

        self.isscalar = False
        if self.shape == ():
            self.isscalar = True


    def extractValMap(self) -> None:
        """
        Parse the output value and extract a valmap.
        """
        if self.isscalar:
            if isinstance(self.val, (bool, np.bool_)):
                self.valmap = {True: 1, False: 0}
            elif not is_num(self.val):
                self.valmap = {hashable_val(self.val): 0}
        else:
            try:
                vals_flattened = np.asanyarray(self.val).flatten()
            except ValueError:
                vals_flattened = flatten([self.val])
            if all(isinstance(x, (bool, np.bool_)) for x in vals_flattened):
                self.valmap = {True: 1, False: 0}
            elif any(not is_num(x) for x in vals_flattened):
                hashable_vals = hashable_val_vectorized(vals_flattened)
                sorted_vals = sorted(set(hashable_vals))
                self.valmap = {val: idx for idx, val in enumerate(sorted_vals)}


    def mapVal(self) -> None:
        """
        Map the output value to a number or array of numbers.
        """
        if self.valmap is None:
            self.num = self.val
        elif self.isscalar:
            self.num = self.valmap[hashable_val(self.val)]
        else:
            num = np.asarray(self.val, dtype='object')
            if len(self.shape) == 1:
                for i in range(self.shape[0]):
                    num[i] = self.valmap[hashable_val(self.val[i])]
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        num[i][j] = self.valmap[hashable_val(self.val[i][j])]
            self.num = np.asarray(num, dtype='float')


    def genNumMap(self) -> None:
        """
        Invert the valmap to get a nummap.
        """
        if self.valmap is None:
            self.nummap = None
        else:
            self.nummap = {hashable_val(num): val for val, num in self.valmap.items()}


    def split(self) -> dict[str, 'OutVal']:  # Quotes in typing to avoid import error
        """
        Split a output value along its outermost dimension,
        and generate individual OutVal objects for each index.

        Returns
        -------
        vals : dict[str : monaco.mc_val.OutVal]
        """
        vals = dict()
        if len(self.shape) > 1:
            for i in range(self.shape[0]):
                name = self.name + f' [{i}]'
                if HAS_PANDAS and isinstance(self.val, (pd.Series, pd.Index)):
                    val = self.val.iloc[i]
                else:
                    val = self.val[i]
                vals[name] = OutVal(name=name, ncase=self.ncase, val=val,
                                    valmap=self.valmap, ismedian=self.ismedian)
        return vals
