# MCVal.py

import numpy as np
import pandas as pd
from itertools import chain
from copy import copy, deepcopy
from monaco.helper_functions import is_num
from typing import Union, Any
from scipy.stats import rv_discrete, rv_continuous
from abc import ABC

### MCVal Base Class ###
class MCVal(ABC):
    def __init__(self, 
                 name  : str, 
                 ncase : int, 
                 ismean : bool,
                 ):
        self.name = name
        self.ncase = ncase
        self.ismean = ismean

        self.val      : Any
        self.valmap   : dict
        self.num      : float
        self.nummap   : dict
        self.isscalar : bool
        self.size     : tuple



### MCInVal Class ###
class MCInVal(MCVal):
    def __init__(self, 
                 name   : str, 
                 ncase  : int, 
                 pct    : float,
                 num    : float,
                 dist   : Union[rv_discrete, rv_continuous], 
                 nummap : dict = None, 
                 ismean : bool = False,
                 ):
        
        super().__init__(name=name, ncase=ncase, ismean=ismean)
        self.dist = dist
        self.pct = pct
        self.num = num
        self.nummap = nummap
        self.isscalar = True
        self.size = (1, 1)
        
        self.mapNum()
        self.genValMap()


    def mapNum(self) -> None:
        if self.nummap is None:
            self.val = self.num
        else:
            self.val = self.nummap[self.num]


    def genValMap(self) -> None:
        if self.nummap is None:
            self.valmap = None
        else:
            self.valmap = {val:num for num, val in self.nummap.items()}



### MCOutVal Class ###
class MCOutVal(MCVal):
    def __init__(self, 
                 name   : str, 
                 ncase  : int, 
                 val    : Any, 
                 valmap : dict = None, 
                 ismean : bool = False,
                 ):
        super().__init__(name=name, ncase=ncase, ismean=ismean)
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
        if isinstance(self.val, pd.Series) or isinstance(self.val, pd.Index):
            self.val = self.val.values


    def genSize(self) -> None:
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
        if self.valmap is None:
            self.nummap = None
        else:
            self.nummap = {num:val for val, num in self.valmap.items()}


    def split(self) -> dict[str, 'MCOutVal']:
        mcvals = dict()
        if self.size[0] > 1:
            for i in range(self.size[0]):
                name = self.name + f' [{i}]'
                mcvals[name] = MCOutVal(name=name, ncase=self.ncase, val=self.val[i], valmap=self.valmap, ismean=self.ismean)
        return mcvals
