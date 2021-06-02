import numpy as np
import pandas as pd
from itertools import chain
from copy import copy, deepcopy
from Monaco.helper_functions import is_num
from typing import Union, Any
from scipy.stats import rv_discrete, rv_continuous

### MCVal Base Class ###
class MCVal():
    def __init__(self, 
                 name  : str, 
                 ncase : int, 
                 isnom : bool,
                 ):
        self.name = name
        self.ncase = ncase
        self.isnom = isnom

        self.val = None
        self.valmap = None
        self.num = None
        self.nummap = None
        self.isscalar = None
        self.size = None



### MCInVal Class ###
class MCInVal(MCVal):
    def __init__(self, 
                 name   : str, 
                 ncase  : int, 
                 pct    : float,
                 num    : float,
                 dist   : Union[rv_discrete, rv_continuous], 
                 nummap : Union[None, dict[int, Any]] = None, 
                 isnom  : bool = False,
                 ):
        
        super().__init__(name=name, ncase=ncase, isnom=isnom)
        self.dist = dist
        self.pct = pct
        self.num = num        
        self.nummap = nummap
        self.isscalar = True
        self.size = (1, 1)
        
        self.mapNum()
        self.genValMap()


    def mapNum(self):
        if self.nummap is None:
            self.val = self.num
        elif self.isscalar:
            self.val = self.nummap[self.num]
        else:
            val = copy(self.num)
            if self.size[0] == 1:
                for i in range(self.num[1]):
                        val[i] = self.nummap[self.num[i]]
            else:
                for i in range(self.size[0]):
                    for j in range(self.size[1]):
                        val[i][j] = self.nummap[self.num[i][j]]
            self.val = val


    def genValMap(self):
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
                 valmap : Union[None, dict[Any, int]] = None, 
                 isnom  : bool = False,
                 ):
        super().__init__(name=name, ncase=ncase, isnom=isnom)
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


    def convertPandas(self):
        if isinstance(self.val, pd.Series) or isinstance(self.val, pd.Index):
            self.val = self.val.values


    def genSize(self):
        if isinstance(self.val,(list, tuple, np.ndarray)):
            self.isscalar = False
            if isinstance(self.val[0],(list, tuple, np.ndarray)):
                self.size = (len(self.val), len(self.val[0]))
            else:
                self.size = (1, len(self.val))
        else:
            self.isscalar = True
            self.size = (1, 1)


    def extractValMap(self):
        if self.isscalar:
            if isinstance(self.val, bool):
                self.valmap = {True:1, False:0}
            elif not is_num(self.val):
                self.valmap = {str(self.val):0}
        else:
            if self.size[0] > 1:
                if all(isinstance(x, bool) for x in chain(*self.val)):
                    self.valmap = {True:1, False:0}                  
                elif not all(is_num(x) for x in chain(*self.val)):
                    self.valmap = {str(key):idx for idx, key in enumerate(set(chain(*self.val)))}
            else:
                if not all(is_num(x) for x in self.val):
                    self.valmap = {True:1, False:0}
                elif not all(is_num(x) for x in self.val):
                    self.valmap = {str(key):idx for idx, key in enumerate(set(self.val))}

                
    def mapVal(self):
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


    def genNumMap(self):
        if self.valmap is None:
            self.nummap = None
        else:
            self.nummap = {num:val for val, num in self.valmap.items()}


    def split(self):
        mcvals = dict()
        if self.size[0] > 1:
            for i in range(self.size[0]):
                name = self.name + f' [{i}]'
                mcvals[name] = MCOutVal(name=name, ncase=self.ncase, val=self.val[i], valmap=self.valmap, isnom=self.isnom)
        return mcvals



'''
### Test ###
if __name__ == '__main__':
    from scipy.stats import norm
    a = MCInVal(name='TestA', ncase=1, pct=0.5, num=0, dist=norm, isnom=True)
    print(a.val) # expected: 0
    b = MCOutVal(name='TestB', ncase=1, val=[[0,0],[0,0],[0,0]], isnom=True)
    print(b.size) # expected: (3, 2)
    print(b.val) # expected: [[0, 0], [0, 0], [0, 0]]
    bsplit = b.split()
    print(bsplit['TestB [0]'].val) # expected: [0, 0]
    c = MCOutVal(name='TestC', ncase=1, val=[['a','a'],['b','b'],['a','b']], isnom=True)
    print(c.valmap) # expected: {'b': 0, 'a': 1}
    print(c.num) # expected: [[1, 1], [0, 0], [1, 0]]
    d = MCOutVal(name='TestD', ncase=1, val=[True, False], valmap={True:2, False:1})
    print(d.val) # expected: [True, False]
    print(d.num) # expected: [2, 1]
    e = MCOutVal(name='TestE', ncase=1, val=[True, False])
    print(e.val) # expected: [True, False]
    print(e.num) # expected: [1, 0]
    
    nvals = 3
    dates = pd.date_range(start='2020-01-01', periods=nvals, freq='YS')
    df = pd.DataFrame({'vals': range(nvals)}, index = dates)
    f = MCOutVal(name='TestF', ncase=1, val=df['vals'], isnom=True)
    print(f.num) # expected: [0 1 2]
    g = MCOutVal(name='TestG', ncase=1, val=df.index, isnom=True)
    print(g.val) # expected: ['2020-01-01T00:00:00.000000000' '2021-01-01T00:00:00.000000000' '2022-01-01T00:00:00.000000000']
#'''
