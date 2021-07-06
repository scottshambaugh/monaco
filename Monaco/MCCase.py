# MCCase.py

from Monaco.MCVar import MCVar
from Monaco.MCVal import MCOutVal
from typing import Union, Any
import numpy as np

class MCCase():
    def __init__(self, 
                 ncase     : int, 
                 isnom     : bool, 
                 mcinvars  : dict[str, MCVar], 
                 constvals : dict[str, Any] = dict(),
                 seed      : int = np.random.get_state()[1][0],
                 ):
        
        self.ncase = ncase
        self.isnom = isnom
        self.mcinvars = mcinvars 
        self.constvals = constvals
        self.mcoutvars = dict()
        self.seed = seed
        
        self.starttime = None
        self.endtime = None
        self.runtime = None
        
        self.filepath = None
        self.runsimid = None
        self.haspreprocessed = False
        self.hasrun = False
        self.haspostprocessed = False
        
        self.mcinvals = self.getMCInVals()
        self.mcoutvals = dict()
        
        self.siminput = None
        self.simrawoutput = None
        

    def getMCInVals(self):
        mcvals = dict()
        for mcvar in self.mcinvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals


    def getMCOutVals(self):
        mcvals = dict()
        for mcvar in self.mcoutvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals
    
    
    def addOutVal(self, 
                  name   : str, 
                  val, # unconstrained type
                  split  : bool = True, 
                  valmap : Union[None, dict[Any, int]] = None
                  ):
        self.mcoutvals[name] = MCOutVal(name=name, ncase=self.ncase, val=val, valmap=valmap, isnom=self.isnom)
        if split:
            self.mcoutvals.update(self.mcoutvals[name].split())

