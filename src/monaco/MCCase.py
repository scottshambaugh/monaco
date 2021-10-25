# MCCase.py

from monaco.MCVar import MCOutVar, MCInVar
from monaco.MCVal import MCOutVal, MCInVal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Any
import numpy as np

class MCCase():
    def __init__(self, 
                 ncase     : int, 
                 isnom     : bool, 
                 mcinvars  : dict[str, MCInVar], 
                 constvals : dict[str, Any] = None,
                 seed      : int = np.random.get_state(legacy=False)['state']['key'][0],
                 ):
        
        self.ncase = ncase
        self.isnom = isnom
        self.mcinvars = mcinvars
        if constvals is None:
            constvals = dict()
        self.constvals = constvals
        self.mcoutvars : dict[str, MCOutVar] = dict()
        self.seed = seed
        
        self.starttime : datetime = None
        self.endtime   : datetime = None
        self.runtime   : timedelta = None
        
        self.filepath : Path = None
        self.runsimid : int = None
        self.haspreprocessed  : bool = False
        self.hasrun           : bool = False
        self.haspostprocessed : bool = False
        
        self.mcinvals  : dict[str, MCInVal]  = self.getMCInVals()
        self.mcoutvals : dict[str, MCOutVal] = dict()
        
        self.siminput     : tuple[Any] = None
        self.simrawoutput : tuple[Any] = None
        

    def getMCInVals(self) -> dict[str, MCInVal]:
        mcvals = dict()
        for mcvar in self.mcinvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals


    def getMCOutVals(self) -> dict[str, MCOutVal]:
        mcvals = dict()
        for mcvar in self.mcoutvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals
    
    
    def addOutVal(self, 
                  name   : str, 
                  val, # unconstrained type
                  split  : bool = True, 
                  valmap : dict[Any, int] = None
                  ) -> None:
        self.mcoutvals[name] = MCOutVal(name=name, ncase=self.ncase, val=val, valmap=valmap, isnom=self.isnom)
        if split:
            self.mcoutvals.update(self.mcoutvals[name].split())
