# MCCase.py

from monaco.MCVar import MCOutVar, MCInVar
from monaco.MCVal import MCOutVal, MCInVal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Any
import numpy as np

class MCCase():
    """
    Object to hold all the data for a single Monte-Carlo case. 

    Init Parameters
    ---------------
    ncase : int
        The number of this case.
    ismedian : bool
        Whether this represents the median case.
    mcinvars : dict[str, monaco.MCVar.MCInVar]
        A dict pointing to all of the input variables.
    constvals : dict[str, Any]
        A dict of any constant values common to all cases.
    seed : int
        The random seed to pass to the run function for this case. Not used in
        as part of any Monte-Carlo sampling.
    
    Other Parameters
    ----------------
    starttime : datetime.datetime
        Timestamp for when this case started running.
    endtime : datetime.datetime
        Timestamp for when this case stopped running.
    runtime : datetime.timedelta
        Total run duration for this case.
    filepath : pathlib.Path
        The filepath for the case data, if saved to disk.
    runsimid : int
        The id for the particular sim run this case was run for.
    haspreprocessed : bool
        Whether this case has been preprocessed.
    hasrun : bool
        Whether this case has run the run function.
    haspostprocessed : bool
        Whether this case has been postprocessed.
    mcinvals : dict[str, monaco.MCVal.MCInVal]
        The input values for this partitcular case.
    mcoutvals : dict[str, monaco.MCVal.MCOutVal]
        The output values for this partitcular case.
    siminput : tuple[Any]
        The preprocessed inputs provided to the run function for this case.
    simrawoutput : tuple[Any]
        The non-postprocessed outputs from the run function for this case.
    """
    def __init__(self, 
                 ncase     : int, 
                 ismedian  : bool, 
                 mcinvars  : dict[str, MCInVar], 
                 constvals : dict[str, Any] = None,
                 seed      : int = np.random.get_state(legacy=False)['state']['key'][0],
                 ):
        
        self.ncase = ncase
        self.ismedian = ismedian
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
        """
        From all the MCInVar's, extract the vals for this case.

        Returns
        -------
        mcvals: dict[str, monaco.MCVal.MCInVal]
            The MCInVal's for this case.
        """
        mcvals = dict()
        for mcvar in self.mcinvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals


    def getMCOutVals(self) -> dict[str, MCOutVal]:
        """
        From all the MCOutVar's, extract the vals for this case.

        Returns
        -------
        mcvals: dict[str, monaco.MCVal.MCOutVal]
            The MCOutVal's for this case.
        """
        mcvals = dict()
        for mcvar in self.mcoutvars.values():
            mcval = mcvar.getVal(self.ncase)
            mcvals[mcval.name] = mcval
        return mcvals
    
    
    def addOutVal(self, 
                  name   : str, 
                  val    : Any,
                  split  : bool = True, 
                  valmap : dict[Any, int] = None
                  ) -> None:
        """
        Generate an MCOutVal and add it to the dict of mcoutvals.

        Parameters
        ----------
        name : str
            The name of the output value.
        val : Any
            The output value.
        split : bool
            Whether to split the value into its components.
        valmap : dict[Any, int]
            A valmap dict mapping nonnumeric values to numbers.
        """

        self.mcoutvals[name] = MCOutVal(name=name, ncase=self.ncase, val=val, valmap=valmap, ismedian=self.ismedian)
        if split:
            self.mcoutvals.update(self.mcoutvals[name].split())
