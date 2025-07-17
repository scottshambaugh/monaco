# mc_case.py
from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from monaco.mc_var import OutVar, InVar
from monaco.mc_val import OutVal, InVal
from monaco.globals import get_global_vars


class Case():
    """
    Object to hold all the data for a single Monte Carlo case.

    `InVal`s and `OutVal`s can be accessed by name, eg `case['Var1']`.

    Parameters
    ----------
    ncase : int
        The number of this case.
    ismedian : bool
        Whether this represents the median case.
    invars : dict[str, monaco.mc_var.InVar]
        A dict pointing to all of the input variables.
    constvals : dict[str, Any], default: None
        A dict of any constant values common to all cases.
    keepsiminput : bool, default: True
        Whether to keep the siminput after running.
    keepsimrawoutput : bool, default: True
        Whether to keep the simrawoutput after postprocessing.
    seed : int, default: np.random.get_state(legacy=False)['state']['key'][0]
        The random seed to pass to the run function for this case. Not used in
        as part of any Monte Carlo sampling.

    Attributes
    ----------
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
    constvals : dict[str, Any]
        The constant values for this case.
    invals : dict[str, monaco.mc_val.InVal]
        The input values for this particular case.
    outvals : dict[str, monaco.mc_val.OutVal]
        The output values for this particular case.
    siminput : tuple[Any]
        The preprocessed inputs provided to the run function for this case.
    simrawoutput : tuple[Any]
        The non-postprocessed outputs from the run function for this case.
    """
    def __init__(self,
                 ncase            : int,
                 ismedian         : bool,
                 invars           : dict[str, InVar],
                 constvals        : dict[str, Any] | None = None,
                 keepsiminput     : bool = True,
                 keepsimrawoutput : bool = True,
                 seed             : int = np.random.get_state(legacy=False)['state']['key'][0],
                 ):

        self.ncase = ncase
        self.ismedian = ismedian
        self.invars = invars
        self.vars : dict[str, InVar | OutVar] = dict(invars)
        if constvals is None:
            constvals = dict()
        self.constvals = constvals
        self.outvars : dict[str, OutVar] = dict()
        self.keepsiminput = keepsiminput
        self.keepsimrawoutput = keepsimrawoutput
        self.seed = seed

        self.starttime : datetime | None = None
        self.endtime   : datetime | None = None
        self.runtime   : timedelta | None = None

        self.filepath : Path | None = None
        self.runsimid : int | None = None
        self.haspreprocessed  : bool = False
        self.hasrun           : bool = False
        self.haspostprocessed : bool = False

        self.invals  : dict[str, InVal]  = self.getInVals()
        self.outvals : dict[str, OutVal] = dict()
        self.vals    : dict[str, InVal | OutVal] = dict(self.invals)

        self.siminput     : tuple[Any] | None = None
        self.simrawoutput : tuple[Any] | None = None


    def __getstate__(self):
        # We delete the large objects to save time multiprocessing
        state = self.__dict__.copy()
        for k in ('invars', 'outvars', 'vars', 'invals', 'vals', 'constvals'):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        invars, outvars, constvals = get_global_vars()
        self.invars  = invars
        self.outvars = outvars
        self.constvals = constvals
        self.vars    = {**invars, **outvars}
        self.invals  = self.getInVals()
        self.vals    = {**self.invals, **self.outvals}


    def __repr__(self):
        return (f"{self.__class__.__name__}(ncase={self.ncase},\n  invals={self.invals} " +
                f"\n  outvals={self.outvals})")


    def __getitem__(self,
                    valname : str,
                    ) -> InVal | OutVal:
        """Get a InVal or OutVal from the case.

        Parameters
        ----------
        valname : str
            The name of the value to get.

        Returns
        -------
        val : InVal | OutVal
            The value requested.
        """
        return self.vals[valname]


    def getInVals(self) -> dict[str, InVal]:
        """
        From all the InVar's, extract the vals for this case.

        Returns
        -------
        vals: dict[str, monaco.mc_val.InVal]
            The InVal's for this case.
        """
        vals = dict()
        for var in self.invars.values():
            val = var.getVal(self.ncase)
            vals[val.name] = val
        return vals


    def addOutVar(self,
                  outvar : OutVar,
                  ) -> None:
        """Add an OutVar to the case.

        Parameters
        ----------
        outvar : OutVar
            The OutVar to add.
        """
        self.outvars[outvar.name] = outvar
        self.vars[outvar.name] = outvar


    def getOutVals(self) -> dict[str, OutVal]:
        """
        From all the OutVar's, extract the vals for this case.

        Returns
        -------
        vals: dict[str, monaco.mc_val.OutVal]
            The OutVal's for this case.
        """
        vals = dict()
        for var in self.outvars.values():
            val = var.getVal(self.ncase)
            vals[val.name] = val
        return vals


    def addOutVal(self,
                  name   : str,
                  val    : Any,
                  split  : bool = True,
                  valmap : dict[Any, float] | None = None
                  ) -> None:
        """
        Generate an OutVal and add it to the dict of outvals.

        Parameters
        ----------
        name : str
            The name of the output value.
        val : Any
            The output value.
        split : bool
            Whether to split the value into its components.
        valmap : dict[Any, float], default: None
            A valmap dict mapping nonnumeric values to numbers.
        """
        if name in self.outvals.keys():
            raise ValueError(f"'{name}' is already an OutVal")

        outval = OutVal(name=name, ncase=self.ncase, val=val,
                        valmap=valmap, ismedian=self.ismedian)
        self.outvals[name] = outval
        self.vals[name] = outval
        if split:
            self.outvals.update(outval.split())
            self.vals.update(outval.split())
