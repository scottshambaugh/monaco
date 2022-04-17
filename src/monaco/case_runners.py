# dask_runners.py
from typing import Callable
from copy import copy
from monaco.mc_case import Case
from monaco.helper_functions import vwrite, vwarn, get_list
from datetime import datetime

def pre_process_case(preProcFcn: Callable,
                     case : Case,
                     debug : bool,
                     verbose : bool,
                     ) -> Case:
    """
    Preprocess a single Monte Carlo case.

    Parameters
    ----------
    case : monaco.mc_case.Case
        The case to preprocess.

    Returns
    -------
    case : monaco.mc_case.Case
        The same case, preprocessed.
    """
    case = copy(case)
    try:
        case.siminput = preProcFcn(case)
        # self.casespreprocessed.add(case.ncase)
        case.haspreprocessed = True

    except Exception:
        if debug:
            raise
        else:
            vwarn(verbose, f'\nPreprocessing case {case.ncase} failed')

    return case



def run_case(runFcn: Callable,
             case : Case,
             debug : bool,
             verbose : bool,
             runsimid : int,
             ) -> None:
    """
    Run a single Monte Carlo case.

    Parameters
    ----------
    case : monaco.mc_case.Case
        The case to run.

    Returns
    -------
    case : monaco.mc_case.Case
        The same case, ran.
    """
    case = copy(case)

    try:
        case.starttime = datetime.now()
        case.simrawoutput = runFcn(*get_list(case.siminput))
        case.endtime = datetime.now()
        case.runtime = case.endtime - case.starttime
        case.runsimid = runsimid
        case.hasrun = True

        '''
        if self.savecasedata:
            filepath = self.resultsdir / f'{self.name}_{case.ncase}.mccase'
            case.filepath = filepath
            try:
                filepath.unlink()
            except FileNotFoundError:
                pass
            with open(filepath, 'wb') as file:
                cloudpickle.dump(case, file)
        '''

        # self.casesrun.add(case.ncase)

    except Exception:
        if debug:
            raise
        vwrite(verbose, f'\nRunning case {case.ncase} failed')

    return case

def post_process_case(postProcFcn: Callable,
                      case : Case,
                      debug : bool,
                      verbose : bool,
                      ) -> None:
    """
    Postprocess a single Monte Carlo case.

    Parameters
    ----------
    case : monaco.mc_case.Case
        The case to postprocess.

    Returns
    -------
    case : monaco.mc_case.Case
        The same case, postprocessed.
    """
    case = copy(case)
    try:
        postProcFcn(case, *get_list(case.simrawoutput))
        # self.casespostprocessed.add(case.ncase)
        case.haspostprocessed = True

    except Exception:
        if debug:
            raise
        else:
            vwrite(verbose, f'\nPostprocessing case {case.ncase} failed')

    return case
