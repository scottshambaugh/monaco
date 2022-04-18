# dask_runners.py
from typing import Callable
from copy import copy
from monaco.mc_case import Case
from monaco.helper_functions import vwrite, vwarn, get_list
from datetime import datetime


def pre_process_case(preprocfcn: Callable,
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
        case.siminput = preprocfcn(case)
        # self.casespreprocessed.add(case.ncase)
        case.haspreprocessed = True

    except Exception:
        if debug:
            raise
        else:
            vwarn(verbose, f'\nPreprocessing case {case.ncase} failed')

    return case



def run_case(runfcn: Callable,
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
        case.simrawoutput = runfcn(*get_list(case.siminput))
        case.endtime = datetime.now()
        case.runtime = case.endtime - case.starttime
        case.runsimid = runsimid
        case.hasrun = True

        # self.casesrun.add(case.ncase)

    except Exception:
        if debug:
            raise
        vwrite(verbose, f'\nRunning case {case.ncase} failed')

    return case

def post_process_case(postprocfcn: Callable,
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
        postprocfcn(case, *get_list(case.simrawoutput))
        # self.casespostprocessed.add(case.ncase)
        case.haspostprocessed = True

    except Exception:
        if debug:
            raise
        else:
            vwrite(verbose, f'\nPostprocessing case {case.ncase} failed')

    return case
