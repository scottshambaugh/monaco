# case_runners.py

from typing import Callable
from copy import copy
from monaco.mc_case import Case
from monaco.helper_functions import vwrite, vwarn, get_list
from datetime import datetime


def preprocess_case(preprocfcn: Callable,
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
             ) -> Case:
    """
    Run a single Monte Carlo case.

    Parameters
    ----------
    case : monaco.mc_case.Case
        The case to run.

    Returns
    -------
    case : monaco.mc_case.Case
        The same case, run.
    """
    case = copy(case)

    try:
        case.starttime = datetime.now()
        case.simrawoutput = runfcn(*get_list(case.siminput))
        case.endtime = datetime.now()
        case.runtime = case.endtime - case.starttime
        case.runsimid = runsimid
        case.hasrun = True
        if not case.keepsiminput:
            case.siminput = ()

    except Exception:
        if debug:
            raise
        vwrite(verbose, f'\nRunning case {case.ncase} failed')

    return case


def postprocess_case(postprocfcn: Callable,
                     case : Case,
                     debug : bool,
                     verbose : bool,
                     ) -> Case:
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
        case.haspostprocessed = True
        if not case.keepsimrawoutput:
            case.simrawoutput = ()

    except Exception:
        if debug:
            raise
        else:
            vwrite(verbose, f'\nPostprocessing case {case.ncase} failed')

    return case
