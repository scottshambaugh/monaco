# case_runners.py

from typing import Callable
from datetime import datetime
from monaco.mc_case import Case
from monaco.helper_functions import vwrite, vwarn, get_list


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


def execute_full_case(preprocfcn: Callable,
                      runfcn: Callable,
                      postprocfcn: Callable,
                      case: Case,
                      debug: bool,
                      verbose: bool,
                      runsimid: int) -> Case:
    """
    Execute the full preprocessing, run, and postprocessing pipeline for a case.

    This reduces data transfer by keeping the case in the worker process
    throughout the entire pipeline, similar to how Dask chains operations.

    Parameters
    ----------
    preprocfcn : Callable
        The preprocessing function.
    runfcn : Callable
        The running function.
    postprocfcn : Callable
        The postprocessing function.
    case : monaco.mc_case.Case
        The case to execute.
    debug : bool
        Whether to raise an error if the case fails.
    verbose : bool
        Whether to print verbose output.
    runsimid : int
        The simulation ID.

    Returns
    -------
    case : monaco.mc_case.Case
        The same case, executed.
    """
    # Preprocess
    case = preprocess_case(preprocfcn, case, debug, verbose)
    if not case.haspreprocessed:
        return case

    # Run
    case = run_case(runfcn, case, debug, verbose, runsimid)
    if not case.hasrun:
        return case

    # Postprocess
    case = postprocess_case(postprocfcn, case, debug, verbose)
    return case
