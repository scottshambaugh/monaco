# global_variables.py
import pickle

_GLOBAL_INVARS  = dict()
_GLOBAL_OUTVARS = dict()
_GLOBAL_CONSTVALS = dict()


def _worker_init(invars_blob, outvars_blob, constvals_blob):
    invars  = pickle.loads(invars_blob)
    outvars = pickle.loads(outvars_blob)
    constvals = pickle.loads(constvals_blob)
    register_global_vars(invars, outvars, constvals)

def register_global_vars(invars, outvars, constvals):
    """
    Called exactly once (in each process) to make the
    heavyweight Var dictionaries available to every Case.

    Parameters
    ----------
    invars : dict
        Dictionary of InVar objects.
    outvars : dict
        Dictionary of OutVar objects.
    constvals : dict
        Dictionary of ConstVal objects.
    """
    global _GLOBAL_INVARS, _GLOBAL_OUTVARS, _GLOBAL_CONSTVALS
    _GLOBAL_INVARS  = invars
    _GLOBAL_OUTVARS = outvars
    _GLOBAL_CONSTVALS = constvals

def get_global_vars():
    """
    Get the global InVar and OutVar dictionaries.

    Returns
    -------
    invars : dict
        Dictionary of InVar objects.
    outvars : dict
        Dictionary of OutVar objects.
    constvals : dict
        Dictionary of ConstVal objects.
    """
    return _GLOBAL_INVARS, _GLOBAL_OUTVARS, _GLOBAL_CONSTVALS
