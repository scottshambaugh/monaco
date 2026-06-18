# global_variables.py
import os
import pickle
import signal
import sys

try:
    from dask.distributed import WorkerPlugin

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

_GLOBAL_INVARS = dict()
_GLOBAL_OUTVARS = dict()
_GLOBAL_CONSTVALS = dict()


if HAS_DASK:

    class GlobalsPlugin(WorkerPlugin):
        def __init__(self, invars_blob, outvars_blob, constvals_blob):
            self.invars_blob = invars_blob
            self.outvars_blob = outvars_blob
            self.constvals_blob = constvals_blob

        def setup(self, worker):
            _worker_init(self.invars_blob, self.outvars_blob, self.constvals_blob)

else:
    GlobalsPlugin = None


def _set_parent_death_signal(parent_pid: int | None = None) -> None:
    """Make this worker exit when its parent dies, so pool workers are not left
    orphaned (holding their memory) if the parent is hard-killed before
    pool.shutdown() can run."""
    if sys.platform.startswith("linux"):
        import ctypes

        # PR_SET_PDEATHSIG: instant, OS-level, but only fires if our real parent
        # dies -- it misses workers reparented to a subreaper (e.g. WSL) at startup.
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)
    if parent_pid is not None:
        _start_parent_watchdog(parent_pid)


def _start_parent_watchdog(parent_pid: int) -> None:
    """Cross-platform fallback: poll the parent PID and exit if it dies. Covers
    reparenting, macOS, and Windows, where PR_SET_PDEATHSIG is unavailable."""
    import threading
    import time

    try:
        import psutil

        parent = psutil.Process(parent_pid)
    except Exception:
        return

    def _watch():
        while True:
            try:
                if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                    os._exit(1)
            except psutil.NoSuchProcess:
                os._exit(1)
            except Exception:
                pass
            time.sleep(0.25)

    threading.Thread(target=_watch, name="monaco-parent-watchdog", daemon=True).start()


def _worker_init(invars_blob, outvars_blob, constvals_blob, parent_pid=None):
    _set_parent_death_signal(parent_pid)
    invars = pickle.loads(invars_blob)
    outvars = pickle.loads(outvars_blob)
    constvals = pickle.loads(constvals_blob)
    register_global_vars(invars, outvars, constvals)


def _fork_worker_init(invars, outvars, constvals, parent_pid=None):
    # Wrapper keeps register_global_vars side-effect-free (it is also called in
    # the main process); PR_SET_PDEATHSIG is cleared by fork() so it is reset here.
    _set_parent_death_signal(parent_pid)
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
    _GLOBAL_INVARS = invars
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
