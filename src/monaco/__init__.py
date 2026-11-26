# ruff: noqa
from importlib import metadata

__version__ = metadata.version(__name__)

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from monaco.mc_case import *
from monaco.mc_sim import *
from monaco.mc_var import *
from monaco.mc_varstat import *
from monaco.mc_val import *
from monaco.mc_enums import *
from monaco.gaussian_statistics import *
from monaco.order_statistics import *
from monaco.integration_statistics import *
from monaco.dvars_sensitivity import *
from monaco.helper_functions import *
from monaco.case_runners import *
from monaco.globals import *
from monaco.mc_sampling import *

_LAZY_MODULES = ("monaco.mc_plot", "monaco.mc_multi_plot", "monaco.dask_plugin")


def __getattr__(name):
    # Load the plotting and dask modules only when used, since matplotlib and dask are slow to import
    import importlib

    if not name.startswith("_"):
        for modname in _LAZY_MODULES:
            try:
                mod = importlib.import_module(modname)
            except ImportError:
                continue
            if name in vars(mod):
                return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
