# ruff: noqa
import importlib as _importlib
import logging as _logging
from importlib import metadata as _metadata

__version__ = _metadata.version(__name__)

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from monaco import (
    mc_case,
    mc_sim,
    mc_var,
    mc_varstat,
    mc_val,
    mc_enums,
    gaussian_statistics,
    order_statistics,
    integration_statistics,
    dvars_sensitivity,
    helper_functions,
    case_runners,
    globals,
    mc_sampling,
)
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

_EAGER_MODULES = (
    mc_case,
    mc_sim,
    mc_var,
    mc_varstat,
    mc_val,
    mc_enums,
    gaussian_statistics,
    order_statistics,
    integration_statistics,
    dvars_sensitivity,
    helper_functions,
    case_runners,
    globals,
    mc_sampling,
)

# matplotlib and dask are slow to import, so these modules are only loaded when one of their
# names is first accessed. Kept in sync with each module's __all__ by test_lazy_names_match_all
_LAZY_NAMES = {
    "monaco.mc_plot": (
        "LINE_COLLECTION_MIN_CASES",
        "plot",
        "plot_hist",
        "plot_cdf",
        "plot_2d_scatter",
        "plot_2d_line",
        "plot_3d_scatter",
        "plot_2p5d_line",
        "plot_3d_line",
        "plot_cov_corr",
        "plot_integration_convergence",
        "plot_integration_error",
        "plot_sensitivities",
        "manage_axis",
        "apply_category_labels",
        "get_hist_lim",
        "plot_rug_marks",
        "plot_2d_cov_ellipse",
        "manage_invar_space",
        "get_plot_points",
        "varstat_to_plotspace",
        "plot_line_ensemble",
        "get_var_steps",
    ),
    "monaco.mc_multi_plot": (
        "multi_plot",
        "multi_plot_2d_scatter_hist",
        "multi_plot_grid_tri",
        "multi_plot_grid_rect",
        "handle_fig",
    ),
    "monaco.dask_plugin": ("GlobalsPlugin",),
}
_LAZY_LOOKUP = {name: modname for modname, names in _LAZY_NAMES.items() for name in names}

# dask is optional, so its plugin is left out of star imports
__all__ = [name for mod in _EAGER_MODULES for name in mod.__all__] + [
    name for name, modname in _LAZY_LOOKUP.items() if modname != "monaco.dask_plugin"
]


def __getattr__(name):
    modname = _LAZY_LOOKUP.get(name)
    if modname is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_importlib.import_module(modname), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_LOOKUP))
