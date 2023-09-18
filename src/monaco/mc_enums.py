# mc_enums.py
from __future__ import annotations

from enum import Enum

class SampleMethod(str, Enum):
    """
    Enum for the possible Monte Carlo sampling methods.
    """
    RANDOM          = 'random'
    SOBOL           = 'sobol'
    SOBOL_RANDOM    = 'sobol_random'
    HALTON          = 'halton'
    HALTON_RANDOM   = 'halton_random'
    LATIN_HYPERCUBE = 'latin_hypercube'


class SimFunctions(str, Enum):
    """
    Enum for the three required user functions.

    Notes
    -----
    The preprocess function must take in only a moncao.mc_case.Case object. It
    then must return a tuple of the input arguments for the run function.
    The run function will take in whatever inputs and return whatever outputs.
    It it recommended to package the outputs into a tuple.
    The postprocess function must take in as its first argument a
    moncao.mc_case.Case object, followed by the outputs from the run function.
    The simulation will attempt to unpack the run function outputs if they are
    stored in a tuple.

    See
    https://github.com/scottshambaugh/monaco/blob/main/template/template_functions.py
    for an example of this.
    """
    PREPROCESS  = 'preprocess'
    RUN         = 'run'
    POSTPROCESS = 'postprocess'


class StatBound(str, Enum):
    """
    Enum for possible statistical bounds. Note that not all of these may be
    valid for a given statistical function.
    """
    NEAREST        = 'nearest'
    BOTH           = 'both'
    ALL            = 'all'
    ONESIDED       = '1-sided'
    TWOSIDED       = '2-sided'
    ONESIDED_UPPER = '1-sided upper'
    ONESIDED_LOWER = '1-sided lower'


class VarStatType(str, Enum):
    """
    Enum for the variable statistics functions.
    """
    MAX         = 'max'
    MIN         = 'min'
    MEDIAN      = 'median'
    MEAN        = 'mean'
    GEOMEAN     = 'geomean'
    MODE        = 'mode'
    VARIANCE    = 'variance'
    SKEWNESS    = 'skewness'
    KURTOSIS    = 'kurtosis'
    MOMENT      = 'moment'
    PERCENTILE  = 'percentile'
    SIGMA       = 'sigma'
    GAUSSIANP   = 'gaussianp'
    ORDERSTATTI = 'orderstatti'
    ORDERSTATP  = 'orderstatp'


class VarStatSide(str, Enum):
    """
    Enum for the variable statistics 'side' (see documentation for each varstat
    function).
    """
    HIGH = 'high'
    LOW  = 'low'
    BOTH = 'both'
    ALL  = 'all'


class PlotOrientation(str, Enum):
    """
    Enum for the plotting functions orientation.
    """
    VERTICAL   = 'vertical'
    HORIZONTAL = 'horizontal'


class InVarSpace(str, Enum):
    """
    Enum for whether to plot invars in number or percentile space.
    """
    NUMS = 'nums'
    PCTS = 'pcts'


class Sensitivities(str, Enum):
    """
    Enum for whether to plot invars in number or percentile space.
    """
    INDICES = 'indices'
    RATIOS = 'ratios'
