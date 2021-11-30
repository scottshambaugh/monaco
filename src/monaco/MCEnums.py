# MCEnums.py

from enum import Enum

class SampleMethod(str, Enum):
    """
    Enum for the possible Monte-Carlo sampling methods.
    """
    RANDOM          = 'random'
    SOBOL           = 'sobol'
    SOBOL_RANDOM    = 'sobol_random'
    HALTON          = 'halton'
    HALTON_RANDOM   = 'halton_random'
    LATIN_HYPERCUBE = 'latin_hypercube'


class MCFunctions(str, Enum):
    """
    Enum for the three required user functions.
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


class VarStat(str, Enum):
    """
    Enum for the variable statistics functions.
    """
    MAX         = 'max'
    MIN         = 'min'
    MEDIAN      = 'median'
    MEAN        = 'mean'
    GEOMEAN     = 'geomean'
    MODE        = 'mode'
    SIGMA       = 'sigma'
    GAUSSIANP   = 'gaussianP'
    ORDERSTATTI = 'orderstatTI'
    ORDERSTATP  = 'orderstatP'


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
