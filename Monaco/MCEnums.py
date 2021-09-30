# MCEnums.py

from enum import Enum

class SampleMethod(str, Enum):
    RANDOM          = 'random'
    SOBOL           = 'sobol'
    SOBOL_RANDOM    = 'sobol_random'
    HALTON          = 'halton'
    HALTON_RANDOM   = 'halton_random'
    LATIN_HYPERCUBE = 'latin_hypercube'


class MCFunctions(str, Enum):
    PREPROCESS  = 'preprocess'
    RUN         = 'run'
    POSTPROCESS = 'postprocess'


class StatBound(str, Enum):
    NEAREST        = 'nearest'
    BOTH           = 'both'
    ALL            = 'all'
    ONESIDED       = '1-sided'
    TWOSIDED       = '2-sided'
    ONESIDED_UPPER = '1-sided upper'
    ONESIDED_LOWER = '1-sided lower'


class VarStat(str, Enum):
    MAX         = 'max'
    MIN         = 'min'
    MEDIAN      = 'median'
    MEAN        = 'mean'
    GEOMEAN     = 'geomean'
    MODE        = 'mode'
    SIGMAP      = 'sigmaP'
    GAUSSIANP   = 'gaussianP'
    ORDERSTATTI = 'orderstatTI'
    ORDERSTATP  = 'orderstatP'


class VarStatSide(str, Enum):
    HIGH = 'high'
    LOW  = 'low'
    BOTH = 'both'
    ALL  = 'all'
    