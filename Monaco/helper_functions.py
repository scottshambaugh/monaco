# helper_functions.py

from collections.abc import Iterable
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Union
from time import time
from functools import wraps
from hashlib import sha512
import warnings


def next_power_of_2(x : int) -> int:
    if x <= 0:
        return 0
    else:    
        return int(2**np.ceil(np.log2(x)))


def hash_str_repeatable(s : str) -> int:
    return int(sha512(s.encode('utf-8')).hexdigest(), 16)


def is_num(val) -> bool:
    if isinstance(val, bool) or isinstance(val, str):
        return False
    else:
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True


def length(x) -> Union[None, int]:
    if isinstance(x, Iterable):
        return len(x)
    elif isinstance(x, (np.float64, float, bool, int)):
        return 1
    else:
        return None


def get_iterable(x) -> Union[tuple, Iterable]:
    if x is None:
        return tuple()
    elif isinstance(x, pd.DataFrame):
        return (x,)
    elif isinstance(x, Iterable):
        return x
    else:
        return (x,)


def slice_by_index(sequence, indices) -> list:
    indices = get_iterable(indices)
    if not sequence or not indices:
        return []
    items = itemgetter(*indices)(sequence)
    if len(indices) == 1:
        return [items]
    return list(items)


def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def warn_short_format(message, category, filename, lineno, file=None, line=None):
    return f'{category.__name__}: {message}\n'


def vwarn(verbose, *args, **kwargs):
    if verbose:
        warn_default_format = warnings.formatwarning
        warnings.formatwarning = warn_short_format
        warnings.warn(*args, **kwargs)
        warnings.formatwarning = warn_default_format


def vwrite(verbose, *args, **kwargs):
    if verbose:
        tqdm.write(*args, **kwargs)
        
        
def timeit(fcn):
    @wraps(fcn)
    def timed(*args, **kw):
        t0 = time()
        output = fcn(*args, **kw)
        t1 = time()
        print(f'"{fcn.__name__}" took {(t1 - t0)*1000 : .3f} ms to execute.\n')
        return output
    return timed

