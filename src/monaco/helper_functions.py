# helper_functions.py

from collections.abc import Sized, Iterable, Sequence
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Callable, Any, Union
from time import time
from functools import wraps
from hashlib import sha512
import warnings


def next_power_of_2(x : int) -> int:
    """
    Returns the next power of two greater than or equal to the input.
    
    Parameters
    ----------
    x : int
        The input number.
    
    Returns
    -------
    _ : int
        The next power of two, _ >= x.
    """
    if x <= 0:
        return 0
    else:
        return int(2**np.ceil(np.log2(x)))


def hash_str_repeatable(s : str) -> int:
    """
    By default, string hashing in python is randomized. This function returns a
    repeatable non-randomized hash for strings.
    See: https://docs.python.org/3/using/cmdline.html#cmdoption-R
    
    Parameters
    ----------
    s : str
        The string to hash.
    
    Returns
    -------
    _ : int
        The hash of str.
    """
    return int(sha512(s.encode('utf-8')).hexdigest(), 16)


def is_num(val : Any) -> bool:
    """
    Type checking function to see if the input is a number.
    
    Parameters
    ----------
    val : Any
        The value to check.
    
    Returns
    -------
    _ : bool
        Returns True if the input is a number, False otherwise.
    """
    if isinstance(val, bool) or isinstance(val, str):
        return False
    else:
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True


def length(x : Any) -> int:
    """
    Genericized length function that works on scalars (which have length 1).
    
    Parameters
    ----------
    x : Any
        The value to check.
    
    Returns
    -------
    _ : int
        The length of the input.
    """
    if isinstance(x, Sized):
        return len(x)
    elif isinstance(x, (np.float64, float, bool, int)):
        return 1
    else:
        return None


def get_sequence(x : Any) -> tuple:
    """
    Converts the input to an iterable tuple.
    
    Parameters
    ----------
    x : Any
        The object to convert.
    
    Returns
    -------
    _ : tuple
        A tuple conversion of the input.
    """
    if x is None:
        return tuple()
    elif isinstance(x, pd.DataFrame):
        return (x,)
    elif isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


def slice_by_index(sequence : Sequence, 
                   indices : Union[int, Iterable]
                   ) -> list:
    """
    Returns a slice of a sequence at the specified indices.
    
    Parameters
    ----------
    sequence : Sequence
        The sequence to slice.
    indices : {int, Iterable}
        The indices to slice at.
    
    Returns
    -------
    _ : list
        A list representing the values of the input sequence at the specified
        indices.
    """
    indices_sequence = get_sequence(indices)
    if sequence is None or indices_sequence == tuple():
        return []
    items = itemgetter(*indices_sequence)(sequence)
    if len(indices_sequence) == 1:
        return [items]
    return list(items)


def vprint(verbose : bool, *args, **kwargs) -> None:
    """
    Print only if verbose == True.
    
    Parameters
    ----------
    verbose : bool
        Flag to determine whether to print something.
    *args, **kwargs
        Must include something to print here!
    """
    if verbose:
        print(*args, **kwargs)


def warn_short_format(message, category, filename, lineno, file=None, line=None) -> str:
    """
    Custom warning format for use in vwarn()
    """
    return f'{category.__name__}: {message}\n'


def vwarn(verbose : bool, *args, **kwargs) -> None:
    """
    Warn only if verbose == True.
    
    Parameters
    ----------
    verbose : bool
        Flag to determine whether to print something.
    *args, **kwargs
        Must include a warning message here!
    """
    if verbose:
        warn_default_format = warnings.formatwarning
        warnings.formatwarning = warn_short_format # type: ignore
        warnings.warn(*args, **kwargs)
        warnings.formatwarning = warn_default_format


def vwrite(verbose : bool, *args, **kwargs) -> None:
    """
    Perform a tqdm.write() only if verbose == True.
    
    Parameters
    ----------
    verbose : bool
        Flag to determine whether to write something.
    *args, **kwargs
        Must include something to write here!
    """
    if verbose:
        tqdm.write(*args, **kwargs)


def timeit(fcn : Callable):
    """
    Function decorator to print out the function runtime in milliseconds.
    
    Parameters
    ----------
    fcn : Callable
        Function to time.
    """
    @wraps(fcn)
    def timed(*args, **kw):
        t0 = time()
        output = fcn(*args, **kw)
        t1 = time()
        print(f'"{fcn.__name__}" took {(t1 - t0)*1000 : .3f} ms to execute.\n')
        return output
    return timed
