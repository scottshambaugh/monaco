# helper_functions.py
from __future__ import annotations

from operator import itemgetter
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Sequence, Iterable, Sized
from time import time
from functools import wraps
from hashlib import sha512
import warnings

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None


def next_power_of_2(x : int) -> int:
    """
    Returns the next power of two greater than or equal to the input.

    Parameters
    ----------
    x : int
        The input number.

    Returns
    -------
    nextpow2 : int
        The next power of two, nextpow2 >= x.
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
    s_hash : int
        The hash of str.
    """
    return int(sha512(s.encode('utf-8')).hexdigest(), 16)


def hashable_val(val : Any) -> Any:
    """
    For `nummap` and `valmap`, we need to use values as keys in a dictionary.
    This function will return the string representation of a value if that
    value is not hashable.

    Parameters
    ----------
    val : Any
        The value to hash.

    Returns
    -------
    hashable_val : Any
        A hashable representation of the val.
    """
    try:
        hash(val)
        return val
    except TypeError:
        return str(val)


def is_num(val : Any) -> bool:
    """
    Type checking function to see if the input is a number.

    Parameters
    ----------
    val : Any
        The value to check.

    Returns
    -------
    isnum : bool
        Returns True if the input is a number, False otherwise.
    """
    if isinstance(val, bool) or isinstance(val, str):
        return False
    else:
        try:
            float(val)
        except (ValueError, TypeError):
            return False
        else:
            return True


def length(x : Any) -> int | None:
    """
    Genericized length function that works on scalars (which have length 1).

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    x_len : int
        The length of the input. If not a sequence or scalar, returns None.
    """
    if isinstance(x, Sized):
        return len(x)
    elif isinstance(x, (np.float64, float, bool, int)):
        return 1
    else:
        return None


def get_list(x : Any) -> list[Any]:
    """
    Converts the input to an iterable list.

    Parameters
    ----------
    x : Any
        The object to convert.

    Returns
    -------
    x_list : list
        A list conversion of the input.
    """
    if x is None:
        return list()
    elif isinstance(x, str):
        return [x, ]
    elif pd and isinstance(x, pd.DataFrame):
        return [x, ]
    elif isinstance(x, Iterable):
        if isinstance(x, np.ndarray) and np.ndim(x) == 0:
            return [x[()], ]
        return list(x)
    else:
        return [x, ]


def slice_by_index(sequence : Sequence[Any],
                   indices  : int | Iterable[int],
                   ) -> list:
    """
    Returns a slice of a sequence at the specified indices.

    Parameters
    ----------
    sequence : Sequence
        The sequence to slice.
    indices : int | Iterable
        The indices to slice at.

    Returns
    -------
    slice : list
        A list representing the values of the input sequence at the specified
        indices.
    """
    indices_list = get_list(indices)
    if sequence is None or indices_list == list():
        return []
    items = itemgetter(*indices_list)(sequence)
    if len(indices_list) == 1:
        return [items]
    return list(items)


def vprint(verbose : bool, *args, **kwargs) -> None:
    """
    Print only if verbose is True.

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
    Warn only if verbose is True.

    Parameters
    ----------
    verbose : bool
        Flag to determine whether to print something.
    *args, **kwargs
        Must include a warning message here!
    """
    if verbose:
        warn_default_format = warnings.formatwarning
        warnings.formatwarning = warn_short_format  # type: ignore
        warnings.warn(*args, **kwargs)
        warnings.formatwarning = warn_default_format


def vwrite(verbose : bool, *args, **kwargs) -> None:
    """
    Perform a tqdm.write() only if verbose is True.

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


def empty_list() -> list:
    """
    Sentinel for default arguments being an empty list.

    Returns
    -------
    empty_list : list
        An empty list.
    """
    return []


def flatten(nested_x : Iterable[Any]) -> list[Any]:
    """
    Flattens a nested interable into a list with all nested items.

    Parameters
    ----------
    nested_x : Iterable
        Nested iterable.

    Returns
    -------
    flattened_x : list
        The nested iterable flattened into a list.
    """
    def flatten_generator(x):
        for element in x:
            if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
                yield from flatten(element)
            else:
                yield element

    flattened_x = list(flatten_generator(nested_x))
    return flattened_x
