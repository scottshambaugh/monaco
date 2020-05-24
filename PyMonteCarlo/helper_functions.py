from collections import Iterable
from operator import itemgetter
import numpy as np

def is_num(val):
    if isinstance(val, bool):
        return False
    else:
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True


def length(x):
    if isinstance(x, Iterable):
        return len(x)
    elif isinstance(x, (np.float, float, bool, int)):
        return 1
    else:
        return None


def get_iterable(x):
    if x == None:
        return tuple()
    elif isinstance(x, Iterable):
        return x
    else:
        return (x,)


def slice_by_index(sequence, indices):
    if not sequence or not indices:
        return []
    items = itemgetter(*indices)(sequence)
    if len(indices) == 1:
        return [items]
    return list(items)
