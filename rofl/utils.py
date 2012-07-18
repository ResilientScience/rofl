#! /usr/bin/env python
# ______________________________________________________________________
"""Module rofl.utils

Utility functions for use in ROFL code.  Think of this as the ROFL
standard library.
"""
# ______________________________________________________________________
# Module imports

from scipy import *
from functools import partial
import pprint

# ______________________________________________________________________
# Function definition(s)

bounds = arange

def mesh (bounds1, bounds2):
    return meshgrid(bounds1, bounds2)

def xstep (arr):
    return arr[0][1] - arr[0][0]

def ystep (arr):
    return arr[1][0] - arr[0][0]

epsilon = 1e-9

np_where = where

def trace (obj):
    pprint.pprint(obj)
    return obj

# ______________________________________________________________________
# "Functional" iteration library

class ROFLHalt (StopIteration):
    '''Used to signal early termination inside a ROFL iteration function.'''

def rofl_halt ():
    raise ROFLHalt()

def rofl_foldl (fn, init, elems):
    ret_val = init
    for elem in elems:
        try:
            ret_val = fn(elem, ret_val)
        except ROFLHalt:
            break
    return ret_val

def rofl_foldr (fn, init, elems):
    rev_elems = list(elems)
    rev_elems.reverse()
    return rofl_foldl(fn, init, rev_elems)

def rofl_repeat (fn, init, count):
    ret_val = init
    for counter in xrange(count):
        try:
            ret_val = fn(ret_val)
        except ROFLHalt:
            break
    return ret_val

# ______________________________________________________________________
# End of utils.py (rofl.utils)
