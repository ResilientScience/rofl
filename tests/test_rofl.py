#! /usr/bin/env python
# ______________________________________________________________________
'''test_rofl

Some basic unit tests for the ROFL translator.
'''
# ______________________________________________________________________
# Module imports

from rofl import frontend as fe
from rofl import backend as be

import sys
import unittest
import traceback

# ______________________________________________________________________
# Utility functions

def rofle (src, *args, **kws):
    '''
    Given ROFL source as a string, translate it to a Python module,
    and call its top-level function, passing along any additional
    arguments given.  Returns the output of the top-level function, or
    an error result string if an exception occured.

    Any optional keyword arguments are passed to the translator (both
    the parser and the code generator).
    '''
    try:
        ret_val = be.codegen_module(fe.parse_string(src, **kws),
                                    **kws).toplevel(*args)
    except:
        traceback.print_exc(file = sys.stderr)
        ret_val = '!Error'
    return ret_val

# ______________________________________________________________________

def roflfe (path, *args, **kws):
    '''
    Given a path to a ROFL source file, translate the file to a Python
    module, and run its top-level function, passing along any
    additional arguments.  Returns the output of the top-level
    function, or an error result string if an exception occured.

    Any optional keyword arguments are passed to the translator (both
    the parser and the code generator).
    '''
    try:
        ret_val = be.codegen_module(fe.parse_file(path, **kws),
                                    **kws).toplevel(*args)
    except:
        traceback.print_exc(file = sys.stderr)
        ret_val = '!Error'
    return ret_val

# ______________________________________________________________________
# Class (test) definitions

class TestROFL (unittest.TestCase):
    def test_empty_list (self):
        self.assertEqual(rofle('f() = []'), [])
        self.assertEqual(rofle('f() = [] + [1, 2]'), [1, 2])

    def test_empty_tuple (self):
        self.assertEqual(rofle('f() = ()'), ())

    def test_identity (self):
        '''Run some sanity checks on the ROFL identity function.
        See also MULT-74.'''
        for test_input in ([], (), 99.99, 3+5j):
            self.assertEqual(rofle('identity(x) = x', test_input), test_input)

# ______________________________________________________________________
# Main (test) routine

def main (*args, **kws):
    unittest.main(*args, **kws)

# ______________________________________________________________________

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_rofl.py
