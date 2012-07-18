#! /usr/bin/env python
# ______________________________________________________________________
'''test_n_body

Test the n-body ROFL program.
'''
# ______________________________________________________________________
# Module imports

import rofl.test.test_rofl as tr

import sys
import os.path
import unittest

# ______________________________________________________________________
# Module data

SRC_PATH = os.path.join(os.path.split(tr.__file__)[0],
                        'n_body.rofl')

# ______________________________________________________________________
# Class (test) definitions

class TestROFLNBody (unittest.TestCase):
    def test_n_body_sanity (self):
        '''Do a sanity check on a 2D n-body simulator written in ROFL.
        This is by no means a functional test of the results of the
        n-body simulator.'''
        self.assertNotEqual(tr.roflfe(SRC_PATH, 3, 10, 3), '!Error')

# ______________________________________________________________________
# Main (test) routine

def main (*args, **kws):
    unittest.main(*args, **kws)

# ______________________________________________________________________

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_n_body.py
