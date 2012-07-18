#! /usr/bin/env python
# ______________________________________________________________________
'''test_raycast

Test the ROFL raycasting demo.
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
                        'raycast.rofl')

# ______________________________________________________________________
# Class (test) definitions

class TestRaycast (unittest.TestCase):
    def test_raycast_sanity (self):
        self.assertNotEqual(tr.roflfe(SRC_PATH, 100, 100, 5, 1), '!Error')

# ______________________________________________________________________
# Main (test) routine

def main (*args, **kws):
    unittest.main(*args, **kws)

# ______________________________________________________________________

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_raycast.py
