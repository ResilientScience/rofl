#! /usr/bin/env python
# ______________________________________________________________________
'''test_mandelbrot

Test the ROFL Mandelbrot set demo.
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
                        'mandelbrot.rofl')

# ______________________________________________________________________
# Class (test) definitionss

class TestMandelbrot (unittest.TestCase):
    def test_mandelbrot_sanity (self):
        self.assertNotEqual(tr.roflfe(SRC_PATH, 120, 80), '!Error')

# ______________________________________________________________________
# Main (test) routine

def main (*args, **kws):
    unittest.main(*args, **kws)

# ______________________________________________________________________

if __name__ == '__main__':
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_mandelbrot.py
