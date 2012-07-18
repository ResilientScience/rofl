#! /usr/bin/env python
# ______________________________________________________________________
'''show_rofl

Run a ROFL program.  If the result is a two dimensional array, show an
image of the scaled array.

Example:

% ./show_rofl.py raycast.rofl 100 100 3 0
'''
# ______________________________________________________________________
# Module imports

import rofl.test.test_rofl as tr

import ast
import Image
import numpy

import sys
import os.path

# ______________________________________________________________________
# Main routine

def main (rofl_path, *args, **kws):
    args = [ast.literal_eval(arg) for arg in args]
    result = tr.roflfe(rofl_path, *args, **kws)
    assert result != "!Error"
    assert hasattr(result, "ndim") and result.ndim == 2
    result -= result.min()
    result *= 255. / result.max()
    image = Image.fromarray(result.astype(numpy.uint8))
    image.show()

# ______________________________________________________________________

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of show_rofl.py
