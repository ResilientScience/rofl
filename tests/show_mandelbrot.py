#! /usr/bin/env python
# ______________________________________________________________________
'''show_mandelbrot

Run the ROFL Mandelbrot set function, and display the results.
'''
# ______________________________________________________________________
# Module imports

import rofl.test.test_rofl as tr

import Image
import numpy

import sys
import os.path

# ______________________________________________________________________
# Module data

SRC_PATH = os.path.join(os.path.split(tr.__file__)[0], 'mandelbrot.rofl')

# ______________________________________________________________________
# Main routine

def main (dx = None, dy = None, *args, **kws):
    if dx is None:
        dx = 640
    else:
        dx = int(dx)
    if dy is None:
        dy = 480
    else:
        dy = int(dy)
    mandelbrot_output = tr.roflfe(SRC_PATH, dx, dy)
    assert mandelbrot_output != '!Error'
    mandelbrot_output -= mandelbrot_output.min()
    mandelbrot_output *= 255. / mandelbrot_output.max()
    image = Image.fromarray(mandelbrot_output.astype(numpy.uint8))
    image.show()

# ______________________________________________________________________

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of show_mandelbrot.py
