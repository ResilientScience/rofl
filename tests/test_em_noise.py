#! /usr/bin/env python
# ______________________________________________________________________
'''test_em_noise script - Test the EM noise code.
'''
# ______________________________________________________________________
# Module imports

import scipy
import rofl.frontend
import rofl.backend
import itertools
import os.path

# ______________________________________________________________________
# Module data

INPUT_SPACE = ((0.069,),
               tuple(scipy.arange(-5., 5., 0.04)),
               (0.14,),
               (0.,),
               (0.,),
               (0.,),
               (13., 14., 15., 16.),
               )

# ______________________________________________________________________
# Main (test) routine

def main (*args):
    bindings = rofl.frontend.parse_string(rofl.frontend.TEST_SRC_ALT)
    rofl_module = rofl.backend.codegen_module(bindings)
    str_inp_results = []
    for g_idx, invec in enumerate(itertools.product(*INPUT_SPACE)):
        result = rofl_module.toplevel(*invec)
        str_inp_results.append(result)
        print g_idx, result
    bindings2 = rofl.frontend.parse_file(
        os.path.join(
            os.path.split(rofl.frontend.__file__)[0], 'test', 'em_noise.rofl'))
    rofl_module2 = rofl.backend.codegen_module(bindings)
    for g_idx, invec in enumerate(itertools.product(*INPUT_SPACE)):
        assert rofl_module2.toplevel(*invec) == str_inp_results[g_idx]

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_em_noise.py
