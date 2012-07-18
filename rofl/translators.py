#! /usr/bin/env python
# ______________________________________________________________________
'''translators.py

Defines utility functions for various modes of translation from ROFL
to Python.
'''
# ______________________________________________________________________
# Module imports

import frontend
import backend

# ______________________________________________________________________
# Function definitions

def rofl_string_to_source (src, *args, **kws):
    return backend.codegen_module_src(frontend.parse_string(src, **kws), *args,
                                      **kws)

rs2s = rofl_string_to_source

# ______________________________________________________________________

def rofl_string_to_module (src, *args, **kws):
    return backend.codegen_module(frontend.parse_string(src, **kws), *args,
                                  **kws)

rs2m = rofl_string_to_module

# ______________________________________________________________________

def rofl_string_to_codeobj (src, *args, **kws):
    return backend.codegen_module_co(frontend.parse_string(src, **kws), *args,
                                     **kws)

rs2c = rofl_string_to_codeobj

# ______________________________________________________________________

def rofl_file_to_source (src_path, *args, **kws):
    return backend.codegen_module_src(frontend.parse_file(src_path, **kws),
                                      *args, **kws)

rf2s = rofl_file_to_source

# ______________________________________________________________________

def rofl_file_to_module (src_path, *args, **kws):
    return backend.codegen_module(frontend.parse_file(src_path, **kws), *args,
                                  **kws)

rf2m = rofl_file_to_module

# ______________________________________________________________________

def rofl_file_to_codeobj (src_path, *args, **kws):
    return backend.codegen_module_co(frontend.parse_file(src_path, **kws),
                                     *args, **kws)

rf2c = rofl_file_to_codeobj

# ______________________________________________________________________
# Main routine

def main (*args, **kws):
    pass

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of rofl2py.py
