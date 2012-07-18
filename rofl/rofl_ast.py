#! /usr/bin/env python
# ______________________________________________________________________
"""Module rofl_ast

Defines the Sympy abstract syntax tree nodes used by ROFL.

Each abstract syntax term is a Sympy expression, constrained to use
one of the following term constructors:

Add
Lambda
Mul
Pow
Symbol

And
App
BitwiseAnd
BitwiseNot
BitwiseOr
BitwiseShiftLeft
BitwiseShiftRight
BitwiseXor
GetAttr
If
Let
Mod
Not
Nth
Or
PartialApp
Tuple

Note that Sympy relational operators are "broken", since they may
return boolean values that are not compatible with the rest of Sympy.

Example:
>>> sympy.sympify('False').is_Float
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'bool' object has no attribute 'is_Float'
"""
# ______________________________________________________________________
# Module imports

import sympy

# ______________________________________________________________________
# Nodes borrowed from Sympy

from sympy import \
     Add, \
     And, \
     Lambda, \
     Mul, \
     Not, \
     Or, \
     Pow, \
     Symbol

# ______________________________________________________________________
# Module data

empty = Symbol('') # Represents an empty slice operand.

ellipsis = Symbol('...') # Represents an ellipsis slice operand.

# Alternates to the following symbols might be to overload the
# Function.__new__() code to permit functions on zero and/or arbitrary
# arguments...  OR, just write our own AST classes (possibly having a
# Sympy compatibility layer...)

empty_list = Symbol('[]') # Work around for empty lists: can't have a
                          # nullary function (as discovered and
                          # already delt with above).

empty_tuple = Symbol('()') # Work around for empty tuples (as above).

# ______________________________________________________________________
# Utility (abstract) class definitions

class BinaryOp (sympy.Function):
    '''
    Abstract class that defines a code generation function for binary
    operators.  Subclasses should define a "__op_str__" member.
    '''
    def my_codegen (self, printer):
        assert len(self.args) == 2
        arg_strs = [printer.doprint(arg) for arg in self.args]
        return '(%s %s %s)' % (arg_strs[0], self.__op_str__, arg_strs[1])

# ______________________________________________________________________

class NaryOp (sympy.Function):
    '''
    Abstract class that defines a code generation function for n-ary
    operators.  Subclassess should define a "__op_str__" member.
    '''
    def my_codegen (self, printer):
        return '(%s)' % (
            (' %s ' % self.__op_str__).join((
                    printer.doprint(arg) for arg in self.args)),)

# ______________________________________________________________________

class UnaryOp (sympy.Function):
    '''
    Abstract class that defines a code generation function for unary
    (prefix) operators.  Subclasses should define a "__op_str__"
    member.
    '''
    def my_codegen (self, printer):
        assert len(self.args) == 1
        return '(%s %s)' % (self.__op_str__, printer.doprint(self.args[0]))

# ______________________________________________________________________
# Class (node) definitions

class And (NaryOp):
    __op_str__ = 'and'

# ______________________________________________________________________

class App (sympy.Function):
    '''
    Call a function with the remaining arguments.
    '''
    @classmethod
    def eval (cls, *args, **kws):
        ret_val = None
        assert len(args) > 0
        first_arg_ty = type(args[0])
        if first_arg_ty is Lambda:
            ret_val = args[0](*args[1:])
        elif issubclass(first_arg_ty, sympy.FunctionClass):
            # Function constructor, go ahead and attempt to apply it.
            ret_val = args[0](*args[1:])
        return ret_val

    def my_fmt_tuple (self, printer):
        return (printer.doprint(self.args[0]),
                ', '.join((printer.doprint(arg)
                           for arg in self.args[1:])))

    def my_codegen (self, printer):
        return '%s(%s)' % self.my_fmt_tuple(printer)

# ______________________________________________________________________

class BitwiseAnd (BinaryOp):
    __op_str__ = '&'

# ______________________________________________________________________

class BitwiseNot (UnaryOp):
    __op_str__ = '~'

# ______________________________________________________________________

class BitwiseOr (BinaryOp):
    __op_str__ = '|'

# ______________________________________________________________________

class BitwiseShiftLeft (BinaryOp):
    __op_str__ = '<<'

# ______________________________________________________________________

class BitwiseShiftRight (BinaryOp):
    __op_str__ = '>>'

# ______________________________________________________________________

class BitwiseXor (BinaryOp):
    __op_str__ = '^'

# ______________________________________________________________________

class CompFor (sympy.Function):
    def my_codegen (self, printer):
        assert len(self.args) == 2
        return 'for %s in %s' % tuple((printer.doprint(arg)
                                       for arg in self.args))

# ______________________________________________________________________

class CompIf (sympy.Function):
    def my_codegen (self, printer):
        assert len(self.args) == 1
        return 'if ' + printer.doprint(self.args[0])

# ______________________________________________________________________

class Equals (BinaryOp):
    __op_str__ = '=='

# ______________________________________________________________________

class GetAttr (sympy.Function):
    def my_codegen (self, printer):
        assert len(self.args) == 2
        attr_sym = self.args[1]
        assert attr_sym.is_Symbol and attr_sym.name[0] == '$'
        return 'getattr(%s, "%s")' % (printer.doprint(self.args[0]),
                                      attr_sym.name[1:])

# ______________________________________________________________________

class Greater (BinaryOp):
    '''
    args[0] > args[1]
    '''
    __op_str__ = '>'

# ______________________________________________________________________

class GreaterEq (BinaryOp):
    '''
    args[0] >= args[1]
    '''
    __op_str__ = '>='

# ______________________________________________________________________

class IDiv (BinaryOp):
    '''
    args[0] // args[1]
    '''
    __op_str__ = '//'

# ______________________________________________________________________

class If (sympy.Function):
    @classmethod
    def eval (cls, *args, **kws):
        ret_val = None
        assert len(args) == 3
        pred_arg = args[0]
        if isinstance(pred_arg, bool):
            if pred_arg:
                ret_val = args[1]
            else:
                ret_val = args[2]
        return ret_val

    def my_codegen (self, printer):
        assert len(self.args) == 3
        args_strs = tuple((printer.doprint(arg) for arg in self.args))
        return '(%s if %s else %s)' % (args_strs[1], args_strs[0],
                                       args_strs[2])

# ______________________________________________________________________

class Lesser (BinaryOp):
    '''
    args[0] < args[1]
    '''
    __op_str__ = '<'

# ______________________________________________________________________

class LesserEq (BinaryOp):
    '''
    args[0] <= args[1]
    '''
    __op_str__ = '<='

# ______________________________________________________________________

class Let (sympy.Function):
    """
    This constructor should desugar as follows:

    Let(sym, exp0, ..., exp1) ==> Lambda(sym, Let(..., exp1))(exp0)

    Let(exp1) ==> exp1
    """
    @classmethod
    def eval (cls, *args, **kws):
        ret_val = None
        assert len(args) % 2 == 1
        arg_len = len(args)
        if arg_len == 1:
            ret_val = args[0]
        #elif arg_len == 2:
        #    vals, exp = args
        #    if len(vals) == 0:
        #        ret_val = exp
        #    else:
        #        name, subexp = vals[0]
        #        ret_val = cls(name, subexp, cls(vals[1:], exp))
        return ret_val

    def desugar (self):
        assert len(self.args) % 2 == 1
        def rec_desugar (args):
            if len(args) == 1:
                return args[0]
            else:
                if type(args[0]) == Tuple:
                    params = args[0].args
                    lamb_args = list(params)
                    eval_args = tuple((Nth(args[1], index)
                                       for index in xrange(len(params))))
                else:
                    lamb_args = [args[0]]
                    eval_args = (args[1],)
                return Lambda(lamb_args, rec_desugar(args[2:]))(*eval_args)
        return rec_desugar(self.args)

# ______________________________________________________________________

class List (sympy.Function):
    '''
    Atomic list.
    '''
    def my_codegen (self, printer):
        return '[%s]' % (','.join((printer.doprint(arg)
                                   for arg in self.args)))

# ______________________________________________________________________

class ListComp (sympy.Function):
    '''
    List comprehension.
    '''
    def my_codegen (self, printer):
        return '[%s %s]' % (printer.doprint(self.args[0]),
                            ' '.join(printer.doprint(arg)
                                     for arg in self.args[1:]))

# ______________________________________________________________________

class Mod (BinaryOp):
    '''
    Modulus operator.
    '''
    __op_str__ = '%'

# ______________________________________________________________________

class Nequals (BinaryOp):
    __op_str__ = '!='

# ______________________________________________________________________

class Not (UnaryOp):
    __op_str__ = 'not'

# ______________________________________________________________________

class Nth (sympy.Function):
    '''
    Accessor for the n-th element in a Tuple.
    '''
    def my_codegen (self, printer):
        lhs = self.args[0]
        rhs = self.args[1:]
        return '%s[%s]' % (printer.doprint(lhs),
                           ', '.join((printer.doprint(rhs_child)
                                      for rhs_child in rhs)))

# ______________________________________________________________________

class Or (NaryOp):
    __op_str__ = 'or'

# ______________________________________________________________________

class PartialApp (App):
    def my_codegen (self, printer):
        # partial will come from functools in a generated prelude.
        return 'partial(%s, %s)' % self.my_fmt_tuple(printer)

# ______________________________________________________________________

class Slice (sympy.Function):
    '''
    Binary or ternary slice operator.
    '''
    def my_codegen (self, printer):
        arg_strs = ['None' if arg == empty else printer.doprint(arg)
                    for arg in self.args]
        return 'slice(%s)' % (','.join(arg_strs))

# ______________________________________________________________________

class Tuple (sympy.Function):
    '''
    Tuple constructor.
    '''
    def __len__ (self):
        return len(self.args)

    def __iter__ (self):
        return iter(self.args)

    def my_codegen (self, printer):
        if len(self.args) == 1:
            return "(%s,)" % (printer.doprint(self.args[0]))
        else:
            return "(%s)" % (', '.join((printer.doprint(arg)
                                        for arg in self.args)))

# ______________________________________________________________________
# Main (self-test) routine

def main (*args):
    pass

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of rofl_ast.py
