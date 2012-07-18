#! /usr/bin/env python
# ______________________________________________________________________
"""frontend.py

Defines the Resilient Optimizing Flow Language front end.
"""
# ______________________________________________________________________
# Module imports

import token

from basil.parsing import PgenParser, PyPgen
from basil.lang.python import DFAParser
from basil.utils.Handler import Handler

import sympy
import rofl_ast

# ______________________________________________________________________
# Module data

# Original starting point for this was the Python 2.3.5 grammar.
grammar_src = """
start: (NEWLINE | defn)* ENDMARKER
defn: defn_lhs (augassign | '=') pyexpr NEWLINE [where_clause]
where_clause: 'where' ':' defns
defns: defn | NEWLINE INDENT (NEWLINE | defn)* DEDENT
defn_lhs: NAME [ '(' [ NAME (',' NAME)* ] ')' | (',' NAME)+ ]

# expr_stmt: testlist (augassign testlist | ('=' testlist)*)

# (which inspired):

# defn: testlist (augassign | '=') testlist NEWLINE [where_clause]

# But we don't allow a lot on the LHS of a definition, so we arrive at
# what's above.

augassign: '+=' | '-=' | '*=' | '/=' | '%=' | '&=' | '|=' | '^=' | '<<=' | '>>=' | '**=' | '//='

pyexpr: testlist

test: or_test ['if' or_test 'else' test]
or_test: and_test ('or' and_test)* # XXX | lambdef
and_test: not_test ('and' not_test)*
not_test: 'not' not_test | comparison
comparison: expr (comp_op expr)*
comp_op: '<'|'>'|'=='|'>='|'<='|'<>'|'!='|'in'|'not' 'in'|'is'|'is' 'not'
expr: xor_expr ('|' xor_expr)*
xor_expr: and_expr ('^' and_expr)*
and_expr: shift_expr ('&' shift_expr)*
shift_expr: arith_expr (('<<'|'>>') arith_expr)*
arith_expr: term (('+'|'-') term)*
term: factor (('*'|'/'|'%'|'//') factor)*
factor: ('+'|'-'|'~') factor | power
power: atom trailer* ['**' factor]
atom: '(' [testlist] ')' | '[' [listmaker] ']' | '{' [dictmaker] '}' | '`' testlist1 '`' | NAME | NUMBER | STRING+
listmaker: test ( list_for | (',' test)* [','] )

#lambdef: 'lambda' [varargslist] ':' test

trailer: '(' [arglist] ')' | '[' subscriptlist ']' | '.' NAME
subscriptlist: subscript (',' subscript)* [',']
subscript: '.' '.' '.' | test | [test] ':' [test] [sliceop]
sliceop: ':' [test]
exprlist: expr (',' expr)* [',']
testlist: test (',' test)* [',']
testlist_safe: or_test [(',' or_test)+ [',']]
dictmaker: test ':' test (',' test ':' test)* [',']

#arglist: (argument ',')* (argument [',']| '*' test [',' '**' test] | '**' test)
arglist: (argument ',')* argument [','] #| '*' test [',' '**' test] | '**' test)
argument: [test '='] test	# Really [keyword '='] test

list_iter: list_for | list_if
list_for: 'for' exprlist 'in' testlist_safe [list_iter]
list_if: 'if' or_test [list_iter]

testlist1: test (',' test)*
"""

# ______________________________________________________________________

# Original TEST_SRC moved to .../rofl/test/em_noise.rofl

TEST_SRC_ALT = '''
fermi_func_deriv (x, TmK) = (((-4. * kbT) ** -1.) *
                             (cosh(x/(2.*kbT)) ** -2.))
where:
    kbT = 8.6173743e-5 * (TmK /1000.) * 1000.

didv_with_noise(ag, Vg, ads, Vds, sigVg, sigVds, TmK) = -dxdy * Z.sum()
where:
    X, Y = mesh(bounds(-5, 5 + epsilon, 0.025), bounds(-5, 5 + epsilon, 0.025))
    X1 = -ag * (Vg + (sigVg * X))
    Y1 = Vds + (sigVds * Y)
    nads = 1. - ads
    Z = (nads * fermi_func_deriv(X1 + (nads * Y1), TmK))
    Z += ads * fermi_func_deriv(X1 + (-ads * Y1), TmK)
    Z *= (1. /sqrt(2 * pi) ** 2 * exp(-1. * (X ** 2 + Y ** 2) / 2))
    dxdy = xstep(X) * ystep(Y)
'''

# ______________________________________________________________________
# Nastiness not required if Basil development was more active...

def pgen_to_grammar_obj (source):
    '''XXX Stolen from PyCon 2010 sprint sandbox.  Move into Basil proper.'''
    pgen = PyPgen.PyPgen()
    nfa_grammar = pgen.handleStart(PgenParser.parseString(source))
    dfa_grammar = pgen.generateDfaGrammar(nfa_grammar)
    pgen.translateLabels(dfa_grammar)
    pgen.generateFirstSets(dfa_grammar)
    dfa_grammar = DFAParser.addAccelerators(dfa_grammar)
    return dfa_grammar

parser_obj = PyPgen.PyPgenParser(pgen_to_grammar_obj(grammar_src))

MAX_STMT_SYMBOL = parser_obj.stringToSymbolMap()['augassign']
DONT_SIMPLIFY_SYMBOLS = [parser_obj.stringMap[nt_name]
                         for nt_name in ('atom', 'arglist', 'subscriptlist')]

# ______________________________________________________________________
# Handler (visitor) base class definitions

class PgenHandler (Handler):
    "XXX Also stolen from PyCon 2010 sprint sandbox.  Move into Basil proper."
    def __init__ (self, parser_obj):
        self.parser = parser_obj
        self.symbolMap = parser_obj.symbolToStringMap()

    def handle_source (self, source):
        return self.handle_node(self.parser.parseString(source))

    def get_nonterminal (self, node):
        ret_val = None
        if not self.is_token(node):
            ret_val = self.symbolMap[node[0][0]]
        return ret_val

    def get_children (self, node):
        ret_val = []
        if not self.is_token(node):
            ret_val = node[1][:]
        return ret_val

    def is_token (self, node):
        return node[0][0] < token.NT_OFFSET

    def make_node (self, node_id, children):
        return tuple([node_id] + children)

    def handle_default (self, node):
        ret_val = self.handle_children(node)
        return ret_val

# ______________________________________________________________________
# Parse tree visitor

class Handler (PgenHandler):
    def __init__ (self):
        super(Handler, self).__init__(parser_obj)
        self.pt_symbols = parser_obj.symbolToStringMap()
        if __debug__:
            self.debug_node = None
    
    if __debug__:
        def handle_default (self, node):
            self.debug_node = node
            return super(Handler, self).handle_default(node) 

    def get_token_str (self, node):
        return node[0][1]

    def handle_start (self, node):
        'Return a list of definitions, be they functions or whatnot.'
        return [self.handle_defn(child)
                for child in self.get_children(node)
                if not self.is_token(child)]

    def handle_defn (self, node):
        # This guard is used since handle_start() shortcuts the method
        # dispatch table.
        assert self.pt_symbols[node[0][0]] == 'defn'
        children = self.get_children(node)
        lhs, params = self.handle_defn_lhs(children[0])
        rhs = self.handle_node(children[2])
        # Handle possible augassign...
        augassigned = False
        if not self.is_token(children[1]):
            # augassign present...
            augassigned = True
            augassignment = self.get_children(children[1])[0]
            assert params is None, ("Cannot use %s in a function definition." %
                                    self.get_token_str(augassignment))
            rhs = self._handle_augassign(lhs, augassignment, rhs)
        # Handle where clause...
        if not self.is_token(children[-1]):
            # Where clause is present
            defns = self.handle_where_clause(children[-1])
            let_args = []
            for sym, subexp in defns:
                let_args.append(sym)
                let_args.append(subexp)
            let_args.append(rhs)
            rhs = rofl_ast.Let(*let_args)
        # Handle function definition...
        if params is not None:
            rhs = rofl_ast.Lambda(params, rhs)
        return (lhs, rhs)

    def handle_where_clause (self, node):
        children = self.get_children(node)
        return self.handle_defns(children[-1])

    def handle_defns (self, node):
        assert self.pt_symbols[node[0][0]] == 'defns'
        children = self.get_children(node)
        if len(children) == 1:
            ret_val = [self.handle_defn(children[0])]
        else:
            ret_val = [self.handle_defn(child)
                       for child in children
                       if not self.is_token(child)]
        return ret_val

    def handle_defn_lhs (self, node):
        params = None
        children = self.get_children(node)
        lhs = sympy.sympify(children[0][0][1])
        if len(children) > 1:
            second_child_str = children[1][0][1]
            if second_child_str == '(':
                params = [sympy.sympify(child[0][1])
                          for child in children[2:]
                          if child[0][0] == token.NAME]
            else:
                assert second_child_str == ',', "Expected '(' or ','..."
                lhs = rofl_ast.Tuple(*(sympy.sympify(child[0][1])
                                       for child in children
                                       if child[0][0] == token.NAME))
        return lhs, params

    def _handle_op (self, node, op_ctor = None):
        children = self.get_children(node)
        if op_ctor is None:
            op_ctor = lambda *args: tuple(args)
        if len(children) == 1:
            ret_val = self.handle_node(children[0])
        else:
            op_ctor_args = [self.handle_node(child)
                            for child in children
                            if not self.is_token(child)]
            ret_val = op_ctor(*op_ctor_args)
        return ret_val

    def get_operator_token_str (self, node):
        if self.is_token(node):
            ret_val = self.get_token_str(node)
        else:
            ret_val = ' '.join((self.get_operator_token_str(child)
                                for child in self.get_children(node)))
        return ret_val

    def _handle_binary_ops (self, node, op_ctor_dispatch = None):
        if op_ctor_dispatch is None:
            op_ctor_dispatch = {}
        children = self.get_children(node)
        if len(children) == 1:
            ret_val = self.handle_node(children[0])
        else:
            ret_val = self.handle_node(children.pop(0))
            while len(children):
                op_tok_str = self.get_operator_token_str(children.pop(0))
                op_ctor = op_ctor_dispatch.get(op_tok_str,
                                               lambda lhs, rhs:
                                                   (op_tok_str, lhs, rhs))
                ret_val = op_ctor(ret_val, self.handle_node(children.pop(0)))
        return ret_val

    def _handle_not_implemented (self, *args, **kws):
        raise NotImplementedError()

    augassign_dispatch = {
        '+=' : rofl_ast.Add,
        '*=' : rofl_ast.Mul,
        '/=' : '_build_div',
        '//=' : rofl_ast.IDiv,
        '-=' : '_build_sub',
        }

    def _handle_augassign (self, lhs, node, rhs):
        assert self.is_token(node)
        augassign_str = self.get_token_str(node)
        augassign_rhs_ctor = self.augassign_dispatch.get(augassign_str,
                                              self._handle_not_implemented)
        if isinstance(augassign_rhs_ctor, str):
            augassign_rhs_ctr = getattr(self, augassign_rhs_ctor)
        return augassign_rhs_ctor(lhs, rhs)

    def handle_augassign (self, node):
        raise NotImplementedError("This should not be called and was "
                                  "therefore not implemented.")

    def handle_test (self, node):
        children = self.get_children(node)
        if len(children) == 1:
            ret_val = self.handle_node(children[0])
        else:
            assert len(children) == 5
            true_expr = self.handle_node(children[0])
            assert self.is_token(children[1])
            pred_expr = self.handle_node(children[2])
            assert self.is_token(children[3])
            false_expr = self.handle_node(children[4])
            ret_val = rofl_ast.If(pred_expr, true_expr, false_expr)
        return ret_val

    def handle_or_test (self, node):
        return self._handle_op(node, rofl_ast.Or)

    def handle_and_test (self, node):
        return self._handle_op(node, rofl_ast.And)

    def handle_not_test (self, node):
        return self._handle_op(node, rofl_ast.Not)

    def handle_comparison (self, node):
        return self._handle_binary_ops(node,
                                       {'<'  : rofl_ast.Lesser,
                                        '>'  : rofl_ast.Greater,
                                        '==' : rofl_ast.Equals,
                                        '>=' : rofl_ast.GreaterEq,
                                        '<=' : rofl_ast.LesserEq,
                                        '<>' : rofl_ast.Nequals,
                                        '!=' : rofl_ast.Nequals,
                                        }) # XXX
    #                                    'in' : FIXME,
    #                                    'not in' : FIXME,
    #                                    'is' : FIXME,
    #                                    'is not' : FIXME})

    def handle_comp_op (self, node):
        raise NotImplementedError("No IR code to be generated for lone "
                                  "comparison operator.")

    def handle_expr (self, node):
        return self._handle_op(node, rofl_ast.BitwiseOr)

    def handle_xor_expr (self, node):
        return self._handle_op(node, rofl_ast.BitwiseXor)

    def handle_and_expr (self, node):
        return self._handle_op(node, rofl_ast.BitwiseAnd)

    def handle_shift_expr (self, node):
        return self._handle_binary_ops(node,
                                       {"<<": rofl_ast.BitwiseShiftLeft,
                                        ">>": rofl_ast.BitwiseShiftRight})

    def _build_sub (self, lhs, rhs):
        return rofl_ast.Add(lhs, rofl_ast.Mul(-1, rhs))

    def handle_arith_expr (self, node):
        return self._handle_binary_ops(node,
                                       {"+" : rofl_ast.Add,
                                        "-" : self._build_sub})

    def _build_div (self, lhs, rhs):
        return rofl_ast.Mul(lhs, rofl_ast.Pow(rhs, -1))

    def handle_term (self, node):
        return self._handle_binary_ops(node,
                                       {"*" : rofl_ast.Mul,
                                        "/" : self._build_div,
                                        "%" : rofl_ast.Mod,
                                        "//" : rofl_ast.IDiv,
                                        })

    def handle_factor (self, node):
        children = self.get_children(node)
        if len(children) == 2:
            first_child_str = self.get_token_str(children[0])
            ret_val = self.handle_node(children[1])
            if first_child_str == "-":
                ret_val *= -1
            elif first_child_str == "~":
                ret_val = rofl_ast.BitwiseNot(ret_val)
        else:
            assert len(children) == 1
            ret_val = self.handle_node(children[0])
        return ret_val

    def handle_power (self, node):
        children = self.get_children(node)
        ret_val = self.handle_node(children.pop(0))
        while ((len(children) > 0) and
               (self.pt_symbols.get(children[0][0][0]) == 'trailer')):
            trailer_ctor = self.handle_trailer(children.pop(0))
            ret_val = trailer_ctor(ret_val)
        if len(children) > 0:
            assert self.is_token(children[0]) and len(children) == 2
            pow_rhs = self.handle_node(children[1])
            ret_val = rofl_ast.Pow(ret_val, pow_rhs)
        return ret_val

    def handle_pyexpr (self, node):
        children = self.get_children(node)
        assert len(children) == 1
        return self.handle_node(children[0])

    def handle_atom (self, node):
        children = self.get_children(node)
        if len(children) == 1:
            child = children[0]
            assert self.is_token(child)
            if child[0][0] == token.NAME:
                ret_val = rofl_ast.Symbol(self.get_token_str(child))
            elif child[0][0] == token.NUMBER:
                ret_val = sympy.sympify(self.get_token_str(child))
            else:
                raise NotImplementedError("Don't know how to handle %s token "
                                          "literals." % child[0])
        elif children[0][0][0] == token.LPAR:
            if len(children) == 3:
                ret_val = self.handle_node(children[1])
            else:
                ret_val = rofl_ast.empty_tuple
        elif children[0][0][0] == token.LSQB:
            if len(children) == 3:
                ret_val = self.handle_node(children[1])
            else:
                ret_val = rofl_ast.empty_list
        else:
            raise NotImplementedError(repr(node))
            ret_val = self.handle_default(node)
        return ret_val

    def handle_listmaker (self, node):
        children = self.get_children(node)
        head = self.handle_node(children[0])
        if len(children) > 1:
            snd = children[1]
            if self.is_token(snd):
                ret_val = rofl_ast.List(head, *[self.handle_node(child)
                                                for child in children[2:]
                                                if not self.is_token(child)])
            else:
                ret_val = rofl_ast.ListComp(head, *self.handle_node(snd))
        else:
            ret_val = rofl_ast.List(head)
        return ret_val


    def _build_app (self, rhs):
        def _build_app_from_lhs (lhs):
            return rofl_ast.App(lhs, *rhs)
        return _build_app_from_lhs

    def _build_nth (self, rhs):
        def _build_nth_from_lhs (lhs):
            return rofl_ast.Nth(lhs, *rhs)
        return _build_nth_from_lhs

    def _build_getattr (self, rhs):
        def _build_getattr_from_lhs (lhs):
            return rofl_ast.GetAttr(lhs, rhs)
        return _build_getattr_from_lhs

    def handle_trailer (self, node):
        children = self.get_children(node)
        assert self.is_token(children[0])
        first_child_str = self.get_token_str(children[0])
        if first_child_str == '(':
            # Function call...
            app_args = []
            if len(children) > 2:
                app_args = self.handle_node(children[1])
            ret_val = self._build_app(app_args)
        elif first_child_str == '[':
            #raise NotImplementedError('Indexing not currently supported.')
            lookup = self.handle_subscriptlist(children[1])
            ret_val = self._build_nth(lookup)
        elif first_child_str == '.':
            # The following prevents the sympification of the attribute name.
            attr_name = rofl_ast.Symbol('$' + self.get_token_str(children[1]))
            ret_val = self._build_getattr(attr_name)
        return ret_val

    def handle_subscriptlist (self, node):
        children = self.get_children(node)
        return tuple((self.handle_node(child) for child in children
                      if not self.is_token(child)))

    def handle_subscript (self, node):
        children = self.get_children(node)
        child_count = len(children)
        if child_count == 1:
            # This should be a test node.
            ret_val = self.handle_node(children[0])
        elif (self.is_token(children[0]) and
              (self.get_token_str(children[0]) == '.')):
            ret_val = rofl_ast.ellipsis
        else:
            index = 1
            arg0 = rofl_ast.empty
            arg1 = rofl_ast.empty
            has_arg2 = False
            arg2 = None
            if not self.is_token(children[0]):
                arg0 = self.handle_node(children[0])
                index = 2
            if index < child_count:
                if self.get_nonterminal(children[index]) == 'test':
                    arg1 = self.handle_test(children[index])
                    index += 1
                if index < child_count:
                    arg2 = self.handle_node(children[index])
                    has_arg2 = True
            if has_arg2:
                ret_val = rofl_ast.Slice(arg0, arg1, arg2)
            else:
                ret_val = rofl_ast.Slice(arg0, arg1)
        return ret_val

    def handle_sliceop (self, node):
        children = self.get_children(node)
        ret_val = rofl_ast.empty
        if len(children) > 1:
            ret_val = self.handle_node(children[1])
        return ret_val

    def _handle_list_stuff (self, node):
        children = self.get_children(node)
        if len(children) == 1:
            ret_val = self.handle_node(children[0])
        else:
            ret_val = rofl_ast.Tuple(*(self.handle_node(child)
                                       for child in children
                                       if not self.is_token(child)))
        return ret_val

    def handle_exprlist (self, node):
        return self._handle_list_stuff(node)

    def handle_testlist (self, node):
        return self._handle_list_stuff(node)

    def handle_testlist_safe (self, node):
        return self._handle_list_stuff(node)

    def handle_dictmaker (self, node):
        return self.handle_default(node)

    def handle_arglist (self, node):
        return [self.handle_node(child)
                for child in self.get_children(node)
                if not self.is_token(child)]

    def handle_argument (self, node):
        children = self.get_children(node)
        if len(children) == 1:
            ret_val = self.handle_node(children[0])
        else:
            lhs = self.handle_node(children[0])
            rhs = self.handle_node(children[2])
            # XXX Note that the backend has no freaking clue how to
            # support this notation...
            ret_val = (lhs, '=', rhs)
        return ret_val

    def handle_list_iter (self, node):
        children = self.get_children(node)
        assert len(children) == 1
        return self.handle_node(children[0])

    def handle_list_for (self, node):
        children = self.get_children(node)
        target = self.handle_node(children[1])
        iter_exp = self.handle_node(children[3])
        ret_val = [rofl_ast.CompFor(target, iter_exp)]
        if len(children) > 4:
            ret_val += self.handle_node(children[4])
        return ret_val

    def handle_list_if (self, node):
        children = self.get_children(node)
        ret_val = [rofl_ast.CompIf(self.handle_node(children[1]))]
        if len(children) > 2:
            ret_val += self.handle_node(children[2])
        return ret_val

    def handle_testlist1 (self, node):
        return self.handle_default(node)

# ______________________________________________________________________
# Function definitions

def concrete_parse_string (src):
    return parser_obj.parseString(src)

# ______________________________________________________________________ 

def concrete_parse_file (filename):
    return concrete_parse_string(open(filename).read())

# ______________________________________________________________________

def parse_string (src, **kws):
    if type(src) != str:
        src = str(src)
    if src[-1] != '\n':
        src += '\n'
    pt = concrete_parse_string(src)
    return Handler().handle_start(pt)

# ______________________________________________________________________

def parse_file (fileobj, **kws):
    if type(fileobj) == str:
        fileobj = open(fileobj)
    # XXX Wrap any errors from the parser into something that makes
    # sense in the context of a file.
    try:
        fileobj_text = fileobj.read()
    finally:
        fileobj.close()
    return parse_string(fileobj_text)

# ______________________________________________________________________

def _really_simplify (node):
    if (node[0][0] not in DONT_SIMPLIFY_SYMBOLS) and len(node[1]) == 1:
        return _really_simplify(node[1][0])
    else:
        return (node[0], [_really_simplify(child) for child in node[1]])

# ______________________________________________________________________

def simplify_pt (pt):
    if pt[0][0] > MAX_STMT_SYMBOL:
        return _really_simplify(pt)
    else:
        return (pt[0], [simplify_pt(child) for child in pt[1]])

# ______________________________________________________________________

def get_main_function (functions):
    '''Given the output of the ROFL front-end parser, return the
    binding 2-tuple, (fn_name_symbol, bound_val), of the main function in the
    parse result.'''
    return functions[-1]

# ______________________________________________________________________

def get_main_function_name (functions):
    '''Given the output of the ROFL front-end parser, return the
    binding name of the main function in the parse result.'''
    return str(get_main_function(functions)[0])

# ______________________________________________________________________

def get_main_function_parameters (functions):
    '''Given the output of the ROFL front-end parser, return a tuple
    of the formal parameters to the main function of the parse
    result.'''
    bound_val = get_main_function(functions)[1]
    assert ((isinstance(bound_val, rofl_ast.Lambda)) and
            (hasattr(bound_val.args[0], '__len__')))
    return tuple((str(sym) for sym in bound_val.args[0]))

# ______________________________________________________________________
# Main (self-test) routine

def main (*args):
    if len(args) > 0:
        for arg in args:
            fns = parse_file(arg)
    else:
        fns = parse_string(TEST_SRC_ALT)
        for fn_name, fn_lambda in fns:
            print "_" * 60
            print fn_name
            sympy.pprint(fn_lambda)

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of frontend.py
