"""Front end: lexer, recursive-descent parser and three-address code generator.

The accepted language is a small C-like subset:

    int i; float x;
    read(i); x = i * 2 + 1;
    if (i < 5) { ... } else { ... }
    while (i < 10) { ... }
    for (i = 0; i < 10; i = i + 1) { ... }
    print(x);

Expressions are side-effect free (no calls, no assignment-expressions), so the
logical operators ``&&`` and ``||`` are evaluated without short-circuiting.
That is semantically indistinguishable here and keeps the generated TAC free of
control flow inside expressions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from . import ir
from .ir import TACProgram

KEYWORDS = {'int', 'float', 'if', 'else', 'while', 'for', 'print', 'read'}
_PUNCT = ['<=', '>=', '==', '!=', '&&', '||',
          '+', '-', '*', '/', '%', '<', '>', '=', '!',
          '(', ')', '{', '}', ';', ',']


class CompileError(Exception):
    pass


# ---------------------------------------------------------------- lexer
@dataclass
class Token:
    kind: str        # 'num' | 'id' | 'kw' | 'punct' | 'eof'
    value: object
    line: int


def tokenize(src: str) -> List[Token]:
    toks: List[Token] = []
    i, line, n = 0, 1, len(src)
    while i < n:
        c = src[i]
        if c == '\n':
            line += 1
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            while i < n and src[i] != '\n':
                i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == '.':
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
                toks.append(Token('num', float(src[i:j]), line))
            else:
                toks.append(Token('num', int(src[i:j]), line))
            i = j
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < n and (src[j].isalnum() or src[j] == '_'):
                j += 1
            word = src[i:j]
            toks.append(Token('kw' if word in KEYWORDS else 'id', word, line))
            i = j
            continue
        for p in _PUNCT:
            if src.startswith(p, i):
                toks.append(Token('punct', p, line))
                i += len(p)
                break
        else:
            raise CompileError(f'line {line}: unexpected character {c!r}')
    toks.append(Token('eof', None, line))
    return toks


# ------------------------------------------------------------------ AST
@dataclass
class Num:
    value: Union[int, float]


@dataclass
class Var:
    name: str


@dataclass
class Bin:
    op: str
    left: object
    right: object


@dataclass
class Un:
    op: str
    operand: object


@dataclass
class Decl:
    typ: str
    name: str
    init: Optional[object]


@dataclass
class Assign:
    name: str
    expr: object


@dataclass
class If:
    cond: object
    then: object
    els: Optional[object]


@dataclass
class While:
    cond: object
    body: object


@dataclass
class For:
    init: object
    cond: object
    step: object
    body: object


@dataclass
class Print:
    expr: object


@dataclass
class Read:
    name: str


@dataclass
class Block:
    stmts: List[object]


# --------------------------------------------------------------- parser
class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    # -- token helpers
    def peek(self) -> Token:
        return self.toks[self.pos]

    def at(self, value) -> bool:
        t = self.peek()
        return t.kind in ('punct', 'kw') and t.value == value

    def accept(self, value) -> bool:
        if self.at(value):
            self.pos += 1
            return True
        return False

    def expect(self, value) -> Token:
        if not self.at(value):
            t = self.peek()
            raise CompileError(f'line {t.line}: expected {value!r}, got {t.value!r}')
        self.pos += 1
        return self.toks[self.pos - 1]

    # -- grammar
    def parse_program(self) -> Block:
        stmts = []
        while self.peek().kind != 'eof':
            stmts.append(self.parse_stmt())
        return Block(stmts)

    def parse_stmt(self):
        if self.at('int') or self.at('float'):
            typ = self.peek().value
            self.pos += 1
            name = self.parse_ident()
            init = None
            if self.accept('='):
                init = self.parse_expr()
            self.expect(';')
            return Decl(typ, name, init)
        if self.at('if'):
            self.pos += 1
            self.expect('(')
            cond = self.parse_expr()
            self.expect(')')
            then = self.parse_stmt()
            els = self.parse_stmt() if self.accept('else') else None
            return If(cond, then, els)
        if self.at('while'):
            self.pos += 1
            self.expect('(')
            cond = self.parse_expr()
            self.expect(')')
            return While(cond, self.parse_stmt())
        if self.at('for'):
            self.pos += 1
            self.expect('(')
            init = self.parse_simple_assign()
            self.expect(';')
            cond = self.parse_expr()
            self.expect(';')
            step = self.parse_simple_assign()
            self.expect(')')
            return For(init, cond, step, self.parse_stmt())
        if self.at('read'):
            self.pos += 1
            self.expect('(')
            name = self.parse_ident()
            self.expect(')')
            self.expect(';')
            return Read(name)
        if self.at('print'):
            self.pos += 1
            self.expect('(')
            e = self.parse_expr()
            self.expect(')')
            self.expect(';')
            return Print(e)
        if self.at('{'):
            self.pos += 1
            stmts = []
            while not self.at('}'):
                if self.peek().kind == 'eof':
                    raise CompileError('unterminated block')
                stmts.append(self.parse_stmt())
            self.expect('}')
            return Block(stmts)
        st = self.parse_simple_assign()
        self.expect(';')
        return st

    def parse_simple_assign(self) -> Assign:
        name = self.parse_ident()
        self.expect('=')
        return Assign(name, self.parse_expr())

    def parse_ident(self) -> str:
        t = self.peek()
        if t.kind != 'id':
            raise CompileError(f'line {t.line}: expected identifier, got {t.value!r}')
        self.pos += 1
        return t.value

    # expression precedence climbing
    _LEVELS = [['||'], ['&&'], ['==', '!='], ['<', '>', '<=', '>='],
               ['+', '-'], ['*', '/', '%']]

    def parse_expr(self, level: int = 0):
        if level == len(self._LEVELS):
            return self.parse_unary()
        node = self.parse_expr(level + 1)
        while True:
            for op in self._LEVELS[level]:
                if self.at(op):
                    self.pos += 1
                    node = Bin(op, node, self.parse_expr(level + 1))
                    break
            else:
                return node

    def parse_unary(self):
        if self.at('-'):
            self.pos += 1
            return Un('-', self.parse_unary())
        if self.at('!'):
            self.pos += 1
            return Un('!', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        t = self.peek()
        if t.kind == 'num':
            self.pos += 1
            return Num(t.value)
        if t.kind == 'id':
            self.pos += 1
            return Var(t.value)
        if self.at('('):
            self.pos += 1
            e = self.parse_expr()
            self.expect(')')
            return e
        raise CompileError(f'line {t.line}: unexpected token {t.value!r}')


def parse(src: str) -> Block:
    return Parser(tokenize(src)).parse_program()


# ------------------------------------------------------- TAC generation
class TACGenerator:
    """Syntax-directed translation of the AST into three-address code.

    The generator is deliberately naive: it introduces a fresh temporary for
    every interior expression node and performs no folding or reuse.  All
    improvement must therefore come from the optimization passes, which is what
    makes the strategy-selection experiment meaningful.
    """

    def __init__(self):
        self.prog = TACProgram()

    def generate(self, ast: Block, source: str = '') -> TACProgram:
        self.prog = TACProgram(source=source)
        self.stmt(ast)
        return self.prog

    # -- statements
    def stmt(self, node):
        if isinstance(node, Block):
            for s in node.stmts:
                self.stmt(s)
        elif isinstance(node, Decl):
            if node.init is not None:
                self.prog.emit(ir.assign(node.name, self.expr(node.init)))
        elif isinstance(node, Assign):
            self.prog.emit(ir.assign(node.name, self.expr(node.expr)))
        elif isinstance(node, Print):
            self.prog.emit(ir.prnt(self.expr(node.expr)))
        elif isinstance(node, Read):
            self.prog.emit(ir.read(node.name))
        elif isinstance(node, If):
            self.gen_if(node)
        elif isinstance(node, While):
            self.gen_while(node)
        elif isinstance(node, For):
            self.gen_for(node)
        else:
            raise CompileError(f'cannot generate code for {node!r}')

    def gen_if(self, node: If):
        p = self.prog
        cond = self.expr(node.cond)
        l_else = p.new_label()
        p.emit(ir.iffalse(cond, l_else))
        self.stmt(node.then)
        if node.els is not None:
            l_end = p.new_label()
            p.emit(ir.goto(l_end))
            p.emit(ir.label(l_else))
            self.stmt(node.els)
            p.emit(ir.label(l_end))
        else:
            p.emit(ir.label(l_else))

    def gen_while(self, node: While):
        p = self.prog
        l_top, l_end = p.new_label(), p.new_label()
        p.emit(ir.label(l_top))
        cond = self.expr(node.cond)
        p.emit(ir.iffalse(cond, l_end))
        self.stmt(node.body)
        p.emit(ir.goto(l_top))
        p.emit(ir.label(l_end))

    def gen_for(self, node: For):
        p = self.prog
        self.stmt(node.init)
        l_top, l_end = p.new_label(), p.new_label()
        p.emit(ir.label(l_top))
        cond = self.expr(node.cond)
        p.emit(ir.iffalse(cond, l_end))
        self.stmt(node.body)
        self.stmt(node.step)
        p.emit(ir.goto(l_top))
        p.emit(ir.label(l_end))

    # -- expressions: return an operand (variable name or literal)
    def expr(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Un):
            a = self.expr(node.operand)
            t = self.prog.new_temp()
            self.prog.emit(ir.unop(t, node.op, a))
            return t
        if isinstance(node, Bin):
            a = self.expr(node.left)
            b = self.expr(node.right)
            t = self.prog.new_temp()
            self.prog.emit(ir.binop(t, a, node.op, b))
            return t
        raise CompileError(f'cannot generate code for expression {node!r}')


def compile_to_tac(src: str) -> TACProgram:
    """Source text -> three-address code."""
    return TACGenerator().generate(parse(src), source=src)
