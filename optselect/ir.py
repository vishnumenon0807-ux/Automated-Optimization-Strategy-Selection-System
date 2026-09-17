"""Three-address code intermediate representation.

An operand is either a ``str`` (a variable or temporary name) or a Python
``int``/``float`` (a literal).  Every instruction is one of a small number of
kinds; keeping the IR structured rather than string-based is what lets the
optimization passes reason about it without re-parsing text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Union

Operand = Union[str, int, float]

BINOPS = {'+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', '!=', '&&', '||'}
UNOPS = {'-', '!'}
COMMUTATIVE = {'+', '*', '==', '!=', '&&', '||'}
RELOPS = {'<', '>', '<=', '>=', '==', '!='}

# Instruction kinds
ASSIGN = 'assign'    # dst = arg1
BINOP = 'binop'      # dst = arg1 op arg2
UNOP = 'unop'        # dst = op arg1
LABEL = 'label'      # label:
GOTO = 'goto'        # goto target
IFGOTO = 'ifgoto'    # if arg1 goto target
IFFALSE = 'iffalse'  # ifFalse arg1 goto target
PRINT = 'print'      # print arg1
READ = 'read'        # dst = next input value


def is_const(o) -> bool:
    return isinstance(o, (int, float)) and not isinstance(o, bool)


def is_var(o) -> bool:
    return isinstance(o, str)


def truthy(v) -> bool:
    return v != 0


def eval_binop(op: str, a, b):
    """Single source of truth for binary-operator semantics.

    Both the constant folder and the reference interpreter call this, so an
    optimization can never silently disagree with execution.  Integer division
    and modulo truncate toward zero (C semantics).  Division or modulo by zero
    is *defined* to yield 0 so that every program in the corpus is total and
    two versions of a program can always be compared.
    """
    both_int = isinstance(a, int) and isinstance(b, int)
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        if b == 0:
            return 0
        if both_int:
            q = abs(a) // abs(b)
            return q if (a >= 0) == (b >= 0) else -q
        return a / b
    if op == '%':
        if b == 0:
            return 0
        if both_int:
            r = abs(a) % abs(b)
            return r if a >= 0 else -r
        return math.fmod(a, b)
    if op == '<':
        return int(a < b)
    if op == '>':
        return int(a > b)
    if op == '<=':
        return int(a <= b)
    if op == '>=':
        return int(a >= b)
    if op == '==':
        return int(a == b)
    if op == '!=':
        return int(a != b)
    if op == '&&':
        return int(truthy(a) and truthy(b))
    if op == '||':
        return int(truthy(a) or truthy(b))
    raise ValueError(f'unknown binary operator {op!r}')


def eval_unop(op: str, a):
    if op == '-':
        return -a
    if op == '!':
        return int(not truthy(a))
    raise ValueError(f'unknown unary operator {op!r}')


@dataclass
class Instr:
    kind: str
    dst: Optional[str] = None
    op: Optional[str] = None
    arg1: Optional[Operand] = None
    arg2: Optional[Operand] = None
    target: Optional[str] = None   # goto / ifgoto / iffalse destination
    name: Optional[str] = None     # label name

    # ---- queries -------------------------------------------------------
    def uses(self) -> List[str]:
        """Variable names read by this instruction."""
        out = []
        for a in (self.arg1, self.arg2):
            if is_var(a):
                out.append(a)
        return out

    def defines(self) -> Optional[str]:
        """Variable name written by this instruction, if any."""
        if self.kind in (ASSIGN, BINOP, UNOP, READ):
            return self.dst
        return None

    def has_side_effect(self) -> bool:
        """True if the instruction must be kept even when its result is dead."""
        return self.kind in (PRINT, READ, GOTO, IFGOTO, IFFALSE, LABEL)

    def is_jump(self) -> bool:
        return self.kind in (GOTO, IFGOTO, IFFALSE)

    def is_computation(self) -> bool:
        return self.kind in (ASSIGN, BINOP, UNOP)

    def copy(self) -> 'Instr':
        return replace(self)

    # ---- rendering -----------------------------------------------------
    def __str__(self) -> str:
        def f(o):
            if is_const(o):
                if isinstance(o, float) and o.is_integer():
                    return f'{o:.1f}'
                return str(o)
            return str(o)

        if self.kind == LABEL:
            return f'{self.name}:'
        if self.kind == GOTO:
            return f'    goto {self.target}'
        if self.kind == IFGOTO:
            return f'    if {f(self.arg1)} goto {self.target}'
        if self.kind == IFFALSE:
            return f'    ifFalse {f(self.arg1)} goto {self.target}'
        if self.kind == PRINT:
            return f'    print {f(self.arg1)}'
        if self.kind == READ:
            return f'    {self.dst} = read()'
        if self.kind == ASSIGN:
            return f'    {self.dst} = {f(self.arg1)}'
        if self.kind == UNOP:
            return f'    {self.dst} = {self.op}{f(self.arg1)}'
        if self.kind == BINOP:
            return f'    {self.dst} = {f(self.arg1)} {self.op} {f(self.arg2)}'
        return f'    <{self.kind}>'


# ---- convenience constructors -----------------------------------------
def assign(dst, src):
    return Instr(ASSIGN, dst=dst, arg1=src)


def binop(dst, a, op, b):
    return Instr(BINOP, dst=dst, op=op, arg1=a, arg2=b)


def unop(dst, op, a):
    return Instr(UNOP, dst=dst, op=op, arg1=a)


def label(name):
    return Instr(LABEL, name=name)


def goto(target):
    return Instr(GOTO, target=target)


def ifgoto(cond, target):
    return Instr(IFGOTO, arg1=cond, target=target)


def iffalse(cond, target):
    return Instr(IFFALSE, arg1=cond, target=target)


def prnt(a):
    return Instr(PRINT, arg1=a)


def read(dst):
    return Instr(READ, dst=dst)


class TACProgram:
    """A flat list of instructions plus the counters used to mint new names."""

    def __init__(self, instrs: Optional[List[Instr]] = None,
                 ntemp: int = 0, nlabel: int = 0, source: str = ''):
        self.instrs: List[Instr] = instrs or []
        self.ntemp = ntemp
        self.nlabel = nlabel
        self.source = source

    def new_temp(self) -> str:
        self.ntemp += 1
        return f't{self.ntemp}'

    def new_label(self) -> str:
        self.nlabel += 1
        return f'L{self.nlabel}'

    def emit(self, instr: Instr) -> Instr:
        self.instrs.append(instr)
        return instr

    def clone(self) -> 'TACProgram':
        return TACProgram([i.copy() for i in self.instrs],
                          self.ntemp, self.nlabel, self.source)

    def size(self) -> int:
        """Instruction count.  Labels are pseudo-instructions and are excluded
        so that the metric measures real executable code."""
        return sum(1 for i in self.instrs if i.kind != LABEL)

    def temps(self):
        names = set()
        for i in self.instrs:
            for n in i.uses():
                names.add(n)
            d = i.defines()
            if d:
                names.add(d)
        return {n for n in names if n.startswith('t') and n[1:].isdigit()}

    def variables(self):
        names = set()
        for i in self.instrs:
            for n in i.uses():
                names.add(n)
            d = i.defines()
            if d:
                names.add(d)
        return names

    def __str__(self) -> str:
        return '\n'.join(str(i) for i in self.instrs)

    def numbered(self) -> str:
        out, n = [], 0
        for i in self.instrs:
            if i.kind == LABEL:
                out.append(f'      {i}')
            else:
                n += 1
                out.append(f'{n:4d}  {i}')
        return '\n'.join(out)
