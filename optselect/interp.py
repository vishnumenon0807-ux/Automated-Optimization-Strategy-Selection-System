"""Reference interpreter for three-address code.

Used only for verification: an optimization is accepted as correct only if the
optimized program produces exactly the same sequence of printed values as the
original.  Because the interpreter and the constant folder share
``ir.eval_binop``, the two can never drift apart on operator semantics.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import ir
from .ir import (ASSIGN, BINOP, GOTO, IFFALSE, IFGOTO, LABEL, PRINT, READ,
                 UNOP, TACProgram, is_const, is_var)

#: Fixed input stream used for every verification run.  Both the original and
#: the optimized program are fed exactly this sequence, so a difference in
#: output can only come from the transformation.
DEFAULT_INPUTS = [7, 3, 12, 5, 9, 2, 14, 6, 11, 4, 8, 13, 1, 10, 15]


class ExecutionLimit(Exception):
    pass


def run(prog: TACProgram, env: Optional[Dict[str, float]] = None,
        max_steps: int = 2_000_000,
        inputs: Optional[List[float]] = None) -> List[float]:
    """Execute the program and return the list of printed values."""
    stream = list(DEFAULT_INPUTS if inputs is None else inputs)
    cursor = [0]
    instrs = prog.instrs
    labels = {ins.name: i for i, ins in enumerate(instrs) if ins.kind == LABEL}
    state: Dict[str, float] = dict(env or {})
    output: List[float] = []

    def val(o):
        if is_const(o):
            return o
        return state.get(o, 0)

    pc, steps = 0, 0
    n = len(instrs)
    while pc < n:
        steps += 1
        if steps > max_steps:
            raise ExecutionLimit('program exceeded the step limit (likely non-terminating)')
        ins = instrs[pc]
        k = ins.kind
        if k == LABEL:
            pc += 1
        elif k == ASSIGN:
            state[ins.dst] = val(ins.arg1)
            pc += 1
        elif k == BINOP:
            state[ins.dst] = ir.eval_binop(ins.op, val(ins.arg1), val(ins.arg2))
            pc += 1
        elif k == UNOP:
            state[ins.dst] = ir.eval_unop(ins.op, val(ins.arg1))
            pc += 1
        elif k == PRINT:
            output.append(val(ins.arg1))
            pc += 1
        elif k == READ:
            state[ins.dst] = stream[cursor[0] % len(stream)] if stream else 0
            cursor[0] += 1
            pc += 1
        elif k == GOTO:
            pc = labels[ins.target]
        elif k == IFGOTO:
            pc = labels[ins.target] if ir.truthy(val(ins.arg1)) else pc + 1
        elif k == IFFALSE:
            pc = pc + 1 if ir.truthy(val(ins.arg1)) else labels[ins.target]
        else:
            raise ValueError(f'cannot execute {ins!r}')
    return output


def outputs_match(a: List[float], b: List[float], tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, float) or isinstance(y, float):
            if abs(float(x) - float(y)) > tol * max(1.0, abs(float(x))):
                return False
        elif x != y:
            return False
    return True
