"""Feature extraction: turn a TAC program into a numeric vector.

``REQUIRED_FEATURES`` are the eight features named in the project
specification.  ``EXTENDED_FEATURES`` adds six more; they exist because the
required set gives constant folding and constant propagation no way to be told
apart (both are driven by ``num_constants``/``const_ratio``), which caps
achievable accuracy.  Both feature sets are evaluated and compared.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from . import ir
from .cfg import CFG, NAC
from .ir import (ASSIGN, BINOP, LABEL, UNOP, COMMUTATIVE, TACProgram,
                 is_const, is_var)
from .optimizations import _canon, _is_temp

REQUIRED_FEATURES = [
    'num_statements',
    'num_constants',
    'num_repeated_expr',
    'num_dead_stmts',
    'num_algebraic_ops',
    'num_loop_ops',
    'num_tmp_vars',
    'const_ratio',
]

EXTENDED_FEATURES = REQUIRED_FEATURES + [
    'num_foldable',
    'num_const_defs',
    'num_prop_sites',
    'num_loop_invariant_ops',
    'max_loop_depth',
    'num_distinct_vars',
]

# arithmetic identity patterns recognised by the algebraic simplifier
def _is_algebraic_identity(ins) -> bool:
    if ins.kind != BINOP:
        return False
    a, b, op = ins.arg1, ins.arg2, ins.op
    if op == '+':
        return (is_const(b) and b == 0) or (is_const(a) and a == 0)
    if op == '-':
        return (is_const(b) and b == 0) or (is_var(a) and a == b) or \
               (is_const(a) and a == 0 and is_var(b))
    if op == '*':
        return (is_const(b) and b in (0, 1)) or (is_const(a) and a in (0, 1))
    if op == '/':
        return is_const(b) and b == 1
    return False


def extract(prog: TACProgram) -> Dict[str, float]:
    """Compute the full feature dictionary for a program."""
    instrs = prog.instrs
    g = CFG(prog)

    num_statements = prog.size()
    denom = max(num_statements, 1)

    # -- lines whose every operand is a numeric literal
    num_constants = 0
    num_foldable = 0
    num_const_defs = 0
    for ins in instrs:
        if ins.kind == LABEL:
            continue
        operands = [o for o in (ins.arg1, ins.arg2) if o is not None]
        if operands and all(is_const(o) for o in operands):
            num_constants += 1
        if ins.kind in (BINOP, UNOP) and operands and all(is_const(o) for o in operands):
            num_foldable += 1
        if ins.kind == ASSIGN and is_const(ins.arg1):
            num_const_defs += 1

    # -- redundant expressions visible to local value numbering
    num_repeated_expr = 0
    for blk in g.blocks:
        avail = {}
        for _k, ins in blk.real_instrs():
            if ins.kind == BINOP:
                key = _canon(ins.op, ins.arg1, ins.arg2)
                if key in avail and avail[key] != ins.dst:
                    num_repeated_expr += 1
            d = ins.defines()
            if d:
                for kk in [k2 for k2, v in avail.items()
                           if v == d or ('v', d) in (k2[1], k2[2])]:
                    del avail[kk]
            if ins.kind == BINOP:
                avail[_canon(ins.op, ins.arg1, ins.arg2)] = ins.dst

    # -- assignments never read afterwards
    _live_in, live_out = g.liveness()
    num_dead_stmts = sum(
        1 for k, ins in enumerate(instrs)
        if ins.is_computation() and not ins.has_side_effect()
        and ins.dst not in live_out.get(k, set()))

    # -- algebraic identities
    num_algebraic_ops = sum(1 for ins in instrs if _is_algebraic_identity(ins))

    # -- loop structure
    loop_idx = g.loop_instruction_indices()
    num_loop_ops = len(loop_idx)
    depths = g.loop_depths()
    max_loop_depth = max(depths.values()) if depths else 0

    # structurally loop-invariant instructions (no dominance/liveness check, so
    # this is a cheap signal rather than an oracle for the LICM pass)
    def_in_loop = defaultdict(int)
    for i in loop_idx:
        d = instrs[i].defines()
        if d:
            def_in_loop[d] += 1
    num_loop_invariant_ops = 0
    for i in loop_idx:
        ins = instrs[i]
        if not ins.is_computation() or ins.has_side_effect():
            continue
        ops = [o for o in (ins.arg1, ins.arg2) if o is not None]
        if ops and all(is_const(o) or def_in_loop.get(o, 0) == 0 for o in ops):
            num_loop_invariant_ops += 1

    # -- sites where a variable use is provably a constant
    env = g.reaching_constants()
    num_prop_sites = 0
    for k, ins in enumerate(instrs):
        state = env.get(k)
        if not state:
            continue
        for o in (ins.arg1, ins.arg2):
            if is_var(o) and state.get(o, NAC) is not NAC:
                num_prop_sites += 1

    temps = prog.temps()
    allvars = prog.variables()

    return {
        'num_statements': float(num_statements),
        'num_constants': float(num_constants),
        'num_repeated_expr': float(num_repeated_expr),
        'num_dead_stmts': float(num_dead_stmts),
        'num_algebraic_ops': float(num_algebraic_ops),
        'num_loop_ops': float(num_loop_ops),
        'num_tmp_vars': float(len(temps)),
        'const_ratio': num_constants / denom,
        'num_foldable': float(num_foldable),
        'num_const_defs': float(num_const_defs),
        'num_prop_sites': float(num_prop_sites),
        'num_loop_invariant_ops': float(num_loop_invariant_ops),
        'max_loop_depth': float(max_loop_depth),
        'num_distinct_vars': float(len(allvars)),
    }


def vector(feats: Dict[str, float], names: List[str]) -> List[float]:
    return [feats[n] for n in names]
