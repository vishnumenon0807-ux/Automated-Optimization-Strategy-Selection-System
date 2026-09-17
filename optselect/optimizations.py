"""The six optimization strategies.

Each pass is *self-contained*: it performs its characteristic transformation
and the immediate clean-up that transformation makes possible, so that applying
it in isolation produces a measurable change.  A pass never runs another pass's
analysis to do that clean-up, which is what keeps the six strategies
distinguishable and makes the selection problem non-trivial.

Every pass has the signature ``pass(prog) -> (new_prog, n_changes)`` and never
mutates its argument.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from . import ir
from .cfg import CFG, NAC
from .ir import (ASSIGN, BINOP, GOTO, IFFALSE, IFGOTO, LABEL, PRINT, UNOP,
                 COMMUTATIVE, Instr, TACProgram, is_const, is_var)

STRATEGIES = [
    'constant_folding',
    'constant_propagation',
    'common_subexpr_elimination',
    'dead_code_elimination',
    'algebraic_simplification',
    'loop_invariant_code_motion',
]

SHORT = {
    'constant_folding': 'CF',
    'constant_propagation': 'CP',
    'common_subexpr_elimination': 'CSE',
    'dead_code_elimination': 'DCE',
    'algebraic_simplification': 'AS',
    'loop_invariant_code_motion': 'LICM',
}

MAX_LOOP_WEIGHT_DEPTH = 3


# ---------------------------------------------------------------- costing
def program_cost(prog: TACProgram) -> int:
    """Loop-weighted static cost: an instruction at nesting depth d counts
    10**d.  This is the usual static estimate of dynamic execution frequency.

    Raw instruction count alone cannot score loop-invariant code motion, which
    relocates instructions without deleting any; the weighted cost makes all
    six strategies comparable on one scale.
    """
    g = CFG(prog)
    depths = g.loop_depths()
    total = 0
    for i, ins in enumerate(prog.instrs):
        if ins.kind == LABEL:
            continue
        total += 10 ** min(depths.get(i, 0), MAX_LOOP_WEIGHT_DEPTH)
    return total


# ---------------------------------------------------------------- helpers
def _def_counts(instrs: List[Instr]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for ins in instrs:
        d = ins.defines()
        if d:
            counts[d] += 1
    return counts


def _use_counts(instrs: List[Instr]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for ins in instrs:
        for v in ins.uses():
            counts[v] += 1
    return counts


def _is_temp(name: str) -> bool:
    return isinstance(name, str) and len(name) > 1 and name[0] == 't' and name[1:].isdigit()


def _substitute(ins: Instr, old: str, new) -> bool:
    changed = False
    if ins.arg1 == old and is_var(ins.arg1):
        ins.arg1 = new
        changed = True
    if ins.arg2 == old and is_var(ins.arg2):
        ins.arg2 = new
        changed = True
    return changed


def _forwardable(value, def_counts: Dict[str, int]) -> bool:
    """May ``value`` be substituted at arbitrary later use sites?

    Literals always may.  A variable may only if it has exactly one definition
    in the whole program, in which case no intervening redefinition can exist.
    """
    if is_const(value):
        return True
    return is_var(value) and def_counts.get(value, 0) == 1


def _rebuild(prog: TACProgram, instrs: List[Instr]) -> TACProgram:
    out = TACProgram(instrs, prog.ntemp, prog.nlabel, prog.source)
    return out


def _drop(instrs: List[Instr], dead: Set[int]) -> List[Instr]:
    return [ins for k, ins in enumerate(instrs) if k not in dead]


# ------------------------------------------------------- 1. constant folding
def constant_folding(prog: TACProgram) -> Tuple[TACProgram, int]:
    """Evaluate expressions whose operands are all literals.

    A folded temporary that has a single definition is then substituted into
    its uses and its definition deleted, so the fold shows up as a real
    reduction.  Branches on literal conditions are resolved as well.
    """
    p = prog.clone()
    instrs = p.instrs
    changes = 0

    folded_at: List[Tuple[int, str, object]] = []
    for k, ins in enumerate(instrs):
        if ins.kind == BINOP and is_const(ins.arg1) and is_const(ins.arg2):
            val = ir.eval_binop(ins.op, ins.arg1, ins.arg2)
            instrs[k] = ir.assign(ins.dst, val)
            folded_at.append((k, ins.dst, val))
            changes += 1
        elif ins.kind == UNOP and is_const(ins.arg1):
            val = ir.eval_unop(ins.op, ins.arg1)
            instrs[k] = ir.assign(ins.dst, val)
            folded_at.append((k, ins.dst, val))
            changes += 1

    # resolve branches with literal conditions
    dead: Set[int] = set()
    for k, ins in enumerate(instrs):
        if ins.kind == IFGOTO and is_const(ins.arg1):
            if ir.truthy(ins.arg1):
                instrs[k] = ir.goto(ins.target)
            else:
                dead.add(k)
            changes += 1
        elif ins.kind == IFFALSE and is_const(ins.arg1):
            if ir.truthy(ins.arg1):
                dead.add(k)
            else:
                instrs[k] = ir.goto(ins.target)
            changes += 1

    # forward single-definition folded temporaries
    dcount = _def_counts(instrs)
    for k, dst, _recorded in folded_at:
        if k in dead or not _is_temp(dst) or dcount.get(dst, 0) != 1:
            continue
        val = instrs[k].arg1           # current value, not the recorded one
        if not _forwardable(val, dcount):
            continue
        for j, other in enumerate(instrs):
            if j != k:
                _substitute(other, dst, val)
        dead.add(k)
        changes += 1

    if dead:
        instrs = _drop(instrs, dead)
    return _rebuild(p, instrs), changes


# --------------------------------------------------- 2. constant propagation
def constant_propagation(prog: TACProgram) -> Tuple[TACProgram, int]:
    """Replace variable uses by constants proved by the reaching-constants
    dataflow analysis, then delete constant definitions left with no users."""
    p = prog.clone()
    g = CFG(p)
    env = g.reaching_constants()
    instrs = p.instrs
    changes = 0

    for k, ins in enumerate(instrs):
        state = env.get(k)
        if state is None:
            continue
        for attr in ('arg1', 'arg2'):
            o = getattr(ins, attr)
            if is_var(o):
                v = state.get(o, NAC)
                if v is not NAC:
                    setattr(ins, attr, v)
                    changes += 1

    if changes:
        ucount = _use_counts(instrs)
        dead = {k for k, ins in enumerate(instrs)
                if ins.kind == ASSIGN and is_const(ins.arg1)
                and ucount.get(ins.dst, 0) == 0}
        if dead:
            instrs = _drop(instrs, dead)
            changes += len(dead)
    return _rebuild(p, instrs), changes


# ------------------------------------------ 3. common subexpression elimination
def _canon(op: str, a, b):
    if op in COMMUTATIVE:
        ka = (0, a) if is_const(a) else (1, str(a))
        kb = (0, b) if is_const(b) else (1, str(b))
        if kb < ka:
            a, b = b, a
    return (op, _key(a), _key(b))


def _key(o):
    return ('c', type(o).__name__, o) if is_const(o) else ('v', o)


def common_subexpr_elimination(prog: TACProgram) -> Tuple[TACProgram, int]:
    """Local (basic-block) common-subexpression elimination by value numbering.

    An available expression is invalidated as soon as any of its operands, or
    the variable holding its result, is redefined.
    """
    p = prog.clone()
    g = CFG(p)
    instrs = p.instrs
    dcount = _def_counts(instrs)
    dead: Set[int] = set()
    replacements: List[Tuple[str, object]] = []
    changes = 0

    for blk in g.blocks:
        avail: Dict[tuple, str] = {}      # expression key -> holding variable
        for k, ins in blk.real_instrs():
            if ins.kind == BINOP:
                key = _canon(ins.op, ins.arg1, ins.arg2)
                holder = avail.get(key)
                if holder is not None and holder != ins.dst:
                    if _is_temp(ins.dst) and dcount.get(ins.dst, 0) == 1 \
                            and _forwardable(holder, dcount):
                        replacements.append((ins.dst, holder))
                        dead.add(k)
                    else:
                        instrs[k] = ir.assign(ins.dst, holder)
                    changes += 1
                    self_dst = ins.dst
                    avail = {kk: vv for kk, vv in avail.items() if vv != self_dst}
                    _invalidate(avail, self_dst)
                    continue
            d = ins.defines()
            if d:
                _invalidate(avail, d)
            if ins.kind == BINOP and k not in dead:
                avail[_canon(ins.op, ins.arg1, ins.arg2)] = ins.dst

    for dst, holder in replacements:
        for j, other in enumerate(instrs):
            if j not in dead:
                _substitute(other, dst, holder)
    if dead:
        instrs = _drop(instrs, dead)
    return _rebuild(p, instrs), changes


def _invalidate(avail: Dict[tuple, str], name: str):
    for key in [k for k, v in avail.items()
                if v == name or ('v', name) in (k[1], k[2])]:
        del avail[key]


# ------------------------------------------------- 4. dead code elimination
def dead_code_elimination(prog: TACProgram) -> Tuple[TACProgram, int]:
    """Delete computations whose result is never read, plus unreachable blocks.

    Deliberately a *single* pass rather than a fixed-point one, so that one
    application of this strategy is comparable in power to one application of
    any other.  A chain of dead definitions is therefore cleared over
    successive rounds of the selection loop, which is where the iteration in
    the driver earns its keep.
    """
    p = prog.clone()
    g = CFG(p)
    _live_in, live_out = g.liveness()
    dead: Set[int] = set()
    for k, ins in enumerate(p.instrs):
        if ins.is_computation() and not ins.has_side_effect():
            if ins.dst not in live_out.get(k, set()):
                dead.add(k)
    # unreachable blocks (labels included: nothing can jump to them once every
    # jump that targeted them is itself unreachable)
    reach = g.reachable()
    for blk in g.blocks:
        if blk.index not in reach:
            for k in range(blk.start, blk.end):
                dead.add(k)
    if not dead:
        return p, 0
    p = _prune_orphan_labels(_rebuild(p, _drop(p.instrs, dead)))
    return p, len(dead)


def _prune_orphan_labels(prog: TACProgram) -> TACProgram:
    targets = {ins.target for ins in prog.instrs if ins.target}
    keep = [ins for ins in prog.instrs
            if ins.kind != LABEL or ins.name in targets]
    return _rebuild(prog, keep)


# ------------------------------------------- 5. algebraic simplification
_FLOAT_SENSITIVE = True


def _has_float_literal(instrs: List[Instr]) -> bool:
    for ins in instrs:
        for o in (ins.arg1, ins.arg2):
            if isinstance(o, float):
                return True
    return False


def algebraic_simplification(prog: TACProgram) -> Tuple[TACProgram, int]:
    """Apply arithmetic identities: x+0, 0+x, x-0, x*1, 1*x, x*0, 0*x, x/1,
    x-x, 0-x.

    Identities that would change a value's type (``x*0`` and ``x-x`` collapsing
    a float to the integer 0) are suppressed if the program contains any float
    literal, since a later integer division would then diverge.
    """
    p = prog.clone()
    instrs = p.instrs
    int_only = not _has_float_literal(instrs)
    simplified: List[Tuple[int, str, object]] = []
    changes = 0

    for k, ins in enumerate(instrs):
        if ins.kind != BINOP:
            continue
        a, b, op = ins.arg1, ins.arg2, ins.op
        new: Optional[object] = None
        as_unop = False
        if op == '+':
            if b == 0 and is_const(b):
                new = a
            elif a == 0 and is_const(a):
                new = b
        elif op == '-':
            if b == 0 and is_const(b):
                new = a
            elif is_var(a) and a == b and int_only:
                new = 0
            elif a == 0 and is_const(a) and is_var(b):
                instrs[k] = ir.unop(ins.dst, '-', b)
                changes += 1
                continue
        elif op == '*':
            if b == 1 and is_const(b):
                new = a
            elif a == 1 and is_const(a):
                new = b
            elif (b == 0 and is_const(b)) or (a == 0 and is_const(a)):
                if int_only:
                    new = 0
        elif op == '/':
            if b == 1 and is_const(b):
                new = a
        if new is None:
            continue
        instrs[k] = ir.assign(ins.dst, new)
        simplified.append((k, ins.dst, new))
        changes += 1

    dcount = _def_counts(instrs)
    dead: Set[int] = set()
    for k, dst, _recorded in simplified:
        if not _is_temp(dst) or dcount.get(dst, 0) != 1:
            continue
        # Read the operand back off the instruction rather than trusting the
        # value recorded during the scan.  An earlier forward in this same pass
        # may have rewritten it, and forwarding the stale name would leave a
        # reference to a definition that has since been deleted.
        val = instrs[k].arg1
        if not _forwardable(val, dcount):
            continue
        for j, other in enumerate(instrs):
            if j != k:
                _substitute(other, dst, val)
        dead.add(k)
        changes += 1

    if dead:
        instrs = _drop(instrs, dead)
    return _rebuild(p, instrs), changes


# ------------------------------------------ 6. loop-invariant code motion
def loop_invariant_code_motion(prog: TACProgram) -> Tuple[TACProgram, int]:
    """Hoist loop-invariant computations into a freshly created preheader.

    An instruction is hoisted only when all four of these hold, which together
    make the move safe even for a loop that executes zero times:

      1. every operand is a literal, or defined outside the loop, or defined by
         an instruction already proved invariant;
      2. its destination is assigned exactly once inside the loop;
      3. its defining block dominates every in-loop use of the destination;
      4. the destination is not live on any edge leaving the loop.
    """
    p = prog.clone()
    g = CFG(p)
    loops = g.natural_loops()
    if not loops:
        return p, 0
    dom = g.dominators()
    live_in, _instr_live_out = g.liveness()

    header, body = loops[0]                      # innermost loop first
    for h, b in loops:
        if len(b) <= len(body):
            header, body = h, b

    hdr_blk = g.blocks[header]
    if not hdr_blk.labels:
        return p, 0
    # guard: the textually preceding block must not fall through from inside
    if header > 0 and (header - 1) in body:
        prev = g.blocks[header - 1]
        if not (prev.instrs and prev.instrs[-1].kind == GOTO):
            return p, 0

    body_indices: Set[int] = set()
    for b in body:
        for i, _ins in g.blocks[b].real_instrs():
            body_indices.add(i)

    def_in_loop: Dict[str, int] = defaultdict(int)
    for i in body_indices:
        d = p.instrs[i].defines()
        if d:
            def_in_loop[d] += 1

    # variables live on any exit edge of the loop
    live_after: Set[str] = set()
    for b in body:
        for s in g.blocks[b].succs:
            if s not in body:
                live_after |= live_in[s]

    block_of: Dict[int, int] = {}
    for b in body:
        for i, _ins in g.blocks[b].real_instrs():
            block_of[i] = b

    invariant: List[int] = []
    invariant_set: Set[int] = set()
    changed = True
    while changed:
        changed = False
        for i in sorted(body_indices):
            if i in invariant_set:
                continue
            ins = p.instrs[i]
            if not ins.is_computation() or ins.has_side_effect():
                continue
            dst = ins.defines()
            if dst is None or def_in_loop.get(dst, 0) != 1 or dst in live_after:
                continue
            if not _operands_invariant(ins, p, def_in_loop, invariant_set, block_of):
                continue
            if not _def_dominates_uses(i, dst, body_indices, block_of, dom, p):
                continue
            invariant.append(i)
            invariant_set.add(i)
            changed = True

    if not invariant:
        return p, 0

    pre_label = p.new_label()
    hoisted = [p.instrs[i].copy() for i in invariant]
    remaining = [ins for k, ins in enumerate(p.instrs) if k not in invariant_set]

    # locate the header's first instruction in the *remaining* list
    hdr_names = hdr_blk.labels
    insert_at = None
    for k, ins in enumerate(remaining):
        if ins.kind == LABEL and ins.name in hdr_names:
            insert_at = k
            break
    if insert_at is None:
        return p, 0

    new_instrs = (remaining[:insert_at] + [ir.label(pre_label)] + hoisted
                  + remaining[insert_at:])

    # every jump from outside the loop that targeted the header now targets the
    # preheader; back edges from inside the loop are left alone
    outside_jumps = [i for i in range(len(p.instrs))
                     if i not in body_indices and p.instrs[i].target in hdr_names]
    outside_targets = {id(p.instrs[i]) for i in outside_jumps}
    old_by_id = {id(ins) for ins in hoisted}
    for ins in new_instrs:
        if ins.target in hdr_names and id(ins) in outside_targets:
            ins.target = pre_label

    out = _rebuild(p, new_instrs)
    out.nlabel = p.nlabel
    return out, len(invariant)


def _operands_invariant(ins: Instr, p: TACProgram, def_in_loop, invariant_set,
                        block_of) -> bool:
    inv_dsts = {p.instrs[i].defines() for i in invariant_set}
    for o in (ins.arg1, ins.arg2):
        if o is None or is_const(o):
            continue
        if def_in_loop.get(o, 0) == 0:
            continue                       # defined outside the loop
        if o in inv_dsts:
            continue                       # defined by an invariant instruction
        return False
    return True


def _def_dominates_uses(idx: int, dst: str, body_indices, block_of, dom,
                        p: TACProgram) -> bool:
    db = block_of[idx]
    for j in body_indices:
        if j == idx:
            continue
        if dst in p.instrs[j].uses():
            ub = block_of[j]
            if ub == db:
                if j < idx:
                    return False           # used before defined in same block
            elif db not in dom.get(ub, set()):
                return False
    return True


# ------------------------------------------------------------------ registry
PASSES = {
    'constant_folding': constant_folding,
    'constant_propagation': constant_propagation,
    'common_subexpr_elimination': common_subexpr_elimination,
    'dead_code_elimination': dead_code_elimination,
    'algebraic_simplification': algebraic_simplification,
    'loop_invariant_code_motion': loop_invariant_code_motion,
}


def apply_strategy(prog: TACProgram, name: str) -> Tuple[TACProgram, int]:
    return PASSES[name](prog)
