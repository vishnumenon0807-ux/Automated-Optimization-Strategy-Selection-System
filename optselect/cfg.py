"""Control-flow analysis: basic blocks, dominators, natural loops, liveness
and a reaching-constants dataflow analysis.

Every analysis here is recomputed from the current instruction list rather than
cached, because the optimization passes rewrite the program between rounds and
stale annotations would silently corrupt later decisions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from . import ir
from .ir import GOTO, IFFALSE, IFGOTO, LABEL, Instr, TACProgram, is_const, is_var


class NACType:
    """Lattice top: 'not a constant'."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'NAC'


NAC = NACType()


class Block:
    def __init__(self, index: int, start: int, end: int, instrs: List[Instr]):
        self.index = index
        self.start = start          # index of first instruction in program
        self.end = end              # exclusive
        self.instrs = instrs
        self.labels: Set[str] = {i.name for i in instrs if i.kind == LABEL}
        self.succs: List[int] = []
        self.preds: List[int] = []

    def real_instrs(self) -> List[Tuple[int, Instr]]:
        """(program index, instruction) pairs, skipping label pseudo-instructions."""
        return [(self.start + k, ins) for k, ins in enumerate(self.instrs)
                if ins.kind != LABEL]

    def __repr__(self):
        return f'<B{self.index} [{self.start}:{self.end}] labels={sorted(self.labels)}>'


class CFG:
    """Basic-block control-flow graph over a flat TAC instruction list."""

    def __init__(self, prog: TACProgram):
        self.prog = prog
        self.instrs = prog.instrs
        self.blocks: List[Block] = []
        self.label_to_block: Dict[str, int] = {}
        self._build()
        self._link()

    # ------------------------------------------------------------ build
    def _build(self):
        instrs = self.instrs
        n = len(instrs)
        if n == 0:
            return
        leaders: Set[int] = {0}
        for i, ins in enumerate(instrs):
            if ins.is_jump() and i + 1 < n:
                leaders.add(i + 1)
            if ins.kind == LABEL:
                leaders.add(i)
        # Merge runs of consecutive labels into a single block header.
        ordered = sorted(leaders)
        merged: List[int] = []
        for i in ordered:
            if merged and i - 1 == merged[-1] and instrs[i - 1].kind == LABEL:
                continue  # this label continues the header of the previous leader
            merged.append(i)
        bounds = merged + [n]
        for bi in range(len(merged)):
            s, e = bounds[bi], bounds[bi + 1]
            blk = Block(bi, s, e, instrs[s:e])
            self.blocks.append(blk)
            for lb in blk.labels:
                self.label_to_block[lb] = bi

    def _link(self):
        nb = len(self.blocks)
        for b in self.blocks:
            last = None
            for ins in reversed(b.instrs):
                last = ins
                break
            succs: List[int] = []
            if last is not None and last.kind == GOTO:
                tgt = self.label_to_block.get(last.target)
                if tgt is not None:
                    succs.append(tgt)
            elif last is not None and last.kind in (IFGOTO, IFFALSE):
                tgt = self.label_to_block.get(last.target)
                if tgt is not None:
                    succs.append(tgt)
                if b.index + 1 < nb:
                    succs.append(b.index + 1)
            else:
                if b.index + 1 < nb:
                    succs.append(b.index + 1)
            seen, uniq = set(), []
            for s in succs:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            b.succs = uniq
        for b in self.blocks:
            for s in b.succs:
                self.blocks[s].preds.append(b.index)

    # ------------------------------------------------------- reachability
    def reachable(self) -> Set[int]:
        if not self.blocks:
            return set()
        seen, stack = {0}, [0]
        while stack:
            b = stack.pop()
            for s in self.blocks[b].succs:
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return seen

    # --------------------------------------------------------- dominators
    def dominators(self) -> Dict[int, Set[int]]:
        """Iterative dominator computation over reachable blocks."""
        reach = self.reachable()
        dom: Dict[int, Set[int]] = {b: set(reach) for b in reach}
        if 0 in dom:
            dom[0] = {0}
        changed = True
        while changed:
            changed = False
            for b in sorted(reach):
                if b == 0:
                    continue
                preds = [p for p in self.blocks[b].preds if p in reach]
                if preds:
                    new = set(dom[preds[0]])
                    for p in preds[1:]:
                        new &= dom[p]
                else:
                    new = set()
                new.add(b)
                if new != dom[b]:
                    dom[b] = new
                    changed = True
        return dom

    # ------------------------------------------------------ natural loops
    def natural_loops(self) -> List[Tuple[int, Set[int]]]:
        """Return (header, body-blocks) for every natural loop, innermost first."""
        dom = self.dominators()
        loops: List[Tuple[int, Set[int]]] = []
        for b in dom:
            for s in self.blocks[b].succs:
                if s in dom and s in dom[b]:          # back edge b -> s
                    body = self._natural_loop_body(s, b)
                    loops.append((s, body))
        # merge loops sharing a header (multiple back edges)
        merged: Dict[int, Set[int]] = {}
        for h, body in loops:
            merged.setdefault(h, set()).update(body)
        out = sorted(merged.items(), key=lambda kv: len(kv[1]))
        return out

    def _natural_loop_body(self, header: int, latch: int) -> Set[int]:
        body = {header, latch}
        stack = [latch] if latch != header else []
        while stack:
            b = stack.pop()
            for p in self.blocks[b].preds:
                if p not in body:
                    body.add(p)
                    stack.append(p)
        return body

    def loop_instruction_indices(self) -> Set[int]:
        """Program indices of every instruction that lies inside some loop body."""
        idxs: Set[int] = set()
        for _, body in self.natural_loops():
            for b in body:
                for i, ins in self.blocks[b].real_instrs():
                    idxs.add(i)
        return idxs

    def loop_depths(self) -> Dict[int, int]:
        """Nesting depth per program instruction index."""
        depth: Dict[int, int] = {}
        for _, body in self.natural_loops():
            for b in body:
                for i, _ins in self.blocks[b].real_instrs():
                    depth[i] = depth.get(i, 0) + 1
        return depth

    # ------------------------------------------------------------ liveness
    def liveness(self) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]]]:
        """Backward liveness analysis.

        Returns (block_live_in, instr_live_out) where instr_live_out maps a
        program instruction index to the set of variables live immediately
        after that instruction.
        """
        use: Dict[int, Set[str]] = {}
        dfn: Dict[int, Set[str]] = {}
        for b in self.blocks:
            u, d = set(), set()
            for _i, ins in b.real_instrs():
                for v in ins.uses():
                    if v not in d:
                        u.add(v)
                dv = ins.defines()
                if dv:
                    d.add(dv)
            use[b.index], dfn[b.index] = u, d

        live_in = {b.index: set() for b in self.blocks}
        live_out = {b.index: set() for b in self.blocks}
        changed = True
        while changed:
            changed = False
            for b in reversed(self.blocks):
                out = set()
                for s in b.succs:
                    out |= live_in[s]
                inn = use[b.index] | (out - dfn[b.index])
                if out != live_out[b.index] or inn != live_in[b.index]:
                    live_out[b.index], live_in[b.index] = out, inn
                    changed = True

        instr_live_out: Dict[int, Set[str]] = {}
        for b in self.blocks:
            live = set(live_out[b.index])
            for i, ins in reversed(b.real_instrs()):
                instr_live_out[i] = set(live)
                dv = ins.defines()
                if dv:
                    live.discard(dv)
                live |= set(ins.uses())
        return live_in, instr_live_out

    # -------------------------------------------------- reaching constants
    def reaching_constants(self) -> Dict[int, Dict[str, object]]:
        """Forward constant-propagation dataflow.

        Returns a map from program instruction index to the constant
        environment holding immediately *before* that instruction.  A variable
        maps either to a Python number or to ``NAC``.  States are total over
        all program variables, and the meet is taken only over predecessors
        already proved reachable, which keeps loop-carried constants precise
        without ever assuming a value for an uninitialised variable.
        """
        all_vars = self.prog.variables()
        entry_state = {v: NAC for v in all_vars}

        block_in: Dict[int, Dict[str, object]] = {}
        block_out: Dict[int, Dict[str, object]] = {}
        visited: Set[int] = set()
        if not self.blocks:
            return {}

        worklist = [0]
        block_in[0] = dict(entry_state)
        while worklist:
            b = worklist.pop(0)
            preds = [p for p in self.blocks[b].preds if p in visited]
            if b == 0:
                state = dict(entry_state)
            elif preds:
                state = dict(block_out[preds[0]])
                for p in preds[1:]:
                    state = _meet(state, block_out[p])
            else:
                state = dict(entry_state)
            block_in[b] = state
            out = self._transfer_block(self.blocks[b], state)
            first_visit = b not in visited
            if first_visit or out != block_out.get(b):
                block_out[b] = out
                visited.add(b)
                for s in self.blocks[b].succs:
                    if s not in worklist:
                        worklist.append(s)

        instr_in: Dict[int, Dict[str, object]] = {}
        for b in self.blocks:
            if b.index not in block_in:
                continue
            state = dict(block_in[b.index])
            for i, ins in b.real_instrs():
                instr_in[i] = dict(state)
                state = _transfer_instr(ins, state)
        return instr_in

    def _transfer_block(self, b: Block, state: Dict[str, object]) -> Dict[str, object]:
        s = dict(state)
        for _i, ins in b.real_instrs():
            s = _transfer_instr(ins, s)
        return s


def _meet(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    """Pointwise meet: a variable keeps its value only if both states agree."""
    out: Dict[str, object] = {}
    for k in set(a) | set(b):
        va, vb = a.get(k, NAC), b.get(k, NAC)
        if va is NAC or vb is NAC:
            out[k] = NAC
        elif va == vb and type(va) is type(vb):
            out[k] = va
        else:
            out[k] = NAC
    return out


def _value_of(o, state: Dict[str, object]):
    if is_const(o):
        return o
    if is_var(o):
        return state.get(o, NAC)
    return NAC


def _transfer_instr(ins: Instr, state: Dict[str, object]) -> Dict[str, object]:
    """Transfer function for the constant-propagation lattice.

    A computed expression yields NAC even when both operands are known
    constants.  *Evaluating* such an expression is constant folding's job, and
    letting the propagation lattice do it as well would make the two strategies
    indistinguishable — constant propagation would silently subsume constant
    folding and the selection problem would collapse.  Propagation here means
    only: replace a variable use by a value the variable is known to hold.
    """
    s = dict(state)
    d = ins.defines()
    if d is None:
        return s
    if ins.kind == ir.ASSIGN:
        s[d] = _value_of(ins.arg1, state)
    else:
        s[d] = NAC
    return s
