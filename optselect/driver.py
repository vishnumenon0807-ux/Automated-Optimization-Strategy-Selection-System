"""Labelling and the iterative optimization loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import interp
from .features import extract
from .ir import TACProgram
from .optimizations import STRATEGIES, apply_strategy, program_cost
from .selectors import Selector

# deterministic tie-break when two strategies achieve the same gain
TIE_ORDER = {s: i for i, s in enumerate(STRATEGIES)}


def strategy_gains(prog: TACProgram) -> Dict[str, Tuple[int, int]]:
    """Apply each strategy independently; return {strategy: (cost gain, size gain)}."""
    base_cost, base_size = program_cost(prog), prog.size()
    out: Dict[str, Tuple[int, int]] = {}
    for s in STRATEGIES:
        try:
            q, _ = apply_strategy(prog, s)
            out[s] = (base_cost - program_cost(q), base_size - q.size())
        except Exception:
            out[s] = (0, 0)
    return out


def label_program(prog: TACProgram) -> Tuple[Optional[str], Dict[str, Tuple[int, int]]]:
    """Ground-truth label: the strategy giving the largest cost reduction.

    Returns (label, gains).  The label is ``None`` when no strategy improves the
    program at all, in which case the program is dropped from the dataset.
    """
    gains = strategy_gains(prog)
    best = max(STRATEGIES,
               key=lambda s: (gains[s][0], gains[s][1], -TIE_ORDER[s]))
    if gains[best][0] <= 0:
        return None, gains
    return best, gains


@dataclass
class IterationRecord:
    iteration: int
    features: Dict[str, float]
    scores: Dict[str, float]
    selected: str
    confidence: float
    applied: Optional[str]
    fell_back: bool
    size_before: int
    size_after: int
    cost_before: int
    cost_after: int
    #: Snapshot of the program as it stood after this iteration.  Only
    #: populated when optimize(record_programs=True); the demo server needs it
    #: to show the code changing step by step, and nothing else does.
    program_after: Optional[object] = None

    @property
    def cost_gain(self) -> int:
        return self.cost_before - self.cost_after

    @property
    def size_gain(self) -> int:
        return self.size_before - self.size_after


@dataclass
class OptimizationRun:
    program_id: str
    selector: str
    iterations: List[IterationRecord] = field(default_factory=list)
    initial_size: int = 0
    final_size: int = 0
    initial_cost: int = 0
    final_cost: int = 0
    correct: bool = True
    final_program: Optional[TACProgram] = None

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def size_reduction(self) -> int:
        return self.initial_size - self.final_size

    @property
    def cost_reduction(self) -> int:
        return self.initial_cost - self.final_cost

    @property
    def size_reduction_pct(self) -> float:
        return 100.0 * self.size_reduction / max(self.initial_size, 1)

    @property
    def cost_reduction_pct(self) -> float:
        return 100.0 * self.cost_reduction / max(self.initial_cost, 1)

    @property
    def n_fallbacks(self) -> int:
        return sum(1 for r in self.iterations if r.fell_back)


def optimize(prog: TACProgram, selector: Selector, mode: str = 'fallback',
             max_iters: int = 30, verify: bool = True,
             program_id: str = '',
             record_programs: bool = False) -> OptimizationRun:
    """Run extract -> select -> apply -> re-measure until nothing improves.

    ``mode='strict'``   stops the moment the selected strategy fails to improve
    the program, so total improvement directly reflects selection quality.
    ``mode='fallback'`` tries the next-highest-scoring strategy instead, and
    stops only when no strategy at all improves the program.
    """
    run = OptimizationRun(program_id=program_id, selector=selector.name)
    cur = prog.clone()
    run.initial_size, run.initial_cost = cur.size(), program_cost(cur)
    reference = interp.run(prog) if verify else None

    for it in range(1, max_iters + 1):
        feats = extract(cur)
        cost_before, size_before = program_cost(cur), cur.size()
        selected, conf, scores = selector.select(feats, cur)
        order = sorted(STRATEGIES, key=lambda s: (-scores[s], TIE_ORDER[s]))
        candidates = [selected] if mode == 'strict' else \
            [selected] + [s for s in order if s != selected]

        applied, new_prog = None, None
        for cand in candidates:
            try:
                q, _ = apply_strategy(cur, cand)
            except Exception:
                continue
            if program_cost(q) < cost_before or (
                    program_cost(q) == cost_before and q.size() < size_before):
                applied, new_prog = cand, q
                break

        rec = IterationRecord(
            iteration=it, features=feats, scores=scores, selected=selected,
            confidence=conf, applied=applied, fell_back=(applied is not None and
                                                         applied != selected),
            size_before=size_before, size_after=size_before,
            cost_before=cost_before, cost_after=cost_before)

        if applied is None:
            run.iterations.append(rec)
            break

        cur = new_prog
        rec.size_after, rec.cost_after = cur.size(), program_cost(cur)
        if record_programs:
            rec.program_after = cur
        run.iterations.append(rec)

    run.final_size, run.final_cost = cur.size(), program_cost(cur)
    run.final_program = cur
    if verify:
        try:
            run.correct = interp.outputs_match(interp.run(cur), reference)
        except Exception:
            run.correct = False
    return run


def format_run(run: OptimizationRun, feature_names: Optional[List[str]] = None,
               show_features: bool = True) -> str:
    """The per-program trace required by section 7 of the specification."""
    from .optimizations import SHORT
    lines = [f'=== {run.program_id}  (selector: {run.selector}) ===']
    for rec in run.iterations:
        lines.append(f'-- iteration {rec.iteration} --')
        if show_features:
            names = feature_names or sorted(rec.features)
            parts = [f'{n}={rec.features[n]:g}' for n in names]
            # wrapped rather than one long line, so the trace stays readable in
            # a terminal and in the report appendix
            for k in range(0, len(parts), 4):
                tag = 'features :' if k == 0 else '           '
                lines.append(f'   {tag} ' + '  '.join(parts[k:k + 4]))
        sc = '  '.join(f'{SHORT[s]}={rec.scores[s]:.3f}'
                       for s in sorted(rec.scores, key=lambda k: -rec.scores[k]))
        lines.append(f'   scores   : {sc}')
        tag = f'{SHORT[rec.selected]}'
        if rec.applied is None:
            lines.append(f'   selected : {tag} (confidence {rec.confidence:.2f}) '
                         f'-> no strategy improves the code, converged')
        else:
            note = '' if not rec.fell_back else \
                f' [selection {SHORT[rec.selected]} gave no gain; fell back to {SHORT[rec.applied]}]'
            lines.append(f'   selected : {tag} (confidence {rec.confidence:.2f}){note}')
            # printed as a signed delta, so a reduction of 2 shows as -2
            lines.append(f'   applied  : {SHORT[rec.applied]}  '
                         f'instructions {rec.size_before} -> {rec.size_after} '
                         f'({-rec.size_gain:+d}),  weighted cost '
                         f'{rec.cost_before} -> {rec.cost_after} '
                         f'({-rec.cost_gain:+d})')
    lines.append(f'   TOTAL    : instructions {run.initial_size} -> {run.final_size} '
                 f'({run.size_reduction_pct:.1f}% smaller), weighted cost '
                 f'{run.initial_cost} -> {run.final_cost} '
                 f'({run.cost_reduction_pct:.1f}% cheaper) in '
                 f'{run.n_iterations} iterations; '
                 f'semantics preserved: {run.correct}')
    return '\n'.join(lines)