"""Verification suite.

Three layers:

1. Unit tests pinning each pass to a hand-written expectation.
2. Randomised *differential* testing -- random programs, random pass
   sequences, several input streams -- asserting that observable output never
   changes.  This is the layer that matters: the dataset build only ever
   applies one pass to a fresh program, so composition bugs (a pass that is
   correct alone but wrong after another has run) would slip past it.  Several
   input streams are used because a single fixed stream can hide a
   transformation that is only wrong on some inputs.
3. Consistency checks on the reported metrics, recomputed from the raw
   predictions rather than trusted.
"""
from __future__ import annotations

import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optselect import interp
from optselect import optimizations as O
from optselect.datagen import ProgramGenerator
from optselect.driver import optimize
from optselect.features import EXTENDED_FEATURES, REQUIRED_FEATURES, extract
from optselect.frontend import compile_to_tac
from optselect.optimizations import SHORT, STRATEGIES
from optselect.selectors import OracleSelector, RuleBasedSelector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASSED = []
FAILED = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        FAILED.append((name, f'{type(exc).__name__}: {exc}'))
        print(f'  FAIL  {name}')
        traceback.print_exc(limit=3)
    else:
        PASSED.append(name)
        print(f'  ok    {name}')


def tac(src):
    return compile_to_tac(src)


def body(prog):
    return [str(i).strip() for i in prog.instrs]


# ------------------------------------------------------------ unit tests
def test_constant_folding():
    p = tac('int a = 2 * 3; print(a);')
    q, n = O.constant_folding(p)
    assert n > 0, 'expected a fold'
    assert 'a = 6' in body(q), body(q)
    assert q.size() < p.size(), 'folding a temporary should remove its definition'


def test_constant_propagation():
    p = tac('int c = 5; int d = c + 1; print(d);')
    q, n = O.constant_propagation(p)
    assert any('5 + 1' in s for s in body(q)), body(q)
    assert not any(s.startswith('c = ') for s in body(q)), 'c should be gone'


def test_cp_does_not_fold():
    """Propagation must substitute but never evaluate -- that is folding's job,
    and the two strategies stop being separable if propagation does both."""
    p = tac('int c = 5; int d = c + 1; print(d);')
    q, _ = O.constant_propagation(p)
    assert not any('= 6' in s for s in body(q)), \
        f'propagation evaluated an expression: {body(q)}'


def test_cse():
    p = tac('read(a); read(b); int u = a * b + 1; int v = a * b + 2; '
            'print(u); print(v);')
    before = sum(1 for i in p.instrs if i.kind == O.BINOP and i.op == '*')
    q, n = O.common_subexpr_elimination(p)
    after = sum(1 for i in q.instrs if i.kind == O.BINOP and i.op == '*')
    assert before == 2 and after == 1, (before, after, body(q))


def test_cse_respects_redefinition():
    """a*b is not available after a is reassigned."""
    p = tac('read(a); read(b); int u = a * b; a = a + 1; int v = a * b; '
            'print(u); print(v);')
    q, _ = O.common_subexpr_elimination(p)
    assert sum(1 for i in q.instrs if i.kind == O.BINOP and i.op == '*') == 2, \
        body(q)


def test_dce():
    """One pass clears one layer of a dead chain, by design.

    `int dead = a * 99;` lowers to `t = a * 99; dead = t;`.  A single pass kills
    `dead = t`, which is what makes `t = a * 99` dead in turn; the second layer
    goes on the next round of the selection loop.  DCE is deliberately not run
    to a fixed point, so that one application of it is comparable in power to
    one application of any other strategy.
    """
    p = tac('read(a); int live = a + 1; int dead = a * 99; print(live);')
    q, n = O.dead_code_elimination(p)
    assert n > 0
    assert not any(s.startswith('dead = ') for s in body(q)), body(q)
    assert any('+ 1' in s for s in body(q)), 'live computation was removed'

    q2, n2 = O.dead_code_elimination(q)
    assert n2 > 0 and not any('* 99' in s for s in body(q2)), body(q2)
    assert q2.size() < q.size() < p.size()


def test_dce_keeps_reads():
    """read() advances the input stream, so it is never dead."""
    from optselect.ir import READ
    p = tac('read(a); read(b); print(b);')
    q, _ = O.dead_code_elimination(p)
    assert sum(1 for i in q.instrs if i.kind == READ) == 2, body(q)


def test_algebraic():
    for src, gone in (('read(a); int v = a + 0; print(v);', '+ 0'),
                      ('read(a); int v = a * 1; print(v);', '* 1'),
                      ('read(a); int v = 1 * a; print(v);', '1 *'),
                      ('read(a); int v = a / 1; print(v);', '/ 1'),
                      ('read(a); int v = a - 0; print(v);', '- 0')):
        q, n = O.algebraic_simplification(tac(src))
        assert n > 0 and not any(gone in s for s in body(q)), (src, body(q))


def test_algebraic_skips_float():
    """x*0 -> 0 would turn a float into an int; suppressed when floats present."""
    p = tac('read(a); float x = a * 1.5; float v = x * 0; print(v);')
    q, _ = O.algebraic_simplification(p)
    assert interp.outputs_match(interp.run(q), interp.run(p))


def test_licm():
    p = tac('read(a); read(b); int s = 0; '
            'for (i = 0; i < 10; i = i + 1) { s = s + a * b; } print(s);')
    before = O.program_cost(p)
    q, n = O.loop_invariant_code_motion(p)
    assert n > 0, 'expected a hoist'
    assert O.program_cost(q) < before, 'hoisting must reduce weighted cost'
    assert q.size() == p.size(), 'hoisting relocates, it does not delete'
    assert interp.outputs_match(interp.run(q), interp.run(p))


def test_licm_refuses_variant():
    """Nothing invariant here: every operand changes with the loop counter."""
    p = tac('int s = 0; for (i = 0; i < 10; i = i + 1) { s = s + i * i; } print(s);')
    _q, n = O.loop_invariant_code_motion(p)
    assert n == 0, 'hoisted a loop-variant computation'


def test_licm_refuses_live_out():
    """`v` is printed after the loop, so hoisting would change its value when
    the loop body never runs."""
    p = tac('read(a); read(b); int v = 99; int s = 0;'
            'for (i = 0; i < 3; i = i + 1) { v = a * b; s = s + v; }'
            'print(s); print(v);')
    q, _n = O.loop_invariant_code_motion(p)
    assert interp.outputs_match(interp.run(q), interp.run(p))


def test_cost_metric():
    flat = tac('read(a); int x = a + 1; print(x);')
    looped = tac('read(a); int s = 0; for (i = 0; i < 5; i = i + 1) '
                 '{ s = s + a; } print(s);')
    assert O.program_cost(flat) == flat.size(), 'no loop -> cost equals size'
    assert O.program_cost(looped) > looped.size(), 'loop body must be weighted'


def test_features_present():
    p = tac('read(a); int v = 2 * 3; int d = a * 7; print(v);')
    f = extract(p)
    for name in EXTENDED_FEATURES:
        assert name in f, name
        assert isinstance(f[name], float)
    assert f['num_dead_stmts'] > 0, 'd is never read'
    assert f['num_foldable'] > 0, '2 * 3 is foldable'


def test_selector_contract():
    p = tac('read(a); int v = 2 * 3; print(v); print(a);')
    f = extract(p)
    for sel in (RuleBasedSelector(), OracleSelector()):
        best, conf, scores = sel.select(f, p)
        assert best in STRATEGIES
        assert 0.0 <= conf <= 1.0, conf
        assert set(scores) == set(STRATEGIES)


# --------------------------------------------- randomised differential tests
INPUT_STREAMS = [
    [7, 3, 12, 5, 9, 2, 14, 6, 11, 4, 8, 13, 1, 10, 15],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [-5, 13, -2, 8, -11, 4, -7, 6, -3, 9],
    [100, -100, 50, -50, 25, -25, 12, -12],
]


def _outputs(prog):
    return [interp.run(prog, inputs=s, max_steps=300_000) for s in INPUT_STREAMS]


def _same(a, b):
    return len(a) == len(b) and all(interp.outputs_match(x, y) for x, y in zip(a, b))


def test_random_pass_sequences():
    """Random programs, random pass sequences, five input streams each."""
    rng = random.Random(1234)
    gen = ProgramGenerator(seed=99)
    n_programs, seq_len = 60, 6
    for k in range(n_programs):
        bias = STRATEGIES[k % len(STRATEGIES)]
        prog = compile_to_tac(gen.generate(bias))
        ref = _outputs(prog)
        cur = prog
        applied = []
        for _ in range(seq_len):
            s = rng.choice(STRATEGIES)
            cur, _ = O.apply_strategy(cur, s)
            applied.append(SHORT[s])
            got = _outputs(cur)
            assert _same(got, ref), (
                f'program {k} diverged after {"->".join(applied)}\n'
                f'  expected {ref}\n  got      {got}')


def test_loop_terminates_and_preserves():
    """The driver must converge and must not change behaviour, under every
    selector, on programs it has never seen."""
    gen = ProgramGenerator(seed=555)
    for k, bias in enumerate(STRATEGIES * 3):
        prog = compile_to_tac(gen.generate(bias))
        ref = _outputs(prog)
        for sel in (RuleBasedSelector(), OracleSelector()):
            for mode in ('strict', 'fallback'):
                run = optimize(prog, sel, mode=mode, verify=False,
                               max_iters=40)
                assert run.n_iterations < 40, 'loop failed to converge'
                assert _same(_outputs(run.final_program), ref), \
                    f'program {k} changed under {sel.name}/{mode}'
                assert run.final_cost <= run.initial_cost, 'loop made it worse'


def test_loop_reaches_fixpoint():
    """On convergence, no strategy may still improve the program."""
    gen = ProgramGenerator(seed=777)
    for bias in STRATEGIES:
        prog = compile_to_tac(gen.generate(bias))
        run = optimize(prog, OracleSelector(), mode='fallback', verify=False)
        final = run.final_program
        base = O.program_cost(final)
        for s in STRATEGIES:
            q, _ = O.apply_strategy(final, s)
            assert O.program_cost(q) >= base, \
                f'{SHORT[s]} still improves a supposedly converged program'


# ------------------------------------------------------- metrics integrity
def test_metrics_match_predictions():
    import json
    import pandas as pd
    from sklearn.metrics import accuracy_score
    out = os.path.join(ROOT, 'out')
    if not os.path.exists(os.path.join(out, 'metrics.json')):
        raise AssertionError('run scripts/train.py first')
    m = json.load(open(os.path.join(out, 'metrics.json')))
    for fs in ('required_8', 'extended_14'):
        path = os.path.join(out, f'predictions_rf_{fs}.csv')
        d = pd.read_csv(path)
        recomputed = accuracy_score(d['best_strategy'], d['predicted'])
        reported = m['feature_sets'][fs]['models']['RandomForest']['test_accuracy']
        assert abs(recomputed - reported) < 1e-9, (fs, recomputed, reported)
        conf = m['feature_sets'][fs]['models']['RandomForest']['confusion']
        assert sum(map(sum, conf)) == len(d), 'confusion matrix total mismatch'


def test_dataset_labels_are_reproducible():
    """Re-derive a sample of labels from source and confirm they still match."""
    import pandas as pd
    from optselect.driver import label_program
    out = os.path.join(ROOT, 'out')
    df = pd.read_csv(os.path.join(out, 'dataset.csv')).sample(
        20, random_state=3)
    for _, row in df.iterrows():
        src = open(os.path.join(ROOT, 'programs',
                                f'{row["program_id"]}.src')).read()
        label, _gains = label_program(compile_to_tac(src))
        assert label == row['best_strategy'], (row['program_id'], label,
                                               row['best_strategy'])


def test_features_are_reproducible():
    import pandas as pd
    out = os.path.join(ROOT, 'out')
    df = pd.read_csv(os.path.join(out, 'dataset.csv')).sample(15, random_state=5)
    for _, row in df.iterrows():
        src = open(os.path.join(ROOT, 'programs',
                                f'{row["program_id"]}.src')).read()
        f = extract(compile_to_tac(src))
        for name in EXTENDED_FEATURES:
            assert abs(f[name] - float(row[name])) < 1e-9, (
                row['program_id'], name, f[name], row[name])


def main():
    tests = [
        ('constant folding', test_constant_folding),
        ('constant propagation', test_constant_propagation),
        ('propagation does not fold', test_cp_does_not_fold),
        ('common subexpression elimination', test_cse),
        ('CSE respects redefinition', test_cse_respects_redefinition),
        ('dead code elimination', test_dce),
        ('DCE keeps read()', test_dce_keeps_reads),
        ('algebraic identities', test_algebraic),
        ('algebraic skips float-typed collapse', test_algebraic_skips_float),
        ('loop-invariant code motion', test_licm),
        ('LICM refuses variant code', test_licm_refuses_variant),
        ('LICM refuses live-out destination', test_licm_refuses_live_out),
        ('loop-weighted cost metric', test_cost_metric),
        ('feature extraction', test_features_present),
        ('selector contract', test_selector_contract),
        ('random pass sequences preserve semantics', test_random_pass_sequences),
        ('loop terminates and preserves semantics',
         test_loop_terminates_and_preserves),
        ('loop reaches a true fixpoint', test_loop_reaches_fixpoint),
        ('reported metrics match raw predictions', test_metrics_match_predictions),
        ('dataset labels reproducible', test_dataset_labels_are_reproducible),
        ('dataset features reproducible', test_features_are_reproducible),
    ]
    print(f'running {len(tests)} checks\n')
    for name, fn in tests:
        check(name, fn)
    print(f'\n{len(PASSED)} passed, {len(FAILED)} failed')
    if FAILED:
        for n, e in FAILED:
            print(f'  - {n}: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
