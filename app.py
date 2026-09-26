"""Flask web demo for the optimization strategy selection system.

    pip install flask
    python app.py            then open http://localhost:8000

Paste a program, pick a selector, and step through the optimization one
iteration at a time: the six scores behind each decision, which pass was chosen,
and exactly which TAC instructions it changed.

This is a thin front end over the real system -- it imports the same passes, the
same selectors and the same loop that produced the report.  Nothing here
reimplements any of it, so the demo cannot show behaviour the project does not
actually have.
"""
from __future__ import annotations

import difflib
import os
import pickle
import sys

from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from optselect import interp
from optselect.driver import optimize
from optselect.features import extract
from optselect.frontend import CompileError, compile_to_tac
from optselect.optimizations import SHORT, STRATEGIES
from optselect.selectors import MLSelector, OracleSelector, RuleBasedSelector

OUT = os.path.join(ROOT, 'out')
MAX_SOURCE_CHARS = 20_000

app = Flask(__name__)


# ----------------------------------------------------------------- selectors
def load_selectors():
    """Rule-based and oracle always work; the ML ones need a trained model."""
    sels = {'rule_based': RuleBasedSelector(), 'oracle': OracleSelector()}
    for key, fname in (('ml_extended', 'model.pkl'),
                       ('ml_required8', 'model_required8.pkl')):
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as fh:
                d = pickle.load(fh)
            sels[key] = MLSelector(d['model'], d['feature_names'], name=key)
        except Exception as exc:                          # noqa: BLE001
            print(f'  ! could not load {fname}: {exc}')
    return sels


SELECTORS = load_selectors()


# ---------------------------------------------------------------- tac diffing
def tac_lines(prog):
    return [str(i) for i in prog.instrs]


def diff_lines(before, after):
    """Merged view of two TAC listings, each line tagged same/added/removed."""
    out = []
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            out += [{'text': t, 'status': 'same'} for t in before[i1:i2]]
        else:
            out += [{'text': t, 'status': 'removed'} for t in before[i1:i2]]
            out += [{'text': t, 'status': 'added'} for t in after[j1:j2]]
    return out


# -------------------------------------------------------------------- the run
def run_optimization(source, selector_key, mode):
    if selector_key not in SELECTORS:
        raise ValueError(f'unknown selector {selector_key!r}')
    if mode not in ('strict', 'fallback'):
        raise ValueError(f'unknown mode {mode!r}')

    prog = compile_to_tac(source)
    before_output = interp.run(prog, max_steps=400_000)
    run = optimize(prog, SELECTORS[selector_key], mode=mode, verify=True,
                   record_programs=True, program_id='demo')

    initial = tac_lines(prog)
    iterations, prev = [], initial
    for rec in run.iterations:
        row = {
            'iteration': rec.iteration,
            'features': rec.features,
            'scores': {SHORT[s]: rec.scores[s] for s in STRATEGIES},
            'selected': SHORT[rec.selected],
            'confidence': rec.confidence,
            'applied': SHORT[rec.applied] if rec.applied else None,
            'fell_back': rec.fell_back,
            'size_before': rec.size_before, 'size_after': rec.size_after,
            'cost_before': rec.cost_before, 'cost_after': rec.cost_after,
            'converged': rec.applied is None,
        }
        if rec.applied is None:
            row['diff'] = [{'text': t, 'status': 'same'} for t in prev]
            iterations.append(row)
            break
        cur = tac_lines(rec.program_after)
        row['diff'] = diff_lines(prev, cur)
        iterations.append(row)
        prev = cur

    final_prog = run.final_program
    return {
        'ok': True,
        'initial': {'size': run.initial_size, 'cost': run.initial_cost},
        'final': {'lines': tac_lines(final_prog), 'size': run.final_size,
                  'cost': run.final_cost},
        'iterations': iterations,
        'summary': {
            'n_iterations': run.n_iterations,
            'n_fallbacks': run.n_fallbacks,
            'size_reduction_pct': run.size_reduction_pct,
            'cost_reduction_pct': run.cost_reduction_pct,
            'correct': run.correct,
        },
        'output_before': before_output,
        'output_after': interp.run(final_prog, max_steps=400_000),
        'strategies': [SHORT[s] for s in STRATEGIES],
        'strategy_names': {SHORT[s]: s.replace('_', ' ') for s in STRATEGIES},
    }


# ------------------------------------------------------------------- routes
@app.get('/')
def index():
    return render_template('index.html',
                           trained=[k for k in SELECTORS if k.startswith('ml_')])


@app.post('/api/optimize')
def api_optimize():
    data = request.get_json(silent=True) or {}
    source = data.get('source', '')
    if len(source) > MAX_SOURCE_CHARS:
        return jsonify(ok=False, error='program too large for the demo')
    try:
        return jsonify(run_optimization(source,
                                        data.get('selector', 'rule_based'),
                                        data.get('mode', 'fallback')))
    except CompileError as exc:
        return jsonify(ok=False, error=str(exc))
    except interp.ExecutionLimit:
        return jsonify(ok=False, error='the program did not terminate within the '
                                       'step limit - check your loop bounds')
    except Exception as exc:                              # noqa: BLE001
        app.logger.exception('optimization failed')
        return jsonify(ok=False, error=f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    trained = [k for k in SELECTORS if k.startswith('ml_')]
    print('Optimization Strategy Selection - demo')
    print(f'  selectors: {", ".join(sorted(SELECTORS))}')
    if not trained:
        print('  note: no trained model in out/. Run `python scripts/train.py` '
              'to enable the ML selectors;')
        print('        rule-based and oracle work regardless.')
    print('\n  open http://localhost:8000   (Ctrl+C to stop)\n')
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8000)), debug=False)