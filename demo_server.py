"""Local web demo for the optimization strategy selection system.

    python demo_server.py          then open http://localhost:8000

Paste a program, pick a selector, and step through the optimization one
iteration at a time: the six scores that drove each decision, which pass was
chosen, and exactly which TAC instructions it changed.

This is a thin front end over the real system -- it imports the same passes,
the same selectors and the same loop that produced the report.  Nothing here
reimplements any of it, so the demo can never show behaviour the project does
not actually have.

Standard library only; no Flask, no extra install.
"""
from __future__ import annotations

import difflib
import json
import os
import pickle
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from optselect import interp
from optselect.driver import optimize
from optselect.features import extract
from optselect.frontend import CompileError, compile_to_tac
from optselect.optimizations import SHORT, STRATEGIES, program_cost
from optselect.selectors import MLSelector, OracleSelector, RuleBasedSelector

OUT = os.path.join(ROOT, 'out')
PAGE = os.path.join(ROOT, 'demo', 'index.html')
MAX_SOURCE_BYTES = 20_000


# ----------------------------------------------------------------- selectors
def load_selectors():
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
        except Exception as exc:                      # noqa: BLE001
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
        if rec.applied is None:
            iterations.append({
                'iteration': rec.iteration,
                'features': rec.features,
                'scores': {SHORT[s]: rec.scores[s] for s in STRATEGIES},
                'selected': SHORT[rec.selected],
                'confidence': rec.confidence,
                'applied': None, 'fell_back': False,
                'size_before': rec.size_before, 'size_after': rec.size_after,
                'cost_before': rec.cost_before, 'cost_after': rec.cost_after,
                'diff': [{'text': t, 'status': 'same'} for t in prev],
                'converged': True,
            })
            break
        cur = tac_lines(rec.program_after)
        iterations.append({
            'iteration': rec.iteration,
            'features': rec.features,
            'scores': {SHORT[s]: rec.scores[s] for s in STRATEGIES},
            'selected': SHORT[rec.selected],
            'confidence': rec.confidence,
            'applied': SHORT[rec.applied],
            'fell_back': rec.fell_back,
            'size_before': rec.size_before, 'size_after': rec.size_after,
            'cost_before': rec.cost_before, 'cost_after': rec.cost_after,
            'diff': diff_lines(prev, cur),
            'converged': False,
        })
        prev = cur

    final_prog = run.final_program
    return {
        'ok': True,
        'initial': {'lines': initial, 'size': run.initial_size,
                    'cost': run.initial_cost},
        'final': {'lines': tac_lines(final_prog), 'size': run.final_size,
                  'cost': run.final_cost},
        'features_initial': extract(prog),
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


# ------------------------------------------------------------------ http glue
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if '/api/' in (args[0] if args else ''):
            sys.stderr.write(f'  {args[0]}\n')

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        payload = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            try:
                with open(PAGE, 'rb') as fh:
                    self._send(200, fh.read(), 'text/html; charset=utf-8')
            except FileNotFoundError:
                self._send(500, b'demo/index.html is missing', 'text/plain')
        elif path == '/api/selectors':
            self._send(200, json.dumps({
                'available': sorted(SELECTORS),
                'trained': [k for k in SELECTORS if k.startswith('ml_')],
            }))
        else:
            self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if self.path.split('?')[0] != '/api/optimize':
            self._send(404, b'not found', 'text/plain')
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            if length > MAX_SOURCE_BYTES:
                raise ValueError('program too large for the demo')
            req = json.loads(self.rfile.read(length) or b'{}')
            result = run_optimization(req.get('source', ''),
                                      req.get('selector', 'rule_based'),
                                      req.get('mode', 'fallback'))
            self._send(200, json.dumps(result))
        except CompileError as exc:
            self._send(200, json.dumps({'ok': False, 'kind': 'compile',
                                        'error': str(exc)}))
        except interp.ExecutionLimit:
            self._send(200, json.dumps({
                'ok': False, 'kind': 'runtime',
                'error': 'the program did not terminate within the step limit '
                         '- check your loop bounds'}))
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            self._send(200, json.dumps({'ok': False, 'kind': 'internal',
                                        'error': f'{type(exc).__name__}: {exc}'}))


def main():
    port = int(os.environ.get('PORT', '8000'))
    url = f'http://localhost:{port}'
    trained = [k for k in SELECTORS if k.startswith('ml_')]
    print(f'Optimization Strategy Selection - demo server')
    print(f'  selectors available : {", ".join(sorted(SELECTORS))}')
    if not trained:
        print('  note: no trained model found in out/. Run '
              '`python scripts/train.py` to enable the ML selectors;')
        print('        the rule-based and oracle selectors work regardless.')
    print(f'\n  open {url}   (Ctrl+C to stop)\n')
    try:
        webbrowser.open(url)
    except Exception:                                  # noqa: BLE001
        pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
        server.server_close()


if __name__ == '__main__':
    main()
