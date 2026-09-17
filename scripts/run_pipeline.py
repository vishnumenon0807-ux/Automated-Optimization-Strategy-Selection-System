"""Run the iterative optimization loop over the whole corpus with each
selector, and emit the per-program traces required by section 7.

Note the distribution shift this exercises: the classifier is trained on
*unoptimized* programs but is queried again at every iteration, on programs
that have already been partly optimized.  Measuring the loop rather than only
the single-shot classification is the only way to see whether that matters.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optselect.driver import format_run, optimize
from optselect.frontend import compile_to_tac
from optselect.optimizations import SHORT, STRATEGIES
from optselect.selectors import MLSelector, OracleSelector, RuleBasedSelector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out')
PROGRAMS = os.path.join(ROOT, 'programs')


def load_selectors():
    with open(os.path.join(OUT, 'model.pkl'), 'rb') as fh:
        best = pickle.load(fh)
    with open(os.path.join(OUT, 'model_required8.pkl'), 'rb') as fh:
        spec = pickle.load(fh)
    return {
        'rule_based': RuleBasedSelector(),
        'ml_required8': MLSelector(spec['model'], spec['feature_names'],
                                   name='ml_required8'),
        'ml_extended': MLSelector(best['model'], best['feature_names'],
                                  name='ml_extended'),
        'oracle': OracleSelector(),
    }, best['name'], spec['name']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', type=int, default=5,
                    help='number of programs to print full traces for')
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(OUT, 'dataset.csv'))
    selectors, best_name, spec_name = load_selectors()
    print(f'ML selectors: extended = {best_name}, required-8 = {spec_name}\n')

    run_rows, iter_rows = [], []
    incorrect = []

    for _, row in df.iterrows():
        pid = row['program_id']
        with open(os.path.join(PROGRAMS, f'{pid}.src')) as fh:
            src = fh.read()
        prog = compile_to_tac(src)
        for sel_name, sel in selectors.items():
            for mode in ('strict', 'fallback'):
                if sel_name == 'oracle' and mode == 'strict':
                    continue          # the oracle never mis-selects
                run = optimize(prog, sel, mode=mode, program_id=pid, verify=True)
                run_rows.append({
                    'program_id': pid, 'selector': sel_name, 'mode': mode,
                    'true_label': row['best_strategy'],
                    'n_iterations': run.n_iterations,
                    'n_fallbacks': run.n_fallbacks,
                    'initial_size': run.initial_size,
                    'final_size': run.final_size,
                    'size_reduction': run.size_reduction,
                    'size_reduction_pct': run.size_reduction_pct,
                    'initial_cost': run.initial_cost,
                    'final_cost': run.final_cost,
                    'cost_reduction': run.cost_reduction,
                    'cost_reduction_pct': run.cost_reduction_pct,
                    'correct': run.correct,
                })
                if not run.correct:
                    incorrect.append((pid, sel_name, mode))
                if mode == 'fallback':
                    for rec in run.iterations:
                        iter_rows.append({
                            'program_id': pid, 'selector': sel_name,
                            'iteration': rec.iteration,
                            'selected': rec.selected,
                            'applied': rec.applied or 'none',
                            'confidence': rec.confidence,
                            'fell_back': rec.fell_back,
                            'size_gain': rec.size_gain,
                            'cost_gain': rec.cost_gain,
                            'num_constants': rec.features['num_constants'],
                            'const_ratio': rec.features['const_ratio'],
                            'score_cf': rec.scores['constant_folding'],
                        })

    runs = pd.DataFrame(run_rows)
    iters = pd.DataFrame(iter_rows)
    runs.to_csv(os.path.join(OUT, 'loop_runs.csv'), index=False)
    iters.to_csv(os.path.join(OUT, 'loop_iterations.csv'), index=False)

    # -------------------------------------------------------------- summary
    print(f'{"selector":14} {"mode":9} {"instr red%":>10} {"cost red%":>10} '
          f'{"iters":>6} {"fallbacks":>10} {"correct":>8}')
    summary = {}
    for (sel, mode), grp in runs.groupby(['selector', 'mode']):
        key = f'{sel}/{mode}'
        summary[key] = {
            'mean_size_reduction_pct': float(grp['size_reduction_pct'].mean()),
            'mean_cost_reduction_pct': float(grp['cost_reduction_pct'].mean()),
            'mean_iterations': float(grp['n_iterations'].mean()),
            'mean_fallbacks': float(grp['n_fallbacks'].mean()),
            'total_instructions_removed': int(grp['size_reduction'].sum()),
            'all_correct': bool(grp['correct'].all()),
        }
        print(f'{sel:14} {mode:9} {grp["size_reduction_pct"].mean():9.2f}% '
              f'{grp["cost_reduction_pct"].mean():9.2f}% '
              f'{grp["n_iterations"].mean():6.2f} {grp["n_fallbacks"].mean():10.2f} '
              f'{str(grp["correct"].all()):>8}')

    # fraction of the oracle's achievable gain that each selector captures
    orc = runs[(runs.selector == 'oracle')].set_index('program_id')['cost_reduction']
    print()
    for sel in ['rule_based', 'ml_required8', 'ml_extended']:
        s = runs[(runs.selector == sel) & (runs['mode'] == 'strict')] \
            .set_index('program_id')['cost_reduction']
        frac = (s / orc.replace(0, pd.NA)).dropna().mean()
        summary[f'{sel}/strict']['oracle_gain_captured'] = float(frac)
        print(f'{sel:14} strict mode captures {100*frac:5.1f}% of the '
              f'oracle achievable weighted-cost gain')

    if incorrect:
        print(f'\n!! {len(incorrect)} runs changed program behaviour: {incorrect[:10]}')
    else:
        print(f'\nall {len(runs)} loop runs preserved program semantics')

    with open(os.path.join(OUT, 'loop_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------------------------ example traces
    sample = df.groupby('best_strategy', group_keys=False).head(1)['program_id'] \
               .tolist()[:args.traces]
    lines = []
    for pid in sample:
        with open(os.path.join(PROGRAMS, f'{pid}.src')) as fh:
            src = fh.read()
        prog = compile_to_tac(src)
        lines.append('#' * 78)
        lines.append(f'# SOURCE {pid}')
        lines.append('#' * 78)
        lines.append(src)
        lines.append('--- initial three-address code ---')
        lines.append(prog.numbered())
        lines.append('')
        for sel_name in ('rule_based', 'ml_extended'):
            run = optimize(prog, selectors[sel_name], mode='fallback',
                           program_id=pid, verify=True)
            lines.append(format_run(run))
            lines.append('')
            if sel_name == 'ml_extended':
                lines.append('--- final optimized three-address code ---')
                lines.append(run.final_program.numbered())
                lines.append('')
    text = '\n'.join(lines)
    with open(os.path.join(OUT, 'example_traces.txt'), 'w') as fh:
        fh.write(text)
    print(f'wrote {OUT}/example_traces.txt ({len(sample)} programs), '
          f'{OUT}/loop_runs.csv, {OUT}/loop_iterations.csv')


if __name__ == '__main__':
    main()
