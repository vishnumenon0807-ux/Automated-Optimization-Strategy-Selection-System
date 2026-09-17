"""Generate the program corpus, verify every optimization on it, extract
features and write the labelled dataset.

Verification is not optional here: for every program and every strategy, the
transformed program is executed against the same input stream as the original
and the printed output must match exactly.  A dataset built on unsound passes
would be measuring nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optselect import interp
from optselect.datagen import ProgramGenerator
from optselect.driver import label_program
from optselect.features import EXTENDED_FEATURES, extract
from optselect.frontend import compile_to_tac
from optselect.optimizations import (SHORT, STRATEGIES, apply_strategy,
                                     program_cost)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out')
PROGRAMS = os.path.join(ROOT, 'programs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=30)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--max-steps', type=int, default=400_000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PROGRAMS, exist_ok=True)

    gen = ProgramGenerator(seed=args.seed)
    corpus = gen.generate_corpus(per_class=args.per_class)
    print(f'generated {len(corpus)} candidate programs')

    rows, gain_rows = [], []
    verified = violations = unlabelled = errors = 0

    for pid, bias, src in corpus:
        try:
            prog = compile_to_tac(src)
            reference = interp.run(prog, max_steps=args.max_steps)
        except Exception as exc:
            errors += 1
            print(f'  ! {pid}: {type(exc).__name__}: {exc}')
            continue

        # -- correctness gate: every pass must preserve observable behaviour
        gains, bad = {}, []
        base_cost, base_size = program_cost(prog), prog.size()
        for s in STRATEGIES:
            try:
                q, _ = apply_strategy(prog, s)
                out = interp.run(q, max_steps=args.max_steps)
            except Exception as exc:
                bad.append(f'{SHORT[s]}:{type(exc).__name__}')
                gains[s] = (0, 0)
                continue
            if not interp.outputs_match(out, reference):
                bad.append(f'{SHORT[s]}:output')
            gains[s] = (base_cost - program_cost(q), base_size - q.size())
            verified += 1
        if bad:
            violations += 1
            print(f'  ! {pid}: semantics violated by {bad}')
            continue

        label, _ = label_program(prog)
        if label is None:
            unlabelled += 1
            continue

        feats = extract(prog)
        row = {'program_id': pid, 'bias': bias}
        row.update({k: feats[k] for k in EXTENDED_FEATURES})
        row['initial_size'] = base_size
        row['initial_cost'] = base_cost
        row['best_strategy'] = label
        rows.append(row)

        for s in STRATEGIES:
            gain_rows.append({'program_id': pid, 'strategy': s,
                              'cost_gain': gains[s][0], 'size_gain': gains[s][1],
                              'best_strategy': label})

        with open(os.path.join(PROGRAMS, f'{pid}.src'), 'w') as fh:
            fh.write(src)

    df = pd.DataFrame(rows)
    gdf = pd.DataFrame(gain_rows)
    df.to_csv(os.path.join(OUT, 'dataset.csv'), index=False)
    gdf.to_csv(os.path.join(OUT, 'strategy_gains.csv'), index=False)

    summary = {
        'n_programs': int(len(df)),
        'n_pass_applications_verified': int(verified),
        'n_semantic_violations': int(violations),
        'n_unlabelled_dropped': int(unlabelled),
        'n_compile_errors': int(errors),
        'class_counts': {k: int(v) for k, v in
                         Counter(df['best_strategy']).items()},
        'mean_initial_size': float(df['initial_size'].mean()),
        'mean_initial_cost': float(df['initial_cost'].mean()),
    }
    with open(os.path.join(OUT, 'dataset_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f'\nkept {len(df)} labelled programs '
          f'({unlabelled} dropped as already optimal, {violations} dropped '
          f'for semantic violations, {errors} compile/run errors)')
    print(f'verified {verified} pass applications, all semantics preserved')
    print('\nclass distribution:')
    for s in STRATEGIES:
        n = int((df['best_strategy'] == s).sum())
        print(f'  {SHORT[s]:5} {s:30} {n:4d}  ({100*n/max(len(df),1):.1f}%)')
    print(f'\nmean program size {df["initial_size"].mean():.1f} instructions, '
          f'mean weighted cost {df["initial_cost"].mean():.1f}')
    print(f'\nwrote {OUT}/dataset.csv and {OUT}/strategy_gains.csv')


if __name__ == '__main__':
    main()
