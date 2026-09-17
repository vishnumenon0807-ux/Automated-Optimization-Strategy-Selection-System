"""Run the whole pipeline end to end.

    python run_all.py

Uses sys.executable, so it works whether you invoke it as `python`, `python3`
or `py -3`, on Windows, macOS or Linux.  Stages run in order and the script
stops at the first failure.

Options:
    --per-class N   programs generated per class (default 40 -> 240 programs)
    --seed N        generator seed (default 7)
    --quick         a small corpus, for checking the setup works (~1 minute)
    --skip-report   don't build the Word report (skips the Node step)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(label, cmd, optional=False):
    print(f'\n{"=" * 70}\n{label}\n{"=" * 70}')
    print('  $ ' + ' '.join(os.path.basename(c) if i == 0 else c
                            for i, c in enumerate(cmd)))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if result.returncode != 0:
        if optional:
            print(f'\n  [skipped] {label} failed after {dt:.0f}s; continuing.')
            return False
        print(f'\n  FAILED: {label} (exit {result.returncode}) after {dt:.0f}s')
        sys.exit(result.returncode)
    print(f'\n  done in {dt:.0f}s')
    return True


def check_imports():
    missing = []
    for mod, pkg in (('numpy', 'numpy'), ('pandas', 'pandas'),
                     ('sklearn', 'scikit-learn'), ('matplotlib', 'matplotlib')):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print('Missing Python packages: ' + ', '.join(missing))
        print(f'Install them with:\n  {PY} -m pip install ' + ' '.join(missing))
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=40)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--quick', action='store_true',
                    help='small corpus, just to check the setup works')
    ap.add_argument('--skip-report', action='store_true')
    args = ap.parse_args()

    per_class = 5 if args.quick else args.per_class
    check_imports()

    print(f'Python  : {sys.version.split()[0]}  ({PY})')
    print(f'Project : {ROOT}')
    print(f'Corpus  : {per_class} programs per class '
          f'({per_class * 6} total), seed {args.seed}')
    if args.quick:
        print('QUICK MODE: the corpus is too small for meaningful accuracy '
              'figures; this only checks that everything runs.')

    started = time.time()
    run('1/5  Build and verify the program corpus',
        [PY, os.path.join('scripts', 'build_dataset.py'),
         '--per-class', str(per_class), '--seed', str(args.seed)])
    run('2/5  Train the classifiers',
        [PY, os.path.join('scripts', 'train.py')])
    run('3/5  Run the iterative optimization loop',
        [PY, os.path.join('scripts', 'run_pipeline.py')])
    run('4/5  Render the figures',
        [PY, os.path.join('scripts', 'make_plots.py')])
    run('5/5  Verification suite',
        [PY, os.path.join('tests', 'test_all.py')])

    if not args.skip_report:
        node = shutil.which('node')
        if node is None:
            print('\n  [skipped] report: Node.js not found on PATH. '
                  'The rest of the pipeline is unaffected.')
        else:
            run('extra  Build the Word report',
                [node, os.path.join('scripts', 'build_report.js')],
                optional=True)

    print(f'\n{"=" * 70}')
    print(f'All stages completed in {time.time() - started:.0f}s.')
    print('Results are in out/  —  metrics.json, loop_summary.json, '
          'example_traces.txt, figures/')


if __name__ == '__main__':
    main()
