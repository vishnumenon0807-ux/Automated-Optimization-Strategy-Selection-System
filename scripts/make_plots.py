"""Produce the six required figures.

Colours come from a categorical palette validated for colour-vision deficiency
(worst adjacent CVD delta-E 9.1, normal-vision 19.6 on the light surface).
Three of the six slots sit below 3:1 contrast against the surface, so every
figure carries visible labels or axis names -- identity is never colour alone.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optselect.optimizations import SHORT, STRATEGIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out')
FIG = os.path.join(OUT, 'figures')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#dedcd6'

# validated categorical slots 1-6, light mode
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
COLOR = {s: SERIES[i] for i, s in enumerate(STRATEGIES)}

# single-hue sequential ramp (blue), light -> dark
BLUES = ['#f2f7fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf',
         '#184f95', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seq_blue', BLUES)


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlepad': 10,
        'legend.frameon': False, 'figure.dpi': 200,
        'font.family': 'DejaVu Sans',
    })


def clean(ax, grid_axis='y'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}')
    return path


# ------------------------------------------------------- 1. confusion matrix
def fig_confusion(metrics):
    labels = [SHORT[s] for s in STRATEGIES]
    panels = [('required_8', 'RandomForest'), ('extended_14', 'RandomForest')]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, (fs, model) in zip(axes, panels):
        m = np.array(metrics['feature_sets'][fs]['models'][model]['confusion'],
                     dtype=float)
        norm = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if m[i, j] == 0:
                    continue
                ax.text(j, i, int(m[i, j]), ha='center', va='center',
                        fontsize=9, weight='bold',
                        color='#ffffff' if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, fontsize=8.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
        ax.set_xlabel('predicted strategy')
        acc = metrics['feature_sets'][fs]['models'][model]['test_accuracy']
        f1 = metrics['feature_sets'][fs]['models'][model]['test_macro_f1']
        n = len(metrics['feature_sets'][fs]['features'])
        ax.set_title(f'{n} features  ·  accuracy {acc:.1%}  ·  macro-F1 {f1:.2f}',
                     fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which='minor', color=SURFACE, linewidth=2)
        ax.tick_params(which='minor', length=0)
    axes[0].set_ylabel('actual best strategy')
    fig.suptitle('Random forest confusion: the required eight features cannot '
                 'separate folding from propagation',
                 fontsize=11.5, weight='bold', y=1.03)
    fig.text(0.5, -0.04, 'Counts on the 72-program held-out test set; shading is '
             'row-normalised recall.', ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, '1_confusion_matrix.png')


# ---------------------------------------------------- 2. feature importance
def fig_importance(metrics):
    imp = metrics['feature_sets']['extended_14']['feature_importance']
    req = set(metrics['feature_sets']['required_8']['features'])
    items = sorted(imp.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [SERIES[0] if n in req else SERIES[1] for n in names]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.barh(names, vals, color=colors, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f'{v:.3f}',
                va='center', fontsize=8, color=INK2)
    clean(ax, grid_axis='x')
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel('mean decrease in impurity')
    ax.set_title('Which features the random forest actually uses')
    ax.legend(handles=[Patch(facecolor=SERIES[0], label='required (8)'),
                       Patch(facecolor=SERIES[1], label='added (6)')],
              loc='lower right', fontsize=8.5)
    fig.text(0.02, -0.03, 'Three of the six added features rank in the top eight; '
             'num_foldable and num_prop_sites are what let folding and '
             'propagation be told apart.', fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, '2_feature_importance.png')


# ------------------------------------------- 3. confidence vs realised gain
def fig_confidence(iters):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    shown = [('ml_extended', 'ML selector (14 features)', SERIES[0]),
             ('rule_based', 'rule-based selector', SERIES[1])]
    for sel, label, col in shown:
        d = iters[(iters.selector == sel) & (iters.applied != 'none')].copy()
        if d.empty:
            continue
        d['bin'] = pd.cut(d['confidence'], bins=np.linspace(
            d['confidence'].min(), d['confidence'].max(), 9))
        grp = d.groupby('bin', observed=True).agg(
            conf=('confidence', 'mean'), gain=('cost_gain', 'mean'),
            n=('cost_gain', 'size')).dropna()
        ax.plot(grp['conf'], grp['gain'], color=col, linewidth=2,
                marker='o', markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
        last = grp.iloc[-1]
        ax.annotate(label, (last['conf'], last['gain']),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=8.5, color=col, weight='bold')
    clean(ax)
    ax.margins(y=0.16)
    ax.set_xlabel('decision confidence (winning score as a share of all scores)')
    ax.set_ylabel('mean weighted-cost gain realised')
    ax.set_title('Confidence predicts realised gain — for the learned selector only')
    ax.legend(loc='lower right', fontsize=8.5)
    fig.text(0.02, -0.08,
             'Each point is a confidence octile over every improving iteration of '
             'every program; y is the weighted cost actually saved by the applied '
             'pass.\nThe classifier\'s probability rises monotonically with the '
             'gain it goes on to achieve, so it can be trusted as a stopping '
             'signal. The rule-based\nscore share is not a probability and does '
             'not behave like one: its largest gains come at its *lowest* '
             'confidence.', fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, '3_confidence_vs_gain.png')


# ------------------------------------ 4. strategy per iteration, 5 programs
def fig_iteration_mix(iters, dataset):
    d = iters[(iters.selector == 'ml_extended') & (iters.applied != 'none')]
    sample = dataset.groupby('best_strategy', group_keys=False).head(1)
    sample = sample['program_id'].tolist()[:5]
    d = d[d.program_id.isin(sample)]

    counts = (d.groupby(['program_id', 'applied']).size()
              .unstack(fill_value=0).reindex(sample))
    for s in STRATEGIES:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[STRATEGIES]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bottom = np.zeros(len(counts))
    xs = np.arange(len(counts))
    for s in STRATEGIES:
        v = counts[s].values
        ax.bar(xs, v, bottom=bottom, color=COLOR[s], width=0.6,
               edgecolor=SURFACE, linewidth=2, label=SHORT[s])
        for x, (h, b) in enumerate(zip(v, bottom)):
            if h >= 2:
                ax.text(x, b + h / 2, SHORT[s], ha='center', va='center',
                        fontsize=8, weight='bold', color='#ffffff')
        bottom += v
    # Program ids carry the *generator bias*, not the true label, so labelling
    # the axis with the id would suggest four folding programs when in fact
    # these are five different labels.
    truth = dataset.set_index('program_id')['best_strategy'].to_dict()
    clean(ax)
    ax.set_xticks(xs, [f'program {i+1}\ntrue label: {SHORT[truth[p]]}'
                       for i, p in enumerate(counts.index)], fontsize=8)
    ax.set_ylabel('iterations in which the pass was applied')
    ax.set_title('Which passes the loop actually applies, per program')
    ax.legend(ncol=6, fontsize=8.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.16))
    fig.text(0.02, -0.13, 'Five programs, one per true label. No program is '
             'improved by a single pass: every one needs a different mixture, '
             'which is the case for selecting rather than fixing an order.',
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, '4_iteration_strategy_mix.png')


# --------------------------------------- 5. instruction reduction by strategy
def fig_box(gains):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    data = [gains[gains.strategy == s]['size_gain'].values for s in STRATEGIES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=INK2, linewidth=1),
                    capprops=dict(color=INK2, linewidth=1),
                    flierprops=dict(marker='o', markersize=3.5,
                                    markerfacecolor=INK2, markeredgecolor='none',
                                    alpha=0.45))
    for patch, s in zip(bp['boxes'], STRATEGIES):
        patch.set_facecolor(COLOR[s])
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    clean(ax)
    ax.set_xticks(range(1, len(STRATEGIES) + 1), [SHORT[s] for s in STRATEGIES])
    ax.set_ylabel('instructions removed by one application')
    ax.set_title('Instruction reduction per strategy, applied alone to every program')
    for i, s in enumerate(STRATEGIES, start=1):
        med = np.median(gains[gains.strategy == s]['size_gain'])
        ax.text(i, med, f'  {med:.0f}', va='bottom', ha='left', fontsize=8,
                color=INK2)
    fig.text(0.02, -0.04, 'Loop-invariant code motion sits at zero throughout: it '
             'relocates instructions rather than deleting them, which is why the '
             'labelling metric is loop-weighted cost, not raw instruction count.',
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, '5_reduction_by_strategy.png')


# ------------------------------ 6. num_constants vs constant-folding score
def fig_scatter(dataset):
    from optselect.selectors import RuleBasedSelector
    from optselect.features import EXTENDED_FEATURES
    sel = RuleBasedSelector()
    scores, is_cf = [], []
    for _, row in dataset.iterrows():
        feats = {k: float(row[k]) for k in EXTENDED_FEATURES}
        scores.append(sel.scores(feats)['constant_folding'])
        is_cf.append(row['best_strategy'] == 'constant_folding')
    d = pd.DataFrame({'num_constants': dataset['num_constants'].values,
                      'score': scores, 'is_cf': is_cf})

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for flag, col, lab in ((False, SERIES[1], 'folding is not the best strategy'),
                           (True, SERIES[0], 'folding IS the best strategy')):
        g = d[d.is_cf == flag]
        ax.scatter(g['num_constants'], g['score'], s=30, color=col, alpha=0.75,
                   edgecolor=SURFACE, linewidth=1.0, label=lab, zorder=3 if flag else 2)
    clean(ax, grid_axis='both')
    ax.set_xlabel('num_constants  (TAC lines whose operands are all literals)')
    ax.set_ylabel('rule-based constant-folding score')
    ax.set_title('The folding score rises with num_constants — but so does '
                 'everything else')
    ax.legend(loc='upper left', fontsize=8.5)
    fig.text(0.02, -0.04, 'The two populations overlap almost completely: '
             'num_constants tells you a constant-related pass will help, not '
             'which one. This is the required feature set\'s blind spot.',
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, '6_constants_vs_fold_score.png')


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    metrics = json.load(open(os.path.join(OUT, 'metrics.json')))
    dataset = pd.read_csv(os.path.join(OUT, 'dataset.csv'))
    gains = pd.read_csv(os.path.join(OUT, 'strategy_gains.csv'))
    iters = pd.read_csv(os.path.join(OUT, 'loop_iterations.csv'))

    print('rendering figures:')
    fig_confusion(metrics)
    fig_importance(metrics)
    fig_confidence(iters)
    fig_iteration_mix(iters, dataset)
    fig_box(gains)
    fig_scatter(dataset)
    print(f'\nfigures in {FIG}')


if __name__ == '__main__':
    main()
