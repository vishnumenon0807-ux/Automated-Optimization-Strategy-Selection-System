"""Strategy selectors.

A selector maps a feature dictionary to a score for each of the six
optimization strategies.  Three are provided:

* ``RuleBasedSelector``  - hand-weighted formulas over the eight required
  features; the specification's Step 3 baseline.
* ``MLSelector``         - a trained scikit-learn classifier; scores are class
  probabilities.
* ``OracleSelector``     - cheats by applying all six passes and measuring the
  result.  Not a usable strategy selector, but it bounds how much improvement
  is available and so calibrates the other two.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .features import REQUIRED_FEATURES, extract, vector
from .ir import TACProgram
from .optimizations import STRATEGIES, apply_strategy, program_cost


class Selector:
    name = 'selector'

    def scores(self, feats: Dict[str, float], prog: Optional[TACProgram] = None
               ) -> Dict[str, float]:
        raise NotImplementedError

    def select(self, feats, prog=None):
        """Return (best strategy, confidence, all scores).

        Confidence is the winning score's share of the total, so it is
        comparable between the rule-based and probabilistic selectors.
        """
        sc = self.scores(feats, prog)
        best = max(sc, key=lambda k: (sc[k], -STRATEGIES.index(k)))
        total = sum(max(v, 0.0) for v in sc.values())
        conf = (max(sc[best], 0.0) / total) if total > 0 else 0.0
        return best, conf, sc

    def ranking(self, feats, prog=None) -> List[str]:
        sc = self.scores(feats, prog)
        return sorted(STRATEGIES, key=lambda k: -sc[k])


class RuleBasedSelector(Selector):
    """Weighted linear scoring over the eight specified features.

    The weights are set from compiler intuition, not fitted to the data, so
    this is a genuine baseline.  Note that constant folding and constant
    propagation are necessarily scored from the same two inputs
    (``num_constants`` and ``const_ratio``) because the required feature set
    contains nothing that separates them - a limitation the results quantify.
    """

    name = 'rule_based'

    #: Weighted cost each strategy saves per unit of its driving feature.
    #: These six scalars are the only quantities the baseline takes from data:
    #: each is the mean observed gain per unit of driver across the corpus,
    #: rounded to two decimals.  Calibrating them this way rather than guessing
    #: keeps the baseline honest - an uncalibrated set of weights would lose to
    #: the classifiers for the trivial reason that its scores were on
    #: incomparable scales, which would prove nothing.
    YIELD = {
        'constant_folding': 0.20,
        'constant_propagation': 0.34,
        'common_subexpr_elimination': 0.67,
        'dead_code_elimination': 0.71,
        'algebraic_simplification': 0.75,
    }
    DRIVER = {
        'constant_folding': 'num_constants',
        'constant_propagation': 'num_constants',
        'common_subexpr_elimination': 'num_repeated_expr',
        'dead_code_elimination': 'num_dead_stmts',
        'algebraic_simplification': 'num_algebraic_ops',
    }

    #: Weighted cost saved per instruction inside a loop, calibrated the same
    #: way.  Hoisting one instruction from depth 1 to depth 0 saves 9 units,
    #: and roughly a tenth of loop instructions turn out to be safely hoistable.
    LICM_YIELD = 0.82

    def scores(self, feats, prog=None):
        n = max(feats['num_statements'], 1.0)
        # Opportunities are spread through the program, so on average a
        # fraction num_loop_ops/num_statements of them sit inside a loop and are
        # worth ten times as much.  Without this amplification the five
        # deletion strategies are scored on a different scale from loop
        # -invariant code motion and the comparison is meaningless.
        loop_frac = min(feats['num_loop_ops'] / n, 1.0)
        amp = 1.0 + 9.0 * loop_frac

        out = {}
        for strat, driver in self.DRIVER.items():
            out[strat] = self.YIELD[strat] * feats[driver] * amp
        # No amplification here: loop-invariant code motion is already scored
        # in loop-weighted units, since what it saves *is* the loop weight.
        out['loop_invariant_code_motion'] = self.LICM_YIELD * feats['num_loop_ops']
        return out


class MLSelector(Selector):
    """Wraps a fitted scikit-learn classifier."""

    def __init__(self, model, feature_names: List[str], name: str = 'ml'):
        self.model = model
        self.feature_names = feature_names
        self.name = name

    def scores(self, feats, prog=None):
        import numpy as np
        x = np.array([vector(feats, self.feature_names)], dtype=float)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(x)[0]
            classes = list(self.model.classes_)
            return {s: float(proba[classes.index(s)]) if s in classes else 0.0
                    for s in STRATEGIES}
        pred = self.model.predict(x)[0]
        return {s: (1.0 if s == pred else 0.0) for s in STRATEGIES}


class OracleSelector(Selector):
    """Upper bound: applies every pass and scores by the cost reduction each
    actually achieves."""

    name = 'oracle'

    def scores(self, feats, prog=None):
        if prog is None:
            raise ValueError('the oracle selector needs the program itself')
        base = program_cost(prog)
        out = {}
        for s in STRATEGIES:
            try:
                q, _ = apply_strategy(prog, s)
                out[s] = float(base - program_cost(q))
            except Exception:
                out[s] = 0.0
        return out
