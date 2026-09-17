"""Synthetic program generator for the training corpus.

Programs are grown from snippet templates, each of which plants a particular
kind of optimization opportunity.  A generation *bias* shifts the mix toward
one strategy, but the ground-truth label always comes from actually running all
six passes -- the bias only shapes the distribution, it never dictates the
label.

Loops are deliberately given to programs of every bias, not only the
loop-invariant-code-motion ones.  If only LICM programs contained loops then
``num_loop_ops > 0`` would be a giveaway and the classification problem would
be trivial; mixing them in forces the classifier to weigh a loop against what
is actually inside it.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .optimizations import STRATEGIES

ARITH = ['+', '-', '*']
NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'm', 'n', 'p', 'q', 'r',
         'u', 'v', 'w', 'x', 'y', 'z']

# Snippet kind that plants each strategy's opportunity.
EMPHASIS = {
    'constant_folding': 'const_expr',
    'constant_propagation': 'const_def_use',
    'common_subexpr_elimination': 'repeat',
    'dead_code_elimination': 'dead',
    'algebraic_simplification': 'algebraic',
    'loop_invariant_code_motion': None,      # planted by the loop body instead
}

SNIPPET_KINDS = ['const_expr', 'const_def_use', 'repeat', 'dead', 'algebraic']

#: Every program gets a background of *all five* snippet kinds and only a modest
#: extra helping of its own.  An earlier version planted one kind almost
#: exclusively; that made every class trivially separable and drove test
#: accuracy to 100%, which measures the generator rather than the selector.
#: With a competitive mixture the label genuinely depends on which opportunity
#: happens to be worth most, and close calls are common.
BACKGROUND = (0, 2)
EMPHASIS_EXTRA = (2, 5)

#: probability that a program of the given bias also contains a loop
LOOP_PROB = {
    'constant_folding': 0.55,
    'constant_propagation': 0.55,
    'common_subexpr_elimination': 0.60,
    'dead_code_elimination': 0.55,
    'algebraic_simplification': 0.55,
    'loop_invariant_code_motion': 1.00,
}

#: probability that a loop body matches the program's bias rather than being
#: drawn at random -- more competition between strategies inside the loop
LOOP_BODY_ON_BIAS = 0.75


class ProgramGenerator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # -- small helpers ----------------------------------------------------
    def lit(self, lo: int = 1, hi: int = 12) -> int:
        return self.rng.randint(lo, hi)

    def pick(self, pool: List[str]) -> str:
        return self.rng.choice(pool)

    def const_expression(self, terms: Optional[int] = None) -> str:
        terms = terms or self.rng.randint(2, 4)
        e = str(self.lit())
        for _ in range(terms - 1):
            e = f'{e} {self.rng.choice(ARITH)} {self.lit()}'
        return e

    def var_expression(self, pool: List[str], terms: int = 2) -> str:
        e = self.pick(pool)
        for _ in range(terms - 1):
            if self.rng.random() < 0.3:
                e = f'{e} {self.rng.choice(ARITH)} {self.lit()}'
            else:
                e = f'{e} {self.rng.choice(ARITH)} {self.pick(pool)}'
        return e

    # -- snippets ---------------------------------------------------------
    def snip_const_expr(self, lines, pool, fresh):
        v = fresh()
        lines.append(f'int {v} = {self.const_expression()};')
        pool.append(v)

    def snip_const_def_use(self, lines, pool, fresh):
        c, v = fresh(), fresh()
        lines.append(f'int {c} = {self.lit(2, 9)};')
        lines.append(f'int {v} = {c} {self.rng.choice(ARITH)} {self.pick(pool)};')
        lines.append(f'{v} = {v} {self.rng.choice(ARITH)} {c};')
        pool.extend([c, v])

    def snip_repeat(self, lines, pool, fresh):
        """Two statements sharing a multi-operator subexpression.

        The shared expression spans two TAC instructions, so each repetition is
        worth two eliminations rather than one; with a single-instruction
        subexpression, common-subexpression elimination is too weak to ever win
        a program outright.
        """
        a, b, c = self.pick(pool), self.pick(pool), self.pick(pool)
        op1, op2 = self.rng.choice(ARITH), self.rng.choice(ARITH)
        shared = f'{a} {op1} {b} {op2} {c}'
        u, v = fresh(), fresh()
        lines.append(f'int {u} = {shared} + {self.lit()};')
        lines.append(f'int {v} = {shared} - {self.lit()};')
        pool.extend([u, v])

    def snip_dead(self, lines, pool, fresh):
        d = 'dead_' + fresh()
        lines.append(f'int {d} = {self.var_expression(pool, 3)};')

    def snip_algebraic(self, lines, pool, fresh):
        v, src = fresh(), self.pick(pool)
        pattern = self.rng.choice(
            [f'{src} + 0', f'0 + {src}', f'{src} - 0', f'{src} * 1',
             f'1 * {src}', f'{src} * 0', f'{src} / 1'])
        lines.append(f'int {v} = {pattern};')
        pool.append(v)

    def snip_plain(self, lines, pool, fresh):
        v = fresh()
        lines.append(f'int {v} = {self.var_expression(pool, self.rng.randint(2, 3))};')
        pool.append(v)

    def snip_branch(self, lines, pool, fresh):
        v, cond = fresh(), self.pick(pool)
        lines.append(f'int {v} = 0;')
        lines.append(f'if ({cond} > {self.lit(3, 20)}) {{')
        lines.append(f'  {v} = {self.var_expression(pool, 2)};')
        lines.append('} else {')
        lines.append(f'  {v} = {self.var_expression(pool, 2)};')
        lines.append('}')
        pool.append(v)

    def snip_loop(self, lines, pool, fresh, bias):
        """A counted loop whose body usually, but not always, matches the bias."""
        if self.rng.random() > LOOP_BODY_ON_BIAS:
            bias = self.rng.choice(STRATEGIES)
        i, acc = 'i_' + fresh(), 'acc_' + fresh()
        bound = self.lit(4, 30)
        lines.append(f'int {acc} = 0;')
        lines.append(f'for ({i} = 0; {i} < {bound}; {i} = {i} + 1) {{')
        body_pool = list(pool)
        if bias == 'loop_invariant_code_motion':
            inv = 'inv_' + fresh()
            a, b, c = self.pick(pool), self.pick(pool), self.pick(pool)
            lines.append(f'  int {inv} = {a} * {b} + {c};')
            lines.append(f'  {acc} = {acc} + {inv} + {i};')
        elif bias == 'common_subexpr_elimination':
            # The shared expression must involve the loop counter.  If it were
            # built only from variables defined outside the loop it would be
            # loop-invariant, and hoisting it once would beat eliminating one of
            # two occurrences -- the program would be a LICM example, not a CSE
            # one.
            a, c = self.pick(pool), self.pick(pool)
            op1, op2 = self.rng.choice(ARITH), self.rng.choice(ARITH)
            shared = f'{a} {op1} {i} {op2} {c}'
            lines.append(f'  {acc} = {acc} + {shared} + {i};')
            lines.append(f'  {acc} = {acc} - ({shared});')
        elif bias == 'dead_code_elimination':
            d = 'dead_' + fresh()
            lines.append(f'  int {d} = {self.var_expression(body_pool, 2)} + {i};')
            lines.append(f'  {acc} = {acc} + {i};')
        elif bias == 'algebraic_simplification':
            a = self.pick(pool)
            lines.append(f'  {acc} = {acc} + {a} * 1 + {i} + 0;')
        elif bias == 'constant_folding':
            lines.append(f'  {acc} = {acc} + {self.const_expression(3)} + {i};')
        else:  # constant_propagation and anything else
            k = 'k_' + fresh()
            lines.append(f'  int {k} = {self.lit(2, 9)};')
            lines.append(f'  {acc} = {acc} + {k} * {i};')
        if self.rng.random() < 0.18:
            j = 'j_' + fresh()
            lines.append(f'  for ({j} = 0; {j} < 3; {j} = {j} + 1) {{')
            lines.append(f'    {acc} = {acc} + {j};')
            lines.append('  }')
        lines.append('}')
        pool.append(acc)

    # -- assembly ---------------------------------------------------------
    def generate(self, bias: str) -> str:
        rng = self.rng
        counter = {'n': 0}

        def fresh() -> str:
            counter['n'] += 1
            return f'{rng.choice(NAMES)}{counter["n"]}'

        lines: List[str] = []
        pool: List[str] = []
        # Most seed variables are read from input rather than initialised to a
        # literal.  If every seed were a literal, constant propagation could
        # dissolve almost any program and would win the label nearly every
        # time; opaque inputs keep the six strategies in genuine competition.
        n_seed = rng.randint(3, 4)
        n_const_seed = 1 if bias != 'constant_propagation' else rng.randint(1, 2)
        for j in range(n_seed):
            v = fresh()
            if j < n_const_seed:
                lines.append(f'int {v} = {self.lit(1, 15)};')
            else:
                lines.append(f'read({v});')
            pool.append(v)

        plan: List[str] = []
        for kind in SNIPPET_KINDS:
            plan += [kind] * rng.randint(*BACKGROUND)
        emph = EMPHASIS[bias]
        if emph is not None:
            plan += [emph] * rng.randint(*EMPHASIS_EXTRA)
        plan += ['plain'] * rng.randint(1, 3)
        rng.shuffle(plan)

        snippets = {
            'const_expr': self.snip_const_expr,
            'const_def_use': self.snip_const_def_use,
            'repeat': self.snip_repeat,
            'dead': self.snip_dead,
            'algebraic': self.snip_algebraic,
            'plain': self.snip_plain,
        }
        for kind in plan:
            snippets[kind](lines, pool, fresh)

        if rng.random() < LOOP_PROB[bias]:
            self.snip_loop(lines, pool, fresh, bias)
        if rng.random() < 0.35:
            self.snip_branch(lines, pool, fresh)

        # Every ordinary variable is printed, so the only dead code in a program
        # is what the `dead_` snippets deliberately planted.  If results were
        # left unprinted at random, dead-code elimination would win almost every
        # program by accident and the corpus would be useless.
        printable = [v for v in pool if not v.startswith(('dead_', 'inv_'))]
        rng.shuffle(printable)
        for v in printable:
            lines.append(f'print({v});')
        return '\n'.join(lines) + '\n'

    def generate_corpus(self, per_class: int = 30) -> List[tuple]:
        """Return [(program_id, bias, source), ...]."""
        out = []
        for bias in STRATEGIES:
            for k in range(per_class):
                out.append((f'{bias}_{k:03d}', bias, self.generate(bias)))
        return out
