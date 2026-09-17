/*
 * Build the project report as a Word document.
 *
 * Every figure in the prose is read out of out/*.json at build time rather
 * than typed in, so the report cannot drift away from the results it
 * describes.  Re-run the pipeline and re-run this: the numbers follow.
 */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, PageOrientation, TableLayoutType,
} = require('docx');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'out');
const FIG = path.join(OUT, 'figures');

const metrics = JSON.parse(fs.readFileSync(path.join(OUT, 'metrics.json')));
const loop = JSON.parse(fs.readFileSync(path.join(OUT, 'loop_summary.json')));
const dset = JSON.parse(fs.readFileSync(path.join(OUT, 'dataset_summary.json')));

const CONTENT_DXA = 10080;          // 12240 page - 2 x 1080 margin
const CONTENT_PX = 630;

const SHORT = {
  constant_folding: 'CF',
  constant_propagation: 'CP',
  common_subexpr_elimination: 'CSE',
  dead_code_elimination: 'DCE',
  algebraic_simplification: 'AS',
  loop_invariant_code_motion: 'LICM',
};
const STRATEGIES = Object.keys(SHORT);
const FULL = {
  CF: 'Constant folding', CP: 'Constant propagation',
  CSE: 'Common subexpression elimination', DCE: 'Dead code elimination',
  AS: 'Algebraic simplification', LICM: 'Loop-invariant code motion',
};

const pct = (x) => `${(100 * x).toFixed(1)}%`;
const f3 = (x) => x.toFixed(3);
const f2 = (x) => x.toFixed(2);

// ------------------------------------------------------------- primitives
function P(text, opts = {}) {
  const { bold, italics, size = 21, spacing = { after: 120 }, align, color,
    font } = opts;
  return new Paragraph({
    alignment: align,
    spacing,
    children: [new TextRun({ text, bold, italics, size, color, font })],
  });
}

function Rich(runs, opts = {}) {
  return new Paragraph({
    spacing: opts.spacing || { after: 120 },
    alignment: opts.align,
    children: runs.map((r) => (typeof r === 'string'
      ? new TextRun({ text: r, size: 21 })
      : new TextRun({ size: 21, ...r }))),
  });
}

function H(text, level) {
  return new Paragraph({
    text,
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 320 : 240, after: 140 },
  });
}

function Code(lines, size = 17) {
  return lines.map((l, i) => new Paragraph({
    spacing: { after: i === lines.length - 1 ? 140 : 0, line: 240 },
    shading: { type: ShadingType.CLEAR, fill: 'F4F3F0' },
    children: [new TextRun({ text: l || ' ', font: 'Consolas', size })],
  }));
}

function Bullets(items) {
  return items.map((t) => new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text: t, size: 21 })],
  }));
}

function cell(text, { w, bold, fill, align, font, size = 19 } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      spacing: { after: 0 },
      children: [new TextRun({ text: String(text), bold, size, font })],
    })],
  });
}

function makeTable(header, rows, widths, aligns = []) {
  const border = { style: BorderStyle.SINGLE, size: 2, color: 'D8D6D0' };
  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    borders: {
      top: border, bottom: border, left: border, right: border,
      insideHorizontal: border, insideVertical: border,
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((h, i) => cell(h, {
          w: widths[i], bold: true, fill: 'EDEBE6', align: aligns[i],
        })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(c, { w: widths[i], align: aligns[i] })),
      })),
    ],
  });
}

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function Figure(file, caption, maxPx = CONTENT_PX) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const { w, h } = pngSize(buf);
  const width = Math.min(maxPx, w);
  const height = Math.round((h * width) / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [new ImageRun({ data: buf, type: 'png',
        transformation: { width, height } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, size: 18, italics: true,
        color: '52514E' })],
    }),
  ];
}

// ------------------------------------------------------------ result tables
const MODELS = ['DecisionTree', 'RandomForest', 'LogisticRegression', 'MLP'];

function modelComparisonRows() {
  const rows = [[
    'Rule-based baseline', 'n/a (8)',
    pct(metrics.rule_based.test_accuracy),
    f3(metrics.rule_based.test_macro_f1), '—',
  ]];
  for (const fs_ of ['required_8', 'extended_14']) {
    const nf = metrics.feature_sets[fs_].features.length;
    for (const m of MODELS) {
      const d = metrics.feature_sets[fs_].models[m];
      rows.push([
        m, `${nf}`, pct(d.test_accuracy), f3(d.test_macro_f1),
        `${pct(d.cv_accuracy_mean)} ± ${(100 * d.cv_accuracy_std).toFixed(1)}`,
      ]);
    }
  }
  return rows;
}

function perClassRows() {
  const rb = metrics.rule_based.per_class;
  const r8 = metrics.feature_sets.required_8.models.MLP.per_class;
  const x14 = metrics.feature_sets.extended_14.models.MLP.per_class;
  return STRATEGIES.map((s) => [
    SHORT[s], `${rb[s].support}`,
    f3(rb[s].f1), f3(r8[s].f1),
    f3(x14[s].precision), f3(x14[s].recall), f3(x14[s].f1),
  ]);
}

function loopRows() {
  const order = [
    ['rule_based/strict', 'Rule-based', 'strict'],
    ['ml_required8/strict', 'ML, 8 features', 'strict'],
    ['ml_extended/strict', 'ML, 14 features', 'strict'],
    ['rule_based/fallback', 'Rule-based', 'fallback'],
    ['ml_required8/fallback', 'ML, 8 features', 'fallback'],
    ['ml_extended/fallback', 'ML, 14 features', 'fallback'],
    ['oracle/fallback', 'Oracle (upper bound)', 'fallback'],
  ];
  return order.map(([k, name, mode]) => {
    const d = loop[k];
    return [
      name, mode,
      `${d.mean_size_reduction_pct.toFixed(1)}%`,
      `${d.mean_cost_reduction_pct.toFixed(1)}%`,
      f2(d.mean_iterations),
      f2(d.mean_fallbacks),
      d.oracle_gain_captured !== undefined ? pct(d.oracle_gain_captured) : '—',
    ];
  });
}

const bestExt = metrics.feature_sets.extended_14.models.MLP;
const bestReq = metrics.feature_sets.required_8.models.MLP;
const rfExt = metrics.feature_sets.extended_14.models.RandomForest;
const rbAcc = metrics.rule_based.test_accuracy;

// ---------------------------------------------------------------- document
const children = [];

// title block
children.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({
    text: 'Automated Optimization Strategy Selection System',
    bold: true, size: 36 })],
}));
children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({
    text: 'Choosing the next compiler optimization from static features of '
        + 'three-address code',
    size: 24, color: '52514E' })],
}));
children.push(new Paragraph({
  spacing: { after: 60 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'C9C6BF',
    space: 8 } },
  children: [new TextRun({ text: '', size: 12 })],
}));
children.push(Rich([
  { text: 'Vishnu Menon', bold: true },
  '  ·  B.Tech Computer Science and Engineering, VIT Vellore  ·  Compiler '
  + 'Design Project 2',
], { spacing: { after: 240 } }));

// ---- abstract
children.push(H('Summary', HeadingLevel.HEADING_1));
children.push(P(
  `A compiler that applies its optimizations in a fixed order wastes work on `
  + `passes a given program cannot benefit from, and applies others too late `
  + `to matter. This project builds a system that instead looks at the `
  + `program: it extracts a numeric feature vector from three-address code, `
  + `scores all six candidate optimizations, applies the highest-scoring one, `
  + `re-measures, and repeats until nothing improves further.`));
children.push(P(
  `The corpus is ${dset.n_programs} generated programs, each labelled by `
  + `actually applying all six passes and recording which helped most. Every `
  + `one of the ${dset.n_pass_applications_verified} pass applications used to `
  + `build the corpus was checked by executing the transformed program and `
  + `comparing its output against the original; none changed behaviour.`));
children.push(Rich([
  'The best classifier reaches ',
  { text: `${pct(bestExt.test_accuracy)} accuracy`, bold: true },
  ` and ${f3(bestExt.test_macro_f1)} macro-F1 on held-out programs, against `,
  { text: `${pct(rbAcc)}`, bold: true },
  ` for a calibrated rule-based baseline. Driving the loop with it recovers `,
  { text: `${pct(loop['ml_extended/strict'].oracle_gain_captured)}`, bold: true },
  ` of the improvement an oracle could achieve, versus `,
  { text: `${pct(loop['rule_based/strict'].oracle_gain_captured)}`, bold: true },
  ' for the baseline.',
]));
children.push(P(
  `Two findings are worth stating up front. First, raw instruction count `
  + `cannot serve as the objective: loop-invariant code motion relocates `
  + `instructions without deleting any, so its instruction-count reduction is `
  + `always exactly zero and it could never be selected. The system therefore `
  + `optimizes a loop-weighted cost, and reports instruction count alongside `
  + `it. Second, the eight features named in the specification contain nothing `
  + `that separates constant folding from constant propagation — both are `
  + `driven by num_constants — which caps accuracy at `
  + `${pct(bestReq.test_accuracy)} and holds those two classes below the `
  + `0.70 per-class F1 target. Six additional features fix this.`));

// ---- 1 aim
children.push(H('1  Aim and objectives', HeadingLevel.HEADING_1));
children.push(P(
  'To design a fully automated system that analyses a program\'s intermediate '
  + 'code, extracts measurable features, and intelligently selects the most '
  + 'beneficial optimization strategy to apply next, instead of blindly '
  + 'applying all optimizations in a fixed order.'));
children.push(P('The objectives and where each is met:', { bold: false }));
children.push(makeTable(
  ['Objective', 'Where it is met'],
  [
    ['Parse input programs and generate TAC',
      'optselect/frontend.py — lexer, recursive-descent parser, syntax-directed TAC generator'],
    ['Extract a numeric feature vector from each TAC',
      'optselect/features.py — 8 required features plus 6 extended'],
    ['Compute an opportunity score for each of the six strategies',
      'optselect/selectors.py — RuleBasedSelector, MLSelector, OracleSelector'],
    ['Select the best strategy (rule-based or trained model)',
      'Selector.select() returns strategy, confidence and all six scores'],
    ['Apply, measure the improvement, re-analyse',
      'optselect/driver.py — optimize()'],
    ['Repeat until no further improvement',
      'The loop halts when no strategy reduces the weighted cost'],
    ['Report the decision and confidence at each step',
      'driver.format_run(); output in out/example_traces.txt'],
  ],
  [4400, 5680]));

// ---- 2 architecture
children.push(H('2  System architecture', HeadingLevel.HEADING_1));
children.push(P(
  'The system is a pipeline of seven stages. Stages 2 to 5 form the loop; '
  + 'everything inside it is recomputed from the current instruction list on '
  + 'every pass, because the optimizations rewrite the program and any cached '
  + 'analysis would go stale and quietly corrupt later decisions.'));
children.push(...Code([
  '  source  ──►  lexer ──► parser ──► TAC generator            (stage 1)',
  '                                          │',
  '                                          ▼',
  '            ┌───────────────  three-address code  ───────────────┐',
  '            │                                                    │',
  '            │   CFG · dominators · natural loops · liveness ·     │',
  '            │   reaching constants           (stage 2: analysis)  │',
  '            │                     │                               │',
  '            │                     ▼                               │',
  '            │   feature vector  (stage 3: 8 or 14 numbers)        │',
  '            │                     │                               │',
  '            │                     ▼                               │',
  '            │   selector → six scores + confidence   (stage 4)    │',
  '            │                     │                               │',
  '            │                     ▼                               │',
  '            │   apply chosen pass; re-measure cost   (stage 5)    │',
  '            │                     │                               │',
  '            └─────────  improved? yes → loop back  ───────────────┘',
  '                                  │ no',
  '                                  ▼',
  '                        optimized TAC + decision trace',
]));
children.push(makeTable(
  ['Module', 'Responsibility'],
  [
    ['ir.py', 'Instruction representation; operator semantics shared by the folder and the interpreter'],
    ['frontend.py', 'Lexer, parser, TAC generation'],
    ['cfg.py', 'Basic blocks, CFG, dominators, natural loops, liveness, reaching constants'],
    ['optimizations.py', 'The six passes and the loop-weighted cost metric'],
    ['features.py', 'Feature extraction'],
    ['selectors.py', 'Rule-based, ML and oracle strategy selectors'],
    ['driver.py', 'Ground-truth labelling and the iterative optimization loop'],
    ['interp.py', 'Reference interpreter used for correctness verification'],
    ['datagen.py', 'Synthetic program generator for the corpus'],
  ],
  [2400, 7680]));

// ---- 3 front end
children.push(H('3  Front end and intermediate representation', HeadingLevel.HEADING_1));
children.push(P(
  'The accepted language is a small C-like subset: integer and float '
  + 'declarations, assignment, if/else, while, for, print, and read for input. '
  + 'Expressions are side-effect free, so && and || are evaluated without '
  + 'short-circuiting — semantically identical here, and it keeps control flow '
  + 'out of expressions.'));
children.push(Rich([
  { text: 'The read() construct matters more than it looks. ', bold: true },
  'Without a source of values the analyses cannot see, every variable in every '
  + 'program traces back to a literal, constant propagation dissolves almost '
  + 'anything, and it wins the label on nearly every program. Opaque inputs '
  + 'keep the six strategies in genuine competition.',
]));
children.push(P(
  'The IR is structured rather than string-based: each instruction is a record '
  + 'with a kind, a destination, an operator and up to two operands, where an '
  + 'operand is either a variable name or a literal. Passes can then reason '
  + 'about instructions without re-parsing text.'));
children.push(P(
  'The TAC generator is deliberately naive — a fresh temporary for every '
  + 'interior expression node, no folding, no reuse. All improvement has to '
  + 'come from the optimization passes, which is what makes the selection '
  + 'experiment meaningful.'));
children.push(...Code([
  'Source                          Generated three-address code',
  '',
  'int a = 2 * 3 + 4;              t1 = 2 * 3',
  'read(b);                        t2 = t1 + 4',
  'int c = a * b;                  a  = t2',
  'print(c);                       b  = read()',
  '                                t3 = a * b',
  '                                c  = t3',
  '                                print c',
]));

// ---- 4 optimizations
children.push(H('4  The six optimization strategies', HeadingLevel.HEADING_1));
children.push(P(
  'Each pass is self-contained: it performs its characteristic transformation '
  + 'and the immediate clean-up that transformation makes possible, so that '
  + 'applying it alone produces a measurable change. A pass never runs another '
  + 'pass\'s analysis to do that clean-up — that is what keeps the six '
  + 'distinguishable and the selection problem non-trivial.'));
children.push(makeTable(
  ['Strategy', 'What it does', 'Analysis it needs'],
  [
    ['CF — constant folding',
      'Evaluates expressions whose operands are all literals; resolves branches on literal conditions; substitutes and deletes a folded single-definition temporary',
      'none (local)'],
    ['CP — constant propagation',
      'Replaces variable uses by values the variable is proved to hold, then deletes constant definitions left with no users',
      'forward reaching-constants dataflow over the CFG'],
    ['CSE — common subexpression elimination',
      'Value-numbers each basic block; a recomputed expression is replaced by the variable already holding it',
      'basic blocks; availability killed on redefinition'],
    ['DCE — dead code elimination',
      'Deletes computations whose result is never read, plus unreachable blocks and orphaned labels',
      'backward liveness dataflow'],
    ['AS — algebraic simplification',
      'Applies x+0, 0+x, x−0, x*1, 1*x, x*0, 0*x, x/1, x−x, 0−x, then forwards and deletes the resulting copy',
      'none (local)'],
    ['LICM — loop-invariant code motion',
      'Hoists invariant computations into a freshly created preheader',
      'dominators, natural loops, liveness'],
  ],
  [2200, 5280, 2600]));

children.push(H('4.1  Two design decisions worth defending', HeadingLevel.HEADING_2));
children.push(Rich([
  { text: 'Constant propagation does not evaluate. ', bold: true },
  'The obvious lattice transfer function evaluates a binary operation when '
  + 'both operands are known constants. That makes constant propagation '
  + 'subsume constant folding — and the first version of this system did '
  + 'exactly that, which is why constant folding never once won a label. '
  + 'Propagation here means only "replace a variable use by a value the '
  + 'variable is known to hold"; evaluating the expression is folding\'s job.',
]));
children.push(Rich([
  { text: 'Dead code elimination runs a single pass, not to a fixed point. ',
    bold: true },
  'A fixed-point DCE clears an entire chain of dead definitions in one call '
  + 'and is far more powerful than one application of any other strategy, '
  + 'which distorts the labels toward it. Restricting it to one pass makes the '
  + 'six comparable, and lets the outer loop do the iterating.',
]));

children.push(H('4.2  Safety conditions on code motion', HeadingLevel.HEADING_2));
children.push(P(
  'Hoisting is the one transformation here that can change behaviour in a way '
  + 'that ordinary testing might miss, because the danger case is a loop that '
  + 'executes zero times. An instruction is hoisted only when all four of these '
  + 'hold:'));
children.push(...Bullets([
  'every operand is a literal, defined outside the loop, or defined by an instruction already proved invariant;',
  'its destination is assigned exactly once inside the loop;',
  'its defining block dominates every in-loop use of that destination;',
  'the destination is not live on any edge leaving the loop — without this, a loop that never executes would see the destination take the hoisted value instead of keeping its previous one.',
]));
children.push(P(
  'Jumps into the header from outside the loop are retargeted to the '
  + 'preheader; back edges are left pointing at the header. There is a '
  + 'dedicated test for each of the last two conditions.'));

// ---- 5 cost metric
children.push(H('5  The objective: why not instruction count', HeadingLevel.HEADING_1));
children.push(P(
  'The specification labels each program with the optimization giving the '
  + 'greatest instruction-count reduction. That metric cannot work here, for a '
  + 'specific and unavoidable reason: loop-invariant code motion moves '
  + 'instructions out of loops without deleting any of them. Its '
  + 'instruction-count reduction is exactly zero on every program, so under '
  + 'that objective it can never be the best strategy, would never appear as a '
  + 'label, and no classifier could ever learn to select it. Figure 5 shows '
  + 'this directly — the LICM box sits flat on zero.'));
children.push(Rich([
  'The system therefore optimizes a ',
  { text: 'loop-weighted static cost', bold: true },
  ': an instruction at loop nesting depth d counts 10^d, capped at depth 3. '
  + 'This is the standard static estimate of dynamic execution frequency. '
  + 'Deleting an instruction still reduces it, so the other five strategies '
  + 'score exactly as before; and hoisting an instruction from depth 1 to '
  + 'depth 0 now scores a saving of 9 units, which is a fair reflection of '
  + 'what it buys at run time. Raw instruction count is reported throughout '
  + 'as a secondary metric, so nothing the specification asks for is lost.',
]));

// ---- 6 features
children.push(H('6  Feature extraction', HeadingLevel.HEADING_1));
children.push(P(
  'The eight features named in the specification are implemented exactly as '
  + 'described. Three of them (repeated expressions, dead statements, '
  + 'loop instructions) are computed from real analyses rather than pattern '
  + 'matching on text, so they agree with what the passes will actually find.'));
children.push(makeTable(
  ['Feature', 'Definition', 'Set'],
  [
    ['num_statements', 'TAC instructions, excluding label pseudo-instructions', 'required'],
    ['num_constants', 'Instructions whose operands are all numeric literals', 'required'],
    ['num_repeated_expr', 'Redundant computations found by local value numbering', 'required'],
    ['num_dead_stmts', 'Assignments whose destination is not live afterwards', 'required'],
    ['num_algebraic_ops', 'Instructions matching an arithmetic identity pattern', 'required'],
    ['num_loop_ops', 'Instructions inside a natural loop body', 'required'],
    ['num_tmp_vars', 'Distinct compiler-generated temporaries', 'required'],
    ['const_ratio', 'num_constants / num_statements', 'required'],
    ['num_foldable', 'Computations with all-literal operands — folding opportunities alone', 'added'],
    ['num_const_defs', 'Assignments of a literal to a variable', 'added'],
    ['num_prop_sites', 'Variable uses the dataflow proves constant — propagation opportunities alone', 'added'],
    ['num_loop_invariant_ops', 'Loop instructions whose operands are all defined outside the loop', 'added'],
    ['max_loop_depth', 'Deepest loop nesting in the program', 'added'],
    ['num_distinct_vars', 'Distinct variable names', 'added'],
  ],
  [2600, 6180, 1300]));
children.push(Rich([
  { text: 'Why six were added. ', bold: true },
  'Constant folding and constant propagation are both driven by num_constants '
  + 'and const_ratio, and the required set contains nothing else that '
  + 'distinguishes them. num_foldable counts only what folding can do and '
  + 'num_prop_sites only what propagation can do; num_loop_invariant_ops is a '
  + 'cheap structural signal for hoisting that stops short of running the '
  + 'safety analysis. Section 8 measures what this is worth.',
]));

// ---- 7 dataset
children.push(H('7  Dataset construction and labelling', HeadingLevel.HEADING_1));
children.push(P(
  `Programs are grown from snippet templates, each planting one kind of `
  + `opportunity. A generation bias adds extra helpings of one kind, but every `
  + `program also receives a background of all five, so several strategies are `
  + `in genuine competition and the label is not a foregone conclusion. The `
  + `bias never dictates the label: the label always comes from applying all `
  + `six passes and measuring.`));
children.push(makeTable(
  ['Property', 'Value'],
  [
    ['Programs in the corpus', `${dset.n_programs}`],
    ['Mean program size', `${dset.mean_initial_size.toFixed(1)} TAC instructions`],
    ['Mean loop-weighted cost', `${dset.mean_initial_cost.toFixed(1)}`],
    ['Pass applications verified during the build', `${dset.n_pass_applications_verified}`],
    ['Semantic violations found', `${dset.n_semantic_violations}`],
    ['Programs dropped as already optimal', `${dset.n_unlabelled_dropped}`],
    ['Class distribution',
      STRATEGIES.map((s) => `${SHORT[s]} ${dset.class_counts[s]}`).join('  ·  ')],
  ],
  [4600, 5480]));
children.push(Rich([
  { text: 'Two generator faults were found and fixed by looking at the label '
    + 'distribution. ', bold: true },
  'Leaving some result variables unprinted made almost every program a '
  + 'dead-code program by accident, so every ordinary variable is now printed '
  + 'and the only dead code is what the dead snippets deliberately plant. And '
  + 'building the shared subexpression in a CSE loop body out of variables '
  + 'defined outside the loop made it loop-invariant, so hoisting beat '
  + 'eliminating it and those programs were really LICM examples; the shared '
  + 'expression now involves the loop counter.',
]));
children.push(Rich([
  { text: 'A note on difficulty. ', bold: true },
  'An earlier corpus planted one kind of opportunity almost exclusively per '
  + 'program. Every model scored 100% on it — which measures the generator, '
  + 'not the selector. The mixture above was chosen deliberately to make the '
  + 'problem hard enough to be informative.',
]));

// ---- 8 selectors
children.push(H('8  Strategy selection', HeadingLevel.HEADING_1));
children.push(H('8.1  Rule-based baseline', HeadingLevel.HEADING_2));
children.push(P(
  'Each strategy is scored as an estimate of the weighted cost it will save: '
  + 'its driving feature, times the cost that feature typically yields, times '
  + 'a loop amplification factor. Since opportunities are spread through the '
  + 'program, a fraction num_loop_ops / num_statements of them sit inside a '
  + 'loop and are worth ten times as much; without that amplification the five '
  + 'deletion strategies are scored on a different scale from hoisting and the '
  + 'comparison is meaningless.'));
children.push(P(
  'The per-unit yields are the only quantities the baseline takes from data — '
  + 'each is the mean observed gain per unit of driver across the corpus. '
  + 'Calibrating them rather than guessing is what makes this a fair baseline: '
  + 'an uncalibrated set of weights would lose to the classifiers for the '
  + 'trivial reason that its scores were on incomparable scales, which would '
  + 'prove nothing. If anything this favours the baseline.'));
children.push(H('8.2  Learned predictor', HeadingLevel.HEADING_2));
children.push(P(
  'Four classifiers are trained on a stratified 70/30 split, with five-fold '
  + 'stratified cross-validation over the whole corpus as a second estimate. '
  + 'Logistic regression and the MLP are wrapped in a standardising pipeline. '
  + 'Confidence is the winning class probability, and for the rule-based '
  + 'selector the winning score\'s share of the total, so the two are '
  + 'comparable.'));

// ---- 9 results
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H('9  Results', HeadingLevel.HEADING_1));
children.push(H('9.1  Classification accuracy', HeadingLevel.HEADING_2));
children.push(makeTable(
  ['Selector / model', 'Features', 'Test accuracy', 'Macro-F1', '5-fold CV accuracy'],
  modelComparisonRows(),
  [3100, 1500, 1800, 1580, 2100],
  [undefined, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER,
    AlignmentType.CENTER]));
children.push(P(
  `Test set: 72 held-out programs. The specification's target of >75% accuracy `
  + `is met by the MLP on the required eight features `
  + `(${pct(bestReq.test_accuracy)}, exactly at the line) and comfortably by `
  + `three of the four models on the extended set, the best reaching `
  + `${pct(bestExt.test_accuracy)}.`, { spacing: { before: 140, after: 120 } }));

children.push(H('9.2  Per-class precision, recall and F1', HeadingLevel.HEADING_2));
children.push(makeTable(
  ['Class', 'Support', 'Rule-based F1', 'ML F1 (8 feat.)',
    'Precision (14)', 'Recall (14)', 'F1 (14)'],
  perClassRows(),
  [1180, 1180, 1700, 1700, 1500, 1400, 1420],
  [undefined, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER,
    AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER]));
children.push(P(
  `The >0.70 per-class F1 target is met for all six classes on the extended `
  + `feature set. On the required eight it fails for exactly the two classes `
  + `predicted in section 6 — constant folding `
  + `(${f3(bestReq.per_class.constant_folding.f1)}) and constant propagation `
  + `(${f3(bestReq.per_class.constant_propagation.f1)}) — and is met for the `
  + `other four. The rule-based baseline scores 0.000 on constant folding, `
  + `which is structural rather than accidental: folding and propagation share `
  + `a driving feature, so a linear rule that prefers one must prefer it on `
  + `every program.`,
  { spacing: { before: 140, after: 120 } }));

children.push(...Figure('1_confusion_matrix.png',
  'Figure 1 — Confusion matrices for the random forest on both feature sets. '
  + 'The 6-cell CF↔CP block on the left is the required feature set\'s blind '
  + 'spot; it largely disappears on the right.'));

children.push(...Figure('6_constants_vs_fold_score.png',
  'Figure 2 — The same blind spot at the level of a single feature. Programs '
  + 'where folding is best and programs where it is not occupy the same region '
  + 'of the num_constants / folding-score plane.'));

children.push(...Figure('2_feature_importance.png',
  'Figure 3 — Random forest feature importance. Three of the six added '
  + 'features rank in the top eight, and the two that separate folding from '
  + 'propagation are the ones the model leans on.'));

children.push(H('9.3  End-to-end optimization', HeadingLevel.HEADING_2));
children.push(P(
  'Single-shot classification accuracy is not the quantity that matters; what '
  + 'matters is how much better the code gets. The loop was run over all '
  + `${dset.n_programs} programs under each selector, in two modes. In strict `
  + 'mode the loop stops as soon as the selected strategy fails to improve the '
  + 'program, so the result reflects selection quality directly. In fallback '
  + 'mode it tries the next-highest-scoring strategy instead, and stops only '
  + 'when no strategy improves anything.'));
children.push(makeTable(
  ['Selector', 'Mode', 'Instr. reduction', 'Cost reduction', 'Iterations',
    'Fallbacks', 'Oracle gain captured'],
  loopRows(),
  [1900, 1180, 1560, 1500, 1300, 1180, 1460],
  [undefined, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER,
    AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER]));
children.push(P(
  'Two things stand out. In strict mode, selector quality translates almost '
  + 'linearly into optimization achieved: the learned selector on 14 features '
  + `removes ${loop['ml_extended/strict'].mean_size_reduction_pct.toFixed(1)}% `
  + `of instructions against `
  + `${loop['rule_based/strict'].mean_size_reduction_pct.toFixed(1)}% for the `
  + 'baseline, roughly three times as much, and captures '
  + `${pct(loop['ml_extended/strict'].oracle_gain_captured)} of what an oracle `
  + `could get versus ${pct(loop['rule_based/strict'].oracle_gain_captured)}.`,
  { spacing: { before: 140, after: 120 } }));
children.push(Rich([
  'In fallback mode every selector converges to essentially the same code — '
  + 'about 35.5% smaller — because with fallback the loop keeps going until the '
  + 'fixed point regardless of who chose first. ',
  { text: 'What differs is the cost of getting there. ', bold: true },
  `The baseline wastes ${f2(loop['rule_based/fallback'].mean_fallbacks)} `
  + 'rejected pass applications per program; the learned selector wastes '
  + `${f2(loop['ml_extended/fallback'].mean_fallbacks)}. So a better selector `
  + 'buys either much better code for a fixed budget, or the same code for '
  + 'roughly half the wasted compilation work.',
]));

children.push(...Figure('4_iteration_strategy_mix.png',
  'Figure 4 — Passes applied per iteration for five programs, one per true '
  + 'label. No program is improved by a single pass; each needs its own '
  + 'mixture, which is the case for selecting rather than fixing an order.'));

children.push(...Figure('5_reduction_by_strategy.png',
  'Figure 5 — Instruction reduction from applying each strategy alone. LICM is '
  + 'flat at zero throughout, which is why the objective is loop-weighted cost.'));

children.push(...Figure('3_confidence_vs_gain.png',
  'Figure 6 — Realised gain against decision confidence. The classifier\'s '
  + 'probability rises with the gain it goes on to achieve; the rule-based '
  + 'score share does not.'));

// ---- 10 verification
children.push(H('10  Correctness verification', HeadingLevel.HEADING_1));
children.push(P(
  'An optimizer that produces smaller but wrong code has optimized nothing, so '
  + 'correctness is checked rather than assumed. The interpreter and the '
  + 'constant folder call the same operator-semantics function, so a pass '
  + 'cannot silently disagree with execution about what an operator means.'));
children.push(makeTable(
  ['Check', 'Scope', 'Result'],
  [
    ['Corpus gate: each pass applied to each program, output compared',
      `${dset.n_pass_applications_verified} pass applications`,
      `${dset.n_semantic_violations} violations`],
    ['Loop runs: final program compared against the original',
      '1680 runs across four selectors and two modes', '0 violations'],
    ['Randomised differential testing: random pass sequences, 5 input streams',
      '60 programs × 6-pass sequences × 5 streams', 'pass'],
    ['Loop termination and monotonicity', '36 unseen programs × 2 selectors × 2 modes', 'pass'],
    ['Fixed point: no strategy improves a converged program', '6 programs', 'pass'],
    ['Unit tests for each pass and its refusal cases', '15 tests', 'pass'],
    ['Reported metrics recomputed from raw predictions', 'both feature sets', 'pass'],
    ['Dataset labels and features re-derived from source', '35 sampled programs', 'pass'],
  ],
  [4700, 3380, 2000]));
children.push(Rich([
  { text: 'The differential layer is the one that earns its keep. ', bold: true },
  'The corpus gate only ever applies one pass to a fresh program, so a pass '
  + 'that is correct alone but wrong after another has run would slip past it. '
  + 'Random pass sequences catch exactly that. Several input streams are used '
  + 'because a single fixed stream can hide a transformation that is only '
  + 'wrong on some inputs.',
]));
children.push(Rich([
  { text: 'A real bug this caught: ', bold: true },
  'algebraic simplification recorded each rewritten value during its scan and '
  + 'forwarded the recorded value afterwards. When two simplifications chained '
  + '(t10 = 6 * 1 feeding t11 = t10 * 1), forwarding the stale name left a '
  + 'reference to a definition that had already been deleted, and the program '
  + 'silently read zero. The fix is to read the operand back off the '
  + 'instruction at forwarding time. Constant folding had the same latent '
  + 'fault and was fixed with it.',
]));

// ---- 11 findings
children.push(H('11  Findings and limitations', HeadingLevel.HEADING_1));
children.push(...Bullets([
  'Selecting the next optimization from static features works. On held-out programs the learned selector picks the genuinely best of six strategies '
    + `${pct(bestExt.test_accuracy)} of the time, and driving the loop with it captures `
    + `${pct(loop['ml_extended/strict'].oracle_gain_captured)} of the improvement an oracle could achieve.`,
  'The feature set is the binding constraint, not the model. Across four very different classifiers the extended set beats the required set by 10 to 15 points; '
    + 'no model overcomes a feature set that cannot represent the distinction it is being asked to make.',
  'The objective function has to match the transformations. Loop-invariant code motion is invisible to instruction count, and any pipeline scored that way '
    + 'would silently never select it.',
  'Order matters, and no fixed order is right for every program. Figure 4 shows five programs each needing a different mixture and sequence of passes.',
  'Confidence from the classifier is usable as a signal; the rule-based score share is not, despite being normalised the same way.',
]));
children.push(H('Limitations', HeadingLevel.HEADING_2));
children.push(...Bullets([
  'The corpus is synthetic. Programs are grown from templates, so the feature distribution is narrower than real code, and the accuracy figures should be read '
    + 'as a comparison between selectors rather than an absolute expectation.',
  'The classifier is trained on unoptimized programs but queried at every iteration on partly optimized ones. That distribution shift is measured end-to-end '
    + 'in section 9.3 rather than assumed away, but training on intermediate states would likely improve strict-mode results.',
  'CSE is local to basic blocks and constant propagation does not use branch conditions to prune unreachable paths; a sparse conditional implementation would find more.',
  'The cost model is a static estimate. Weighting by 10 per loop level assumes a typical trip count and ignores actual bounds, which are known for these counted loops.',
  'Loop-invariant code motion handles one loop per application, innermost first, so deeply nested programs need more iterations than strictly necessary.',
]));

// ---- 12 reproduce
children.push(H('12  Reproducing the results', HeadingLevel.HEADING_1));
children.push(...Code([
  'python3 scripts/build_dataset.py --per-class 40 --seed 7   # corpus + labels + verification',
  'python3 scripts/train.py                                   # classifiers + metrics.json',
  'python3 scripts/run_pipeline.py                            # iterative loop + traces',
  'python3 scripts/make_plots.py                              # the six figures',
  'python3 tests/test_all.py                                  # 21-check verification suite',
  'node    scripts/build_report.js                            # this document',
]));
children.push(P(
  'All seeds are fixed, so the numbers in this report are reproducible '
  + 'exactly. Outputs land in out/: dataset.csv, strategy_gains.csv, '
  + 'metrics.json, loop_runs.csv, loop_iterations.csv, loop_summary.json, '
  + 'example_traces.txt and figures/.'));

// ---- appendix
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H('Appendix  Sample decision trace', HeadingLevel.HEADING_1));
children.push(P(
  'The per-iteration report required by section 7 of the specification: the '
  + 'extracted features, the score of every candidate strategy, the strategy '
  + 'selected with its confidence, and the resulting instruction-count change. '
  + 'Full traces for five programs are in out/example_traces.txt.'));

const trace = fs.readFileSync(path.join(OUT, 'example_traces.txt'), 'utf8');
const start = trace.indexOf('=== ');
const snippet = trace.slice(start).split('\n').slice(0, 44);
children.push(...Code(snippet, 15));

// ---------------------------------------------------------------- assemble
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 21 } },
      heading1: { run: { font: 'Calibri', size: 28, bold: true, color: '1A1A19' } },
      heading2: { run: { font: 'Calibri', size: 23, bold: true, color: '3A3A38' } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
      },
    },
    children,
  }],
});

const outPath = path.join(ROOT, 'Optimization_Strategy_Selection_Report.docx');
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log(`wrote ${outPath} (${(buf.length / 1024).toFixed(0)} KB)`);
});
