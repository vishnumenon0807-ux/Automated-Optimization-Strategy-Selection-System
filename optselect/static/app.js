'use strict';

const $ = id => document.getElementById(id);

const EXAMPLES = {
  'Mixed — several strategies compete': `read(a);
read(b);
int k = 5;
int c = 2 * 3 + 4;
int d = a + 0;
int s = 0;
for (i = 0; i < 12; i = i + 1) {
  int inv = a * b;
  s = s + inv + k * i;
}
int e = a * b + 1;
int f = a * b + 2;
print(c); print(d); print(s); print(e); print(f);`,

  'Constant folding': `int a = 2 * 3 + 4;
int b = 10 * 10 - 50;
int c = 7 * 8 + 6 * 2;
read(x);
int d = a + b + c + x;
print(d);`,

  'Constant propagation': `int k = 6;
read(a);
int u = k * a;
int v = u + k;
int w = k + k + a;
print(u); print(v); print(w);`,

  'Common subexpression': `read(a);
read(b);
read(c);
int u = a * b + c + 1;
int v = a * b + c - 2;
int w = a * b + c * 3;
print(u); print(v); print(w);`,

  'Dead code': `read(a);
read(b);
int live = a + b;
int unused1 = a * 99 + b;
int unused2 = unused1 * 3 + a;
int unused3 = unused2 - b;
print(live);`,

  'Algebraic identities': `read(a);
read(b);
int p = a + 0;
int q = b * 1;
int r = a * 0 + b;
int t = a / 1 - 0;
print(p); print(q); print(r); print(t);`,

  'Loop-invariant code motion': `read(a);
read(b);
int s = 0;
for (i = 0; i < 25; i = i + 1) {
  int inv = a * b + 7;
  s = s + inv + i;
}
print(s);`
};

let DATA = null;
let STEP = 0;

// ---- examples -------------------------------------------------------------
const exSel = $('ex');
for (const name of Object.keys(EXAMPLES)) {
  const o = document.createElement('option');
  o.value = name;
  o.textContent = name;
  exSel.appendChild(o);
}
exSel.addEventListener('change', () => { $('src').value = EXAMPLES[exSel.value]; });
$('src').value = EXAMPLES[Object.keys(EXAMPLES)[0]];

// ---- run ------------------------------------------------------------------
$('run').addEventListener('click', async () => {
  const btn = $('run');
  btn.disabled = true;
  btn.textContent = 'Optimizing…';
  showMessage('', null);

  try {
    const res = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: $('src').value,
        selector: $('sel').value,
        mode: $('mode').value
      })
    });
    const data = await res.json();
    if (!data.ok) {
      $('results').hidden = true;
      showMessage(data.error, 'error');
    } else {
      DATA = data;
      STEP = 0;
      $('results').hidden = false;
      renderSummary();
      renderTimeline();
      renderStep();
    }
  } catch (err) {
    showMessage('Could not reach the server: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Optimize';
  }
});

function showMessage(text, kind) {
  const el = $('msg');
  if (!text) { el.hidden = true; return; }
  el.className = 'msg' + (kind ? ' ' + kind : '');
  el.textContent = text;
  el.hidden = false;
}

// ---- summary --------------------------------------------------------------
function renderSummary() {
  const s = DATA.summary;
  const outputsMatch =
    JSON.stringify(DATA.output_before) === JSON.stringify(DATA.output_after);
  const safe = outputsMatch && s.correct;

  $('summary').innerHTML = `
    <div>
      <dt>Instructions</dt>
      <dd>${DATA.initial.size} &rarr; ${DATA.final.size}
        <small>${s.size_reduction_pct.toFixed(1)}% smaller</small></dd>
    </div>
    <div>
      <dt>Weighted cost</dt>
      <dd>${DATA.initial.cost} &rarr; ${DATA.final.cost}
        <small>${s.cost_reduction_pct.toFixed(1)}% cheaper</small></dd>
    </div>
    <div>
      <dt>Iterations</dt>
      <dd>${s.n_iterations}
        <small>${s.n_fallbacks} fallback${s.n_fallbacks === 1 ? '' : 's'}</small></dd>
    </div>
    <div>
      <dt>Semantics</dt>
      <dd class="${safe ? 'ok' : 'bad'}">${safe ? 'preserved' : 'CHANGED'}
        <small>output verified</small></dd>
    </div>`;
}

// ---- timeline -------------------------------------------------------------
function renderTimeline() {
  const box = $('timeline');
  box.innerHTML = '';
  DATA.iterations.forEach((it, idx) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip' + (idx === STEP ? ' on' : '');
    b.textContent = it.converged ? 'done' : `${it.iteration} ${it.applied}`;
    b.title = it.converged
      ? 'converged — no strategy improves the program further'
      : `iteration ${it.iteration}: applied ${it.applied}`;
    b.addEventListener('click', () => {
      STEP = idx;
      renderTimeline();
      renderStep();
    });
    box.appendChild(b);
  });
}

// ---- one step -------------------------------------------------------------
function renderStep() {
  const it = DATA.iterations[STEP];

  $('decTitle').textContent = it.converged
    ? 'Decision — converged'
    : `Decision — iteration ${it.iteration}`;

  // scores, highest first
  const ordered = [...DATA.strategies].sort((a, b) => it.scores[b] - it.scores[a]);
  $('scores').innerHTML = ordered.map(k => {
    const win = k === it.selected;
    return `<tr class="${win ? 'win' : ''}" title="${DATA.strategy_names[k]}">
      <td>${k}</td>
      <td class="r">${it.scores[k].toFixed(2)}</td>
      <td class="mark">${win ? '&larr;' : ''}</td>
    </tr>`;
  }).join('');

  // verdict
  const parts = [];
  parts.push(`<p>Highest score: <b>${DATA.strategy_names[it.selected]}</b>
    (${it.selected}) &middot; confidence
    <b>${(it.confidence * 100).toFixed(0)}%</b></p>`);

  if (it.converged) {
    parts.push(`<p class="aside">No strategy reduces the cost any further &mdash;
      the loop stops here.</p>`);
  } else {
    if (it.fell_back) {
      parts.push(`<p class="aside">${it.selected} gave no improvement, so the loop
        fell back to the next-highest scorer, ${it.applied}.</p>`);
    }
    const delta = d => d === 0 ? 'unchanged' : (d > 0 ? '−' : '+') + Math.abs(d);
    parts.push(`<p>Applied <b>${it.applied}</b> &mdash; instructions
      <b>${it.size_before} &rarr; ${it.size_after}</b>
      (${delta(it.size_before - it.size_after)}), weighted cost
      <b>${it.cost_before} &rarr; ${it.cost_after}</b>
      (${delta(it.cost_before - it.cost_after)}).</p>`);

    if (it.applied === 'LICM' && it.size_before === it.size_after) {
      parts.push(`<p class="aside">Hoisting moves instructions out of the loop
        rather than deleting them, so the instruction count does not change
        &mdash; only the loop-weighted cost does. This is why raw instruction
        count cannot be the objective.</p>`);
    }
  }
  $('verdict').innerHTML = parts.join('');

  // the eight required features
  const REQUIRED = ['num_statements', 'num_constants', 'num_repeated_expr',
    'num_dead_stmts', 'num_algebraic_ops', 'num_loop_ops', 'num_tmp_vars',
    'const_ratio'];
  $('feat').textContent = REQUIRED.map(n => {
    const v = it.features[n];
    return `${n}=${Number.isInteger(v) ? v : v.toFixed(3)}`;
  }).join('   ');

  // code
  $('codeTitle').textContent = it.converged
    ? `Three-address code — final (${DATA.final.size} instructions)`
    : `Three-address code — what ${it.applied} changed`;

  let n = 0;
  $('code').innerHTML = it.diff.map(line => {
    const isLabel = /^\S.*:$/.test(line.text.trim()) && !line.text.includes('=');
    if (line.status !== 'removed' && !isLabel) n++;
    const num = (line.status === 'removed' || isLabel) ? '' : n;
    return `<div class="ln ${line.status}"><span class="n">${num}</span>${escapeHtml(line.text)}</div>`;
  }).join('');

  const firstChange = $('code').querySelector('.added, .removed');
  if (firstChange) firstChange.scrollIntoView({ block: 'center' });

  $('io').textContent = `program output (unchanged): [${DATA.output_after.join(', ')}]`;
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

// ---- keyboard -------------------------------------------------------------
document.addEventListener('keydown', e => {
  if (!DATA) return;
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight' && STEP < DATA.iterations.length - 1) STEP++;
  else if (e.key === 'ArrowLeft' && STEP > 0) STEP--;
  else return;
  e.preventDefault();
  renderTimeline();
  renderStep();
});