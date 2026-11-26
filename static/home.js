const { animate, scroll, stagger } = window.Motion;
const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function esc(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

// ---------------------------------------------------------------- start a report
for (const form of document.querySelectorAll('[data-start]')) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = form.querySelector('button[type=submit]');
    const error = form.parentElement.querySelector('.form-error');
    button.disabled = true;
    error.textContent = '';
    const response = await fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic: form.topic.value }),
    });
    if (response.ok) {
      location.href = '/research/' + (await response.json()).id;
      return;
    }
    const body = await response.json().catch(() => ({}));
    error.textContent = typeof body.detail === 'string' ? body.detail : 'Please enter a topic of at least 4 characters.';
    button.disabled = false;
  });
}
for (const chip of document.querySelectorAll('.tries button')) {
  chip.addEventListener('click', () => {
    const input = document.querySelector('.ask input');
    input.value = chip.textContent;
    input.focus();
  });
}

// ---------------------------------------------------------------- report list
const LABELS = {
  planning: 'planning', awaiting_outline: 'needs your review', researching: 'researching', revising: 'revising',
  ready: 'ready', failed: 'failed', blocked: 'blocked', interrupted: 'paused',
};
fetch('/api/research').then((r) => (r.ok ? r.json() : [])).then((reports) => {
  const list = document.getElementById('reports');
  if (!reports.length) {
    list.innerHTML = '<li class="empty">No reports yet. Your first one will appear here.</li>';
    return;
  }
  list.innerHTML = reports.slice(0, 8).map((r) => `
    <li><a href="/research/${r.id}">
      <span class="t">${esc(r.topic)}</span>
      <span class="m"><span>${LABELS[r.status] || r.status}</span><span>${r.sections} sections</span>
      <span>${r.sources} sources</span>${r.trust ? `<span>trust ${Math.round(r.trust * 100)}%</span>` : ''}</span>
    </a></li>`).join('');
  if (!calm) animate('#reports li', { opacity: [0, 1], y: [8, 0] }, { delay: stagger(0.05), duration: 0.35 });
});

// ---------------------------------------------------------------- sample run, driven by scroll
const OUTLINE = [
  { label: 'Market growth' },
  { label: 'Key manufacturers' },
  { label: 'History of scooters', cut: true },
  { label: 'Charging and batteries' },
  { label: 'Government policy', added: true },
];
const SOURCES = [
  { site: 'industry-report.example', quote: 'Electric two-wheeler sales reached 1.28 million units in 2025.', score: '0.91' },
  { site: 'auto-news.example', quote: 'First-quarter registrations rose 37.4% compared with a year earlier.', score: '0.88' },
  { site: 'policy-brief.example', quote: 'Purchase incentives under the national scheme run through 2026.', score: '0.84' },
];
const SENTENCES = [
  { text: 'Electric two-wheeler sales reached 1.28 million units in 2025', source: 0 },
  { text: 'Registrations rose 37.4% year on year in the first quarter', source: 1 },
  { text: 'This makes India the largest electric vehicle market in the world', source: 1, unsupported: true },
  { text: 'Purchase incentives are in place through 2026', source: 2 },
];

const rowsBox = document.getElementById('o-rows');
const docBox = document.getElementById('doc-text');
const srcBox = document.getElementById('srcs');
const threads = document.getElementById('threads');
const body = document.getElementById('win-body');
const steps = [...document.querySelectorAll('#steps span')];

rowsBox.innerHTML = OUTLINE.map((row, i) =>
  `<div class="o-row"><span class="n">${i + 1}</span><span class="l">${row.label}</span></div>`).join('');
docBox.innerHTML = SENTENCES.map((s) =>
  `<span class="s">${s.text} <span class="cite">${s.source + 1}</span>. </span>`).join('');
srcBox.innerHTML = SOURCES.map((s, i) =>
  `<div class="src"><span class="sc">match ${s.score}</span><span class="d">[${i + 1}] ${s.site}</span><div class="q">${s.quote}</div></div>`).join('');
threads.innerHTML = SENTENCES.map((s) => `<path class="${s.unsupported ? 'no' : ''}"></path>`).join('');

const rows = [...rowsBox.children];
const sentences = [...docBox.children];
const cards = [...srcBox.children];
const paths = [...threads.children];

// part(p, a, b): how far p has travelled from a to b, as 0..1
function part(p, a, b) {
  return Math.min(1, Math.max(0, (p - a) / (b - a)));
}

function drawThread(path, sentence, card, amount) {
  const frame = body.getBoundingClientRect();
  const chip = sentence.querySelector('.cite').getBoundingClientRect();
  const quote = card.querySelector('.q').getBoundingClientRect();
  const x1 = chip.right - frame.left, y1 = chip.top + chip.height / 2 - frame.top;
  const x2 = quote.left - frame.left, y2 = quote.top + quote.height / 2 - frame.top;
  const stacked = quote.left < chip.left;  // phone layout: evidence sits under the text
  const curve = stacked
    ? `M${chip.left + 9 - frame.left},${chip.bottom - frame.top} C${chip.left - frame.left},${y2 - 30} ${x2 - 30},${y2} ${x2},${y2}`
    : `M${x1},${y1} C${x1 + 70},${y1} ${x2 - 70},${y2} ${x2},${y2}`;
  path.setAttribute('d', curve);
  const length = path.getTotalLength();
  path.style.strokeDasharray = path.classList.contains('no') ? '' : length;
  path.style.strokeDashoffset = path.classList.contains('no') ? '' : length * (1 - amount);
  path.style.opacity = amount > 0 ? 1 : 0;
}

const SAY = [
  [0.00, 'Planning the outline'],
  [0.14, 'Waiting for you: edit or approve the outline'],
  [0.27, 'Reading sources for each section'],
  [0.42, 'Writing from the evidence only'],
  [0.56, 'Checking every sentence against the source it cites'],
  [0.78, 'One sentence had no support in its source. Removed.'],
  [0.90, 'Ready. Ask for changes to any section.'],
];

function render(p) {
  const stage = [0, 0.14, 0.27, 0.42, 0.56, 0.9].filter((edge) => p >= edge).length - 1;
  steps.forEach((el, i) => { el.className = i === stage ? 'on' : i < stage ? 'was' : ''; });
  document.getElementById('say').textContent = SAY.filter(([edge]) => p >= edge).pop()[1];

  // 1 plan: outline rows arrive
  rows.forEach((row, i) => {
    const added = OUTLINE[i].added;
    const t = added ? part(p, 0.2, 0.24) : part(p, 0.02 + i * 0.025, 0.06 + i * 0.025);
    row.style.opacity = t;
    row.style.transform = `translateY(${(1 - t) * 6}px)`;
    row.classList.toggle('cut', Boolean(OUTLINE[i].cut) && p > 0.17);
    row.classList.toggle('new', Boolean(added));
  });
  document.getElementById('o-note').style.opacity = part(p, 0.22, 0.26);

  // 2 read: sources slide in
  cards.forEach((card, i) => {
    const t = part(p, 0.28 + i * 0.04, 0.34 + i * 0.04);
    card.style.opacity = t;
    card.style.transform = `translateX(${(1 - t) * 24}px)`;
  });

  // 3 write: sentences appear in order
  sentences.forEach((sentence, i) => {
    sentence.style.opacity = part(p, 0.42 + i * 0.03, 0.46 + i * 0.03);
  });

  // 4 verify: a thread runs from each citation to the passage that backs it
  let kept = 0;
  SENTENCES.forEach((data, i) => {
    const from = 0.57 + i * 0.05;
    const drawn = part(p, from, from + 0.045);
    const sentence = sentences[i], card = cards[data.source];
    const bad = Boolean(data.unsupported);  // toggle() needs a real true/false, never undefined
    const failed = bad && p > from + 0.06;
    const removed = bad && p > 0.8;
    sentence.classList.toggle('gone', removed);
    sentence.classList.toggle('no', failed && !removed);
    sentence.classList.toggle('ok', !bad && drawn === 1);
    if (!bad && drawn === 1) kept += 1;
    if (!bad) {
      card.classList.toggle('hit', drawn === 1);
      card.querySelector('.sc').style.opacity = drawn;
    }
    drawThread(paths[i], sentence, card, removed ? 0 : drawn);
  });
  document.getElementById('kept').textContent = p > 0.57 ? `${kept} kept${p > 0.8 ? ', 1 removed' : ''}` : '';
  document.getElementById('trust').style.transform = `scaleX(${part(p, 0.57, 0.9) * 0.9})`;
}

if (calm) {
  render(1);
} else {
  render(0);
  scroll(render, { target: document.getElementById('run'), offset: ['start start', 'end end'] });
  window.addEventListener('resize', () => render(lastProgress()));
  animate('.desk h1, .desk .lede, .desk .ask, .desk .tries', { opacity: [0, 1], y: [14, 0] }, { delay: stagger(0.07), duration: 0.5 });
}

function lastProgress() {
  const run = document.getElementById('run');
  const total = run.offsetHeight - window.innerHeight;
  return Math.min(1, Math.max(0, -run.getBoundingClientRect().top / total));
}
