const { animate, stagger } = window.Motion;
const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const ID = document.getElementById('work').dataset.report;
const BUSY = ['planning', 'researching', 'revising'];
const STATUS = {
  planning: 'Planning', awaiting_outline: 'Needs your review', researching: 'Researching', revising: 'Revising',
  ready: 'Ready', failed: 'Failed', blocked: 'Blocked', interrupted: 'Paused',
};
const STAGES = { search: 'searching', read: 'reading', write: 'writing', verify: 'checking' };

const state = {
  report: null,
  cursor: 0,      // how many progress events we have already received
  log: [],
  live: {},       // section id -> what it is doing right now
  tab: 'sources',
  drawn: '',
  draft: null,    // outline being edited
  fresh: [],      // sections changed by the last run
  wasBusy: false,
  chatFor: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const sheet = $('#sheet');
const side = $('#side');
const pop = $('#pop');

function esc(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

async function api(path, options = {}) {
  if (options.body) {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch('/api/research/' + ID + path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed');
  return data;
}

function enter(selector, root = document) {
  const items = root.querySelectorAll(selector);
  if (calm || !items.length) return;
  animate(items, { opacity: [0, 1], y: [10, 0] }, { delay: stagger(0.04), duration: 0.35 });
}

// ---------------------------------------------------------------- loading and polling
async function load() {
  let report;
  try {
    report = await api('?after=' + state.cursor);
  } catch (error) {
    sheet.innerHTML = `<p class="notice bad">${esc(error.message)}</p>`;
    return;
  }
  for (const event of report.events) {
    state.log.push(event);
    if (event.section) {
      if (event.stage === 'done') delete state.live[event.section];
      else state.live[event.section] = STAGES[event.stage] || event.stage;
    }
  }
  state.cursor += report.events.length;

  const busy = BUSY.includes(report.status);
  if (!busy) state.live = {};
  if (state.wasBusy && report.status === 'ready') {
    const last = report.history[report.history.length - 1];
    state.fresh = last ? last.changed.map((change) => change.id) : [];
  }
  state.wasBusy = busy;
  state.report = report;
  draw();
  if (busy) setTimeout(load, 1000);
}

function draw() {
  const report = state.report;
  drawToc();
  drawLog();
  const key = [report.status, report.updated_at, JSON.stringify(state.live), Boolean(report.analysis), report.session_id].join('|');
  if (key === state.drawn) return;
  state.drawn = key;
  drawSheet();
  drawSide();
  $('#composer').classList.toggle('hidden', report.status !== 'ready');
}

// ---------------------------------------------------------------- left rail
function drawToc() {
  const report = state.report;
  $('#toc').innerHTML = report.outline.map((item, i) => {
    const section = report.sections[item.id];
    const doing = state.live[item.id];
    const trust = section && section.text ? Math.round(section.trust * 100) + '%' : '';
    return `<a href="#sec-${item.id}">
      <span class="row"><span class="ttl">${i + 1}. ${esc(item.title)}</span>
      <span class="${doing ? 'live' : 'mono soft'}">${doing || trust}</span></span>
      <span class="meter"><i style="transform: scaleX(${section ? section.trust : 0})"></i></span></a>`;
  }).join('') || '<p class="soft">The outline will appear here.</p>';
}

function drawLog() {
  const box = $('#log');
  box.innerHTML = state.log.slice(-40).map((e) => `<span><b>${esc(e.stage)}</b> ${esc(e.message)}</span>`).join('')
    || '<span>No activity in this session yet.</span>';
  box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------- the report itself
function drawSheet() {
  const report = state.report;
  const step = { planning: 0, awaiting_outline: 1, researching: 2, revising: 2, ready: 4 }[report.status] ?? -1;
  const checking = Object.values(state.live).includes('checking');
  const names = ['Plan', 'Your review', 'Research', 'Verify', 'Ready'];
  const current = step === 2 && checking ? 3 : step;

  let html = `<header><h1>${esc(report.topic)}</h1>
    <div class="facts">
      <span class="pill ${report.status === 'ready' ? 'ok' : ''}">${STATUS[report.status] || report.status}</span>
      ${report.outline.length ? `<span class="pill">${report.outline.length} sections</span>` : ''}
      ${Object.keys(report.numbers).length ? `<span class="pill">${Object.keys(report.numbers).length} sources</span>` : ''}
      ${report.trust ? `<span class="pill ok" title="Supported claims, source quality and source variety">trust ${Math.round(report.trust * 100)}%</span>` : ''}
      ${report.status === 'ready' ? `<a class="btn small quiet" href="/api/research/${ID}/export.md">Markdown</a>
        <button class="btn small quiet" data-print>Print or PDF</button>` : ''}
    </div>
    <div class="steps big">${names.map((n, i) => `<span class="${i === current ? 'on' : i < current ? 'was' : ''}">${n}</span>`).join('')}</div>
  </header>`;

  if (report.status === 'planning') html += skeleton(5);
  if (report.status === 'blocked') html += `<p class="notice bad">${esc(report.error)}</p>`;
  if (report.status === 'failed') html += `<p class="notice bad">The run failed: ${esc(report.error)}</p>`;
  if (report.status === 'ready' && report.error) html += `<p class="notice">The last run stopped early: ${esc(report.error)}</p>`;
  if (report.status === 'interrupted') {
    html += `<p class="notice">This run was cut off, probably by a server restart. The work so far is saved.
      <button class="btn small" data-resume>Resume from checkpoint</button></p>`;
  }
  if (report.status === 'awaiting_outline') html += outlineEditor();
  else html += report.outline.map(sectionHtml).join('') + references();

  sheet.innerHTML = html;
  if (state.fresh.length) {
    for (const id of state.fresh) $('#sec-' + id)?.classList.add('fresh');
    $('#sec-' + state.fresh[0])?.scrollIntoView({ block: 'start' });
    state.fresh = [];
  } else {
    enter('.sec, .outline-edit .item', sheet);
  }
}

function skeleton(lines) {
  return '<div class="sec-body">' + '<div class="skel"></div>'.repeat(lines) + '</div>';
}

function sectionHtml(item, index) {
  const report = state.report;
  const section = report.sections[item.id];
  const doing = state.live[item.id];
  const waiting = !section && BUSY.includes(report.status);
  let body = skeleton(4);
  if (section) body = `<div class="sec-body">${paragraphs(section) || `<p class="soft">${esc(section.note)}</p>`}</div>`;
  else if (!waiting) body = '<div class="sec-body"><p class="soft">Not written yet.</p></div>';

  const tools = report.status === 'ready' && section ? `<div class="sec-tools">
      <button class="btn small quiet" data-act="expand" data-id="${item.id}">More detail</button>
      <button class="btn small quiet" data-act="replace_sources" data-id="${item.id}">Better sources</button>
      <button class="btn small quiet" data-act="rewrite" data-id="${item.id}">Rewrite</button>
      <button class="btn small quiet" data-act="remove" data-id="${item.id}">Remove</button>
    </div>` : '';
  return `<section class="sec ${doing ? 'busy' : ''}" id="sec-${item.id}">
    <div class="sec-head">
      <h2>${index + 1}. ${esc(item.title)}</h2>
      ${doing ? `<span class="pill warn">${doing}</span>` : ''}
      ${section && section.version > 1 ? `<span class="pill">v${section.version}</span>` : ''}
      ${section && section.text ? `<span class="pill ok">trust ${Math.round(section.trust * 100)}%</span>` : ''}
      ${section && section.stats.dropped ? `<span class="pill" title="Sentences removed because their source did not back them">${section.stats.dropped} removed</span>` : ''}
    </div>${tools}${body}</section>`;
}

function paragraphs(section) {
  const numbers = state.report.numbers;
  const groups = {};
  section.claims.forEach((claim, i) => {
    const chips = claim.source_ids.filter((sid) => numbers[sid]).map((sid) =>
      `<a class="cite" href="#" data-sec="${section.id}" data-claim="${i}" data-sid="${sid}">${numbers[sid]}</a>`).join('');
    const end = /[.!?]$/.test(claim.text) ? claim.text.slice(-1) : '';
    const text = end ? claim.text.slice(0, -1) : claim.text;
    (groups[claim.para] ||= []).push(`<span class="c ${claim.status}">${esc(text)}</span> ${chips}${end}`);
  });
  return Object.values(groups).map((lines) => `<p>${lines.join(' ')}</p>`).join('');
}

function references() {
  const report = state.report;
  const ids = Object.keys(report.numbers);
  if (!ids.length) return '';
  return `<section class="refs"><h2>References</h2><ol>${ids.map((sid) => {
    const source = report.sources[sid];
    return `<li><a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">${esc(source.title)}</a>
      <span class="soft">(${esc(source.domain)})</span></li>`;
  }).join('')}</ol></section>`;
}

// ---------------------------------------------------------------- human step 1: the outline
function outlineEditor() {
  if (!state.draft) state.draft = state.report.outline.map((item) => ({ ...item }));
  return `<div class="outline-edit">
    <p>This is the plan. Rename, reorder, remove or add sections, then approve it. Nothing is researched until you do.</p>
    ${state.draft.map((item, i) => `<div class="item">
      <span class="num">${i + 1}</span>
      <div><input class="field" data-title="${i}" value="${esc(item.title)}" maxlength="120" aria-label="Section ${i + 1} title">
        <div class="qs">${esc((item.questions || []).join('  ·  '))}</div></div>
      <div class="acts">
        <button class="btn small quiet" data-move="${i}" data-by="-1" aria-label="Move up" ${i === 0 ? 'disabled' : ''}>↑</button>
        <button class="btn small quiet" data-move="${i}" data-by="1" aria-label="Move down" ${i === state.draft.length - 1 ? 'disabled' : ''}>↓</button>
        <button class="btn small quiet" data-drop="${i}" aria-label="Remove section">Remove</button>
      </div></div>`).join('')}
    <div class="foot">
      <button class="btn" data-add>Add a section</button>
      <button class="btn go" data-approve>Approve and start research</button>
    </div></div>`;
}

function redrawOutline() {
  state.drawn = '';
  draw();
}

// ---------------------------------------------------------------- clicks inside the report
sheet.addEventListener('input', (event) => {
  const index = event.target.dataset.title;
  if (index !== undefined) state.draft[index].title = event.target.value;
});

sheet.addEventListener('click', async (event) => {
  const target = event.target.closest('button, a.cite');
  if (!target) return;
  const data = target.dataset;

  if (target.matches('a.cite')) {
    event.preventDefault();
    showEvidence(target);
  } else if (data.move !== undefined) {
    const from = Number(data.move), to = from + Number(data.by);
    [state.draft[from], state.draft[to]] = [state.draft[to], state.draft[from]];
    redrawOutline();
  } else if (data.drop !== undefined) {
    if (state.draft.length > 1) state.draft.splice(Number(data.drop), 1);
    redrawOutline();
  } else if (data.add !== undefined) {
    state.draft.push({ id: '', title: '', questions: [], queries: [] });
    redrawOutline();
    sheet.querySelector(`[data-title="${state.draft.length - 1}"]`).focus();
  } else if (data.approve !== undefined) {
    const outline = state.draft.filter((item) => item.title.trim());
    if (!outline.length) return;
    target.disabled = true;
    await send('/outline', { outline });
    state.draft = null;
  } else if (data.act) {
    let text = '';
    if (data.act === 'rewrite') {
      text = window.prompt('How should this section change?') || '';
      if (!text.trim()) return;
    }
    if (data.act === 'remove' && !window.confirm('Remove this section from the report?')) return;
    await send('/feedback', { text, section_id: data.id, action: data.act });
  } else if (data.resume !== undefined) {
    await send('/resume');
  } else if (data.print !== undefined) {
    window.print();
  }
});

async function send(path, body) {
  try {
    await api(path, { method: 'POST', body });
  } catch (error) {
    window.alert(error.message);
  }
  load();
}

$('#composer form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = event.target.text;
  if (!input.value.trim()) return;
  const text = input.value;
  input.value = '';
  await send('/feedback', { text });
});

// ---------------------------------------------------------------- evidence popover
function showEvidence(chip) {
  const report = state.report;
  const source = report.sources[chip.dataset.sid];
  const claim = report.sections[chip.dataset.sec].claims[chip.dataset.claim];
  const checked = claim.status === 'supported' ? 'ok' : 'warn';
  pop.innerHTML = `<a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer"><b>${esc(source.title)}</b></a>
    <div class="row" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
      <span class="pill">${esc(source.domain)}</span>
      <span class="pill">source trust ${Math.round(source.trust * 100)}%</span>
      <span class="pill ${checked}">${claim.status}${claim.score ? ' · match ' + claim.score.toFixed(2) : ''}</span>
    </div>
    ${claim.quote ? `<p class="q">${esc(claim.quote)}</p><p class="soft" style="margin-top:6px">Closest passage in the sources this sentence cites.</p>` : ''}`;
  pop.classList.remove('hidden');
  const box = chip.getBoundingClientRect();
  const left = Math.min(window.innerWidth - pop.offsetWidth - 12, Math.max(12, box.left - 40));
  const below = box.bottom + 8 + pop.offsetHeight < window.innerHeight;
  pop.style.left = left + 'px';
  pop.style.top = (below ? box.bottom + 8 : box.top - pop.offsetHeight - 8) + 'px';
  if (!calm) animate(pop, { opacity: [0, 1], scale: [0.96, 1] }, { duration: 0.16 });
  document.querySelectorAll('.cite.on, .card.on').forEach((el) => el.classList.remove('on'));
  chip.classList.add('on');
  const card = side.querySelector(`[data-source="${chip.dataset.sid}"]`);
  if (card) {
    card.classList.add('on');
    card.scrollIntoView({ block: 'nearest' });
  }
}

function hideEvidence() {
  pop.classList.add('hidden');
  document.querySelectorAll('.cite.on, .card.on').forEach((el) => el.classList.remove('on'));
}
document.addEventListener('click', (event) => {
  if (!event.target.closest('.cite, .pop')) hideEvidence();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') hideEvidence();
});
$('#main').addEventListener('scroll', hideEvidence, { passive: true });

// ---------------------------------------------------------------- right panel
document.querySelector('.side-tabs').addEventListener('click', (event) => {
  const tab = event.target.dataset.tab;
  if (!tab) return;
  state.tab = tab;
  document.querySelectorAll('.side-tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  state.chatFor = null;
  drawSide();
});

function drawSide() {
  if (!state.report) return;
  if (state.tab === 'chat') {
    if (state.chatFor !== state.report.session_id + state.report.status) drawChat();
    return;
  }
  side.innerHTML = { sources: sourcesHtml, analysis: analysisHtml, history: historyHtml, runs: runsHtml }[state.tab]();
  enter('.card, .insight, .senti', side);
}

function sourcesHtml() {
  const report = state.report;
  const tone = report.analysis ? report.analysis.sources : {};
  const ids = Object.keys(report.numbers);
  if (!ids.length) return '<p class="soft">Sources appear here once sections are written. Only pages that were actually fetched can be cited.</p>';
  return ids.map((sid) => {
    const source = report.sources[sid];
    const used = report.outline.filter((item) => report.sections[item.id]?.source_ids.includes(sid)).map((item) => item.title);
    return `<div class="card" data-source="${sid}">
      <a class="ttl" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">[${report.numbers[sid]}] ${esc(source.title)}</a>
      <div class="row"><span class="pill">${esc(source.domain)}</span>
        <span class="pill">trust ${Math.round(source.trust * 100)}%</span>
        ${tone[sid] ? `<span class="pill ${toneClass(tone[sid].label)}">${tone[sid].label}</span>` : ''}</div>
      <div class="row soft" style="font-size:13px">Used in: ${esc(used.join(', '))}</div></div>`;
  }).join('');
}

function toneClass(label) {
  return { positive: 'ok', negative: 'bad' }[label] || '';
}

function scale(score, lexicon) {
  const at = (value) => Math.round((Math.max(-1, Math.min(1, value)) + 1) * 50);
  const tick = lexicon === undefined ? '' : `<u style="left:${at(lexicon)}%" title="Lexicon score ${lexicon}"></u>`;
  return `<div class="scale" title="Score ${score}">${tick}<i style="left:${at(score)}%"></i></div>`;
}

function analysisHtml() {
  const report = state.report;
  const analysis = report.analysis;
  if (report.status !== 'ready') return '<p class="soft">Analysis is available once the report is ready.</p>';
  if (!analysis) {
    return `<p>Sentiment for the topic, each section and each source, plus takeaways, opportunities, risks and next steps. Every point links back to the sections it came from.</p>
      <p style="margin-top:12px"><button class="btn go" data-analyse>Run analysis</button></p>`;
  }
  const titles = Object.fromEntries(report.outline.map((item, i) => [item.id, `${i + 1}. ${item.title}`]));
  const group = (name, items) => !items.length ? '' : `<h3 style="margin-top:20px">${name}</h3>` + items.map((insight) =>
    `<div class="insight">${esc(insight.text)}<div class="row">
      ${insight.sections.map((sid) => `<a class="pill" href="#sec-${sid}">${esc(titles[sid] || sid)}</a>`).join('')}
      ${insight.sources.map((n) => `<span class="cite">${n}</span>`).join('')}</div></div>`).join('');

  return `${report.analysis_stale ? `<p class="notice">The report changed after this analysis. <button class="btn small" data-analyse>Run again</button></p>` : ''}
    <h3>Overall sentiment</h3>
    <div class="senti"><span><span class="pill ${toneClass(analysis.overall.label)}">${analysis.overall.label}</span></span>
      ${scale(analysis.overall.score)}<span class="why">${esc(analysis.overall.why)}</span></div>
    <h3 style="margin-top:20px">By section</h3>
    ${analysis.sections.map((row) => `<div class="senti">
      <span>${esc(titles[row.id] || row.id)} <span class="pill ${toneClass(row.label)}">${row.label}</span>
        ${row.agrees ? '' : '<span class="pill warn" title="The word-list score points the other way. Read this section yourself.">models disagree</span>'}</span>
      ${scale(row.score, row.lexicon_score)}<span class="why">${esc(row.why)}</span></div>`).join('')}
    <p class="soft" style="font-size:12.5px;margin-top:8px">Dot: LLM reading. Tick: word-list score (VADER), a free second opinion.</p>
    ${group('Key takeaways', analysis.takeaways)}${group('Opportunities', analysis.opportunities)}
    ${group('Risks', analysis.risks)}${group('Next steps', analysis.next_steps)}`;
}

function historyHtml() {
  const entries = [...state.report.history].reverse();
  if (!entries.length) return '<p class="soft">Each run is recorded here with what changed.</p>';
  return entries.map((entry) => `<div class="card">
    <b>${entry.kind === 'initial' ? 'First draft' : 'Revision'}</b> <span class="soft mono">${entry.at.replace('T', ' ').slice(0, 16)}</span>
    ${entry.request ? `<div style="margin-top:4px">“${esc(entry.request)}”</div>` : ''}
    ${entry.note ? `<div class="soft" style="margin-top:4px">${esc(entry.note)}</div>` : ''}
    ${entry.changed.map((change) => `<div style="margin-top:10px"><span class="pill">${esc(change.title)} · v${change.version}</span>
      ${change.before ? `<div class="diff">${wordDiff(change.before, change.after)}</div>` : ''}</div>`).join('')}
    ${entry.kind !== 'initial' ? `<div class="soft" style="margin-top:8px;font-size:13px">${entry.changed.length} of ${state.report.outline.length} sections were redone. The rest were not touched.</div>` : ''}
  </div>`).join('');
}

// Word-level diff using the longest common subsequence of the two texts.
function wordDiff(before, after) {
  const a = before.split(/\s+/).slice(0, 500), b = after.split(/\s+/).slice(0, 500);
  const table = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const out = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) { out.push(esc(a[i])); i++; j++; }
    else if (j < b.length && (i === a.length || table[i][j + 1] >= table[i + 1][j])) { out.push(`<ins>${esc(b[j])}</ins>`); j++; }
    else { out.push(`<del>${esc(a[i])}</del>`); i++; }
  }
  return out.join(' ');
}

function runsHtml() {
  const runs = state.report.runs;
  if (!runs.length) return '<p class="soft">Latency, tokens and tool calls for every run show up here.</p>';
  const sum = (key) => runs.reduce((total, run) => total + (run[key] || 0), 0);
  const row = (run) => `<tr><td>${run.kind}</td><td>${run.seconds}s</td><td>${run.llm_calls}</td>
    <td>${run.input_tokens + run.output_tokens}</td><td>${run.search_calls ?? 0}</td><td>${run.pages_read ?? 0}</td><td>$${run.cost_usd.toFixed(4)}</td></tr>`;
  return `<table class="runs"><thead><tr><th>Run</th><th>Time</th><th>LLM</th><th>Tokens</th><th>Search</th><th>Pages</th><th>Cost</th></tr></thead>
    <tbody>${runs.map(row).join('')}
    <tr><td><b>Total</b></td><td>${sum('seconds').toFixed(1)}s</td><td>${sum('llm_calls')}</td><td>${sum('input_tokens') + sum('output_tokens')}</td>
      <td>${sum('search_calls')}</td><td>${sum('pages_read')}</td><td>$${sum('cost_usd').toFixed(4)}</td></tr></tbody></table>
    <p class="soft" style="font-size:12.5px;margin-top:10px">Cost is an estimate at list prices. Every run is also traced in LangSmith with the report id in its metadata.</p>`;
}

side.addEventListener('click', async (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  if (button.dataset.analyse !== undefined) {
    button.disabled = true;
    button.textContent = 'Analysing';
    try {
      state.report = { ...(await api('/analysis', { method: 'POST' })), events: [] };
    } catch (error) {
      window.alert(error.message);
    }
    state.drawn = '';
    draw();
  } else if (button.dataset.ingest !== undefined) {
    button.disabled = true;
    button.textContent = 'Indexing';
    try {
      const result = await api('/kb', { method: 'POST' });
      state.report.session_id = result.session_id;
      state.report.kb = result.kb;
    } catch (error) {
      window.alert(error.message);
    }
    drawChat();
  } else if (button.dataset.ask !== undefined) {
    $('#chat-form input').value = button.textContent;
    $('#chat-form').requestSubmit();
  }
});

// ---------------------------------------------------------------- chat with the report
async function drawChat() {
  const report = state.report;
  state.chatFor = report.session_id + report.status;
  if (report.status !== 'ready') {
    side.innerHTML = '<p class="soft">Chat is available once the report is ready.</p>';
    return;
  }
  if (!report.session_id) {
    side.innerHTML = `<p>Add this report to the knowledge base to ask questions about it. Each section is chunked, embedded and stored with its title and source links.</p>
      <p style="margin-top:12px"><button class="btn go" data-ingest>Add to knowledge base</button></p>`;
    return;
  }
  const session = await fetch('/api/sessions/' + report.session_id).then((r) => (r.ok ? r.json() : { messages: [] }));
  const asks = report.outline.flatMap((item) => item.questions || []).slice(0, 3);
  side.innerHTML = `<div class="chat">
    <div class="chat-log" id="chat-log"></div>
    ${session.messages.length ? '' : `<div class="asks">${asks.map((q) => `<button data-ask>${esc(q)}</button>`).join('')}</div>`}
    <p class="soft" style="font-size:12.5px;margin-bottom:6px">${report.kb ? report.kb.chunks + ' chunks indexed. ' : ''}Answers come only from this report.</p>
    <form id="chat-form"><input class="field" name="content" placeholder="Ask about this report" maxlength="2000" required aria-label="Question">
      <button class="btn go" type="submit">Ask</button></form></div>`;
  for (const message of session.messages) addMessage(message.role, message.content, message.sources);
  $('#chat-form').addEventListener('submit', ask);
}

function addMessage(role, text, sources = []) {
  const log = $('#chat-log');
  const bubble = document.createElement('div');
  bubble.className = 'msg ' + role;
  bubble.textContent = text;
  log.append(bubble);
  showFrom(bubble, sources);
  log.scrollTop = log.scrollHeight;
  return bubble;
}

function showFrom(bubble, sources) {
  if (!sources || !sources.length) return;
  const sections = [...new Set(sources.map((s) => s.section).filter(Boolean))];
  const urls = [...new Set(sources.flatMap((s) => s.urls || []))].slice(0, 4);
  const from = document.createElement('span');
  from.className = 'from';
  from.innerHTML = `From: ${sections.map(esc).join(', ') || esc(sources[0].source)}` +
    (urls.length ? ' · ' + urls.map((url) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(new URL(url).hostname.replace('www.', ''))}</a>`).join(', ') : '');
  bubble.append(from);
}

async function ask(event) {
  event.preventDefault();
  const form = event.target;
  const content = form.content.value.trim();
  if (!content) return;
  form.content.value = '';
  form.querySelector('button').disabled = true;
  side.querySelector('.asks')?.remove();
  addMessage('user', content);
  const reply = addMessage('assistant', '');
  try {
    const response = await fetch(`/api/sessions/${state.report.session_id}/messages/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content }),
    });
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let end;
      while ((end = buffer.indexOf('\n\n')) !== -1) {
        const [eventLine, dataLine] = buffer.slice(0, end).split('\n');
        buffer = buffer.slice(end + 2);
        const data = JSON.parse(dataLine.slice(6));
        if (eventLine.slice(7) === 'token') reply.textContent += data;
        if (eventLine.slice(7) === 'done') { reply.textContent = data.content; showFrom(reply, data.sources); }
        $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
      }
    }
  } catch (error) {
    reply.textContent = 'Something went wrong: ' + error.message;
  }
  form.querySelector('button').disabled = false;
  form.content.focus();
}

load();
