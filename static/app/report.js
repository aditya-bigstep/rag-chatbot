/* The report workspace: loads the report, polls while it is running, and draws every panel. */
(() => {
  const { animate, stagger } = window.Motion || {};
  const calm = !animate || matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ID = document.getElementById("work").dataset.report;
  const BUSY = ["planning", "researching", "revising"];
  const STATUS = {
    planning: "Planning", awaiting_outline: "Needs your review", researching: "Researching", revising: "Revising",
    ready: "Ready", failed: "Failed", blocked: "Blocked", interrupted: "Paused",
  };
  const DOING = { search: "searching", read: "reading", write: "writing", verify: "checking" };
  const STAGE = { plan: 0, search: 2, read: 3, write: 4, verify: 5, done: 5, assemble: 6 };

  const state = {
    report: null,
    cursor: 0,        // progress events already received
    log: [],
    live: {},         // section id -> what it is doing now
    stage: 0,         // which box of the waiting diagram is lit
    tab: "outline",
    drawn: "",
    draft: null,      // outline being edited
    fresh: [],        // sections changed by the last run
    wasBusy: false,
    chatFor: null,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const sheet = $("#sheet");
  const side = $("#side");
  const pop = $("#pop");
  const composer = $("#composer");

  function esc(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  }

  async function api(path, options = {}) {
    if (options.body) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(options.body);
    }
    const response = await fetch("/api/research/" + ID + path, options);
    if (response.status === 401) { location.href = "/auth"; return {}; }
    const data = response.status === 204 ? {} : await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
    return data;
  }

  function enter(selector, root) {
    const items = root.querySelectorAll(selector);
    if (calm || !items.length) return;
    animate(items, { opacity: [0, 1], y: [8, 0] }, { delay: stagger(0.04), duration: 0.32 });
  }

  // ---------------------------------------------------------------- loading and polling
  async function load() {
    let report;
    try {
      report = await api("?after=" + state.cursor);
    } catch (error) {
      sheet.innerHTML = `<p class="notice bad">${esc(error.message)}</p>`;
      return;
    }
    for (const event of report.events) {
      state.log.push(event);
      if (event.stage in STAGE) state.stage = STAGE[event.stage];
      if (event.section) {
        if (event.stage === "done") delete state.live[event.section];
        else state.live[event.section] = DOING[event.stage] || event.stage;
      }
    }
    state.cursor += report.events.length;

    const busy = BUSY.includes(report.status);
    if (!busy) state.live = {};
    if (report.status === "planning") state.stage = 0;
    if (state.wasBusy && report.status === "ready") {
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
    drawHeader();
    drawWait();
    if (state.tab === "outline" || state.tab === "activity") drawSide();
    const key = [report.status, report.updated_at, JSON.stringify(state.live), Boolean(report.analysis), report.session_id].join("|");
    if (key === state.drawn) return;
    state.drawn = key;
    drawSheet();
    drawSide();
    composer.hidden = report.status !== "ready";
  }

  function drawHeader() {
    const report = state.report;
    const pill = $("#r-status");
    pill.textContent = STATUS[report.status] || report.status;
    pill.className = "pill " + ({ ready: "ok", failed: "bad", blocked: "bad" }[report.status] || "live");
    $("#r-menu").classList.toggle("is-ready", report.status === "ready");
    const sources = Object.keys(report.numbers).length;
    $("#r-meta").innerHTML = [
      report.outline.length ? `<span class="pill">${report.outline.length} sections</span>` : "",
      sources ? `<span class="pill">${sources} sources</span>` : "",
      report.trust ? `<span class="pill ok" title="Supported claims, source quality and source variety">trust ${Math.round(report.trust * 100)}%</span>` : "",
    ].join("");
  }

  // ---------------------------------------------------------------- waiting card
  let factTimer = null;
  let factIndex = Math.floor(Math.random() * 1000);

  function nextFact() {
    const facts = window.PILOT_FACTS || [];
    if (!facts.length) return;
    const box = $("#wait-fact");
    factIndex = (factIndex + 1) % facts.length;
    box.textContent = facts[factIndex];
    if (!calm) animate(box, { opacity: [0, 1], y: [6, 0] }, { duration: 0.4 });
    const timer = $("#wait-fact-timer");  // restart the bar that counts down to the next fact
    timer.classList.remove("run");
    void timer.offsetWidth;
    timer.classList.add("run");
  }

  function drawWait() {
    const report = state.report;
    const card = $("#wait");
    const busy = BUSY.includes(report.status);
    card.hidden = !busy;
    if (!busy) {
      clearInterval(factTimer);
      factTimer = null;
      return;
    }
    if (!factTimer) {
      nextFact();
      factTimer = setInterval(nextFact, 7000);
    }
    for (const node of card.querySelectorAll(".w-node")) {
      const index = Number(node.dataset.stage);
      node.classList.toggle("on", index === state.stage);
      node.classList.toggle("was", index < state.stage);
    }
    const packet = $("#w-packet");
    const x = 50 + state.stage * 100;
    if (calm) packet.setAttribute("cx", x);
    else animate(packet, { cx: x }, { duration: 0.6, easing: [0.22, 1, 0.36, 1] });

    const last = state.log[state.log.length - 1];
    const done = Object.values(report.sections).filter((s) => !state.live[s.id]).length;
    const total = report.outline.length;
    const progress = total && report.status !== "planning" ? ` · ${Math.min(done, total)} of ${total} sections written` : "";
    $("#wait-status").innerHTML = `<b>${STATUS[report.status]}</b> ${esc(last ? last.message : "Starting")}${progress}`;
  }

  // ---------------------------------------------------------------- the report itself
  function drawSheet() {
    const report = state.report;
    let html = "";
    if (report.status === "blocked") html += `<p class="notice bad">${esc(report.error)}</p>`;
    if (report.status === "failed") html += `<p class="notice bad">The run failed: ${esc(report.error)}</p>`;
    if (report.status === "ready" && report.error) html += `<p class="notice">The last run stopped early: ${esc(report.error)}</p>`;
    if (report.status === "interrupted") {
      html += `<p class="notice">This run was cut off, probably by a server restart. The work so far is saved.
        <button class="btn small" data-resume>Resume from checkpoint</button></p>`;
    }
    if (report.status === "awaiting_outline") html += outlineEditor();
    else html += report.outline.map(sectionHtml).join("") + references();

    sheet.innerHTML = html;
    if (state.fresh.length) {
      for (const id of state.fresh) $("#sec-" + id)?.classList.add("fresh");
      $("#sec-" + state.fresh[0])?.scrollIntoView({ block: "start" });
      state.fresh = [];
    } else if (!BUSY.includes(report.status)) {
      enter(".sec, .outline-item", sheet);
    }
  }

  function skeleton() {
    return '<div class="sec-body">' + '<div class="skel"></div>'.repeat(4) + "</div>";
  }

  function sectionHtml(item, index) {
    const report = state.report;
    const section = report.sections[item.id];
    const doing = state.live[item.id];
    const waiting = !section && BUSY.includes(report.status);
    let body = skeleton();
    if (section) body = `<div class="sec-body">${paragraphs(section) || `<p class="hint">${esc(section.note)}</p>`}</div>`;
    else if (!waiting) body = '<div class="sec-body"><p class="hint">Not written yet.</p></div>';

    const tools = report.status === "ready" && section ? `<details class="menu">
        <summary class="icon-btn" aria-label="Change this section" title="Change this section">
          <svg class="icon" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>
        </summary>
        <div class="menu-list">
          <button type="button" data-act="expand" data-id="${item.id}">More detail</button>
          <button type="button" data-act="replace_sources" data-id="${item.id}">Find better sources</button>
          <button type="button" data-act="rewrite" data-id="${item.id}">Rewrite with instructions</button>
          <button type="button" class="danger" data-act="remove" data-id="${item.id}">Remove section</button>
        </div></details>` : "";
    const removed = section && section.stats.dropped
      ? `<span class="pill" title="Sentences removed because their source did not back them">${section.stats.dropped} removed</span>` : "";
    return `<section class="sec ${doing ? "busy" : ""}" id="sec-${item.id}">
      <div class="sec-head">
        <h2>${index + 1}. ${esc(item.title)}</h2>
        ${doing ? `<span class="pill live">${doing}</span>` : ""}
        ${section && section.version > 1 ? `<span class="pill">v${section.version}</span>` : ""}
        ${removed}${tools}
      </div>${body}</section>`;
  }

  function paragraphs(section) {
    const numbers = state.report.numbers;
    const groups = {};
    section.claims.forEach((claim, i) => {
      const chips = claim.source_ids.filter((sid) => numbers[sid]).map((sid) =>
        `<a class="cite" href="#" data-sec="${section.id}" data-claim="${i}" data-sid="${sid}">${numbers[sid]}</a>`).join("");
      const end = /[.!?]$/.test(claim.text) ? claim.text.slice(-1) : "";
      const text = end ? claim.text.slice(0, -1) : claim.text;
      (groups[claim.para] ||= []).push(`<span class="c ${claim.status}">${esc(text)}</span> ${chips}${end}`);
    });
    return Object.values(groups).map((lines) => `<p>${lines.join(" ")}</p>`).join("");
  }

  function references() {
    const report = state.report;
    const ids = Object.keys(report.numbers);
    if (!ids.length) return "";
    return `<section class="refs"><h2>References</h2><ol>${ids.map((sid) => {
      const source = report.sources[sid];
      return `<li><a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">${esc(source.title)}</a> (${esc(source.domain)})</li>`;
    }).join("")}</ol></section>`;
  }

  // ---------------------------------------------------------------- human step 1: the outline
  function outlineEditor() {
    if (!state.draft) state.draft = state.report.outline.map((item) => ({ ...item }));
    const last = state.draft.length - 1;
    return `<div class="outline-edit">
      <p>This is the plan. Rename, reorder, remove or add sections, then approve it. Nothing is researched until you do.</p>
      ${state.draft.map((item, i) => `<div class="outline-item">
        <span class="num">${i + 1}</span>
        <div><input class="input" data-title="${i}" value="${esc(item.title)}" maxlength="120" aria-label="Section ${i + 1} title">
          <div class="qs">${esc((item.questions || []).join("  ·  "))}</div></div>
        <div class="acts">
          <button class="icon-btn" data-move="${i}" data-by="-1" aria-label="Move up" ${i === 0 ? "disabled" : ""}>↑</button>
          <button class="icon-btn" data-move="${i}" data-by="1" aria-label="Move down" ${i === last ? "disabled" : ""}>↓</button>
          <button class="icon-btn" data-drop="${i}" aria-label="Remove section" title="Remove section">✕</button>
        </div></div>`).join("")}
      <div class="outline-foot">
        <button class="btn primary" data-approve>Approve and start research</button>
        <button class="btn" data-add>Add a section</button>
      </div></div>`;
  }

  function redrawOutline() {
    state.drawn = "";
    draw();
  }

  // ---------------------------------------------------------------- clicks
  sheet.addEventListener("input", (event) => {
    const index = event.target.dataset.title;
    if (index !== undefined) state.draft[index].title = event.target.value;
  });

  sheet.addEventListener("click", async (event) => {
    const target = event.target.closest("button, a.cite");
    if (!target) return;
    const data = target.dataset;

    if (target.matches("a.cite")) {
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
      state.draft.push({ id: "", title: "", questions: [], queries: [] });
      redrawOutline();
      sheet.querySelector(`[data-title="${state.draft.length - 1}"]`).focus();
    } else if (data.approve !== undefined) {
      const outline = state.draft.filter((item) => item.title.trim());
      if (!outline.length) return;
      target.disabled = true;
      await send("/outline", { outline });
      state.draft = null;
    } else if (data.act) {
      let text = "";
      if (data.act === "rewrite") {
        text = window.prompt("How should this section change?") || "";
        if (!text.trim()) return;
      }
      if (data.act === "remove" && !window.confirm("Remove this section from the report?")) return;
      await send("/feedback", { text, section_id: data.id, action: data.act });
    } else if (data.resume !== undefined) {
      await send("/resume");
    }
  });

  async function send(path, body) {
    try {
      await api(path, { method: "POST", body });
    } catch (error) {
      window.alert(error.message);
    }
    load();
  }

  const request = composer.querySelector("textarea");
  async function revise() {
    const text = request.value.trim();
    if (!text) return;
    request.value = "";
    await send("/feedback", { text });
  }
  composer.addEventListener("submit", (event) => { event.preventDefault(); revise(); });
  request.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); revise(); }
  });

  document.addEventListener("click", async (event) => {
    const target = event.target;
    if (target.closest("[data-print]")) window.print();
    if (target.closest("[data-delete]") && window.confirm("Delete this report? This cannot be undone.")) {
      await api("", { method: "DELETE" }).catch((error) => window.alert(error.message));
      location.href = "/";
    }
    const toggle = target.closest("[data-toggle]");
    if (toggle && toggle.dataset.toggle === "panel") $("#panel").classList.toggle("is-open");
    if (toggle && toggle.dataset.toggle === "sidebar") {
      const sidebar = $("#sidebar");
      const open = !sidebar.classList.contains("is-open");
      sidebar.classList.toggle("is-open", open);
      $(".scrim").hidden = !open;
    }
    // close any open "..." menu when clicking elsewhere
    for (const menu of document.querySelectorAll("details.menu[open]")) {
      if (!menu.contains(target) || target.closest(".menu-list")) menu.open = false;
    }
    if (!target.closest(".cite, .pop")) hideEvidence();
  });

  // ---------------------------------------------------------------- evidence popover
  function showEvidence(chip) {
    const report = state.report;
    const source = report.sources[chip.dataset.sid];
    const claim = report.sections[chip.dataset.sec].claims[chip.dataset.claim];
    const checked = claim.status === "supported" ? "ok" : "warn";
    pop.innerHTML = `<a class="card-title" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">${esc(source.title)}</a>
      <div class="card-row">
        <span class="pill">${esc(source.domain)}</span>
        <span class="pill">source trust ${Math.round(source.trust * 100)}%</span>
        <span class="pill ${checked}">${claim.status}${claim.score ? " · match " + claim.score.toFixed(2) : ""}</span>
      </div>
      ${claim.quote ? `<p class="q">${esc(claim.quote)}</p><p class="hint" style="margin-top:6px">Closest passage in the sources this sentence cites</p>` : ""}`;
    pop.hidden = false;
    const box = chip.getBoundingClientRect();
    const left = Math.min(innerWidth - pop.offsetWidth - 12, Math.max(12, box.left - 40));
    const below = box.bottom + 8 + pop.offsetHeight < innerHeight;
    pop.style.left = left + "px";
    pop.style.top = (below ? box.bottom + 8 : box.top - pop.offsetHeight - 8) + "px";
    if (!calm) animate(pop, { opacity: [0, 1], scale: [0.96, 1] }, { duration: 0.16 });
    document.querySelectorAll(".cite.on, .card.on").forEach((el) => el.classList.remove("on"));
    chip.classList.add("on");
    const card = side.querySelector(`[data-source="${chip.dataset.sid}"]`);
    if (card) {
      card.classList.add("on");
      card.scrollIntoView({ block: "nearest" });
    }
  }

  function hideEvidence() {
    pop.hidden = true;
    document.querySelectorAll(".cite.on, .card.on").forEach((el) => el.classList.remove("on"));
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideEvidence();
  });
  $("#scroll").addEventListener("scroll", hideEvidence, { passive: true });

  // ---------------------------------------------------------------- right panel
  $(".panel-tabs").addEventListener("click", (event) => {
    const tab = event.target.dataset.tab;
    if (!tab) return;
    state.tab = tab;
    document.querySelectorAll(".panel-tab").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
    state.chatFor = null;
    drawSide();
    enter(".card, .an-card, .toc-link", side);
  });

  function drawSide() {
    if (!state.report) return;
    if (state.tab === "chat") {
      if (state.chatFor !== state.report.session_id + state.report.status) drawChat();
      return;
    }
    side.innerHTML = { outline: outlineHtml, sources: sourcesHtml, analysis: analysisHtml, activity: activityHtml }[state.tab]();
    const log = side.querySelector(".log");
    if (log) log.scrollTop = log.scrollHeight;
  }

  function outlineHtml() {
    const report = state.report;
    if (!report.outline.length) return '<p class="hint">The outline appears here once it is planned.</p>';
    const rows = report.outline.map((item, i) => {
      const section = report.sections[item.id];
      const doing = state.live[item.id];
      const trust = section && section.text ? Math.round(section.trust * 100) + "%" : "";
      return `<a class="toc-link" href="#sec-${item.id}">
        <span class="toc-row"><span class="t">${i + 1}. ${esc(item.title)}</span>
        <span class="n ${doing ? "live" : ""}">${doing || trust}</span></span>
        <span class="meter"><i style="transform: scaleX(${section ? section.trust : 0})"></i></span></a>`;
    }).join("");
    const busy = BUSY.includes(report.status);
    const log = busy && state.log.length
      ? `<h3>Live activity</h3><div class="log">${state.log.slice(-12).map((e) => `<span><b>${esc(e.stage)}</b> ${esc(e.message)}</span>`).join("")}</div>` : "";
    return `<h3>Sections and trust</h3>${rows}${log}`;
  }

  function sourcesHtml() {
    const report = state.report;
    const tone = report.analysis ? report.analysis.sources : {};
    const ids = Object.keys(report.numbers);
    if (!ids.length) return '<p class="hint">Sources appear here once sections are written. Only pages that were actually fetched can be cited.</p>';
    return ids.map((sid) => {
      const source = report.sources[sid];
      const used = report.outline.filter((item) => report.sections[item.id]?.source_ids.includes(sid)).map((item) => item.title);
      return `<div class="card" data-source="${sid}">
        <a class="card-title" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">[${report.numbers[sid]}] ${esc(source.title)}</a>
        <div class="card-row"><span class="pill">${esc(source.domain)}</span>
          <span class="pill">trust ${Math.round(source.trust * 100)}%</span>
          ${tone[sid] ? `<span class="pill ${toneClass(tone[sid].label)}">${tone[sid].label}</span>` : ""}</div>
        <div class="card-row">Used in ${esc(used.join(", "))}</div></div>`;
    }).join("");
  }

  function toneClass(label) {
    return { positive: "ok", negative: "bad" }[label] || "";
  }

  // Where a score from -1 to +1 sits on a 0-100% axis.
  function axis(value) {
    return (Math.max(-1, Math.min(1, value)) + 1) * 50;
  }

  // A bar from the zero line to the score.
  function bar(score, tone) {
    const at = axis(score);
    return `<b class="${tone}" style="left:${Math.min(50, at)}%;width:${Math.abs(at - 50)}%"></b>`;
  }

  const INSIGHT_GROUPS = [
    ["takeaways", "Key takeaways", "tk"], ["opportunities", "Opportunities", "op"],
    ["risks", "Risks", "rk"], ["next_steps", "Next steps", "nx"],
  ];

  function signed(score) {
    return (score > 0 ? "+" : score < 0 ? "\u2212" : "") + Math.abs(score).toFixed(2);
  }

  function analysisHtml() {
    const report = state.report;
    const analysis = report.analysis;
    if (report.status !== "ready") return '<p class="hint">Analysis is available once the report is ready.</p>';
    if (!analysis) {
      return `<p class="hint" style="margin-bottom:12px">Sentiment for the topic, each section and each source, plus takeaways, opportunities, risks and next steps. Every point links back to the sections it came from.</p>
        <button class="btn primary" data-analyse>Run analysis</button>`;
    }
    const titles = Object.fromEntries(report.outline.map((item, i) => [item.id, `${i + 1}. ${item.title}`]));
    const names = Object.fromEntries(report.outline.map((item) => [item.id, item.title]));
    const order = Object.fromEntries(report.outline.map((item, i) => [item.id, i + 1]));
    const overall = analysis.overall;
    const rows = analysis.sections;
    const sources = Object.entries(analysis.sources || {}).filter(([sid]) => report.sources[sid]);
    const tally = (list, label) => list.filter((row) => row.label === label).length;
    const sourceTones = sources.map(([, tone]) => tone);
    const agree = rows.filter((row) => row.agrees).length;

    // Summary: three figures, then the reason in words.
    const stat = (name, figure, note, tone = "") => `<div class="an-stat">
        <span class="an-stat-name">${name}</span>
        <span class="an-fig">${figure}</span>
        <span class="an-stat-note ${tone}">${note}</span></div>`;
    const summary = `<div class="an-card an-summary">
        <div class="an-stats">
          ${stat("Overall", signed(overall.score), overall.label, toneClass(overall.label))}
          ${stat("Sections", `${tally(rows, "positive")}<small> of ${rows.length}</small>`, "positive")}
          ${sources.length ? stat("Sources", `${tally(sourceTones, "positive")}<small> of ${sources.length}</small>`, "positive") : ""}
        </div>
        <p class="an-why">${esc(overall.why)}</p>
      </div>`;

    // Main chart: one diverging bar per section, the word-list score as a tick, the overall score as a dashed line.
    const chartRows = rows.map((row) => {
      const tick = row.lexicon_score === undefined ? ""
        : `<u style="left:${axis(row.lexicon_score)}%" title="Word-list score ${signed(row.lexicon_score)}"></u>`;
      return `<a class="ch-row" href="#sec-${row.id}" title="${esc(titles[row.id] || row.id)}: ${signed(row.score)}">
          <span class="ch-n">${order[row.id] ?? ""}</span>
          <span class="ch-plot">${bar(row.score, toneClass(row.label))}${tick}<s style="left:${axis(overall.score)}%"></s></span>
          <span class="ch-v">${signed(row.score)}</span></a>`;
    }).join("");
    const chart = `<div class="chart" role="img" aria-label="Sentiment score of each section, from -1 to +1">
        ${chartRows}
        <div class="ch-axis"><span></span><span class="ch-ticks"><i style="left:0">−1</i><i style="left:25%">−0.5</i><i style="left:50%">0</i><i style="left:75%">+0.5</i><i style="left:100%">+1</i></span><span></span></div>
      </div>
      <p class="an-legend"><span><i></i>LLM score</span><span><u></u>Word-list score</span><span><s></s>Overall</span></p>`;

    const notes = rows.map((row) => `<li class="an-row">
        <a class="an-title" href="#sec-${row.id}"><span class="an-n">${order[row.id] ?? ""}</span>${esc(names[row.id] || row.id)}</a>
        <p class="an-why">${esc(row.why)}</p>
        ${row.agrees ? "" : '<p class="an-warn">The word-list score points the other way. Read this section yourself</p>'}
      </li>`).join("");

    // Sources: the share of each tone as one bar, then the three least positive sources to read for the other side.
    const shares = [["positive", "ok"], ["neutral", ""], ["negative", "bad"]]
      .map(([label, tone]) => [label, tone, tally(sourceTones, label)]).filter(([, , n]) => n);
    const critical = [...sources].sort((a, b) => a[1].score - b[1].score).slice(0, 3).map(([sid, tone]) => {
      const source = report.sources[sid];
      const n = report.numbers ? report.numbers[sid] : "";
      return `<li class="sl-row">
          <span class="sl-n">${n ?? ""}</span>
          <a class="sl-title" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer" title="${esc(source.title)}">${esc(source.title)}</a>
          <span class="sl-v">${signed(tone.score)}</span>
          <span class="sl-d">${esc(source.domain)}</span>
          <span class="sl-bar">${bar(tone.score, toneClass(tone.label))}</span></li>`;
    }).join("");
    const sourceBlock = sources.length ? `
      <h3>Tone across ${sources.length} sources</h3>
      <div class="an-card an-pad">
        <div class="share" role="img" aria-label="${shares.map(([l, , n]) => `${n} ${l}`).join(", ")}">${shares.map(([label, tone, n]) => `<b class="${tone}" style="flex:${n}" title="${n} ${label}"></b>`).join("")}</div>
        <p class="an-legend share-legend">${shares.map(([label, tone, n]) => `<span><i class="${tone}"></i><em>${n}</em> ${label}</span>`).join("")}</p>
        <p class="sl-head">Least positive, worth reading for the other side</p>
        <ul class="sl">${critical}</ul>
      </div>` : "";

    const groups = INSIGHT_GROUPS.filter(([key]) => analysis[key].length).map(([key, name, tone]) => `
      <h3 class="an-head"><i class="an-dot ${tone}"></i>${name}<span class="an-count">${analysis[key].length}</span></h3>
      <ul class="an-card an-list">${analysis[key].map((insight) => `<li class="an-row">
        <p>${esc(insight.text)}</p>
        <div class="an-src">
          <span class="an-from">${insight.sections.map((sid) => `<a href="#sec-${sid}">${esc(titles[sid] || sid)}</a>`).join("")}</span>
          <span class="an-cites">${insight.sources.map((n) => `<span class="cite" title="Source ${n}">${n}</span>`).join("")}</span>
        </div></li>`).join("")}</ul>`).join("");

    return `${report.analysis_stale ? `<p class="notice" style="margin:0 0 16px">The report changed after this analysis. <button class="btn small" data-analyse>Run again</button></p>` : ""}
      <h3>Overall sentiment</h3>
      ${summary}
      <h3>Tone by section</h3>
      <div class="an-card an-pad">
        ${chart}
        <p class="an-note">The word-list check agrees with the LLM on ${agree} of ${rows.length} sections</p>
        <ul class="an-notes">${notes}</ul>
      </div>
      ${sourceBlock}
      ${groups}`;
  }

  // Activity = what each revision changed, then what each run cost.
  function activityHtml() {
    const report = state.report;
    const entries = [...report.history].reverse();
    const history = entries.length ? entries.map((entry) => `<div class="card">
      <span class="card-title">${entry.kind === "initial" ? "First draft" : "Revision"} <span class="source-score">${entry.at.replace("T", " ").slice(0, 16)}</span></span>
      ${entry.request ? `<div class="card-row">“${esc(entry.request)}”</div>` : ""}
      ${entry.note ? `<div class="card-row">${esc(entry.note)}</div>` : ""}
      ${entry.changed.map((change) => `<div style="margin-top:10px"><span class="pill">${esc(change.title)} · v${change.version}</span>
        ${change.before ? `<div class="diff">${wordDiff(change.before, change.after)}</div>` : ""}</div>`).join("")}
      ${entry.kind !== "initial" ? `<div class="card-row">${entry.changed.length} of ${report.outline.length} sections were redone. The rest were not touched.</div>` : ""}
    </div>`).join("") : '<p class="hint">Each run is recorded here with what changed.</p>';

    const runs = report.runs;
    const sum = (key) => runs.reduce((total, run) => total + (run[key] || 0), 0);
    const row = (run) => `<tr><td>${run.kind}</td><td>${run.seconds}s</td><td>${run.llm_calls}</td>
      <td>${run.input_tokens + run.output_tokens}</td><td>${run.search_calls ?? 0}</td><td>${run.pages_read ?? 0}</td></tr>`;
    const table = runs.length ? `<table class="runs"><thead><tr><th>Run</th><th>Time</th><th>LLM</th><th>Tokens</th><th>Search</th><th>Pages</th></tr></thead>
      <tbody>${runs.map(row).join("")}
      <tr><td>Total</td><td>${sum("seconds").toFixed(1)}s</td><td>${sum("llm_calls")}</td><td>${sum("input_tokens") + sum("output_tokens")}</td>
        <td>${sum("search_calls")}</td><td>${sum("pages_read")}</td></tr></tbody></table>
      <p class="hint" style="margin-top:10px">Estimated cost $${sum("cost_usd").toFixed(4)} at list prices. Every run is also traced in LangSmith with the report id.</p>` : "";
    return `<h3>Changes</h3>${history}<h3>Runs</h3>${table || '<p class="hint">No runs yet.</p>'}`;
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
    return out.join(" ");
  }

  side.addEventListener("click", async (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.dataset.analyse !== undefined) {
      button.disabled = true;
      button.textContent = "Analysing";
      try {
        state.report = { ...(await api("/analysis", { method: "POST" })), events: [] };
      } catch (error) {
        window.alert(error.message);
      }
      state.drawn = "";
      draw();
    } else if (button.dataset.ingest !== undefined) {
      button.disabled = true;
      button.textContent = "Indexing";
      try {
        const result = await api("/kb", { method: "POST" });
        state.report.session_id = result.session_id;
        state.report.kb = result.kb;
      } catch (error) {
        window.alert(error.message);
      }
      drawChat();
    } else if (button.dataset.ask !== undefined) {
      $("#chat-form input").value = button.textContent;
      $("#chat-form").requestSubmit();
    }
  });

  // ---------------------------------------------------------------- chat with the report
  async function drawChat() {
    const report = state.report;
    state.chatFor = report.session_id + report.status;
    if (report.status !== "ready") {
      side.innerHTML = '<p class="hint">Chat is available once the report is ready.</p>';
      return;
    }
    if (!report.session_id) {
      side.innerHTML = `<p class="hint" style="margin-bottom:12px">Add this report to the knowledge base to ask questions about it. Each section is chunked, embedded and stored with its title and source links.</p>
        <button class="btn primary" data-ingest>Add to knowledge base</button>`;
      return;
    }
    const session = await fetch("/api/sessions/" + report.session_id).then((r) => (r.ok ? r.json() : { messages: [] }));
    const asks = report.outline.flatMap((item) => item.questions || []).slice(0, 3);
    side.innerHTML = `<div class="pchat">
      <div class="messages" id="chat-log"></div>
      ${session.messages.length ? "" : `<div class="starters" id="asks">${asks.map((q) => `<button class="starter" data-ask>${esc(q)}</button>`).join("")}</div>`}
      <form id="chat-form"><div class="composer-box">
        <input name="content" placeholder="Ask about this report" maxlength="2000" required aria-label="Question">
        <button class="send" type="submit" aria-label="Ask"><svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg></button>
      </div></form>
      <p class="hint">${report.kb ? report.kb.chunks + " chunks indexed. " : ""}Answers come only from this report.</p></div>`;
    for (const message of session.messages) addMessage(message.role, message.content, message.sources);
    $("#chat-form").addEventListener("submit", ask);
  }

  function addMessage(role, text, sources = []) {
    const log = $("#chat-log");
    const bubble = document.createElement("article");
    bubble.className = "message " + role;
    bubble.innerHTML = '<div class="content"></div>';
    bubble.firstChild.textContent = text;
    log.append(bubble);
    showFrom(bubble, sources);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  function showFrom(bubble, sources) {
    if (!sources || !sources.length) return;
    const sections = [...new Set(sources.map((s) => s.section).filter(Boolean))];
    const urls = [...new Set(sources.flatMap((s) => s.urls || []))].slice(0, 4);
    const from = document.createElement("div");
    from.className = "msg-from";
    from.innerHTML = `From ${sections.map(esc).join(", ") || esc(sources[0].source)}` +
      (urls.length ? " · " + urls.map((url) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(new URL(url).hostname.replace("www.", ""))}</a>`).join(", ") : "");
    bubble.append(from);
  }

  async function ask(event) {
    event.preventDefault();
    const form = event.target;
    const content = form.content.value.trim();
    if (!content) return;
    form.content.value = "";
    form.querySelector("button").disabled = true;
    $("#asks")?.remove();
    addMessage("user", content);
    const reply = addMessage("assistant", "");
    const body = reply.firstChild;
    reply.insertAdjacentHTML("afterbegin", '<div class="thinking"><span class="slots" aria-hidden="true">' + "<i></i>".repeat(6) + "</span><span>Searching the report</span></div>");
    try {
      const response = await fetch(`/api/sessions/${state.report.session_id}/messages/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }),
      });
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        let end;
        while ((end = buffer.indexOf("\n\n")) !== -1) {
          const [eventLine, dataLine] = buffer.slice(0, end).split("\n");
          buffer = buffer.slice(end + 2);
          const kind = eventLine.slice(7);
          const data = JSON.parse(dataLine.slice(6));
          if (kind === "token" || kind === "done") reply.querySelector(".thinking")?.remove();
          if (kind === "token") body.textContent += data;
          if (kind === "done") { body.textContent = data.content; showFrom(reply, data.sources); }
          $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
        }
      }
    } catch (error) {
      reply.querySelector(".thinking")?.remove();
      body.textContent = "Something went wrong: " + error.message;
    }
    form.querySelector("button").disabled = false;
    form.content.focus();
  }

  load();
})();
