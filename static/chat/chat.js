/* Document chat: streams the answer, shows sources, and asks for approval before a document is deleted. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const scroller = $("#scroll");
  const messages = $("#messages");
  const composer = $("#composer");
  const textarea = composer?.querySelector("textarea");
  const send = composer?.querySelector(".send");
  const sessionId = composer?.dataset.session;
  let busy = false;

  const WAITING = ["Reading your question", "Searching the documents", "Checking the answer against the sources"];

  function toBottom() {
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }

  function grow() {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
    send.disabled = !textarea.value.trim() || busy;
  }

  function addMessage(role, text) {
    const article = document.createElement("article");
    article.className = "message " + role;
    article.innerHTML = '<div class="content"></div>';
    article.querySelector(".content").textContent = text;
    messages.append(article);
    toBottom();
    return article;
  }

  function addSources(article, sources) {
    if (!sources || !sources.length) return;
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = sources.length + (sources.length === 1 ? " source" : " sources");
    const list = document.createElement("ul");
    for (const source of sources) {
      const item = document.createElement("li");
      const head = document.createElement("div");
      head.className = "source-head";
      const name = document.createElement("span");
      name.className = "source-name";
      name.textContent = source.section || source.source;
      head.append(name);
      if (source.score !== null) {
        const score = document.createElement("span");
        score.className = "source-score";
        score.textContent = source.score.toFixed(2);
        head.append(score);
      }
      item.append(head);
      if (source.snippet) {
        const snippet = document.createElement("div");
        snippet.className = "snippet";
        snippet.textContent = source.snippet;
        item.append(snippet);
      }
      list.append(item);
    }
    details.append(summary, list);
    article.append(details);
  }

  // The row of six ticks that scans while the bot works, with a line saying what it is doing.
  function thinking(article) {
    const row = document.createElement("div");
    row.className = "thinking";
    row.innerHTML = '<span class="slots" aria-hidden="true">' + "<i></i>".repeat(6) + "</span><span></span>";
    article.prepend(row);
    let step = 0;
    const label = row.lastChild;
    label.textContent = WAITING[0];
    const timer = setInterval(() => {
      step = Math.min(step + 1, WAITING.length - 1);
      label.textContent = WAITING[step];
    }, 1800);
    return () => { clearInterval(timer); row.remove(); };
  }

  async function streamReply(content, question, reply) {
    const body = reply.querySelector(".content");
    const stopThinking = thinking(reply);
    const response = await fetch("/api/sessions/" + sessionId + "/messages/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    let started = false;
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
        if (kind === "user") question.querySelector(".content").textContent = data.content;  // personal data is masked
        if (kind === "token") {
          if (!started) { started = true; stopThinking(); reply.classList.add("streaming"); }
          body.textContent += data;
        }
        if (kind === "done") {
          stopThinking();
          reply.classList.remove("streaming");
          body.textContent = data.content;
          addSources(reply, data.sources);
        }
        toBottom();
      }
    }
  }

  // Human in the loop: when the chat is paused before a risky action, show Approve and Reject.
  async function checkApproval() {
    if (!sessionId) return;
    $("#approval")?.remove();
    const response = await fetch("/api/sessions/" + sessionId + "/approval");
    if (!response.ok) return;
    const { pending } = await response.json();
    if (!pending) return;
    const bar = document.createElement("div");
    bar.id = "approval";
    bar.className = "approval";
    const text = document.createElement("p");
    text.textContent = pending.message;
    bar.append(text);
    for (const [label, approved] of [["Approve", true], ["Reject", false]]) {
      const button = document.createElement("button");
      button.className = approved ? "btn primary small" : "btn small";
      button.textContent = label;
      button.onclick = async () => {
        bar.querySelectorAll("button").forEach((b) => (b.disabled = true));
        const result = await fetch("/api/sessions/" + sessionId + "/approval", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approved }),
        });
        bar.remove();
        if (result.ok) addMessage("assistant", (await result.json()).content);
        checkApproval();
      };
      bar.append(button);
    }
    messages.append(bar);
    toBottom();
  }

  async function submit() {
    const content = textarea.value.trim();
    if (!content || busy) return;
    busy = true;
    $("#intro")?.remove();
    textarea.value = "";
    grow();
    const question = addMessage("user", content);
    const reply = addMessage("assistant", "");
    try {
      await streamReply(content, question, reply);
      await checkApproval();
    } catch (error) {
      reply.querySelector(".thinking")?.remove();
      reply.insertAdjacentHTML("beforeend", '<p class="turn-error">The answer did not come through. Your question is back in the box, try sending it again</p>');
      textarea.value = content;
    }
    busy = false;
    grow();
    textarea.focus();
  }

  if (composer) {
    textarea.addEventListener("input", grow);
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        composer.requestSubmit();
      }
    });
    composer.addEventListener("submit", (event) => { event.preventDefault(); submit(); });
    grow();
    checkApproval();
  }

  document.addEventListener("click", (event) => {
    const starter = event.target.closest("[data-ask]");
    if (starter && composer) {
      textarea.value = starter.dataset.ask;
      grow();
      composer.requestSubmit();
      return;
    }
    if (event.target.closest("[data-toggle=sidebar]")) {
      const sidebar = $("#sidebar");
      const open = !sidebar.classList.contains("is-open");
      sidebar.classList.toggle("is-open", open);
      $(".scrim").hidden = !open;
    }
  });

  // Renaming saves when you leave the field; Escape puts the old title back.
  const rename = $(".rename");
  if (rename) {
    const input = rename.querySelector("input");
    let saved = input.value;
    input.addEventListener("keydown", (event) => {
      if (event.key === "Escape") { input.value = saved; input.blur(); }
    });
    input.addEventListener("blur", () => {
      const value = input.value.trim();
      if (!value) { input.value = saved; return; }
      if (value !== saved) { saved = value; rename.submit(); }
    });
  }

  const drop = $("#dropzone");
  if (drop) {
    const file = drop.querySelector("input[type=file]");
    ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, () => drop.classList.remove("is-over")));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      if (!event.dataTransfer.files.length) return;
      file.files = event.dataTransfer.files;
      file.dispatchEvent(new Event("change", { bubbles: true }));
    });
    drop.addEventListener("htmx:afterRequest", () => drop.reset());
  }

  toBottom();
})();
