/* The new-report screen. What you type is routed on the server (app/research/intake.py):
   a greeting or a quick question is answered right here, a real topic opens a report. */
(() => {
  const form = document.getElementById("start");
  const input = form.querySelector("textarea");
  const send = form.querySelector(".send");
  const messages = document.getElementById("messages");
  const scroller = document.getElementById("scroll");
  const TAGS = { direct: "direct reply", tool: "tool", history: "from memory", llm: "quick answer", blocked: "blocked" };
  // The box's conversation on the server. Kept only while this page is open: "New report" or a reload
  // starts a clean one (the server drops the old messages). Facts such as your name live in the user memory.
  let sessionId = null;
  let busy = false;

  function addMessage(role, text) {
    const article = document.createElement("article");
    article.className = "message " + role;
    article.innerHTML = '<div class="content"></div>';
    article.querySelector(".content").textContent = text;
    messages.append(article);
    scroller.scrollTop = scroller.scrollHeight;
    return article;
  }

  function fail(article, text) {
    article.querySelector(".thinking")?.remove();
    const error = document.createElement("p");
    error.className = "turn-error";
    error.setAttribute("role", "alert");
    error.textContent = text;
    article.append(error);
  }

  async function post(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (response.status === 401) { location.href = "/auth"; return new Promise(() => {}); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "That could not be started. Please try again.");
    return data;
  }

  // "Research this in depth" under a quick answer: skip the routing and start the report.
  async function research(topic, article, button) {
    button.disabled = true;
    try {
      location.href = "/research/" + (await post("/api/research", { topic })).id;
    } catch (error) {
      button.disabled = false;
      fail(article, error.message);
    }
  }

  async function start() {
    const text = input.value.trim();
    if (!text || busy) return;
    busy = true;
    send.disabled = true;
    document.getElementById("intro")?.remove();
    input.value = "";
    addMessage("user", text);
    const reply = addMessage("assistant", "");
    reply.insertAdjacentHTML("afterbegin", '<div class="thinking"><span class="slots" aria-hidden="true">' + "<i></i>".repeat(6) + "</span><span>Working out how to handle this</span></div>");
    try {
      const result = await post("/api/research/ask", { text, session_id: sessionId });
      sessionId = result.session_id;
      if (result.route === "agent") {
        reply.querySelector(".thinking span:last-child").textContent = "Starting the research";
        location.href = "/research/" + result.report_id;
        return;
      }
      reply.querySelector(".thinking").remove();
      const tag = document.createElement("span");
      tag.className = "route-tag " + result.route;
      tag.textContent = TAGS[result.route] || result.route;
      reply.prepend(tag);
      reply.querySelector(".content").textContent = result.reply;
      if (result.route === "llm" && text.length >= 4) {
        const actions = document.createElement("div");
        actions.className = "message-actions";
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn small";
        button.textContent = "Research this in depth";
        button.addEventListener("click", () => research(text, reply, button));
        actions.append(button);
        reply.append(actions);
      }
    } catch (error) {
      fail(reply, error.message);
      input.value = text;
    }
    busy = false;
    send.disabled = false;
    scroller.scrollTop = scroller.scrollHeight;
    input.focus();
  }

  form.addEventListener("submit", (event) => { event.preventDefault(); start(); });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); start(); }
  });
  document.addEventListener("click", (event) => {
    const starter = event.target.closest("[data-topic]");
    if (starter) { input.value = starter.dataset.topic; input.focus(); }
    if (event.target.closest("[data-toggle=sidebar]")) {
      const sidebar = document.getElementById("sidebar");
      const open = !sidebar.classList.contains("is-open");
      sidebar.classList.toggle("is-open", open);
      document.querySelector(".scrim").hidden = !open;
    }
  });
  // A topic typed on the welcome page before signing in is waiting here.
  const saved = sessionStorage.getItem("pilot-topic");
  if (saved && !input.value) { input.value = saved; sessionStorage.removeItem("pilot-topic"); }
  input.focus();
})();
