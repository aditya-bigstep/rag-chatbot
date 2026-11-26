(() => {
  const form = document.getElementById("start");
  const input = form.querySelector("textarea");
  const error = document.getElementById("start-error");

  async function start() {
    const topic = input.value.trim();
    if (topic.length < 4) return;
    form.querySelector(".send").disabled = true;
    error.hidden = true;
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    });
    if (response.ok) {
      location.href = "/research/" + (await response.json()).id;
      return;
    }
    const body = await response.json().catch(() => ({}));
    error.textContent = typeof body.detail === "string" ? body.detail : "That topic could not be started. Try a longer one.";
    error.hidden = false;
    form.querySelector(".send").disabled = false;
  }

  form.addEventListener("submit", (event) => { event.preventDefault(); start(); });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); start(); }
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
