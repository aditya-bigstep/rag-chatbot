/* Shared by every signed-in page: the confirmation dialog, log out, and deleting from the sidebar. */
(() => {
  let dialog;

  // Ask before something that cannot be undone. Resolves true when the person confirms.
  // Escape, the Cancel button and a click outside the box all mean "no".
  window.confirmDialog = ({ title, text = "", confirm = "Confirm", cancel = "Cancel", danger = false }) => {
    if (!window.HTMLDialogElement) return Promise.resolve(window.confirm(title + (text ? "\n\n" + text : "")));
    if (!dialog) {
      dialog = document.createElement("dialog");
      dialog.className = "dialog";
      dialog.setAttribute("aria-labelledby", "dialog-title");
      dialog.innerHTML =
        '<h2 id="dialog-title"></h2><p></p><form method="dialog" class="dialog-actions">' +
        '<button class="btn" value="no"></button><button class="btn" value="yes"></button></form>';
      dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close("no"); });
      document.body.append(dialog);
    }
    const [no, yes] = dialog.querySelectorAll("button");
    dialog.querySelector("h2").textContent = title;
    dialog.querySelector("p").textContent = text;
    dialog.querySelector("p").hidden = !text;
    no.textContent = cancel || "";
    no.hidden = !cancel;
    yes.textContent = confirm;
    yes.className = "btn " + (danger ? "danger" : "primary");
    return new Promise((resolve) => {
      dialog.addEventListener("close", () => resolve(dialog.returnValue === "yes"), { once: true });
      dialog.returnValue = "no";
      dialog.showModal();
      (cancel ? no : yes).focus();  // the safe choice has the focus, so a stray Enter does nothing
    });
  };

  // <form data-confirm="Title" data-confirm-text="..." data-confirm-label="..." [data-confirm-danger]>
  document.addEventListener("submit", async (event) => {
    const form = event.target.closest("form[data-confirm]");
    if (!form || form.dataset.confirmed) return;
    event.preventDefault();
    const yes = await window.confirmDialog({
      title: form.dataset.confirm,
      text: form.dataset.confirmText,
      confirm: form.dataset.confirmLabel,
      danger: "confirmDanger" in form.dataset,
    });
    if (!yes) return;
    form.dataset.confirmed = "1";
    form.submit();
  });

  // The bin on a sidebar row: delete a report or a chat without opening it.
  const KINDS = {
    report: { api: "/api/research/", home: "/", title: "Delete this report?", what: "The report, its sources and its chat" },
    chat: { api: "/api/sessions/", home: "/chat", title: "Delete this chat?", what: "The chat and its messages" },
  };
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-remove]");
    if (!button) return;
    const kind = KINDS[button.dataset.remove];
    const row = button.closest(".session-row");
    const yes = await window.confirmDialog({
      title: kind.title,
      text: `“${button.dataset.name}”: ${kind.what.toLowerCase()} will be removed for good.`,
      confirm: "Delete",
      danger: true,
    });
    if (!yes) return;
    row.classList.add("is-leaving");
    const response = await fetch(kind.api + encodeURIComponent(button.dataset.id), { method: "DELETE" }).catch(() => null);
    if (response && response.status === 401) { location.href = "/auth"; return; }
    if (!response || !(response.ok || response.status === 404)) {
      row.classList.remove("is-leaving");
      window.confirmDialog({ title: "That could not be deleted", text: "Nothing was removed. Please try again.", confirm: "OK", cancel: "" });
      return;
    }
    if (row.querySelector(".session-link.active")) { location.href = kind.home; return; }  // it was the open page
    const list = row.parentElement;
    row.remove();
    if (button.dataset.remove === "report" && !list.querySelector(".session-row")) {
      list.insertAdjacentHTML("beforeend", '<p class="hint">No reports yet</p>');
    }
  });
})();
