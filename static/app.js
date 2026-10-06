const $ = (s) => document.querySelector(s);
const escapeHtml = (text) =>
  String(text).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icons = () => window.lucide?.createIcons();
const sentimentGroup = (label) => label.toLowerCase().replace(/^very /, "");
let mode = "single",
  records = [],
  busy = false;
const examples = {
  positive:
    "The battery lasts all day and the camera is excellent. I am really happy with this iPhone.",
  mixed:
    "The screen looks great, but the battery does not last as long as I expected for the price.",
  negative:
    "My new phone keeps freezing and the battery drains in a few hours. Very disappointed with the quality.",
};
function switchMode(next) {
  if (busy) return;
  mode = next;
  for (const name of ["single", "batch"]) {
    $(`#${name}`).hidden = name !== next;
    $(`#${name}-tab`).setAttribute("aria-selected", String(name === next));
  }
  $("#error").hidden = true;
}
for (const name of ["single", "batch"])
  $(`#${name}-tab`).onclick = () => switchMode(name);
document.querySelector('[role="tablist"]').onkeydown = (e) => {
  if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
    e.preventDefault();
    switchMode(mode === "single" ? "batch" : "single");
    $(`#${mode}-tab`).focus();
  }
};
$("#review").oninput = () => {
  $("#char-count").textContent =
    `${$("#review").value.length.toLocaleString()} / 5,000`;
};
document.querySelectorAll("[data-example]").forEach(
  (button) =>
    (button.onclick = () => {
      $("#review").value = examples[button.dataset.example];
      $("#review").oninput();
      $("#review").focus();
    }),
);
function showResult(row) {
  $("#result-tag").textContent = row.needs_review
    ? "Review recommended"
    : "Analysis complete";
  $("#result").innerHTML =
    `<div class="result-summary"><div><p class="eyebrow">PREDICTED SENTIMENT</p><div class="sentiment-label">${escapeHtml(row.label)}</div></div><div class="score-large">${(row.confidence * 100).toFixed(1)}%<small>top model score</small></div></div><div>${Object.entries(
      row.scores,
    )
      .map(
        ([label, score]) =>
          `<div class="score-row"><span>${escapeHtml(label)}</span><div class="bar-track"><span style="width:${score * 100}%"></span></div><span>${(score * 100).toFixed(1)}%</span></div>`,
      )
      .join(
        "",
      )}</div><div class="quality"><div>Vocabulary coverage<strong>${Math.round(row.coverage * 100)}%</strong></div><div>Recognized tokens<strong>${row.tokens}</strong></div><div>Model version<strong>${escapeHtml(row.model_version.slice(0, 8))}</strong></div></div>${row.warnings.map((w) => `<div class="flag">${escapeHtml(w)}</div>`).join("")}`;
  $("#model-version").textContent =
    `Model SHA-256: ${row.model_version}. Score threshold for human review: 70%.`;
}
function render() {
  const count = records.length;
  for (const id of ["total", "nav-count", "queue-count"])
    $(`#${id}`).textContent = count;
  $("#flagged").textContent = records.filter((r) => r.needs_review).length;
  for (const name of ["positive", "negative"])
    $(`#${name}`).textContent = count
      ? `${Math.round((records.filter((r) => sentimentGroup(r.label) === name).length / count) * 100)}%`
      : "--";
  $("#export").disabled = !count;
  $("#clear").disabled = !count;
  const filter = $("#filter").value;
  const visible = records
    .map((r, i) => ({ ...r, index: i }))
    .filter(
      (r) =>
        filter === "all" ||
        (filter === "flagged"
          ? r.needs_review
          : sentimentGroup(r.label) === filter),
    );
  $("#queue-body").innerHTML = visible.length
    ? `<table class="table"><thead><tr><th>REVIEW</th><th>SENTIMENT</th><th>MODEL SCORE</th><th>REVIEW STATUS</th></tr></thead><tbody>${visible.map((r) => `<tr><td><button class="review-button" data-row="${r.index}">${escapeHtml(r.text.length > 130 ? r.text.slice(0, 130) + "..." : r.text)}</button></td><td><span class="pill ${["positive", "negative", "neutral"].includes(sentimentGroup(r.label)) ? sentimentGroup(r.label) : ""}">${escapeHtml(r.label)}</span></td><td>${(r.confidence * 100).toFixed(1)}%</td><td>${r.needs_review ? "Needs review" : "No quality flags"}</td></tr>`).join("")}</tbody></table>`
    : '<div class="empty-queue"><i data-lucide="inbox"></i><p>No reviews match this view.</p></div>';
  document.querySelectorAll("[data-row]").forEach(
    (button) =>
      (button.onclick = () => {
        showResult(records[Number(button.dataset.row)]);
        $(".result-panel").scrollIntoView({ block: "center" });
      }),
  );
  icons();
}
$("#analyze-form").onsubmit = async (e) => {
  e.preventDefault();
  if (busy) return;
  $("#error").hidden = true;
  let body,
    headers = {};
  if (mode === "single") {
    if (!$("#review").value.trim()) {
      $("#error").textContent = "Enter a review first.";
      $("#error").hidden = false;
      $("#review").focus();
      return;
    }
    body = JSON.stringify({ reviews: [$("#review").value] });
    headers["Content-Type"] = "application/json";
  } else {
    if (!$("#csv").files[0]) {
      $("#error").textContent = "Choose a CSV file first.";
      $("#error").hidden = false;
      $("#csv").focus();
      return;
    }
    body = new FormData();
    body.append("file", $("#csv").files[0]);
  }
  busy = true;
  $("#submit").disabled = true;
  $("#submit span").textContent = "Analyzing...";
  $("#result-tag").textContent = "Running inference";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers,
      body,
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.error || "Analysis failed. Please retry.");
    records.unshift(...data.results);
    showResult(data.results[0]);
    render();
  } catch (error) {
    $("#error").textContent = error.message;
    $("#error").hidden = false;
    $("#result-tag").textContent = "Analysis unavailable";
  } finally {
    busy = false;
    $("#submit").disabled = false;
    $("#submit span").textContent = "Analyze sentiment";
  }
};
$("#filter").onchange = render;
$("#clear").onclick = () => {
  if (confirm("Clear all reviews from this session?")) location.reload();
};
const csvCell = (value) =>
  '"' +
  String(value)
    .replace(/^[=+@\-\t\r]/, "'$&")
    .replace(/"/g, '""') +
  '"';
$("#export").onclick = () => {
  const rows = [
    [
      "review",
      "sentiment",
      "model_score",
      "vocabulary_coverage",
      "needs_review",
      "warnings",
      "model_version",
    ],
    ...records.map((r) => [
      r.text,
      r.label,
      r.confidence,
      r.coverage,
      r.needs_review,
      r.warnings.join("; "),
      r.model_version,
    ]),
  ];
  const url = URL.createObjectURL(
    new Blob([rows.map((row) => row.map(csvCell).join(",")).join("\r\n")], {
      type: "text/csv;charset=utf-8",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "signal-review-analysis.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("#model-open").onclick = () => $("#model-dialog").showModal();
$("#model-open-mobile").onclick = () => $("#model-dialog").showModal();
$("#model-close").onclick = () => $("#model-dialog").close();
icons();
