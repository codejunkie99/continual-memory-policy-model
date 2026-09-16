"use strict";

const main = document.getElementById("main");
const dialog = document.getElementById("memory-dialog");
const detailBody = document.getElementById("detail-body");
const announcer = document.getElementById("announcer");
const updatedLabel = document.getElementById("updated");
const labels = { overview: "Overview", memories: "Memories", decisions: "Decisions", audit: "History", checkpoints: "Learning" };
const OPS = ["WRITE", "UPDATE", "LINK", "COMPACT", "DELETE", "NOOP"];
const OP_NAMES = { WRITE: "Saved", UPDATE: "Updated", LINK: "Connected", COMPACT: "Combined", DELETE: "Put aside", NOOP: "Left unchanged" };
const STATUS_NAMES = { active: "Available to recall", compacted: "Combined", tombstoned: "Put aside" };
const VERSION_STATUS = { active: "In use", candidate: "Awaiting review", rejected: "Declined", retired: "Retired", rolled_back: "Rolled back" };
const METRIC_ORDER = ["operation_accuracy", "macro_f1", "structured_output_validity", "harmful_memory_rate", "privacy_leakage", "downstream_utility", "net_downstream_reward", "latency_s", "n_cases"];
const HISTORY_LIMIT = 50;
const state = {
  page: "overview",
  q: "", status: "all", signal: "all", scope: "", offset: 0, limit: 20,
  op: "", version: "", decisionOffset: 0,
  historyType: "", since: 0, historyPages: [],
};
let pageController, listController, detailController, searchTimer;

// User data is inserted only through textContent, never HTML interpolation.
function node(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
function append(parent, ...children) { parent.append(...children.filter(Boolean)); return parent; }
function button(text, handler, className = "button") {
  const b = node("button", className, text);
  b.type = "button";
  b.addEventListener("click", handler);
  return b;
}
function link(text, page, className = "") { const a = node("a", className, text); a.href = "#" + page; return a; }
function announce(text) { announcer.textContent = text; }
function validDate(ts) {
  if (typeof ts !== "number" || !Number.isFinite(ts)) return null;
  const d = new Date(ts * 1000);
  return Number.isFinite(d.getTime()) ? d : null;
}
function date(ts) {
  const d = validDate(ts);
  return d ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(d) : "Date unavailable";
}
function timeOnly(ts) {
  const d = validDate(ts);
  return d ? new Intl.DateTimeFormat(undefined, { timeStyle: "medium" }).format(d) : "Time unavailable";
}
function dayLabel(ts) {
  const d = validDate(ts);
  return d ? new Intl.DateTimeFormat(undefined, { dateStyle: "full" }).format(d) : "Date unavailable";
}
function relative(ts) {
  const d = validDate(ts);
  if (!d) return "at an unknown time";
  const seconds = Math.round((d.getTime() - Date.now()) / 1000);
  const units = [["year", 31536000], ["month", 2592000], ["week", 604800], ["day", 86400], ["hour", 3600], ["minute", 60]];
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of units) {
    if (Math.abs(seconds) >= size) return rtf.format(Math.round(seconds / size), unit);
  }
  return "just now";
}
function number(n) { return new Intl.NumberFormat().format(Number(n) || 0); }
function count(n, singular, plural) { return number(n) + " " + (Number(n) === 1 ? singular : plural); }
function fixed(n, digits = 2) { const v = Number(n); return Number.isFinite(v) ? v.toFixed(digits) : "n/a"; }
function signed(n, digits = 2) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "n/a";
  const text = Math.abs(v).toFixed(digits);
  return v > 0 ? "+" + text : v < 0 ? "-" + text : text;
}
function percent(n) { const v = Number(n); return Number.isFinite(v) ? (v * 100).toFixed(1) + "%" : "not measured"; }
function num(text) { return node("span", "num", text); }
function content(value) {
  const text = String(value || "");
  return text.startsWith("[concat] ") ? text.slice(9).replaceAll(" || ", "; ") : text;
}
function tag(text, kind = "") { return node("span", kind ? "tag " + kind : "tag", text); }
function memoryStatus(status) { return STATUS_NAMES[status] || String(status || "Unknown"); }
function signal(reward, tested) {
  if (reward > 0) return tag("Appears helpful", "good");
  if (reward < 0) return tag("Needs review", "concern");
  return tested ? tag("No clear signal") : tag("No result yet", "caution");
}
function versionTag(status) {
  return tag(VERSION_STATUS[status] || String(status || "Unknown"), status === "active" ? "ink" : status === "rejected" ? "concern" : "");
}
function payloadOf(record) { return record && record.payload && typeof record.payload === "object" ? record.payload : {}; }
function blockedReason(reason) {
  const text = String(reason || "");
  return text.startsWith("refuse-harmful:") ? text.slice("refuse-harmful:".length) : null;
}
function eventName(type) {
  return ({
    "memory.write": "Saved a memory", "memory.update": "Updated a memory",
    "memory.delete": "Put a memory aside", "memory.link": "Connected two memories",
    "memory.compact": "Combined memories", "policy.decision": "Recorded a policy decision",
    "policy.noop": "Left memory unchanged", retrieval: "Recalled a memory",
    outcome: "Recorded a task result", attribution: "Estimated a memory’s effect",
    "checkpoint.add": "Registered a learning version", "checkpoint.activate": "Activated a learning version",
    "checkpoint.reject": "Declined a learning version", "checkpoint.rollback": "Returned to an earlier version",
  })[type] || String(type || "Activity").replaceAll(".", " ");
}
function eventDetail(e) {
  const p = payloadOf(e);
  switch (e.event_type) {
    case "memory.write": return p.scope ? "In the " + p.scope + " area." : "";
    case "memory.update": return p.rev ? "Now at revision " + number(p.rev) + "." : "";
    case "memory.link": return p.kind ? "Relationship: " + p.kind + "." : "";
    case "memory.compact": return (Array.isArray(p.memory_ids) ? count(p.memory_ids.length, "memory", "memories") + " combined" : "Memories combined") + (p.strategy ? " using the " + p.strategy + " strategy." : ".");
    case "policy.decision": return (OP_NAMES[p.op] ? "Operation: " + OP_NAMES[p.op].toLowerCase() + "." : "") + (p.policy_version ? " Policy version " + p.policy_version + "." : "");
    case "policy.noop": { const blocked = blockedReason(p.reason); return blocked ? "Blocked: the observation looked like " + blocked + " data." : p.reason ? "Reason: " + p.reason + "." : ""; }
    case "outcome": return (p.kind ? "A " + p.kind + " result" : "A result") + (typeof p.value === "number" ? " with value " + signed(p.value) + "." : ".");
    case "attribution": return typeof p.reward === "number" ? "Reward " + signed(p.reward) + "." : "";
    case "checkpoint.add": case "checkpoint.activate": case "checkpoint.reject": case "checkpoint.rollback": return p.version ? "Version " + p.version + "." : "";
    default: return "";
  }
}
function eventMemoryId(e) {
  const p = payloadOf(e);
  if (typeof p.memory_id === "string") return p.memory_id;
  if (e.event_type === "policy.decision" && ["WRITE", "UPDATE", "DELETE", "COMPACT"].includes(p.op) && typeof p.target === "string") return p.target;
  if (e.event_type === "memory.compact" && typeof p.result_memory_id === "string") return p.result_memory_id;
  if (e.event_type === "memory.link" && typeof p.source_id === "string") return p.source_id;
  return null;
}
function eventEntry(e, technical) {
  const detail = eventDetail(e);
  const actor = e.actor === "system" ? "" : e.actor ? " Recorded by " + e.actor + "." : "";
  return { ts: e.ts, title: eventName(e.event_type), description: (detail + actor).trim(), memoryId: eventMemoryId(e), technical: technical ? e : null };
}
function disclosure(title, value) {
  return append(node("details"), node("summary", "", title), node("pre", "", JSON.stringify(value, null, 2)));
}
function facts(pairs) {
  const dl = node("dl", "facts");
  pairs.forEach(([key, value, mono]) => {
    const dd = node("dd", mono ? "num" : "");
    if (value instanceof Node) dd.append(value); else dd.textContent = String(value);
    append(dl, node("dt", "", key), dd);
  });
  return dl;
}
function heading(title, description) {
  return append(node("div", "page-head"), node("h1", "", title), description ? node("p", "lede", description) : null);
}
function empty(title, description, action) {
  return append(node("div", "empty"), node("h2", "", title), node("p", "", description), action);
}
function loading(target, text) {
  const shell = append(node("div", "loading"), node("p", "", text));
  for (let i = 0; i < 3; i++) {
    const bar = node("div", "skeleton");
    bar.setAttribute("aria-hidden", "true");
    shell.append(bar);
  }
  target.replaceChildren(shell);
  target.setAttribute("aria-busy", "true");
}
function errorView(target, retry) {
  target.removeAttribute("aria-busy");
  const box = empty("We couldn’t load this information", "Check that the local memory server is running, then try again.", button("Try again", retry));
  box.setAttribute("role", "alert");
  target.replaceChildren(box);
}
async function api(path, abortSignal) {
  const response = await fetch(path, { headers: { Accept: "application/json" }, signal: abortSignal, cache: "no-store" });
  if (!response.ok) throw new Error("Request failed: " + response.status);
  return response.json();
}
function completed(target) { target.removeAttribute("aria-busy"); }
function sectionHead(title, action) { return append(node("div", "section-head"), node("h2", "", title), action); }
function table(caption, columns, rows) {
  const t = node("table");
  if (caption) t.append(node("caption", "sr-only", caption));
  const head = node("tr");
  columns.forEach(c => {
    const th = node("th", c.numeric ? "num" : "");
    if (c.label instanceof Node) th.append(c.label); else th.textContent = String(c.label);
    th.scope = "col";
    head.append(th);
  });
  t.append(append(node("thead"), head));
  const body = node("tbody");
  rows.forEach(cells => {
    const tr = node("tr");
    cells.forEach((cell, i) => {
      const td = node("td", columns[i] && columns[i].numeric ? "num" : "");
      if (cell instanceof Node) td.append(cell); else td.textContent = cell === null || cell === undefined ? "" : String(cell);
      tr.append(td);
    });
    body.append(tr);
  });
  t.append(body);
  return append(node("div", "table-wrap"), t);
}
function metricLabel(key) { const text = String(key).replaceAll("_", " "); return text.charAt(0).toUpperCase() + text.slice(1); }
function metricValue(key, value) {
  if (value === null || value === undefined) return "not measured";
  if (typeof value !== "number") return typeof value === "object" ? JSON.stringify(value) : String(value);
  const k = String(key);
  if (/accuracy|f1|rate|utility|validity|leakage/.test(k) && value >= 0 && value <= 1) return percent(value);
  if (/latency_s$/.test(k)) return (value * 1000).toFixed(1) + " ms";
  if (Number.isInteger(value)) return number(value);
  return value.toFixed(3);
}
function metricKeys(metricsList) {
  const keys = new Set();
  metricsList.forEach(m => { if (m && typeof m === "object") Object.keys(m).forEach(k => keys.add(k)); });
  const ordered = METRIC_ORDER.filter(k => keys.has(k));
  const rest = [...keys].filter(k => !METRIC_ORDER.includes(k)).sort();
  return ordered.concat(rest);
}
function ledger(entries, options = {}) {
  const list = node("ol", options.days ? "timeline dated" : "timeline");
  let lastDay = null;
  entries.forEach(e => {
    if (options.days) {
      const day = dayLabel(e.ts);
      if (day !== lastDay) { lastDay = day; list.append(node("li", "day", day)); }
    }
    const time = node("time", "", options.relative ? relative(e.ts) : options.days ? timeOnly(e.ts) : date(e.ts));
    const d = validDate(e.ts);
    if (d) { time.dateTime = d.toISOString(); if (options.relative || options.days) time.title = date(e.ts); }
    const body = append(node("div"), node("h3", "", e.title), e.description ? node("p", "", e.description) : null);
    if (e.memoryId) body.append(append(node("div", "event-actions"), button("Open memory", () => openMemory(e.memoryId), "text-button")));
    if (e.technical) body.append(disclosure("Inspect record", e.technical));
    list.append(append(node("li"), time, body));
  });
  return list;
}
function selectField(id, label, options, value, change) {
  const select = node("select"); select.id = id;
  options.forEach(([v, t]) => { const opt = node("option", "", t); opt.value = v; select.append(opt); });
  select.value = value;
  select.addEventListener("change", change);
  const l = node("label", "", label); l.htmlFor = id;
  return append(node("div", "field"), l, select);
}
function inputField(id, label, placeholder, value, change, type = "text") {
  const input = node("input"); input.id = id; input.type = type; input.placeholder = placeholder; input.value = value;
  input.maxLength = /scope|version/.test(id) ? 64 : 200;
  input.autocomplete = "off";
  input.addEventListener("input", change);
  const l = node("label", "", label); l.htmlFor = id;
  return append(node("div", "field"), l, input);
}

// -- Home -------------------------------------------------------------
async function overview(abortSignal) {
  const [status, recent] = await Promise.all([
    api("/api/status", abortSignal), api("/api/audit?newest=1&limit=8", abortSignal),
  ]);
  if (abortSignal.aborted) return;
  const total = status.counts.memories, active = status.active_memories, concerning = status.active_concerning_memories;
  const untested = status.active_untested_memories, retired = status.retired_memories;
  let title, lede;
  if (!total) {
    title = "Memory overview";
    lede = "Nothing has been remembered yet. When a connected agent saves information, it appears here with its full history.";
  } else {
    title = "Memory overview";
    const parts = [count(active, "memory is", "memories are") + " ready to recall; " + (concerning ? count(concerning, "needs", "need") + " review." : "none need review."), "Last activity " + relative(status.last_event_ts) + "."];
    if (untested) parts.push(count(untested, "has", "have") + " no recorded result yet, so " + (untested === 1 ? "its" : "their") + " effect is unknown.");
    if (retired) parts.push(count(retired, "memory was", "memories were") + " put aside or combined.");
    lede = parts.join(" ");
  }
  const attention = concerning ? append(node("section", "attention"),
    append(node("div"), node("h2", "", count(concerning, "memory is", "memories are") + " linked to worse results"),
      node("p", "", "Open them to see the recorded outcomes and the credit each one received. The estimate is a recorded heuristic, not proof of cause.")),
    button("Review memories", () => { Object.assign(state, { status: "active", signal: "concerning", q: "", scope: "", offset: 0 }); location.hash = "memories"; }, "button primary")) : null;
  const stats = node("section", "stats");
  stats.setAttribute("aria-label", "Memory summary");
  [[active, "Ready to recall"], [status.active_helpful_memories, "Appear helpful"], [untested, "Awaiting a result"], [concerning, "Need review"]]
    .forEach(([value, label]) => stats.append(append(node("div", "stat"), node("strong", "", number(value)), node("span", "", label))));

  const activity = append(node("section"), sectionHead("Recent activity", link("See full history", "audit")));
  if (recent.items.length) activity.append(ledger(recent.items.map(e => eventEntry(e, false)), { relative: true }));
  else activity.append(empty("No activity recorded", "Events appear here as soon as a connected agent saves, recalls, or evaluates a memory."));

  const policy = append(node("section"), sectionHead("Policy in use", link("See learning versions", "checkpoints")));
  const cp = status.active_checkpoint;
  if (cp) {
    const metrics = cp.metrics && typeof cp.metrics === "object" ? cp.metrics : {};
    const rows = [["Name", cp.name || cp.version], ["Version", cp.version, true], ["Safety gate", cp.gate_approved ? tag("Passed", "good") : tag("Not approved", "caution")]];
    ["operation_accuracy", "macro_f1", "harmful_memory_rate", "downstream_utility"].forEach(k => { if (k in metrics) rows.push([metricLabel(k), metricValue(k, metrics[k]), true]); });
    policy.append(facts(rows));
  } else policy.append(node("p", "lede", "No learning version is registered as active in this database."));

  const mix = append(node("section"), sectionHead("Decision mix", link("See all decisions", "decisions")));
  if (status.decisions_total) {
    const bar = node("div", "mix");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", OPS.map(op => number(status.decision_ops[op] || 0) + " " + OP_NAMES[op].toLowerCase()).join(", "));
    const legend = node("ul", "legend");
    OPS.forEach(op => {
      const n = status.decision_ops[op] || 0;
      if (n) { const seg = node("span", "op-" + op); seg.style.flexGrow = String(n); bar.append(seg); }
      legend.append(append(node("li"), append(node("span", "legend-name"), node("span", "swatch op-" + op), node("span", "", OP_NAMES[op])), num(number(n))));
    });
    mix.append(node("p", "lede", count(status.decisions_total, "decision", "decisions") + " recorded so far."), bar, legend);
  } else mix.append(node("p", "lede", "No policy decisions have been recorded yet."));

  main.replaceChildren();
  append(main, heading(title, lede), attention, stats,
    append(node("div", "overview-columns"), activity, append(node("div"), policy, mix)),
    disclosure("System details", status));
}

// -- Memories ---------------------------------------------------------
function memoryRow(m) {
  const tested = m.n_attributions > 0;
  const copy = append(node("div", "memory-copy"), node("strong", "", content(m.preview.content)));
  const meta = append(node("div", "memory-meta"), node("span", "", m.scope ? "Area: " + m.scope : "No area"), node("span", "", memoryStatus(m.status)),
    node("span", "", "Revision " + number(m.current_revision)));
  if (m.preview.truncated) meta.append(node("span", "", "Preview only"));
  copy.append(meta);
  const result = append(node("div", "memory-result"), signal(m.total_reward, tested),
    node("span", "", m.n_retrievals ? "Recalled " + count(m.n_retrievals, "time", "times") : "Not recalled yet"),
    tested ? num("Net effect " + signed(m.total_reward)) : null);
  const row = button("", () => openMemory(m.memory_id), "memory-row");
  row.setAttribute("aria-haspopup", "dialog");
  return append(row, copy, result);
}
function resetFilters() {
  Object.assign(state, { q: "", scope: "", status: "all", signal: "all", offset: 0 });
  render();
}
async function memoriesPage(abortSignal) {
  const filters = node("div", "filters");
  const change = (key, debounce = false) => e => {
    state[key] = e.target.value; state.offset = 0;
    clearTimeout(searchTimer); listController?.abort();
    if (debounce) searchTimer = setTimeout(() => loadMemories(abortSignal), 180);
    else loadMemories(abortSignal);
  };
  append(filters,
    inputField("mem-search", "Search memories", "Find a word or phrase", state.q, change("q", true), "search"),
    selectField("mem-status", "Availability", [["all", "Everything"], ["active", "Available to recall"], ["retired", "Put aside or combined"]], state.status, change("status")),
    selectField("mem-signal", "Result", [["all", "Any result"], ["helpful", "Appears helpful"], ["concerning", "Needs review"], ["untested", "No result yet"]], state.signal, change("signal")),
    inputField("mem-scope", "Area", "e.g. work", state.scope, change("scope", true)));
  const results = node("div"); results.id = "memory-results";
  main.replaceChildren(heading("Memory library", "Search saved information. Open a memory to inspect its revisions, recalls, outcomes, and credit."), filters, results);
  await loadMemories(abortSignal);
}
async function loadMemories(pageSignal) {
  if (pageSignal.aborted) return;
  listController?.abort(); listController = new AbortController();
  const controller = listController;
  const target = document.getElementById("memory-results");
  if (!target) return;
  loading(target, "Finding memories");
  const params = new URLSearchParams({ q: state.q, status: state.status, signal: state.signal, scope: state.scope, offset: state.offset, limit: state.limit });
  try {
    const data = await api("/api/memories?" + params, controller.signal);
    if (pageSignal.aborted || controller.signal.aborted) return;
    target.replaceChildren();
    const countText = data.total ? number(data.offset + 1) + "-" + number(data.offset + data.items.length) + " of " + count(data.total, "memory", "memories") : "No memories found";
    target.append(append(node("div", "results-head"), node("span", "", countText), button("Clear filters", resetFilters, "text-button")));
    if (data.items.length) {
      const list = node("div", "memory-list");
      data.items.forEach(m => list.append(memoryRow(m)));
      target.append(list);
    } else {
      const filtered = state.q || state.scope || state.status !== "all" || state.signal !== "all";
      target.append(empty(filtered ? "No matching memories" : "Nothing has been remembered yet",
        filtered ? "Try a different word or clear the filters to see everything." : "When a connected agent saves information, it appears here.",
        filtered ? button("Clear filters", resetFilters) : null));
    }
    const prev = button("Previous", () => { state.offset = Math.max(0, state.offset - state.limit); loadMemories(pageSignal); });
    const next = button("Next", () => { state.offset += state.limit; loadMemories(pageSignal); });
    prev.disabled = data.offset === 0; next.disabled = data.offset + data.items.length >= data.total;
    target.append(append(node("div", "pager"), prev, next));
    completed(target); announce(countText);
  } catch (e) {
    if (!pageSignal.aborted && !controller.signal.aborted) errorView(target, () => loadMemories(pageSignal));
  }
}

// -- Memory record sheet ---------------------------------------------
function detailSection(title, note) { return append(node("section", "detail-section"), node("h3", "", title), note ? node("p", "", note) : null); }
async function openMemory(id) {
  detailController?.abort(); detailController = new AbortController();
  const controller = detailController;
  if (!dialog.open) dialog.showModal();
  loading(detailBody, "Loading this memory");
  try {
    const [d, decided] = await Promise.all([
      api("/api/memories/" + encodeURIComponent(id), controller.signal),
      api("/api/decisions?" + new URLSearchParams({ target: id, limit: 20 }), controller.signal).catch(() => ({ items: [] })),
    ]);
    if (controller.signal.aborted || !dialog.open) return;
    const tested = d.attributions.length > 0;
    const status = append(node("div", "memory-meta"), signal(d.total_reward, tested), tag(memoryStatus(d.status)), tag(d.scope ? "Area: " + d.scope : "No area"));
    detailBody.replaceChildren(status, node("p", "memory-text", content(d.content.content)),
      facts([["Saved", date(d.created_at)], ["Last changed", date(d.updated_at)], ["Revision", number(d.current_revision), true], ["Key", d.key || "None"],
        ["Times recalled", number(d.retrievals.length), true], ["Results linked", number(d.outcomes.length), true],
        ["Net effect", tested ? signed(d.total_reward) : "Not measured", true], ["Connections", number(d.links.length), true]]));
    if (d.content.truncated) detailBody.append(node("p", "notice", "Showing the first part of this memory. The full text has " + count(d.content.length, "character", "characters") + "."));
    if (d.content.injection_flags?.length) detailBody.append(node("p", "notice error", "Automatic checks flagged instruction-like text in this memory: " + d.content.injection_flags.join(", ") + ". The MCP server neutralizes this text before agents read it."));

    const outcome = detailSection("Did it help?", "A linked task result is evidence to inspect. It does not prove the memory caused that result.");
    if (!d.outcomes.length) outcome.append(node("p", "", "No task result has been linked to this memory yet."));
    else outcome.append(table("Task results linked to this memory",
      [{ label: "When" }, { label: "Result" }, { label: "Value", numeric: true }, { label: "Confidence", numeric: true }, { label: "Share", numeric: true }],
      d.outcomes.map(o => [date(o.ts), o.kind === "positive" ? "Positive" : o.kind === "negative" ? "Negative" : String(o.kind || "Neutral"), signed(o.value), percent(o.confidence), percent(o.contribution)])));

    const credit = detailSection("Credit ledger", "Each row is the reward this memory received from one result, after weighting by its share and confidence.");
    if (!d.attributions.length) credit.append(node("p", "", "No credit has been assigned yet."));
    else credit.append(table("Credit assigned to this memory",
      [{ label: "When" }, { label: "Reward", numeric: true }, { label: "Weight", numeric: true }, { label: "Confidence", numeric: true }],
      d.attributions.map(a => [date(a.ts), signed(a.reward), fixed(a.weight), percent(a.confidence)])));

    const decisions = detailSection("Policy decisions about this memory", "Why the policy saved, updated, or put this memory aside.");
    if (!decided.items.length) decisions.append(node("p", "", "No recorded decision targets this memory directly."));
    else decisions.append(ledger(decided.items.map(x => ({
      ts: x.ts, title: OP_NAMES[x.op] || String(x.op),
      description: [x.rationale ? "Reasoning: " + x.rationale : "", "Policy " + (x.policy_version || "unknown") + ", confidence " + percent(x.confidence) + "."].filter(Boolean).join(" "),
      technical: x,
    }))));

    const revisions = detailSection("How it changed");
    revisions.append(ledger(d.revisions.map(r => ({ ts: r.ts, title: (OP_NAMES[r.op] || r.op) + ", revision " + number(r.rev), description: content(r.content.content) }))));

    const recalls = detailSection("When it was recalled");
    recalls.append(d.retrievals.length
      ? ledger(d.retrievals.map(r => ({ ts: r.ts, title: "Included in a later task", description: "Ranked " + number(Number(r.rank) + 1) + " with score " + fixed(r.score, 3) + ".", technical: r })))
      : node("p", "", "This memory has not been recalled yet."));

    const connections = detailSection("Connected memories");
    const links = node("div", "link-list");
    d.links.forEach(l => {
      const other = l.direction === "outbound" ? l.target_id : l.source_id;
      links.append(button("Open the " + l.kind + " memory (" + l.direction + ", weight " + fixed(l.weight) + ")", () => openMemory(other), "text-button"));
    });
    connections.append(d.links.length ? links : node("p", "", "There are no recorded connections."));

    append(detailBody, outcome, credit, decisions, revisions, recalls, connections, disclosure("Raw record", d));
    completed(detailBody); dialog.scrollTop = 0;
    announce("Memory record loaded");
  } catch (e) { if (!controller.signal.aborted && dialog.open) errorView(detailBody, () => openMemory(id)); }
}

// -- Decisions --------------------------------------------------------
function decisionSummary(d) {
  const p = payloadOf(d);
  switch (d.op) {
    case "WRITE": case "UPDATE": return d.preview ? content(d.preview.content) : OP_NAMES[d.op] + " a memory.";
    case "DELETE": return "Put a memory aside.";
    case "LINK": return "Connected two memories" + (p.kind ? " as " + p.kind : "") + (typeof p.weight === "number" ? " with weight " + fixed(p.weight) : "") + ".";
    case "COMPACT": return (Array.isArray(p.memory_ids) ? "Combined " + count(p.memory_ids.length, "memory", "memories") : "Combined memories") + (p.strategy ? " using the " + p.strategy + " strategy." : ".");
    case "NOOP": { const blocked = blockedReason(p.reason); return blocked ? "Blocked: the observation looked like " + blocked + " data." : p.reason ? "Left unchanged: " + p.reason + "." : "Left memory unchanged."; }
    default: return String(d.op || "Decision");
  }
}
function decisionTarget(d) {
  const p = payloadOf(d);
  if (["WRITE", "UPDATE", "DELETE", "COMPACT"].includes(d.op) && typeof d.target === "string") return d.target;
  if (d.op === "LINK" && typeof p.source_id === "string") return p.source_id;
  return null;
}
function decisionRow(d) {
  const p = payloadOf(d);
  const ops = append(node("div", "decision-ops"), tag(OP_NAMES[d.op] || String(d.op)));
  if (d.op === "NOOP" && blockedReason(p.reason)) ops.append(tag("Blocked by safety gate", "caution"));
  const copy = append(node("div", "decision-copy"), node("p", "", decisionSummary(d)));
  if (d.rationale) copy.append(node("p", "why", "Reasoning: " + d.rationale));
  const meta = append(node("div", "decision-meta"), node("span", "", "Policy " + (d.policy_version || "unknown")),
    append(node("span"), "Confidence ", num(percent(d.confidence))), node("span", "", date(d.ts)));
  if (d.session_id) meta.append(node("span", "", "Session " + d.session_id));
  copy.append(meta, disclosure("Inspect record", { decision_id: d.decision_id, event_seq: d.event_seq, payload: d.payload, features: d.features }));
  const actions = node("div", "decision-actions");
  const target = decisionTarget(d);
  if (target) actions.append(button("Open memory", () => openMemory(target), "text-button"));
  return append(node("article", "decision-row"), ops, copy, actions);
}
function resetDecisionFilters() {
  Object.assign(state, { op: "", version: "", decisionOffset: 0 });
  render();
}
async function decisionsPage(abortSignal) {
  const filters = node("div", "filters two");
  const change = (key, debounce = false) => e => {
    state[key] = e.target.value; state.decisionOffset = 0;
    clearTimeout(searchTimer); listController?.abort();
    if (debounce) searchTimer = setTimeout(() => loadDecisions(abortSignal), 180);
    else loadDecisions(abortSignal);
  };
  append(filters,
    selectField("decision-op", "Operation", [["", "All operations"]].concat(OPS.map(op => [op, OP_NAMES[op]])), state.op, change("op")),
    inputField("decision-version", "Policy version", "e.g. baseline", state.version, change("version", true)));
  const results = node("div"); results.id = "decision-results";
  main.replaceChildren(heading("Policy decisions", "Every choice the memory policy made, including changes that were blocked or left unchanged. Newest first."), filters, results);
  await loadDecisions(abortSignal);
}
async function loadDecisions(pageSignal) {
  if (pageSignal.aborted) return;
  listController?.abort(); listController = new AbortController();
  const controller = listController;
  const target = document.getElementById("decision-results");
  if (!target) return;
  loading(target, "Finding decisions");
  const params = new URLSearchParams({ op: state.op, version: state.version, offset: state.decisionOffset, limit: state.limit });
  try {
    const data = await api("/api/decisions?" + params, controller.signal);
    if (pageSignal.aborted || controller.signal.aborted) return;
    target.replaceChildren();
    const countText = data.total ? number(data.offset + 1) + "-" + number(data.offset + data.items.length) + " of " + count(data.total, "decision", "decisions") : "No decisions found";
    target.append(append(node("div", "results-head"), node("span", "", countText), button("Clear filters", resetDecisionFilters, "text-button")));
    if (data.items.length) {
      const list = node("div", "decision-list");
      data.items.forEach(d => list.append(decisionRow(d)));
      target.append(list);
    } else {
      const filtered = state.op || state.version;
      target.append(empty(filtered ? "No matching decisions" : "No decisions recorded yet",
        filtered ? "Choose another operation or clear the filters." : "Decisions appear here once a policy observes an interaction.",
        filtered ? button("Clear filters", resetDecisionFilters) : null));
    }
    const prev = button("Previous", () => { state.decisionOffset = Math.max(0, state.decisionOffset - state.limit); loadDecisions(pageSignal); });
    const next = button("Next", () => { state.decisionOffset += state.limit; loadDecisions(pageSignal); });
    prev.disabled = data.offset === 0; next.disabled = data.offset + data.items.length >= data.total;
    target.append(append(node("div", "pager"), prev, next));
    completed(target); announce(countText);
  } catch (e) {
    if (!pageSignal.aborted && !controller.signal.aborted) errorView(target, () => loadDecisions(pageSignal));
  }
}

// -- History ----------------------------------------------------------
async function historyPage(abortSignal) {
  const filters = node("div", "filters two");
  filters.append(selectField("history-type", "Show activity", [
    ["", "All activity"], ["memory.write,memory.update,memory.delete,memory.link,memory.compact", "Memory changes"],
    ["policy.decision,policy.noop", "Policy decisions"], ["retrieval", "Recalls"], ["outcome,attribution", "Results and credit"],
    ["checkpoint.add,checkpoint.activate,checkpoint.reject,checkpoint.rollback", "Learning versions"],
  ], state.historyType, e => { state.historyType = e.target.value; state.since = 0; state.historyPages = []; render(); }));
  const data = await api("/api/audit?" + new URLSearchParams({ types: state.historyType, since: state.since, limit: HISTORY_LIMIT }), abortSignal);
  if (abortSignal.aborted) return;
  main.replaceChildren(heading("Activity history", "Every event in the order it was written. This ledger is read-only."), filters);
  if (data.items.length) main.append(node("p", "results-head", "Showing up to " + HISTORY_LIMIT + " events at a time."), ledger(data.items.map(e => eventEntry(e, true)), { days: true }));
  else main.append(empty("No activity in this view", "Choose another activity type, or check again after a connected agent uses memory."));
  const previous = button("Previous", () => { state.since = state.historyPages.pop() || 0; render(); });
  previous.disabled = !state.historyPages.length;
  const next = button("Next", () => { state.historyPages.push(state.since); state.since = data.items.at(-1).seq; render(); });
  next.disabled = data.items.length < HISTORY_LIMIT;
  main.append(append(node("div", "pager"), previous, next));
}

// -- Learning ---------------------------------------------------------
async function learningPage(abortSignal) {
  const [data, status] = await Promise.all([api("/api/checkpoints", abortSignal), api("/api/status", abortSignal)]);
  if (abortSignal.aborted) return;
  const current = append(node("section"), sectionHead("In use now"));
  if (data.active) {
    const cp = data.active;
    current.append(facts([["Name", cp.name || cp.version], ["Version", cp.version, true], ["Registered", date(cp.created_at)],
      ["Safety gate", cp.gate_approved ? tag("Passed", "good") : tag("Not approved", "caution")], ["Model file", cp.has_artifact ? "Attached" : "Not attached"], ["Parent", cp.parent || "None", true]]));
    if (cp.note) current.append(node("p", "lede", cp.note));
  } else current.append(node("p", "lede", "No learning version is registered as active in this database."));
  const training = append(node("section"), sectionHead("Training status"),
    node("p", "lede", status.live_training ? "The server reports live training." : "No training is running. Saving memories and training the policy are separate steps, and promotion happens on the command line."));
  main.replaceChildren(heading("Policy versions", "Each version changes how the system decides what to remember. Candidates stay inactive until they pass evaluation and a person promotes them."),
    append(node("div", "learning-grid"), current, training));
  if (!data.items.length) {
    main.append(empty("No learning versions registered", "After a training run registers a version in this database, its status and test results appear here."));
    return;
  }
  const compare = append(node("section"), sectionHead("Test results side by side"));
  const keys = metricKeys(data.items.map(cp => cp.metrics));
  if (keys.length) {
    const columns = [{ label: "Metric" }].concat(data.items.map(cp => ({ label: append(node("div", "col-head"), num(cp.version), versionTag(cp.status)), numeric: true })));
    compare.append(table("Test results for each learning version", columns,
      keys.map(k => [metricLabel(k)].concat(data.items.map(cp => metricValue(k, cp.metrics && typeof cp.metrics === "object" ? cp.metrics[k] : undefined))))));
    compare.append(node("p", "notice", "Metrics come from the evaluation recorded with each version. A “not measured” cell means that run did not report the metric."));
  } else compare.append(node("p", "lede", "No version has recorded test results yet."));
  const versions = append(node("section"), sectionHead("All versions"));
  const list = node("div", "version-list");
  data.items.forEach(cp => {
    const row = append(node("article", "version"),
      append(node("div"), node("h3", "", cp.name || cp.version),
        append(node("div", "version-meta"), num(cp.version), node("span", "", "Registered " + date(cp.created_at)),
          node("span", "", cp.gate_approved ? "Gate passed" : "Gate not approved"), node("span", "", cp.has_artifact ? "Model file attached" : "No model file"),
          cp.parent ? node("span", "", "Parent " + cp.parent) : null)),
      versionTag(cp.status));
    if (cp.note) row.append(node("p", "", cp.note));
    if (cp.metrics) row.append(disclosure("Recorded metrics", cp.metrics));
    list.append(row);
  });
  versions.append(list);
  main.append(compare, versions, node("p", "notice", "This page shows version records only. Promotion, activation, and rollback happen through the project’s command-line tools."));
}

// -- Router and shell -------------------------------------------------
async function render() {
  clearTimeout(searchTimer);
  pageController?.abort(); listController?.abort();
  pageController = new AbortController();
  const controller = pageController;
  const requested = location.hash.slice(1);
  state.page = Object.hasOwn(labels, requested) ? requested : "overview";
  document.querySelectorAll("[data-page]").forEach(a => {
    if (a.dataset.page === state.page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.getElementById("page-label").textContent = labels[state.page];
  document.title = labels[state.page] + " | Memory Center";
  updatedLabel.textContent = "Reading local data";
  loading(main, "Loading " + labels[state.page].toLowerCase());
  try {
    await ({ overview, memories: memoriesPage, decisions: decisionsPage, audit: historyPage, checkpoints: learningPage })[state.page](controller.signal);
    if (controller.signal.aborted) return;
    completed(main);
    updatedLabel.textContent = "Loaded at " + new Intl.DateTimeFormat(undefined, { timeStyle: "short" }).format(new Date());
  } catch (e) {
    if (controller.signal.aborted) return;
    updatedLabel.textContent = "Connection unavailable";
    errorView(main, render);
  }
}
const theme = document.getElementById("theme");
const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
try { const saved = localStorage.getItem("memory-center-theme"); if (["light", "dark", "system"].includes(saved)) theme.value = saved; } catch {}
function applyTheme() {
  document.documentElement.dataset.theme = theme.value === "system" ? (systemTheme.matches ? "dark" : "light") : theme.value;
}
theme.addEventListener("change", () => { applyTheme(); try { localStorage.setItem("memory-center-theme", theme.value); } catch {} });
systemTheme.addEventListener("change", applyTheme);
applyTheme();
document.getElementById("refresh").addEventListener("click", render);
document.getElementById("close-detail").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => { detailController?.abort(); });
window.addEventListener("hashchange", () => { if (dialog.open) dialog.close(); render(); });
render();
