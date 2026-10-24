/* Basal Station console */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const state = {
  models: [],
  currentModel: null,
  backends: [],
  selectedBackend: null,
  recommended: null,
  downloads: {},
  qSeq: 0,
  defaultModel: "omni-1.0-max",
  recipePolicy: null,
};

/* ---------------- helpers ---------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch { /* not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

let toastTimer;
function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

function fmtGB(x) { return x == null ? "?" : `${x} GB`; }

/* ---------------- header / system ---------------- */

async function loadSystem() {
  try {
    const sys = await api("/api/system");
    const hw = sys.hardware;
    const gpu = (hw.gpus || []).map((g) => `${g.name} · ${g.vram_gb}GB`).join(" + ") || "CPU only";
    $("#hwSummary").textContent =
      `${hw.os} ${hw.arch}${hw.chip ? " · " + hw.chip : ""} · ${hw.ram_gb}GB RAM · ${gpu}`;

    const ep = $("#enginePill");
    const healthy = sys.engine_healthy;
    ep.textContent = healthy ? `engine · ${sys.engine.backend} / ${sys.engine.model}` : "engine stopped";
    ep.className = `pill ${healthy ? "on" : "off"}`;
    if (sys.engine.last_error) ep.title = sys.engine.last_error;

    const kb = sys.knowledge;
    state.defaultModel = sys.default_model || state.defaultModel;
    const kp = $("#kbPill");
    kp.textContent = kb.connected
      ? `knowledge-palace · ${(kb.tools || []).length} tools`
      : "knowledge offline";
    kp.className = `pill ${kb.connected ? "on" : "warn"}`;
    kp.title = kb.error || (kb.tools || []).join(", ");

    $("#kbStatus").innerHTML = kb.connected
      ? `Connected via MCP: <code>${kb.command.join(" ")}</code><br>Tools: ${(kb.tools || []).join(", ")}`
      : `Not connected. Command: <code>${kb.command.join(" ")}</code>` +
        (kb.error ? `<br><span class="muted">${kb.error}</span>` : "");
  } catch (e) {
    toast(`system: ${e.message}`, true);
  }
}

/* ---------------- models & backends ---------------- */

async function loadModels() {
  state.models = await api("/api/models");
  const sel = $("#modelSelect");
  sel.innerHTML = "";
  const byFamily = {};
  for (const m of state.models) {
    (byFamily[m.family] = byFamily[m.family] || []).push(m);
  }
  for (const family of Object.keys(byFamily).sort()) {
    const og = document.createElement("optgroup");
    og.label = `basal-${family}`;
    for (const m of byFamily[family]) {
      const o = document.createElement("option");
      o.value = m.key;
      o.textContent = m.label;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  sel.addEventListener("change", () => selectModel(sel.value));
  const initial = sel.querySelector(`option[value="${CSS.escape(state.defaultModel)}"]`)
    ? state.defaultModel : sel.querySelector("option")?.value;
  selectModel(initial);
}

function selectModel(key) {
  state.currentModel = state.models.find((m) => m.key === key) || null;
  if (!state.currentModel) return;
  state.selectedBackend = null; // re-recommend for the new model
  $("#modelSelect").value = key;
  const m = state.currentModel;
  $("#modelInfo").innerHTML =
    `${m.params_b ? m.params_b + "B params · " : ""}` +
    (m.media ? "text + image/audio (omni)" : "text-only decisions");

  renderDownloads(m);
  loadBackends();
}

function renderDownloads(m) {
  const box = $("#downloadList");
  box.innerHTML = "";
  for (const f of m.formats) {
    const row = document.createElement("div");
    row.className = "dl-row";
    const left = document.createElement("span");
    left.className = "fmt";
    left.textContent = f.format;
    const right = document.createElement("span");
    if (f.local) {
      right.innerHTML = `<span class="local">cached</span>`;
    } else {
      const size = document.createElement("span");
      size.className = "size";
      size.textContent = f.size_gb ? `~${f.size_gb} GB` : "";
      const btn = document.createElement("button");
      btn.className = "small";
      btn.textContent = "download";
      btn.onclick = () => startDownload(m, f, btn);
      right.append(size, btn);
    }
    row.append(left, right);
    box.appendChild(row);
  }
}

async function startDownload(m, f, btn) {
  const quant = f.format === "gguf" ? $("#quantSelect").value : null;
  btn.disabled = true;
  btn.textContent = "queued…";
  try {
    await api("/api/models/download", { method: "POST", body: { model: m.key, format: f.format, quant } });
    toast(`downloading ${f.format} weights for ${m.key}`);
    pollDownloads();
  } catch (e) {
    toast(`download: ${e.message}`, true);
    btn.disabled = false;
    btn.textContent = "download";
  }
}

async function pollDownloads() {
  const jobs = await api("/api/models/downloads");
  state.downloads = jobs;
  const active = Object.values(jobs).filter((j) => j.status === "downloading" || j.status === "queued");
  for (const j of active) {
    toast(`${j.model}/${j.format}: ${j.pct ?? 0}%`);  // lightweight progress hint
  }
  if (active.length) {
    setTimeout(async () => {
      await pollDownloads();
    }, 5000);
  } else if (Object.values(jobs).some((j) => j.status === "done")) {
    const models = await api("/api/models?refresh=true");
    state.models = models;
    if (state.currentModel) selectModel(state.currentModel.key);
  }
}

async function loadBackends() {
  if (!state.currentModel) return;
  const data = await api(
    `/api/backends?model=${encodeURIComponent(state.currentModel.key)}&prefer_media=false`
  );
  state.backends = data.backends;
  state.recommended = data.recommended;
  if (!state.selectedBackend || !data.backends.some((b) => b.key === state.selectedBackend && b.eligible)) {
    state.selectedBackend = data.recommended;
  }
  renderBackends();
}

function renderBackends() {
  const box = $("#backendList");
  box.innerHTML = "";
  const anyGguf = state.backends.some((b) => b.key === state.selectedBackend && b.format === "gguf");
  $("#quantRow").classList.toggle("hidden", !anyGguf);

  for (const b of state.backends) {
    const el = document.createElement("div");
    el.className = "backend" + (b.eligible ? "" : " disabled") +
      (b.key === state.selectedBackend ? " selected" : "");
    const head = document.createElement("div");
    head.className = "b-head";
    const name = document.createElement("span");
    name.textContent = b.label;
    head.appendChild(name);
    if (b.key === state.recommended && b.eligible) {
      const badge = document.createElement("span");
      badge.className = "badge rec";
      badge.textContent = "recommended";
      head.appendChild(badge);
    } else if (b.local && b.eligible) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "cached";
      head.appendChild(badge);
    }
    const reason = document.createElement("div");
    reason.className = "b-reason";
    const bits = [b.reason];
    if (b.ngl != null) bits.push(`ngl=${b.ngl}`);
    if (b.est_vram_gb != null) bits.push(`~${b.est_vram_gb} GB`);
    reason.textContent = b.eligible ? bits.join(" · ") : b.reason;
    el.append(head, reason);
    if (b.eligible) {
      el.onclick = () => { state.selectedBackend = b.key; renderBackends(); };
    }
    box.appendChild(el);
  }
}

/* ---------------- engine control ---------------- */

async function startEngine() {
  if (!state.currentModel || !state.selectedBackend) {
    toast("pick a model and backend first", true);
    return;
  }
  const btn = $("#btnStart");
  btn.disabled = true;
  btn.textContent = "starting…";
  try {
    const body = {
      backend: state.selectedBackend,
      model: state.currentModel.key,
      quant: $("#quantSelect").value,
    };
    await api("/api/engine/start", { method: "POST", body });
    toast("engine is up");
  } catch (e) {
    toast(`engine: ${e.message}`, true);
    $("#logDetails").open = true;
  } finally {
    btn.disabled = false;
    btn.textContent = "Start engine";
    loadSystem();
    loadLogs();
  }
}

async function stopEngine() {
  try {
    await api("/api/engine/stop", { method: "POST" });
    toast("engine stopped");
  } catch (e) {
    toast(`engine: ${e.message}`, true);
  }
  loadSystem();
}

async function loadLogs() {
  if (!$("#logDetails").open) return;
  try {
    const data = await api("/api/engine/logs?n=200");
    $("#logBox").textContent = data.lines.join("\n") || "(no logs yet)";
  } catch { /* ignore */ }
}

/* ---------------- question builder ---------------- */

const QUESTION_TYPES = ["choice", "noul", "score", "multi", "act", "tool"];

function addCriterionRow(box, key = "", value = "") {
  const row = document.createElement("div");
  row.className = "criterion";
  const k = document.createElement("input");
  k.className = "c-key";
  k.placeholder = "criterion key";
  k.value = key;
  const v = document.createElement("input");
  v.className = "c-val";
  v.placeholder = "description";
  v.value = value;
  const x = document.createElement("button");
  x.className = "ghost small";
  x.textContent = "✕";
  x.onclick = () => row.remove();
  row.append(k, v, x);
  box.appendChild(row);
}

function addQuestion(spec = {}) {
  state.qSeq += 1;
  const id = `q${state.qSeq}`;
  const card = document.createElement("div");
  card.className = "qcard";
  card.dataset.id = id;

  const row1 = document.createElement("div");
  row1.className = "row";
  const name = document.createElement("input");
  name.className = "qname";
  name.placeholder = "question name";
  name.value = spec.name || "";
  const type = document.createElement("select");
  type.className = "qtype";
  for (const t of QUESTION_TYPES) {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = t;
    if (t === (spec.type || "choice")) o.selected = true;
    type.appendChild(o);
  }
  const maxerr = document.createElement("input");
  maxerr.className = "qmaxerr";
  maxerr.type = "number";
  maxerr.step = "0.01";
  maxerr.min = "0";
  maxerr.placeholder = "max_error";
  maxerr.style.width = "7em";
  if (spec.max_error != null) maxerr.value = spec.max_error;
  row1.append(name, type, maxerr);

  const instr = document.createElement("input");
  instr.className = "qinstr";
  instr.placeholder = "instructions (how to decide)";
  instr.value = spec.instructions || "";

  const critBox = document.createElement("div");
  critBox.className = "criteria";
  const crits = spec.criteria && typeof spec.criteria === "object" ? spec.criteria : {};
  const entries = Object.entries(crits);
  const needsOptions = ["choice", "multi", "tool", "score"].includes(type.value);
  if (entries.length) {
    for (const [k, v] of entries) addCriterionRow(critBox, k, v);
  } else if (needsOptions) {
    if (type.value === "score") {
      // the engine requires a scale of 2+ anchors for score questions
      addCriterionRow(critBox, "0.0", "scale low end");
      addCriterionRow(critBox, "1.0", "scale high end");
    } else {
      addCriterionRow(critBox);
    }
  }
  type.addEventListener("change", () => {
    if (type.value === "score" && !$$(".criterion .c-key", critBox).some((i) => i.value.trim())) {
      addCriterionRow(critBox, "0.0", "scale low end");
      addCriterionRow(critBox, "1.0", "scale high end");
    }
  });
  const addCrit = document.createElement("button");
  addCrit.className = "ghost small";
  addCrit.textContent = "+ criterion";
  addCrit.onclick = () => addCriterionRow(critBox);

  const del = document.createElement("button");
  del.className = "qdel";
  del.textContent = "✕";
  del.onclick = () => card.remove();

  card.append(del, row1, instr, critBox, addCrit);
  $("#questionList").appendChild(card);
  name.focus();
}

function collectQuestions() {
  const out = {};
  for (const card of $$(".qcard")) {
    const name = $(".qname", card).value.trim();
    if (!name) continue;
    const spec = {
      type: $(".qtype", card).value,
      instructions: $(".qinstr", card).value.trim(),
      criteria: {},
    };
    for (const row of $$(".criterion", card)) {
      const k = $(".c-key", row).value.trim();
      if (k) spec.criteria[k] = $(".c-val", row).value.trim();
    }
    const maxerr = $(".qmaxerr", card).value;
    if (maxerr !== "") spec.max_error = parseFloat(maxerr);
    out[name] = spec;
  }
  return out;
}

/* ---------------- decision ---------------- */

function renderResult(answers, usage) {
  const box = $("#results");
  box.innerHTML = "";
  for (const [name, a] of Object.entries(answers || {})) {
    const q = document.createElement("div");
    q.className = "result-q";
    const head = document.createElement("div");
    head.className = "r-head";
    head.innerHTML = `<span class="r-name">${escapeHtml(name)}</span>` +
      `<span class="r-type">${escapeHtml(a.type || "")}</span>` +
      (a.choice != null ? `<span class="r-choice">→ ${escapeHtml(a.choice)}</span>` : "") +
      (a.deferred ? `<span class="tag-deferred">deferred</span>` : "");
    q.appendChild(head);

    const probs = a.probabilities || {};
    const entries = Object.entries(probs).sort((x, y) => y[1] - x[1]);
    if (entries.length) {
      const bars = document.createElement("div");
      bars.className = "bars";
      const winner = a.choice ?? entries[0][0];
      for (const [label, p] of entries) {
        const row = document.createElement("div");
        row.className = "bar-row" + (label === winner ? " winner" : "");
        row.innerHTML =
          `<span class="bar-label${label === winner ? " winner" : ""}">${escapeHtml(label)}</span>` +
          `<div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>` +
          `<span class="bar-pct">${(p * 100).toFixed(1)}%</span>`;
        bars.appendChild(row);
        requestAnimationFrame(() =>
          requestAnimationFrame(() => { $(".bar-fill", row).style.width = `${Math.max(p * 100, 1)}%`; })
        );
      }
      q.appendChild(bars);
    }
    if (a.confidence != null) {
      const conf = document.createElement("div");
      conf.className = "confidence";
      conf.innerHTML = `confidence <b>${(a.confidence * 100).toFixed(1)}%</b>`;
      q.appendChild(conf);
    }
    box.appendChild(q);
  }
  if (usage) {
    $("#usageLine").textContent =
      `${usage.input_tokens ?? "?"} input tokens · ${usage.questions ?? Object.keys(answers || {}).length} questions` +
      ` · ${usage.branches ?? 1} branches · ${usage.latency_ms ?? "?"} ms`;
  }
}

function renderPolicy(policy) {
  const old = $("#policyBox");
  if (old) old.remove();
  if (!policy) return;
  const box = document.createElement("div");
  box.id = "policyBox";
  box.className = "qcard";
  box.style.marginTop = "10px";
  const badge = policy.flagged
    ? `<span class="tag-deferred">${escapeHtml(policy.action)}</span>`
    : `<span class="r-choice">→ ${escapeHtml(policy.action)}</span>`;
  box.innerHTML =
    `<div class="r-head"><span class="r-name" style="color:var(--accent2)">policy</span>${badge}` +
    (policy.min_confidence != null ? `<span class="small muted">min confidence ${(policy.min_confidence * 100).toFixed(0)}%</span>` : "") +
    `</div><div class="small muted">${escapeHtml((policy.reasons || []).join(" · "))}</div>`;
  $("#results").appendChild(box);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function runDecision() {
  const stateText = $("#stateInput").value.trim();
  const questions = collectQuestions();
  if (!stateText) return toast("state is empty", true);
  if (!Object.keys(questions).length) return toast("add at least one question", true);

  const body = {
    state: stateText,
    questions,
    knowledge: {
      enabled: $("#chkGround").checked,
      query: $("#kbQuery").value.trim() || null,
      top_k: parseInt($("#topk").value, 10) || 5,
    },
  };
  if (state.recipePolicy) body.policy = state.recipePolicy;
  if ($("#chkOrders").checked) body.orders = 2;

  const btn = $("#btnRun");
  btn.disabled = true;
  btn.textContent = "deciding…";
  try {
    const result = await api("/api/decide", { method: "POST", body });
    renderResult(result.answers, result.usage);
    renderPolicy(result.policy);
    if (result.knowledge?.used) {
      const k = result.knowledge;
      $("#usageLine").textContent =
        `grounded: ${k.included} kb hits (${k.latency_ms} ms) · query: ${k.query.slice(0, 80)} — ` +
        $("#usageLine").textContent;
      renderKbHits(k.hits);
    } else if (result.knowledge?.error) {
      toast(`knowledge: ${result.knowledge.error}`, true);
    }
  } catch (e) {
    toast(`decide: ${e.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run decision";
  }
}

/* ---------------- knowledge panel ---------------- */

function renderKbHits(payload) {
  const box = $("#kbResults");
  box.innerHTML = "";
  const hits = Array.isArray(payload) ? payload
    : (payload && (payload.results || payload.hits || [payload])) || [];
  if (!hits.length) {
    box.innerHTML = `<div class="muted small">no hits</div>`;
    return;
  }
  for (const h of hits.slice(0, 12)) {
    const el = document.createElement("div");
    el.className = "kb-hit";
    const src = h.source || h.title || h.author || h.id || "";
    const text = (h.text || h.content || (typeof h === "string" ? h : "")).toString();
    el.innerHTML =
      (src ? `<div class="hit-src">${escapeHtml(src)}</div>` : "") +
      `<div class="hit-text">${escapeHtml(text.slice(0, 400))}</div>`;
    box.appendChild(el);
  }
}

async function kbSearch() {
  const q = $("#kbSearchInput").value.trim();
  if (!q) return;
  try {
    const out = await api("/api/knowledge/search", { method: "POST", body: { query: q, limit: 8 } });
    renderKbHits(out);
  } catch (e) {
    toast(`search: ${e.message}`, true);
  }
}

async function kbAsk() {
  const q = $("#kbAskInput").value.trim();
  if (!q) return;
  try {
    const out = await api("/api/knowledge/ask", { method: "POST", body: { question: q } });
    const text = typeof out === "string" ? out : JSON.stringify(out, null, 2);
    $("#kbResults").innerHTML = `<div class="kb-hit"><div class="hit-text">${escapeHtml(text.slice(0, 1200))}</div></div>`;
  } catch (e) {
    toast(`ask: ${e.message}`, true);
  }
}

/* ---------------- cookbook ---------------- */

let recipeCache = [];

async function loadCookbook() {
  try {
    recipeCache = await api("/api/recipes");
    const sel = $("#recipeSelect");
    const roles = [...new Set(recipeCache.map((r) => r.role))].sort();
    for (const role of roles) {
      const og = document.createElement("optgroup");
      og.label = role;
      for (const r of recipeCache.filter((x) => x.role === role)) {
        const o = document.createElement("option");
        o.value = r.key;
        o.textContent = r.title;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
  } catch (e) { /* cookbook is optional sugar */ }
}

function loadRecipe() {
  const key = $("#recipeSelect").value;
  const r = recipeCache.find((x) => x.key === key);
  if (!r) return;
  $("#stateInput").value = r.sample_state;
  $("#questionList").innerHTML = "";
  for (const [name, spec] of Object.entries(r.questions)) {
    addQuestion({ name, ...spec });
  }
  const hint = $("#recipeHint");
  hint.textContent = `${r.title} — ${r.description}`;
  hint.classList.remove("hidden");
  if (r.policy) {
    state.recipePolicy = r.policy;
  }
}

/* ---------------- example ---------------- */

function loadExample() {
  $("#stateInput").value =
    "From: ops@meridian-bank\n" +
    "Subject: KYC escalation for account 4471-XX\n\n" +
    "Client Iwona K. opened a business account two weeks ago. Screening hit a " +
    "PEP match with 71% fuzzy score on a name with common spelling variants. " +
    "Docs provided: passport, extract from CEIDG, utility bill. Address in the " +
    "bill differs from the application form (old flat, moved in January).";
  $("#questionList").innerHTML = "";
  addQuestion({
    name: "routing",
    type: "choice",
    instructions: "Where should this KYC escalation go first?",
    criteria: {
      auto_clear: "PEP match is false positive, clear automatically",
      compliance_officer: "route to a human compliance officer",
      edd_team: "enhanced due diligence team, high risk",
      reject: "reject the application",
    },
  });
  addQuestion({
    name: "needs_edd",
    type: "noul",
    instructions: "Does this case require enhanced due diligence?",
  });
}

/* ---------------- wiring ---------------- */

function wire() {
  $("#btnStart").onclick = startEngine;
  $("#btnStop").onclick = stopEngine;
  $("#btnRun").onclick = runDecision;
  $("#btnAddQ").onclick = () => addQuestion();
  $("#btnExample").onclick = loadExample;
  $("#btnLoadRecipe").onclick = loadRecipe;
  $("#btnSearch").onclick = kbSearch;
  $("#btnAsk").onclick = kbAsk;
  $("#kbSearchInput").addEventListener("keydown", (e) => e.key === "Enter" && kbSearch());
  $("#kbAskInput").addEventListener("keydown", (e) => e.key === "Enter" && kbAsk());
  $("#logDetails").addEventListener("toggle", loadLogs);
  $("#quantSelect").addEventListener("change", loadBackends);
  addQuestion();
}

wire();
loadSystem();
loadModels();
loadCookbook();
setInterval(loadSystem, 3000);
setInterval(loadLogs, 4000);
