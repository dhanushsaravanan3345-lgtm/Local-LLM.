"use strict";
const $ = (id) => document.getElementById(id);
const state = { cid: null, busy: false, abort: null };

/* ---------- helpers ---------- */
async function api(path, opts = {}) {
  let r;
  try { r = await fetch(path, opts); }
  catch { throw new Error("Cannot reach the app server. Is it still running?"); }
  let data = null;
  try { data = await r.json(); } catch { /* non-JSON */ }
  if (!r.ok) throw new Error(data?.error?.message || "Request failed.");
  return data;
}
const jsonOpts = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function renderText(el, text) {           // safe: only textContent is ever used
  el.textContent = "";
  text.split("```").forEach((part, i) => {
    if (i % 2) {
      const pre = document.createElement("pre"), code = document.createElement("code");
      code.textContent = part.replace(/^[\w+-]*\n/, "");
      pre.appendChild(code); el.appendChild(pre);
    } else if (part) el.appendChild(document.createTextNode(part));
  });
}
function scrollDown() { const m = $("messages"); m.scrollTop = m.scrollHeight; }
function showBanner(msg) { const b = $("banner"); b.textContent = msg || ""; b.hidden = !msg; }

/* ---------- messages ---------- */
function addMsg(role, text = "", sources = []) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "ai");
  const body = document.createElement("div"); body.className = "body"; el.appendChild(body);
  if (text) renderText(body, text);
  $("messages").querySelector(".empty")?.remove();
  $("messages").appendChild(el);
  if (role === "assistant" && sources.length) addMeta(el, { sources });
  scrollDown();
  return el;
}
function addMeta(el, { sources = [], stats = null, text = "" }) {
  el.querySelector(".meta")?.remove();
  const m = document.createElement("div"); m.className = "meta";
  if (sources.length) { const s = document.createElement("span"); s.textContent = "📄 Source: " + sources.join(", "); m.appendChild(s); }
  if (stats?.tokens_per_s) {
    const s = document.createElement("span");
    s.textContent = `${stats.tokens_per_s} tok/s · first token ${stats.first_token_s ?? "?"}s`; m.appendChild(s);
  }
  const copy = document.createElement("button"); copy.textContent = "Copy";
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(el.querySelector(".body").textContent); copy.textContent = "Copied ✓"; }
    catch { copy.textContent = "Copy failed"; }
    setTimeout(() => (copy.textContent = "Copy"), 1500);
  };
  m.appendChild(copy);
  if (el === $("messages").lastElementChild) {
    const rg = document.createElement("button"); rg.textContent = "↻ Regenerate"; rg.onclick = () => send({ regenerate: true });
    m.appendChild(rg);
  }
  el.appendChild(m);
}
function emptyState() {
  $("messages").innerHTML = '<div class="empty"><h2>🤖 My Local AI</h2><p>Everything runs on this computer.<br>Ask a question or upload a document.</p></div>';
}

/* ---------- send / stream ---------- */
function setBusy(b) { state.busy = b; $("sendBtn").hidden = b; $("stopBtn").hidden = !b; $("input").disabled = b; if (!b) $("input").focus(); }

async function send({ regenerate = false } = {}) {
  if (state.busy) return;
  const text = $("input").value.trim();
  if (!regenerate && !text) { showBanner("Please type a message first."); return; }
  showBanner("");
  if (regenerate) {
    const last = $("messages").lastElementChild; if (last?.classList.contains("ai")) last.remove();
  } else { addMsg("user", text); $("input").value = ""; autosize(); }

  const ai = addMsg("assistant");
  const body = ai.querySelector(".body");
  body.className = "body thinking"; body.textContent = "AI is thinking";
  setBusy(true);
  state.abort = new AbortController();
  let acc = "", sources = [], stats = null, failed = null;
  try {
    const r = await fetch("/api/chat/stream", { ...jsonOpts("POST", {
      message: text, conversation_id: state.cid, use_documents: $("useDocs").checked, regenerate }), signal: state.abort.signal });
    if (!r.ok) { let m = "Request failed."; try { m = (await r.json()).error.message; } catch {} throw new Error(m); }
    const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "meta") { state.cid = ev.conversation_id; sources = ev.sources; }
        else if (ev.type === "token") { acc += ev.text; body.className = "body"; renderText(body, acc); scrollDown(); }
        else if (ev.type === "done") stats = ev.stats;
        else if (ev.type === "error") failed = ev.message;
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") failed = e.message || "Something went wrong.";
  }
  body.className = "body";
  if (failed) { ai.classList.add("err"); if (!acc) body.textContent = failed; else showBanner(failed); }
  else if (!acc) body.textContent = "(stopped)";
  if (acc || !failed) addMeta(ai, { sources, stats });
  setBusy(false);
  loadChats(); refreshStatus();
}

/* ---------- sidebar data ---------- */
async function loadChats() {
  try {
    const { conversations } = await api("/api/conversations");
    const ul = $("chatList"); ul.textContent = "";
    conversations.forEach((c) => {
      const li = document.createElement("li"); if (c.id === state.cid) li.className = "active";
      const t = document.createElement("span"); t.className = "t"; t.textContent = c.title; li.appendChild(t);
      const x = document.createElement("button"); x.textContent = "✕"; x.title = "Delete chat";
      x.onclick = async (e) => { e.stopPropagation(); await api(`/api/conversations/${c.id}`, { method: "DELETE" }); if (state.cid === c.id) newChat(); loadChats(); };
      li.appendChild(x); li.onclick = () => openChat(c.id); ul.appendChild(li);
    });
  } catch (e) { showBanner(e.message); }
}
async function openChat(id) {
  if (state.busy) return;
  try {
    const { messages } = await api(`/api/conversations/${id}`);
    state.cid = id; $("messages").textContent = ""; showBanner("");
    messages.forEach((m) => addMsg(m.role, m.content, m.sources));
    const last = $("messages").lastElementChild;
    if (last?.classList.contains("ai")) addMeta(last, { sources: messages.at(-1).sources });
    $("sidebar").classList.remove("open"); loadChats(); scrollDown();
  } catch (e) { showBanner(e.message); }
}
function newChat() { if (state.busy) return; state.cid = null; emptyState(); showBanner(""); $("sidebar").classList.remove("open"); loadChats(); }

async function loadDocs() {
  try {
    const { documents } = await api("/api/documents");
    const ul = $("docList"); ul.textContent = "";
    if (!documents.length) { const li = document.createElement("li"); li.className = "muted small"; li.textContent = "No documents yet"; ul.appendChild(li); }
    documents.forEach((d) => {
      const li = document.createElement("li"); li.style.cursor = "default";
      const t = document.createElement("span"); t.className = "t"; t.textContent = "📄 " + d.name; t.title = `${d.chunk_count} chunks`; li.appendChild(t);
      const x = document.createElement("button"); x.textContent = "✕"; x.title = "Remove document";
      x.onclick = async () => { await api(`/api/documents/${d.id}`, { method: "DELETE" }); loadDocs(); };
      li.appendChild(x); ul.appendChild(li);
    });
  } catch (e) { showBanner(e.message); }
}
async function upload(file) {
  const msg = $("uploadMsg"); msg.textContent = `Processing ${file.name} locally…`;
  const fd = new FormData(); fd.append("file", file);
  try { const r = await api("/api/documents/upload", { method: "POST", body: fd }); msg.textContent = `✓ ${r.name} indexed (${r.chunks} chunks)`; loadDocs(); }
  catch (e) { msg.textContent = "✗ " + e.message; }
}

/* ---------- status & settings ---------- */
async function refreshStatus() {
  try {
    const m = await api("/api/models");
    const badge = $("localBadge");
    badge.textContent = m.status === "ready" ? "● Local" : m.status === "runtime_down" ? "● Runtime offline" : "● Model missing";
    badge.className = "badge" + (m.status === "ready" ? "" : " down");
    const s = m.system, g = s.gpu;
    $("modelInfo").textContent = `${m.current} · ${m.runtime} · ${m.status} · CPU ${s.cpu_percent}% · RAM ${s.ram_used_gb}/${s.ram_total_gb} GB` +
      (g ? ` · GPU ${g.util_percent}% ${Math.round(g.vram_used_mb)}/${Math.round(g.vram_total_mb)} MB` : "");
    const sel = $("modelSelect"); const names = m.installed.map((x) => x.name).filter((n) => !n.includes("embed") && !n.includes("minilm"));
    if (document.activeElement !== sel) {
      sel.textContent = "";
      (names.includes(m.current) ? names : [m.current, ...names]).forEach((n) => { const o = document.createElement("option"); o.value = o.textContent = n; o.selected = n === m.current; sel.appendChild(o); });
    }
    if (m.status === "runtime_down") showBanner("Ollama is not running. Start it with: ollama serve");
    else if (m.status === "model_missing") showBanner(`Model "${m.current}" is not installed. Run: ollama pull ${m.current}`);
    else if ($("banner").textContent.startsWith("Ollama is not") || $("banner").textContent.startsWith("Model \"")) showBanner("");
  } catch { $("localBadge").textContent = "● Server offline"; $("localBadge").className = "badge down"; }
}
async function loadSettings() { try { $("sysPrompt").value = (await api("/api/settings")).system_prompt; } catch {} }

/* ---------- wiring ---------- */
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 180) + "px"; }
$("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("input").addEventListener("input", autosize);
$("stopBtn").onclick = () => state.abort?.abort();
$("newChat").onclick = newChat;
$("clearBtn").onclick = async () => { if (state.busy) return; if (state.cid) await api(`/api/chat/history?conversation_id=${state.cid}`, { method: "DELETE" }); newChat(); };
$("menuBtn").onclick = () => $("sidebar").classList.toggle("open");
$("fileInput").onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; };
$("savePrompt").onclick = async () => { try { await api("/api/settings", jsonOpts("PUT", { system_prompt: $("sysPrompt").value })); $("savePrompt").textContent = "Saved ✓"; } catch (e) { showBanner(e.message); } setTimeout(() => ($("savePrompt").textContent = "Save prompt"), 1500); };
$("modelSelect").onchange = async (e) => { try { await api("/api/models/select", jsonOpts("POST", { model: e.target.value })); } catch (er) { showBanner(er.message); } refreshStatus(); };
function applyTheme(t) { document.documentElement.dataset.theme = t; try { localStorage.setItem("theme", t); } catch {} }
$("themeBtn").onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
let saved = null; try { saved = localStorage.getItem("theme"); } catch {}
applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

emptyState(); loadChats(); loadDocs(); loadSettings(); refreshStatus();
setInterval(() => { if (!state.busy) refreshStatus(); }, 5000);
