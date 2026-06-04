/* ============================================================================
 * tutor.js — a tiny in-page AI tutor for the FinetuningLearn animations.
 *
 * Drop-in: each animation.html sets `window.TUTOR_CONFIG = {...}` then loads
 * this script. It injects a floating "Ask tutor" button + chat panel that calls
 * an LLM API directly from the browser (your key stays in YOUR browser).
 *
 * Supported providers:
 *   - "anthropic" (Claude)  -> uses the official browser-access header
 *   - "openai"   (GPT)
 *
 * Your API key is saved in localStorage on THIS machine only. It is never sent
 * anywhere except directly to the provider you choose. Don't commit a key, and
 * don't share an HTML file after typing one (it isn't stored in the HTML, only
 * in your browser's localStorage — but be careful on shared computers).
 * ==========================================================================*/
(function () {
  const CFG = Object.assign({
    caseTitle: "FinetuningLearn",
    // Extra context handed to the model so answers fit the current lesson.
    context: "A hands-on course teaching neural-network fine-tuning.",
    // Suggested questions shown as clickable chips.
    suggestions: [
      "Explain this page like I'm new to ML.",
      "What's the single most important idea here?",
      "Give me a tiny code example."
    ]
  }, window.TUTOR_CONFIG || {});

  const LS = {
    key:   "ftlearn_api_key",
    prov:  "ftlearn_provider",
    model: "ftlearn_model"
  };
  const DEFAULT_MODELS = {
    anthropic: "claude-sonnet-4-6",
    openai:    "gpt-4o-mini"
  };
  const MODEL_PRESETS = {
    anthropic: ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"],
    openai:    ["gpt-4o-mini", "gpt-4o"]
  };

  // ---- conversation state -------------------------------------------------
  let history = []; // {role:'user'|'assistant', content:string}

  function systemPrompt() {
    return [
      `You are a friendly, rigorous tutor embedded in an interactive lesson titled "${CFG.caseTitle}".`,
      `The learner knows Python and ML basics but is newer to deep-learning training and fine-tuning.`,
      `Lesson context: ${CFG.context}`,
      ``,
      `Guidelines:`,
      `- Be concise and concrete. Prefer short paragraphs and small, runnable code snippets (PyTorch).`,
      `- Build intuition first, then the math, then code. Use analogies sparingly but well.`,
      `- When relevant, tie answers back to the learner's goal: continually fine-tuning a self-driving`,
      `  model cheaply and accurately without catastrophic forgetting.`,
      `- If a question is ambiguous, make a reasonable assumption and say so. Don't refuse.`,
      `- Use plain text and Markdown. Keep replies focused; avoid walls of text.`
    ].join("\n");
  }

  // ---- styles -------------------------------------------------------------
  const css = `
  #tutor-fab{position:fixed;right:20px;bottom:20px;z-index:99999;background:#5cc8ff;color:#04222f;
    border:0;border-radius:999px;padding:12px 18px;font-weight:700;font-size:14px;cursor:pointer;
    box-shadow:0 6px 24px rgba(0,0,0,.35);font-family:ui-sans-serif,system-ui,sans-serif}
  #tutor-fab:hover{filter:brightness(1.05)}
  #tutor-panel{position:fixed;right:20px;bottom:20px;z-index:100000;width:460px;max-width:calc(100vw - 32px);
    height:640px;max-height:calc(100vh - 32px);background:#12151d;color:#e7e9ee;border:1px solid #283044;
    border-radius:16px;display:none;flex-direction:column;overflow:hidden;font-family:ui-sans-serif,system-ui,sans-serif;
    box-shadow:0 12px 48px rgba(0,0,0,.5);resize:both}
  #tutor-panel.open{display:flex}
  #tutor-panel.max{width:min(960px,calc(100vw - 32px));height:calc(100vh - 32px);resize:none}
  #tutor-panel.collapsed{height:auto !important;min-height:0;resize:none}
  #tutor-panel.collapsed .tt-cfg,#tutor-panel.collapsed .tt-body,
  #tutor-panel.collapsed .tt-sugg,#tutor-panel.collapsed .tt-warn,#tutor-panel.collapsed .tt-foot{display:none}
  .tt-head{display:flex;align-items:center;gap:8px;padding:12px 14px;background:#161a24;border-bottom:1px solid #283044;cursor:default}
  .tt-head b{font-size:14px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .tt-icon{width:9px;height:9px;border-radius:50%;background:#6ee7a8;flex:none}
  .tt-x{background:none;border:0;color:#9aa3b2;font-size:18px;cursor:pointer;line-height:1}
  .tt-gear,.tt-min,.tt-max{background:none;border:0;color:#9aa3b2;font-size:15px;cursor:pointer;line-height:1}
  .tt-gear:hover,.tt-min:hover,.tt-max:hover,.tt-x:hover{color:#e7e9ee}
  .tt-cfg{display:none;padding:12px 14px;background:#0e1118;border-bottom:1px solid #283044;font-size:13px}
  .tt-cfg.open{display:block}
  .tt-cfg label{display:block;color:#9aa3b2;margin:8px 0 3px;font-size:12px}
  .tt-cfg input,.tt-cfg select{width:100%;padding:7px 9px;background:#171c28;border:1px solid #2a3346;
    border-radius:8px;color:#e7e9ee;font-size:13px}
  .tt-cfg .hint{color:#6b7689;font-size:11px;margin-top:6px;line-height:1.4}
  .tt-body{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
  .tt-msg{padding:9px 12px;border-radius:12px;font-size:13.5px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
  .tt-user{align-self:flex-end;background:#1f3a4d;border:1px solid #2b5066;max-width:85%}
  .tt-bot{align-self:flex-start;background:#181d29;border:1px solid #283044;max-width:92%}
  .tt-bot code{background:#0b0d13;padding:1px 5px;border-radius:5px;color:#6ee7a8;font-size:12.5px}
  .tt-bot pre{background:#0b0d13;padding:10px;border-radius:8px;overflow-x:auto;margin:6px 0}
  .tt-bot pre code{background:none;padding:0}
  .tt-sugg{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 8px}
  .tt-chip{background:#1a2030;border:1px solid #2a3346;color:#bcc6d6;border-radius:999px;padding:5px 10px;
    font-size:12px;cursor:pointer}
  .tt-chip:hover{background:#222a3d}
  .tt-foot{display:flex;gap:8px;padding:10px;border-top:1px solid #283044;background:#0e1118}
  .tt-foot textarea{flex:1;resize:none;height:42px;background:#171c28;border:1px solid #2a3346;border-radius:10px;
    color:#e7e9ee;padding:10px;font-size:13.5px;font-family:inherit}
  .tt-send{background:#5cc8ff;color:#04222f;border:0;border-radius:10px;padding:0 14px;font-weight:700;cursor:pointer}
  .tt-send:disabled{opacity:.5;cursor:default}
  .tt-warn{color:#ffb86b;font-size:12px;padding:0 14px 8px}
  .tt-cursor::after{content:'▋';animation:ttblink 1s steps(2) infinite;color:#6ee7a8}
  @keyframes ttblink{0%{opacity:1}50%{opacity:0}}
  /* rendered-markdown styling (shared by chat bubbles + board) */
  .md h1,.md h2,.md h3,.md h4,.md .tt-h{margin:10px 0 4px;color:#e7e9ee;font-size:14px;line-height:1.3}
  .md p{margin:6px 0}
  .md ul,.md ol{margin:6px 0;padding-left:20px}
  .md li{margin:3px 0}
  .md a{color:#5cc8ff}
  .md blockquote{margin:6px 0;padding:4px 10px;border-left:3px solid #2b5066;background:#0e1626;color:#bcc6d6}
  .md hr{border:0;border-top:1px solid #283044;margin:10px 0}
  .md code{background:#0b0d13;padding:1px 5px;border-radius:5px;color:#6ee7a8;font-size:12.5px}
  .md pre{background:#0b0d13;padding:10px;border-radius:8px;overflow-x:auto;margin:6px 0}
  .md pre code{background:none;padding:0}
  /* Lesson explanations added from your questions (woven into the tutorial) */
  #tutor-board-panel{grid-column:1/-1;background:#181b24;border:1px solid #232838;border-radius:14px;padding:16px 18px}
  #tutor-board-panel h2{font-size:15px;margin:0 0 4px;color:#5cc8ff}
  #tutor-board-panel .tb-sub{font-size:12.5px;color:#9aa3b2;margin-bottom:10px}
  #tutor-board-panel .tb-clear{float:right;background:#222838;color:#e7e9ee;border:0;border-radius:8px;padding:5px 10px;font-size:12px;cursor:pointer}
  #tutor-board{display:flex;flex-direction:column;gap:10px}
  .tb-empty{color:#6b7689;font-size:13px;padding:10px 0}
  .tb-card{border:1px solid #28304a;border-radius:10px;overflow:hidden;background:#0e1118}
  .tb-card>summary{cursor:pointer;list-style:none;padding:10px 12px;font-size:13.5px;color:#e7e9ee;font-weight:600;display:flex;gap:8px;align-items:flex-start}
  .tb-card>summary::-webkit-details-marker{display:none}
  .tb-card>summary::before{content:"▶ ";color:#5cc8ff;font-size:11px}
  .tb-card[open]>summary::before{content:"▼ ";color:#5cc8ff;font-size:11px}
  .tb-card .tb-num{color:#5cc8ff;flex:none}
  .tb-a{padding:2px 14px 12px;font-size:13.5px;color:#c4ccda;line-height:1.55}
  `;
  const style = document.createElement("style"); style.textContent = css; document.head.appendChild(style);

  // ---- DOM ----------------------------------------------------------------
  const fab = document.createElement("button");
  fab.id = "tutor-fab"; fab.textContent = "💬 Ask tutor";
  document.body.appendChild(fab);

  const panel = document.createElement("div");
  panel.id = "tutor-panel";
  panel.innerHTML = `
    <div class="tt-head">
      <span class="tt-icon"></span><b>Tutor · ${escapeHtml(CFG.caseTitle)}</b>
      <button class="tt-gear" title="Settings">⚙</button>
      <button class="tt-max" title="Maximize / restore">⤢</button>
      <button class="tt-min" title="Collapse / expand">▁</button>
      <button class="tt-x" title="Close">✕</button>
    </div>
    <div class="tt-cfg">
      <label>Provider</label>
      <select class="tt-prov">
        <option value="anthropic">Anthropic (Claude)</option>
        <option value="openai">OpenAI (GPT)</option>
      </select>
      <label>API key <span style="color:#6b7689">(stored only in this browser)</span></label>
      <input class="tt-key" type="password" placeholder="sk-..." autocomplete="off">
      <label>Model</label>
      <input class="tt-model" list="tt-models" placeholder="model id">
      <datalist id="tt-models"></datalist>
      <div class="hint">Your key is saved in localStorage on this device and sent only to the provider
        you pick. Anthropic calls use the official direct-browser-access header. Don't use a shared computer.</div>
    </div>
    <div class="tt-body"></div>
    <div class="tt-sugg"></div>
    <div class="tt-warn" style="display:none"></div>
    <div class="tt-foot">
      <textarea class="tt-input" placeholder="Ask about this page... (Enter to send)"></textarea>
      <button class="tt-send">Send</button>
    </div>`;
  document.body.appendChild(panel);

  const $ = (s) => panel.querySelector(s);
  const body = $(".tt-body"), input = $(".tt-input"), sendBtn = $(".tt-send"),
        cfgBox = $(".tt-cfg"), provSel = $(".tt-prov"), keyInp = $(".tt-key"),
        modelInp = $(".tt-model"), warn = $(".tt-warn"), suggBox = $(".tt-sugg"),
        modelsList = panel.querySelector("#tt-models");

  // ---- settings load/save -------------------------------------------------
  function loadCfg() {
    provSel.value = localStorage.getItem(LS.prov) || "anthropic";
    keyInp.value  = localStorage.getItem(LS.key) || "";
    modelInp.value = localStorage.getItem(LS.model) || DEFAULT_MODELS[provSel.value];
    refreshModelPresets();
  }
  function refreshModelPresets() {
    const p = provSel.value;
    modelsList.innerHTML = (MODEL_PRESETS[p] || []).map(m => `<option value="${m}">`).join("");
    if (!modelInp.value || Object.values(DEFAULT_MODELS).includes(modelInp.value) ||
        (MODEL_PRESETS.anthropic.concat(MODEL_PRESETS.openai).includes(modelInp.value) &&
         !MODEL_PRESETS[p].includes(modelInp.value))) {
      modelInp.value = DEFAULT_MODELS[p];
    }
  }
  provSel.onchange = () => { localStorage.setItem(LS.prov, provSel.value); refreshModelPresets(); saveCfg(); };
  function saveCfg() {
    localStorage.setItem(LS.key, keyInp.value.trim());
    localStorage.setItem(LS.prov, provSel.value);
    localStorage.setItem(LS.model, modelInp.value.trim());
  }
  keyInp.onchange = saveCfg; modelInp.onchange = saveCfg;

  // ---- open/close ---------------------------------------------------------
  fab.onclick = () => {
    panel.classList.add("open"); fab.style.display = "none"; loadCfg(); applyWinState();
    if (!localStorage.getItem(LS.key)) cfgBox.classList.add("open");
    input.focus();
  };
  $(".tt-x").onclick = () => { panel.classList.remove("open"); fab.style.display = ""; };
  $(".tt-gear").onclick = () => cfgBox.classList.toggle("open");

  // ---- window controls: maximize + collapse (persisted) -------------------
  const LS_MAX = "ftlearn_max", LS_COL = "ftlearn_collapsed";
  function applyWinState() {
    panel.classList.toggle("max", localStorage.getItem(LS_MAX) === "1");
    panel.classList.toggle("collapsed", localStorage.getItem(LS_COL) === "1");
    $(".tt-min").textContent = panel.classList.contains("collapsed") ? "▢" : "▁";
  }
  $(".tt-max").onclick = () => {
    const on = !panel.classList.contains("max");
    localStorage.setItem(LS_MAX, on ? "1" : "0");
    if (on) localStorage.setItem(LS_COL, "0");      // maximizing un-collapses
    panel.style.width = ""; panel.style.height = ""; // drop any manual resize
    applyWinState();
  };
  $(".tt-min").onclick = () => {
    const on = !panel.classList.contains("collapsed");
    localStorage.setItem(LS_COL, on ? "1" : "0");
    if (on) localStorage.setItem(LS_MAX, "0");       // collapsing un-maximizes
    applyWinState();
  };

  // ---- suggestions --------------------------------------------------------
  (CFG.suggestions || []).forEach(s => {
    const c = document.createElement("button");
    c.className = "tt-chip"; c.textContent = s;
    c.onclick = () => { input.value = s; send(); };
    suggBox.appendChild(c);
  });

  // ---- Q&A board: pins each answer onto the PAGE as study notes -----------
  // Lives in the page flow (inside .wrap if present), persists per page URL.
  const BOARD_KEY = "ftlearn_board_" + location.pathname;
  let board = [];
  try { board = JSON.parse(localStorage.getItem(BOARD_KEY) || "[]"); } catch { board = []; }

  const boardPanel = document.createElement("section");
  boardPanel.id = "tutor-board-panel";
  boardPanel.innerHTML =
    `<button class="tb-clear" title="Remove these added explanations">Clear</button>` +
    `<h2>📚 Lesson explanations — added from your questions</h2>` +
    `<div class="tb-sub">As you ask the 💬 tutor, its answers are folded into the tutorial here as new, ` +
    `collapsible explanation sections — so this lesson grows to cover <em>your</em> questions. Saved for this page.</div>` +
    `<div id="tutor-board"></div>`;
  const wrap = document.querySelector(".wrap");
  if (wrap) wrap.appendChild(boardPanel); else document.body.insertBefore(boardPanel, fab);
  const boardEl = boardPanel.querySelector("#tutor-board");
  boardPanel.querySelector(".tb-clear").onclick = () => {
    if (!board.length || confirm("Remove the explanations added from your questions on this page?")) {
      board = []; saveBoard(); renderBoard();
    }
  };

  function saveBoard() { try { localStorage.setItem(BOARD_KEY, JSON.stringify(board)); } catch {} }
  function renderBoard() {
    if (!board.length) {
      boardEl.innerHTML = `<div class="tb-empty">Nothing added yet. Open the tutor (💬, bottom-right) and ask about this page — your questions become explanation sections here.</div>`;
      return;
    }
    boardEl.innerHTML = board.map((c, i) =>
      `<details class="tb-card" open><summary><span class="tb-num">Q${i + 1}.</span>` +
      `<span>${escapeHtml(c.q)}</span></summary>` +
      `<div class="tb-a md">${renderMd(c.a)}</div></details>`).join("");
  }
  function addToBoard(q, a) { board.push({ q, a, t: Date.now() }); saveBoard(); renderBoard(); }
  renderBoard();

  // ---- send ---------------------------------------------------------------
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  sendBtn.onclick = send;

  function showWarn(msg) { warn.style.display = "block"; warn.textContent = msg; }
  function clearWarn() { warn.style.display = "none"; warn.textContent = ""; }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    saveCfg();
    const key = keyInp.value.trim();
    if (!key) { cfgBox.classList.add("open"); showWarn("Add your API key in ⚙ settings first."); return; }
    clearWarn();
    suggBox.style.display = "none";

    addMsg("user", text);
    history.push({ role: "user", content: text });
    input.value = ""; setBusy(true);

    const botEl = addMsg("bot", "");
    botEl.classList.add("tt-cursor");

    try {
      const provider = provSel.value;
      const model = modelInp.value.trim() || DEFAULT_MODELS[provider];
      let full = "";
      const onDelta = (chunk) => { full += chunk; botEl.innerHTML = renderMd(full); body.scrollTop = body.scrollHeight; };
      if (provider === "anthropic") await streamAnthropic(key, model, onDelta);
      else await streamOpenAI(key, model, onDelta);
      botEl.classList.remove("tt-cursor");
      history.push({ role: "assistant", content: full });
      if (full.trim()) addToBoard(text, full);   // pin this Q&A onto the page board
    } catch (err) {
      botEl.classList.remove("tt-cursor");
      botEl.innerHTML = renderMd("⚠️ **Error:** " + (err && err.message ? err.message : String(err)) +
        "\n\nCheck your API key, model id, and network. Open ⚙ settings to fix.");
      // drop the failed user turn so retry doesn't double it
      history.pop();
    } finally {
      setBusy(false); input.focus();
    }
  }

  function setBusy(b) { sendBtn.disabled = b; sendBtn.textContent = b ? "…" : "Send"; }

  // ---- providers (streaming via fetch + SSE) ------------------------------
  async function streamAnthropic(key, model, onDelta) {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true"
      },
      body: JSON.stringify({
        model,
        max_tokens: 1024,
        system: systemPrompt(),
        stream: true,
        messages: history.map(m => ({ role: m.role, content: m.content }))
      })
    });
    if (!res.ok) throw new Error(await errText(res));
    await readSSE(res, (evt) => {
      if (evt.type === "content_block_delta" && evt.delta && evt.delta.type === "text_delta")
        onDelta(evt.delta.text);
    });
  }

  async function streamOpenAI(key, model, onDelta) {
    const msgs = [{ role: "system", content: systemPrompt() }]
      .concat(history.map(m => ({ role: m.role, content: m.content })));
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", "authorization": "Bearer " + key },
      body: JSON.stringify({ model, stream: true, messages: msgs })
    });
    if (!res.ok) throw new Error(await errText(res));
    await readSSE(res, (evt) => {
      const d = evt.choices && evt.choices[0] && evt.choices[0].delta;
      if (d && d.content) onDelta(d.content);
    });
  }

  async function errText(res) {
    let t = "";
    try { const j = await res.json(); t = (j.error && (j.error.message || j.error.type)) || JSON.stringify(j); }
    catch { t = await res.text().catch(() => ""); }
    return `HTTP ${res.status} — ${t || res.statusText}`;
  }

  // Generic SSE reader: calls cb(parsedJSON) for each `data:` line.
  async function readSSE(res, cb) {
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try { cb(JSON.parse(data)); } catch { /* ignore keep-alives */ }
      }
    }
  }

  // ---- tiny markdown renderer (safe-ish: escapes first) -------------------
  function escapeHtml(s) { return s.replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  // Inline markdown: code, bold, italic, links. Input is ALREADY html-escaped.
  function inlineMd(s) {
    return s
      .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
               '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  // Block-level markdown renderer (no deps): fenced code, headings,
  // ordered/unordered lists, blockquotes, hr, and paragraphs.
  function renderMd(src) {
    const blocks = [];
    src = src.replace(/```[ \t]*([\w+-]*)\r?\n?([\s\S]*?)```/g, (_, lang, code) => {
      blocks.push(`<pre><code>${escapeHtml(code.replace(/\r?\n$/, ""))}</code></pre>`);
      return ` B${blocks.length - 1} `;
    });

    const lines = escapeHtml(src).split(/\r?\n/);
    let html = "", list = null;
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

    for (let raw of lines) {
      const line = raw.replace(/ B(\d+) /g, (_, i) => blocks[+i]);
      if (/<pre>/.test(line)) { closeList(); html += line; continue; }
      if (/^\s*$/.test(line)) { closeList(); continue; }
      if (/^(---|\*\*\*|___)\s*$/.test(line)) { closeList(); html += "<hr>"; continue; }

      let m;
      if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
        closeList(); const lvl = Math.min(m[1].length, 4);
        html += `<h${lvl} class="tt-h">${inlineMd(m[2])}</h${lvl}>`; continue;
      }
      if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
        if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
        html += `<li>${inlineMd(m[1])}</li>`; continue;
      }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
        if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
        html += `<li>${inlineMd(m[1])}</li>`; continue;
      }
      if ((m = line.match(/^\s*&gt;\s?(.*)$/))) {
        closeList(); html += `<blockquote>${inlineMd(m[1])}</blockquote>`; continue;
      }
      closeList(); html += `<p>${inlineMd(line)}</p>`;
    }
    closeList();
    return html;
  }

  function addMsg(who, text) {
    const el = document.createElement("div");
    el.className = "tt-msg " + (who === "user" ? "tt-user" : "tt-bot md");
    el.innerHTML = who === "user" ? escapeHtml(text) : renderMd(text);
    body.appendChild(el); body.scrollTop = body.scrollHeight;
    return el;
  }
})();
