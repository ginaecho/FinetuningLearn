# CLAUDE.md — working agreement for FinetuningLearn

This file is the durable memory for anyone (human or agent) building in this repo.
It records **what the maintainer expects** and **what we've learned** so the bar stays high.

## What this project is

A hands-on fine-tuning curriculum, Cases 00 → 07, each a self-contained lesson with
three learning layers:

| Layer | File | Purpose |
|-------|------|---------|
| 🎬 Animation | `animation.html` | Build intuition with interactive canvases **before** any math. |
| 📓 Notebook | `*.ipynb` | Runnable code. Cases 00–02 run on CPU; **Case 03+ are Colab-first** (free T4). |
| 🧪 Challenge | `challenge.md` | An open problem the learner solves. |
| 📖 Theory | `THEORY.md` | Glossary + curated links; ⭐ Promote from the in-page tutor appends here. |

The whole path lands on one capstone question: *how do you keep fine-tuning a
self-driving model as new data arrives — cheaply, accurately, without forgetting?*
Keep that **self-driving narrative thread** running through every case (sunny→snowy
roads, highway/urban/snow adapters, streaming experiences, etc.).

## The animation design contract (NON-NEGOTIABLE)

The maintainer has repeatedly asked for the same things. Every `animation.html` must:

1. **Actually animate the dynamic, not just draw a static picture.** Show the *process*
   (training curves moving, a weight settling, data flowing, a query funneling). If a
   panel only renders one frozen state, it's not done. Use `requestAnimationFrame` for
   continuous motion and `setInterval`/step buttons for stepped processes.

2. **Be genuinely interactive.** Sliders, method toggles, "▶ Train / Send / Run"
   buttons, clickable elements. The learner should be able to *cause the problem* and
   then *cause the fix* and watch the difference. Prefer "try it and see" over "read about it."

3. **Explain what the visual MEANS and HOW to interact — inline, not buried.** Every panel
   gets a `.howto` block (defined in `assets/lesson.css`) with two cards:
   - `<div class="card see"><span class="tag">👁 What you're seeing</span> …</div>`
     — decode every color/axis/shape in plain language.
   - `<div class="card try"><span class="tag">🎮 Try this</span> …</div>`
     — concrete active-learning steps ("set X to 0, train, watch Y collapse; now set X to 20% and retrain").
   This is the maintainer's top recurring request. Do not skip it.

4. **Keep the 📖 theory expander** (`details.theory`) for the deeper "why," and define
   every term and every control there.

5. **Set `window.TUTOR_CONFIG`** with an accurate `context` describing each panel (the
   in-page tutor uses it) plus 3–4 `suggestions`, including at least one
   "how do I read this visual?" question.

### Layout / canvas gotchas learned the hard way
- A canvas in a **non-full** `.panel` (half-width grid column) can size to ~0 width via
  `getBoundingClientRect()` and silently render nothing. If a panel's animation "doesn't
  show," first suspect width — make it `.panel.full` or verify `fit()` runs after layout.
- Always `fit(canvas)` (set `canvas.width/height *= dpr`) **after** the element is in the
  DOM and on `resize`. Existing pattern: `window.addEventListener('resize', …)` re-fits or reloads.
- Guard against zero: `c.width = Math.max(1, r.width*dpr)`.
- **Syntax-check every `<script>` block** before committing:
  ```
  node -e "const fs=require('fs');const h=fs.readFileSync('FILE','utf8');
  [...h.matchAll(/<script>([\s\S]*?)<\/script>/g)].forEach((m,i)=>{try{new Function(m[1]);console.log(i,'OK')}catch(e){console.log(i,e.message)}})"
  ```

### Two valid ways to drive a "real model" animation
- **Live JS simulation** with closed-form/approximate dynamics (Cases 00, 01, 06, 07).
  Cheap, fully interactive, good when exact numbers don't matter.
- **Precomputed-data playback**: a Python script trains a real model, dumps the
  trajectory to a JSON blob embedded in the HTML, and the page interpolates frames
  (Case 05's `window.CASE05_DATA`). Use when authenticity matters. Keep the blob on its
  own line; it can be large.

## Shared building blocks (`cases/assets/`)
- `lesson.css` — page/panel layout, `.howto` callouts, `.metric` cards, `.formula`,
  `details.theory` styling. New cases should link this, not inline styles.
- `tutor.js` — floating 💬 tutor + self-updating "📚 Lesson explanations" board. Driven
  entirely by `window.TUTOR_CONFIG`. Bring-your-own-key (Anthropic/OpenAI).

## Security rules (still in force)
- API keys live in `localStorage` only. **Never** write a key into any file or commit one.
- `.gitignore` blocks `.key/.pem/.env/secrets`. Don't enter keys on shared machines.

## Git workflow
- Develop on the assigned feature branch; commit with clear messages; push with
  `git push -u origin <branch>`. Do **not** open a PR unless explicitly asked.
- Don't put model IDs or internal identifiers in commits/PRs/code — chat only.

## Case status
- 00 Tensors & NN basics · 01 Foundations · 02 LoRA from scratch · 03 QLoRA (Colab) — built.
- 05 Catastrophic forgetting — real-data playback, 3 panels + `.howto`.
- 06 Continual learning toolkit — live "train & measure" sandbox: naive/replay/EWC/adapters.
- 07 Capstone autocar — hybrid architecture, replay-vs-retrieval stream, memory funnel,
  cost/accuracy/forgetting simulator.
- (04 Instruction/SFT is reserved in the roadmap; folder not yet built.)

## When asked to "improve" an animation, the checklist
- [ ] Does each panel move / respond, or is it static? (make it move)
- [ ] Does each panel have a `.howto` (see + try)? (add it)
- [ ] Can the learner cause the problem AND the fix?
- [ ] Are all colors/axes decoded in words?
- [ ] Is the self-driving thread present?
- [ ] `TUTOR_CONFIG.context` updated to match the new panels?
- [ ] All `<script>` blocks pass `node` syntax check?
