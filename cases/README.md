# Cases

Work through these in order — each builds on the last and they all lead to the self-driving capstone.

| # | Folder | What you build | Status |
|---|--------|----------------|--------|
| 01 | [`01_foundations`](./01_foundations/) | Pre-train vs fine-tune; warm start beats cold start | ✅ |
| 02 | [`02_lora_from_scratch`](./02_lora_from_scratch/) | Implement LoRA yourself; match full tuning at ~1% cost | ✅ |
| 03 | `03_qlora_real_llm` | QLoRA on a real Llama via `peft` + 4-bit | 🔜 |
| 04 | `04_instruction_sft` | Instruction/SFT tuning + real evaluation | 🔜 |
| 05 | `05_catastrophic_forgetting` | Make forgetting happen and **measure** it | 🔜 |
| 06 | `06_continual_learning` | Replay · EWC · per-task adapters · distillation | 🔜 |
| 07 | `07_capstone_autocar` | Streaming self-driving updates: cost vs accuracy vs forgetting | 🔜 |

Each folder: `README.md` (start here) · `animation.html` (intuition) · `*.ipynb` (code) · `challenge.md` (your turn).

Built so far: **01 and 02**. Finish their challenges, then say the word and I'll build the next case.

---

## Shared building blocks (`assets/`)

Every case reuses these, so new cases are cheap to add and behave consistently:

| File | What it gives every page |
|------|--------------------------|
| `assets/lesson.css` | Page/panel layout + the 📖 "Read the theory" expander styling. |
| `assets/tutor.js` | The floating 💬 tutor **and** the self-updating "📚 Lesson explanations" board (keep / remove / ⭐ promote-to-Markdown). Just set `window.TUTOR_CONFIG`. |

## Adding a new case

1. **Copy the template:** `cp -r cases/_TEMPLATE cases/NN_short_name`.
2. **Rename placeholders:** replace `NN` and `TITLE` in `README.md`, `THEORY.md`, `challenge.md`, `animation.html`.
3. **Fill `animation.html`:** duplicate the example `.panel` per idea; give each a 📖 `details.theory` expander
   that defines every term and every control. Set `TUTOR_CONFIG.context` to describe the panels.
4. **Add code:** a runnable `*.py` and/or a `*.ipynb` (keep it CPU-friendly where possible).
5. **Write `THEORY.md`:** glossary mapped to the animation + curated links. Learners' **⭐ Promote** exports
   append here.
6. **Link it:** add a row to the table above and to the root `README.md` roadmap.

The `_TEMPLATE/` folder is not a lesson — it's the starting skeleton (note the leading underscore so it sorts first and is easy to spot).
