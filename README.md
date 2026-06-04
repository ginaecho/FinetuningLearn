# FinetuningLearn

Learning fine-tuning from real hands-on examples — understand the theory, then implement it, step by step.

Each **case** has three layers so you can learn the way that suits you:

| Layer | File | What it gives you |
|-------|------|-------------------|
| 🎬 **Animation** | `animation.html` | Open in a browser. Sliders + animated canvases build intuition *before* any math. |
| 📓 **Notebook** | `*.ipynb` | Runnable Jupyter notebook: theory refresher → code → experiment. Runs on CPU. |
| 🧪 **Your turn** | `challenge.md` | An open problem to solve yourself, with hints and a checklist. |

> **New here?** Read [`SETUP.md`](./SETUP.md) first to install PyTorch + LoRA/QLoRA and hook up your local Llama.

---

## The roadmap (medium → professional)

The cases are ordered so each one earns the next. The whole path is built to land on **your** question:
*how do you keep fine-tuning a self-driving model as new data arrives — cheaply, accurately, and without it forgetting what it already knew?*

| # | Case | Level | Core idea you'll own | Status |
|---|------|-------|----------------------|--------|
| 01 | **Foundations** — what *is* fine-tuning? | Medium | Pre-train vs fine-tune; which weights move; over/underfitting | ✅ Built |
| 02 | **LoRA from scratch** | Medium | Why low-rank adapters work; implement LoRA in ~30 lines | ✅ Built |
| 03 | **QLoRA on a real LLM** | Medium+ | 4-bit quantization + LoRA; fine-tune Llama on one GPU | 🔜 Next |
| 04 | **Instruction / SFT tuning** | Pro-ish | Datasets, chat templates, eval; measure if it actually learned | 🔜 |
| 05 | **Catastrophic forgetting — see it happen** | Pro-ish | Train task B, watch task A collapse; *measure* forgetting | 🔜 |
| 06 | **Continual learning toolkit** | Pro | Replay, EWC, LoRA-per-task, distillation — when to use which | 🔜 |
| 07 | **Capstone: Auto-car continual fine-tuning** | Pro | Cost vs accuracy vs forgetting on a streaming-update problem | 🔜 |

✅ = ready to learn now. 🔜 = built as you progress (so each builds on what you actually ran).

---

## How to use this repo

1. **Install once:** follow [`SETUP.md`](./SETUP.md).
2. **Pick a case folder** under [`cases/`](./cases/), start with `01_foundations`.
3. In each folder:
   - Open `animation.html` in your browser → play with it for 5 min.
   - Open the `.ipynb` in Jupyter/VS Code → run top to bottom, then tweak.
   - Try `challenge.md` → solve it, then compare with notes.
4. Come back and we build the next case together.

---

## 💬 Ask-the-tutor (in the animations)

Every `animation.html` has a **floating "💬 Ask tutor" button**. Click it, paste your API key once
(⚙ settings), and ask questions about whatever you're looking at — the tutor is given the page's context
so its answers fit the lesson.

- **Resizable window:** drag the bottom-right corner, **⤢ maximize**, or **▁ collapse** to just the title
  bar. Your size/state is remembered.
- **Readable answers:** replies render as **Markdown** (headings, lists, code blocks, links).
- **📚 Self-updating lesson:** every answer is folded into the tutorial as a new collapsible section, so the
  lesson grows to cover *your* questions. **Curate it:** click **👍 Useful** to keep a section or
  **🗑 This can be removed** to drop it. Kept sections are fed back to the tutor as memory, so it builds on
  what you found useful (saved per page).

- **Bring your own key:** **Anthropic (Claude)** by default, or **OpenAI**. Pick the provider in ⚙.
- **Your key stays local.** It's saved in your browser's `localStorage` on your machine and sent *only*
  to the provider you choose (Claude calls use Anthropic's official direct-browser-access header). It is
  **never** written into the HTML or committed to git.
- **Where to get a key:** Anthropic → <https://console.anthropic.com/settings/keys> · OpenAI → <https://platform.openai.com/api-keys>.
- ⚠️ Don't type your key on a shared/public computer, and don't paste a key into any file in this repo.

See [`SETUP.md` §7](./SETUP.md#7-in-page-ai-tutor-optional) for details.

---

## Your capstone question (Case 07)

> *"In fine-tuning for an autonomous car, when new memory/data and new training arrive, how do I fine-tune in the best way for cost, efficiency, and accuracy — and how do I overcome catastrophic forgetting?"*

We don't hand-wave this. By Case 07 you'll have built every piece:
- **Cost/efficiency** ← LoRA & QLoRA (Cases 02–03): tune <1% of weights.
- **Accuracy** ← SFT + eval discipline (Case 04).
- **Forgetting** ← measure it (Case 05), then defeat it (Case 06) with replay / regularization / modular adapters.

Case 07 puts them together on a simulated streaming self-driving task and makes you trade the three off against each other — the real engineering decision.

---

## Glossary quick-reference

- **Fine-tuning** — continue training a pre-trained model on new, narrower data.
- **LoRA** (Low-Rank Adaptation) — freeze the big model, train two small matrices per layer.
- **QLoRA** — LoRA on top of a 4-bit quantized model → fits big models on small GPUs.
- **Catastrophic forgetting** — a model loses old skills when trained on new data.
- **Continual / lifelong learning** — training on a *stream* of tasks without forgetting.
