# SETUP — environment for fine-tuning (LoRA / QLoRA)

You said you have **Python + ML basics** and **Llama installed**. This gets you from there to
running real LoRA/QLoRA fine-tuning. Everything here also runs **CPU-only** with tiny models, so
you can do Cases 01–02 even with no GPU.

---

## 0. The mental model (30 seconds)

There are two different "Llama" things people install — make sure you know which you have:

| If you installed… | It's for… | Can it fine-tune? |
|---|---|---|
| **Ollama** (`ollama run llama3`) | *running* models locally for chat | No — inference only. We'll use it later just to test. |
| **`llama.cpp`** | fast CPU/GPU *inference* of GGUF files | Inference only (plus tiny LoRA merge). |
| **Meta Llama weights** (HuggingFace `meta-llama/...`) | research / training | ✅ Yes — this is what we fine-tune. |

> Check what you have: run `ollama --version` and `python -c "import transformers"` (next section).
> Whatever you have, the setup below gives you the **training** stack. We can still point inference at your Ollama later.

---

## 1. Create an isolated environment

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 2. Install the training stack

### CPU-only or Mac (works for Cases 01–02, slow but fine)
```bash
pip install -r requirements.txt
```

### NVIDIA GPU (needed for QLoRA / real Llama, Cases 03+)
```bash
# 1) install a CUDA build of torch (pick the CUDA that matches your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) the rest
pip install -r requirements.txt
# 3) QLoRA needs bitsandbytes for 4-bit quantization (GPU only)
pip install bitsandbytes
```

> **No NVIDIA GPU?** Skip `bitsandbytes`. Use **Google Colab** (free T4 GPU) for Cases 03+:
> upload the notebook, then `Runtime → Change runtime type → T4 GPU`.

---

## 3. The three libraries that matter (and why)

| Library | Role | One-liner |
|---|---|---|
| `transformers` | loads pre-trained models (Llama, GPT-2, BERT…) | the model zoo |
| `peft` | **P**arameter-**E**fficient **F**ine-**T**uning = LoRA/QLoRA lives here | the adapter toolkit |
| `bitsandbytes` | 4-bit/8-bit quantization | makes big models fit small GPUs |
| `datasets` | load & stream training data | the data plumbing |
| `accelerate` | device/precision/multi-GPU glue | makes training "just work" |

LoRA is **not a separate install** — it's a feature of `peft`. "Installing LoRA" = `pip install peft`.

---

## 4. Verify it works

```bash
python -c "import torch, transformers, peft, datasets; \
print('torch', torch.__version__); \
print('cuda available:', torch.cuda.is_available()); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__)"
```

You want no import errors. `cuda available: False` is OK for Cases 01–02.

Then run the smoke test for Case 01:
```bash
python cases/01_foundations/train.py
```

---

## 5. (Optional) get a real Llama to fine-tune later

For Cases 03+ you'll want a small, gated-free model. Good CPU/GPU-friendly choices:

```bash
# tiny, ungated, great for learning (no license click needed):
#   TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   Qwen/Qwen2.5-0.5B-Instruct
# real Llama (needs a free HF account + accept license):
#   meta-llama/Llama-3.2-1B-Instruct
huggingface-cli login        # paste a token from huggingface.co/settings/tokens
```

We'll download these inside the Case 03 notebook — no need to do it now.

---

## 6. Jupyter (for the `.ipynb` notebooks)

```bash
pip install jupyterlab
jupyter lab            # opens in your browser
```
Or just open the `.ipynb` files in **VS Code** (install the Jupyter extension) — same thing, nicer UI.

---

## 7. In-page AI tutor (optional)

The interactive `animation.html` pages include a **💬 Ask tutor** button so you can ask questions
without leaving the lesson. It calls an LLM API **directly from your browser** — no server, works from a
`file://` page.

**Setup (once):**
1. Open any `cases/*/animation.html` in your browser.
2. Click **💬 Ask tutor → ⚙** and choose a provider:
   - **Anthropic (Claude)** — get a key at <https://console.anthropic.com/settings/keys>
   - **OpenAI (GPT)** — get a key at <https://platform.openai.com/api-keys>
3. Paste the key and pick a model (defaults: `claude-sonnet-4-6` / `gpt-4o-mini`). Cheaper option:
   `claude-haiku-4-5-20251001`.
4. Ask away. Replies stream in as rendered **Markdown** and the tutor knows which lesson page you're on.

**Window controls:** drag the bottom-right corner to resize, **⤢** to maximize, **▁** to collapse to the
title bar (state is remembered). **📚 Lesson grows with you:** each answer is folded into the tutorial as a
new collapsible explanation section on the page — the lesson expands to cover your own questions; use
**Clear** to reset.

**How your key is handled (read this):**
- Stored in your browser's `localStorage` on **this device only**.
- Sent **only** to the provider you select. Anthropic requests use the official
  `anthropic-dangerous-direct-browser-access` header (that's the supported way to call the API from a
  browser).
- It is **never** written into the HTML files and **never** committed to git.
- 🔒 Don't enter a key on a shared computer. The repo's `.gitignore` also blocks common key files.

**Why a browser call is OK here:** these are personal, local learning pages. For a real app you'd proxy
the key through a backend instead of exposing it client-side — but for a single-user local tutor this is
the simplest setup.

---

## Troubleshooting

- **`bitsandbytes` errors on Mac/CPU** → expected; it's GPU-only. Skip it; do QLoRA on Colab.
- **`torch.cuda.is_available()` is False but you have a GPU** → you installed the CPU wheel; redo step 2 with the CUDA `--index-url`.
- **Out of memory on GPU** → lower `batch_size`, raise `gradient_accumulation_steps`, or use a smaller model / QLoRA.
- **HF download is gated** → use the ungated `TinyLlama`/`Qwen` models above while you wait for Llama access.

Done? → open [`cases/01_foundations/`](./cases/01_foundations/).
