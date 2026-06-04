# Case 03 · QLoRA on a real LLM

**Level:** Medium-Advanced · **Runs on:** Colab free T4 (16 GB) or any >=12 GB GPU · **Prereq:** Case 02

---

## What you'll walk away knowing
- **Quantization** squeezes a 16 GB model into ~4 GB so it fits a cheap GPU.
- **QLoRA** = LoRA on a 4-bit model — the standard trick for fine-tuning big LLMs affordably.
- How `BitsAndBytesConfig`, `peft`, and `trl`'s `SFTTrainer` wire together in practice.
- How to **save, load, and merge** LoRA adapters from a real model.
- That you can fine-tune an 8B LLM on a **free Colab T4** in under 30 minutes.

## How to run the notebook on Google Colab (free GPU)

> Never used Colab? It's a free Jupyter notebook in the cloud with a GPU — no install on your machine.
> See also [`SETUP.md` §6b](../../SETUP.md#6b-google-colab-cases-03) for the full walkthrough.

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.
2. **Upload the notebook:**
   - Click **File → Upload notebook**.
   - Pick `qlora_finetune.ipynb` from this folder on your computer.
   - (Alternative: if the repo is on GitHub, click **File → Open notebook → GitHub** tab, paste the repo URL, and select the notebook.)
3. **Switch to a GPU runtime:**
   - Click **Runtime → Change runtime type** → set **Hardware accelerator** to **T4 GPU** → **Save**.
   - You should see "T4" in the top-right status area.
4. **Run all cells:** Click **Runtime → Run all** (or Ctrl+F9). The first cell installs packages (~2 min), then everything runs top to bottom.
5. **Save your work:** Colab auto-saves to your Google Drive. You can also **File → Download → Download .ipynb**.

> **Session limits:** free Colab sessions disconnect after ~90 min idle or ~12 hours total. The default notebook (Qwen-0.5B, 1K examples) finishes in ~5 min, so this isn't a problem. For longer runs (Llama-3-8B), keep the tab active.

---

## Do it in this order (~40 min)

1. 🎬 Open **`animation.html`** in your browser (local file, no GPU needed):
   - Panel 1 — drag the bit-width slider; watch precision vs compression trade off.
   - Panel 2 — see where VRAM goes for full fine-tune vs LoRA vs QLoRA; slide model size.
   - Panel 3 — the QLoRA pipeline: load 4-bit → freeze → attach LoRA → train → merge.
2. 📓 **Upload `qlora_finetune.ipynb` to Google Colab** (see steps above):
   - Installs everything, loads a model in 4-bit, attaches LoRA, fine-tunes on a small dataset.
   - Default: ungated `Qwen/Qwen2.5-0.5B-Instruct` (runs in minutes, no license needed).
   - Swap to `meta-llama/Meta-Llama-3-8B` once comfortable (see `GET_LLAMA3.md` for access).
3. ✅ Do **`challenge.md`**.

## The one idea to remember
> You don't need an expensive GPU to fine-tune a large language model. Quantize the frozen
> base to 4-bit, bolt on a tiny LoRA adapter, and train just that — same quality, 1/4 the memory.

## Files
| File | What it is |
|------|------------|
| `animation.html` | Quantization, VRAM budgets, QLoRA pipeline — interactive. Each panel has a **📖 Read the theory** expander + a **💬 Ask tutor** button. |
| `THEORY.md` | Every term explained + curated links (QLoRA paper, bitsandbytes, PEFT). |
| `qlora_finetune.ipynb` | Colab-ready notebook: 4-bit load → LoRA → SFTTrainer → merge. |
| `GET_LLAMA3.md` | Step-by-step guide to download the gated Llama-3-8B. |
| `download_llama3.py` | Helper script for downloading / verifying Llama access. |
| `challenge.md` | Your turn. |

➡️ Next: Case 04 — Instruction SFT + evaluation (built when you're ready).
