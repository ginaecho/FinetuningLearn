# Getting `meta-llama/Meta-Llama-3-8B` onto your machine

This is a **gated** model (~16 GB). You'll: accept the license → make a token → install tools → log in →
verify access cheaply → download. Do this on the machine where you'll fine-tune (not a throwaway cloud box).

> ⚠️ Don't run this inside an ephemeral cloud session — the 16 GB download is wiped when the container is
> reclaimed. Do it on your own computer (or a persistent cloud disk / Colab with a mounted Drive).

---

## 0. Reality check — can your machine run an 8B model?

| What you want to do | Needs (approx) |
|---|---|
| **Download** the weights | ~16 GB free disk |
| **Inference**, full precision (bf16/fp16) | ~16 GB GPU VRAM |
| **Inference**, 4-bit | ~6 GB VRAM |
| **QLoRA fine-tune** (4-bit base + LoRA) ← *what Case 03 does* | **~6–10 GB VRAM** |
| **LoRA** on 16-bit base | ~18–24 GB VRAM |
| **Full** fine-tune | multiple 40–80 GB GPUs (not a laptop) |
| **CPU only** | ~32 GB RAM, *very* slow — fine for a tiny test, painful for real work |

**No big GPU?** Two good options:
- Use **Google Colab** (free T4 = 16 GB) for Case 03's QLoRA notebook.
- Or learn on a **smaller sibling** first: `meta-llama/Llama-3.2-1B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`
  (same workflow, far lighter). The ungated `Qwen/Qwen2.5-0.5B-Instruct` needs no license at all.

> **Base vs Instruct:** `Meta-Llama-3-8B` is the **base** (text completion) model. For chat-style fine-tuning
> you usually want `meta-llama/Meta-Llama-3-8B-Instruct`. Newer is `meta-llama/Llama-3.1-8B-Instruct`. The
> install steps are identical — just swap the name.

---

## 1. Accept the license (only you can do this)

1. Log in / sign up at <https://huggingface.co>.
2. Open <https://huggingface.co/meta-llama/Meta-Llama-3-8B> and click **“Agree and access repository.”**
3. Fill Meta’s short form. Use a **real name/email** that matches your account — mismatches get rejected.
4. Wait for approval (often minutes, sometimes hours). You’ll see **“You have been granted access”** on the
   model page and get an email.

You can't download until the page shows you have access.

---

## 2. Create an access token

1. Go to <https://huggingface.co/settings/tokens> → **New token**.
2. Type: **Read** is enough. Name it e.g. `llama-download`. Copy it (starts with `hf_...`).
3. Keep it secret — it's like a password. Don't paste it into code or commit it.

---

## 3. Install the tools

```bash
# in your project venv (see ../../SETUP.md)
pip install -U "huggingface_hub[cli]" transformers accelerate
# faster downloads (optional but recommended for 16 GB):
pip install -U hf_transfer
# for QLoRA later (NVIDIA GPU only):
pip install -U peft bitsandbytes
```

---

## 4. Log in

```bash
huggingface-cli login        # paste your hf_... token when prompted
# (newer versions also accept:  hf auth login )
```
This caches the token in `~/.cache/huggingface` so you don't paste it again.

---

## 5. Verify access **without** the 16 GB download (do this first!)

```bash
python download_llama3.py --check-only
```
This fetches only the tokenizer + config (a few MB). If it prints **“ACCESS OK”**, your license + token work.
If you get a 401/403, see Troubleshooting below.

---

## 6. Download the full weights

```bash
# easiest: cache it (transformers will find it automatically later)
python download_llama3.py

# or pick an explicit folder:
python download_llama3.py --local-dir ./models/Meta-Llama-3-8B

# or pure CLI, no script:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Meta-Llama-3-8B
```
Downloads land in `~/.cache/huggingface/hub` unless you pass `--local-dir`.

---

## 7. Load it the cheap (4-bit) way — preview of Case 03

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
print("loaded in 4-bit on:", model.device)
```
That `BitsAndBytesConfig` is the **Q** in QLoRA — it squeezes the 16 GB model into ~6 GB so it fits a small
GPU. We attach LoRA adapters on top of this in Case 03.

---

## Troubleshooting

- **401 / `Repository not found` / 403 gated** → your access isn't active yet, or the token isn't loaded.
  Re-check the model page shows access, and run `huggingface-cli whoami`.
- **`OSError: ... is not a local folder and is not a valid model identifier`** → usually the gated/token issue above.
- **Out of memory loading** → use the 4-bit snippet in §7, or switch to `Llama-3.2-1B-Instruct`.
- **No NVIDIA GPU / `bitsandbytes` won't install** → it's GPU-only; do QLoRA on Colab, or use the tiny models on CPU.
- **Slow download** → `export HF_HUB_ENABLE_HF_TRANSFER=1` (after `pip install hf_transfer`).
- **Disk full** → 8B needs ~16 GB; check with `df -h ~/.cache/huggingface`.

Done? You're ready for **Case 03 (QLoRA)** — tell me and I'll build the notebook against this model.
