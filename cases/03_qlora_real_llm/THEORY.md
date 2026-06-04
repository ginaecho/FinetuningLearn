# Case 03 · Theory & deeper reading

A map from each thing in `animation.html` to the concept behind it and where to read more.

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **Quantization** | Rounding each weight from 16-bit (65 536 levels) to fewer bits (e.g. 16 levels for 4-bit). Trades tiny accuracy loss for huge memory savings. | Panel 1 |
| **NF4 (Normal Float 4-bit)** | A 4-bit format where the 16 levels are spaced for normally-distributed data (which weights are). Better than naive uniform 4-bit. | Panel 1 |
| **Double quantization** | The quantization constants themselves are quantized a second time, saving another ~0.4 GB on a 7B model. | Panel 1 |
| **VRAM** | GPU memory. The bottleneck for LLM fine-tuning — everything (weights, optimizer state, gradients, activations) must fit. | Panel 2 |
| **Optimizer state** | Adam stores two extra numbers per trainable param (momentum + variance). Full fine-tune: 2× the trainable weight memory. QLoRA: only for the tiny LoRA params. | Panel 2 |
| **QLoRA** | Quantized LoRA. Freeze the base in 4-bit, train LoRA adapters in fp16/bf16. Same quality as full fine-tune at ~1/4 the memory. | Panel 3 |
| **BitsAndBytesConfig** | The HuggingFace object that tells `from_pretrained` to load the model in 4-bit NF4 with compute in bf16. | Panel 3 |
| **peft / get_peft_model** | Library that wraps a frozen model with LoRA (or other adapters). One function call. | Panel 3 |
| **SFTTrainer** | From HuggingFace `trl`. A Trainer subclass that handles tokenization, formatting, and LoRA training in one go. | Notebook |
| **Adapter** | The small set of LoRA weights you save after training. ~10-50 MB vs 16 GB for the full model. | Notebook |
| **Merge** | Fold the adapter back into the base weights: `W ← W + B·A·(α/r)`. One model, no adapter overhead at inference. | Notebook |

---

## The core idea

A 7-8B parameter model in fp16 takes ~16 GB of VRAM just for the weights — before you even start training. Training in full precision needs 3-4× that for optimizer states and gradients.

**QLoRA collapses this** by:
1. **Quantizing** the base model to 4-bit → ~4 GB instead of 16 GB.
2. **Freezing** those 4-bit weights (no optimizer state for them).
3. **Training only LoRA adapters** in full precision — a few million params, not billions.

The result: fine-tune an 8B model in **~6-10 GB VRAM** — comfortably on a free Colab T4 (16 GB).

---

## Go deeper

- **QLoRA paper** — Dettmers et al. 2023: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
  The paper that introduced NF4 + double quantization + paged optimizers. The key result: QLoRA matches full 16-bit fine-tuning quality on benchmarks.

- **LoRA paper** — Hu et al. 2021: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
  The original. Read §4 (which layers to adapt) and §7 (why low rank works).

- **bitsandbytes docs** — [huggingface.co/docs/bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index)
  The library that does the 4-bit quantization under the hood. GPU-only (NVIDIA).

- **PEFT docs** — [huggingface.co/docs/peft](https://huggingface.co/docs/peft/main/en/index)
  `get_peft_model`, `LoraConfig`, saving/loading adapters, merging.

- **TRL / SFTTrainer** — [huggingface.co/docs/trl](https://huggingface.co/docs/trl/main/en/sft_trainer)
  The high-level trainer we use in the notebook.

- **Tim Dettmers' blog** — "Which GPU for deep learning?" and "LLM.int8() and bitsandbytes"
  Practical GPU advice from the QLoRA author himself.

- **Sebastian Raschka's LoRA/QLoRA comparison** — [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
  Practical tips: rank, alpha, which layers, dataset size.

<!-- ⭐ Promote: the tutor's "Promote kept → Markdown" button exports your kept explanations as a
     Markdown file. Paste them below this line to make them a permanent part of the tutorial. -->
