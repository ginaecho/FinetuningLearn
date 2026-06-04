# Case 02 · Theory & deeper reading

A map from each thing in `animation.html` to the concept behind it and where to read more.

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **ΔW (delta-W)** | How much a weight matrix changes during fine-tuning. | Panel 1 left heatmap. |
| **Rank** | How many independent directions a matrix actually uses. | Panel 1 slider `r`. |
| **Low-rank** | A matrix secretly built from a few patterns → cheap to store. | Panel 1 right heatmap. |
| **A, B (factors)** | Two skinny matrices whose product approximates ΔW: `ΔW ≈ B·A`. | Panel 1/2 green boxes. |
| **Frozen weights** | Parameters the optimizer is told never to change. | Panel 2 gray `W`. |
| **Forward pass** | `y = Wx + (B·A)x·(α/r)` — input → output. | Panel 2 flowing dots. |
| **B = 0 init** | Adapter starts as zero so the model begins == pretrained. | Panel 2 note. |
| **α (alpha) / scale** | `α/r` volume knob on the adapter's strength. | Panel 2/3. |
| **Merging** | Fold `B·A` back into `W` → zero extra inference cost. | Panel 2 note. |
| **Trainable fraction** | `2·d·r / d²` — tiny for big models. | Panel 3 green bar. |
| **Per-task adapters** | Keep one small adapter per task; swap as needed. | Panel 2/3; Case 06. |

---

## Why low-rank works (the one insight)

Fine-tuning a model means adding a change `ΔW` to each weight matrix. The surprising empirical finding
behind LoRA: the `ΔW` a task needs has a **low intrinsic rank** — it can be reconstructed from just a few
directions. So instead of learning the full `d×d` matrix (`d²` numbers), you learn `B (d×r)` and
`A (r×d)` (`2dr` numbers). With `d=4096, r=8` that's 65K vs 16.7M — about **0.4%**.

Two practical tricks make it safe and free:
- **`B = 0` at init** → `B·A = 0` → training starts exactly at the pretrained model (no jump).
- **Merge after training** → `W ← W + B·A·(α/r)` → inference is a normal layer, no overhead.

And because the base `W` is **frozen**, the original knowledge can't be overwritten — which is why
adapters are a natural defense against **catastrophic forgetting** (Case 06).

---

## Go deeper

**LoRA & PEFT (read these)**
- Hu et al. (2021), **LoRA: Low-Rank Adaptation of Large Language Models** — the original paper, very
  approachable: <https://arxiv.org/abs/2106.09685>
- Hugging Face **PEFT** docs (the library you'll use in Case 03):
  <https://huggingface.co/docs/peft/index> · conceptual guide:
  <https://huggingface.co/docs/peft/conceptual_guides/lora>
- Sebastian Raschka, *Practical tips for finetuning LLMs using LoRA* (what r/alpha to pick, where to apply):
  <https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms>

**The "why is it low-rank?" backstory**
- Aghajanyan et al. (2020), *Intrinsic Dimensionality… of Fine-Tuning* — evidence that adaptation lives in
  a low-dim subspace: <https://arxiv.org/abs/2012.13255>
- Li et al. (2018), *Measuring the Intrinsic Dimension of Objective Landscapes*:
  <https://arxiv.org/abs/1804.08838>

**What comes next**
- Dettmers et al. (2023), **QLoRA: Efficient Finetuning of Quantized LLMs** — LoRA on a 4-bit model,
  the subject of Case 03: <https://arxiv.org/abs/2305.14314>

**Linear-algebra refresher (rank / low-rank approximation)**
- 3Blue1Brown, *Essence of Linear Algebra* (rank, span): <https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab>
- Low-rank approximation via SVD (Wikipedia, the math under "best rank-r"):
  <https://en.wikipedia.org/wiki/Low-rank_approximation>

> Next: **Case 03** swaps this hand-written LoRA for `peft` + 4-bit QLoRA on a real Llama.
