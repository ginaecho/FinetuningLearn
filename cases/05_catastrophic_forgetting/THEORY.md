# Case 05 · Theory & deeper reading

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **Catastrophic forgetting** | A model loses previously learned skills when trained on new data. | Panel 1 green line. |
| **Forgetting score** | `loss_old_after − loss_old_before`. Positive = forgetting happened. | Script output. |
| **Weight drift** | How far parameters move away from values that worked for old tasks. | Panel 3 heatmap. |
| **Frozen base** | Original weights the optimizer is not allowed to change. | Panel 2/3; LoRA. |
| **Naive fine-tuning** | Training on new data with no protection for old knowledge. | Baseline in script. |

---

## Why forgetting is the default

Gradient descent minimizes the loss on the data it sees. If that data is only Task B,
the optimizer has no reason to keep parameters in a region that also works for Task A.
The parameters "drift" toward the B optimum, and A's performance collapses.

This happens even in tiny models. In billion-parameter LLMs, the effect is the same —
just harder to see because you need benchmarks to measure it.

---

## Go deeper

**Core papers on measuring & understanding forgetting**
- Goodfellow et al. (2013), *An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks*: <https://arxiv.org/abs/1312.6211>
- Kirkpatrick et al. (2017), **Overcoming catastrophic forgetting in neural networks** (EWC): <https://www.nature.com/articles/nature24286>
- McCloskey & Cohen (1989), *Catastrophic Interference in Connectionist Networks* — the original observation.

**What comes next**
- Case 06: replay buffers, EWC implementation, and per-task adapters.
- Case 07: hybrid architectures that combine retrieval + periodic replay.
