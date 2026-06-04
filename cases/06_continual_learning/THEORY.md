# Case 06 · Theory & deeper reading

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **Replay buffer** | A stored set of old samples mixed into new training. | Panel 1 slider. |
| **Replay ratio** | Fraction of a batch that comes from old data. | Panel 1 `%` slider. |
| **EWC** | Elastic Weight Consolidation — adds penalty for changing important weights. | Panel 2. |
| **Fisher information** | Diagonal approximation of how sensitive the loss is to each parameter. | EWC computation. |
| **λ (lambda)** | Penalty strength in EWC. Higher = more protection, harder to learn new tasks. | Panel 2 slider. |
| **Per-task adapter** | A separate LoRA trained and saved for each task. | Panel 3 boxes. |
| **Continual accuracy** | Average performance across all tasks seen so far. | Script output. |

---

## Why replay beats fancy methods

Surprising empirical finding: a well-designed replay buffer often outperforms
sophisticated regularization methods. Why?
- The optimizer actually **sees** old examples, so it stays in a parameter region
  that works for both old and new tasks.
- Regularization methods (EWC, SI, MAS) only **penalize** change — they constrain
  the search space, which can make learning new tasks harder.

The trade-off: replay needs storage. If you can't store old data (privacy, memory limits),
regularization is your fallback.

---

## Go deeper

**Core papers**
- Kirkpatrick et al. (2017), **Overcoming catastrophic forgetting in neural networks** (EWC): <https://www.nature.com/articles/nature24286>
- Rebuffi et al. (2017), *iCaRL: Incremental Classifier and Representation Learning*: <https://arxiv.org/abs/1611.07725>
- Hu et al. (2021), **LoRA**: <https://arxiv.org/abs/2106.09685>
- SuRe (ICLR 2026): surprise-driven prioritized replay + dual LoRA — see tutorial references.

**Parameter-efficient continual learning**
- O-LoRA, SAPT, CLoRA, PIECE — surveyed in the ACM Computing Surveys 2025 paper
  and the Emergent Mind article linked in `continual_learning_tutorial.md`.

**What comes next**
- Case 07 combines retrieval + replay into a hybrid production architecture.
