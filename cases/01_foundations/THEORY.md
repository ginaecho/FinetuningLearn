# Case 01 · Theory & deeper reading

A map from each thing in `animation.html` to the concept behind it and where to read more.
Read this *after* playing with the animation — it'll make far more sense.

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **Weights / parameters** | The numbers inside a model that get learned. | Panel 1 ball position; Panel 3 squares. |
| **Loss** | A single number: how wrong the model is right now (lower = better). | Panel 1/2 curve height. |
| **Loss landscape** | Loss plotted over all possible weights — the surface we descend. | Panel 1/2 curve. |
| **Gradient** | The slope of the loss; points uphill. We step the opposite way. | Panel 1 orange arrow. |
| **Gradient descent** | The update rule `w ← w − lr·gradient`, repeated. | Panel 1 ▶ Train. |
| **Learning rate (lr)** | Step size. Too small = slow; too big = unstable/overshoot. | Panel 1 slider. |
| **Pre-training** | Train from *random* weights on a big general task. Done once, expensive. | Panel 2 ① |
| **Fine-tuning** | Continue training a *pre-trained* model on a new, narrower task. Cheap, few steps, small lr. | Panel 2 ③ |
| **Warm vs cold start** | Starting from good weights vs random weights. | Panel 2 ① vs ③ |
| **Full fine-tuning** | Update *all* weights. | Panel 3 |
| **LoRA / PEFT** | Update a tiny add-on instead of all weights. | Panel 3; full case 02 |
| **Catastrophic forgetting** | Adapting to a new task makes the model worse at old tasks. | Panel 2 footnote; Case 05 |

---

## The core idea (why fine-tuning is cheap)

Pre-training pays a huge one-time cost to land the model in a good region of the loss landscape.
When a new task arrives, its best solution is usually **nearby** — so you only need a few small steps to
reach it, instead of climbing all the way down from random again. That "nearby" assumption is what
fine-tuning exploits, and it's why a **small learning rate** and **few steps** are the norm.

The flip side, visible in Panel 2: moving to the new task's minimum can move you *away* from the old
task's minimum → **forgetting**. Cases 05–06 are entirely about this trade-off, which is the heart of your
self-driving capstone.

---

## Go deeper (curated, beginner-friendly first)

**Foundations**
- 3Blue1Brown, *Gradient descent, how neural networks learn* (video) — the best visual intuition for
  loss + gradients: <https://www.youtube.com/watch?v=IHZwWFHWa-w>
- Andrej Karpathy, *The spelled-out intro to neural networks and backprop* (micrograd) — build gradient
  descent from scratch: <https://www.youtube.com/watch?v=VMj-3S1tku0>
- Google ML Crash Course, *Reducing Loss / Learning rate*:
  <https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate>

**Transfer learning & fine-tuning (the warm-start idea)**
- Stanford CS231n, *Transfer Learning* notes: <https://cs231n.github.io/transfer-learning/>
- Hugging Face, *Fine-tune a pretrained model*: <https://huggingface.co/docs/transformers/training>
- Howard & Ruder, *ULMFiT* (2018) — early, readable case for fine-tuning + discriminative/small LRs:
  <https://arxiv.org/abs/1801.06146>

**Catastrophic forgetting (the villain you'll meet)**
- French (1999), *Catastrophic forgetting in connectionist networks* (survey, very readable):
  <https://www.sciencedirect.com/science/article/abs/pii/S1364661399012942>
- Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting (EWC)* — we implement ideas like this in
  Case 06: <https://arxiv.org/abs/1612.00796>

> Next concept-wise: **Case 02** explains *how* Panel 3's "LoRA" actually works.
