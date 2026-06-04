# Case 00 · Tensors & Neural Network Basics

**Level:** Beginner · **Runs on:** CPU, seconds · **Prereq:** Python basics

---

## Why this case exists

You asked the right question: *"Can't an LLM agent just write all the training code for me?"*

**Yes — mostly.** But you'll hit a wall the moment something breaks if you can't read shapes, understand what the training loop is doing, or tell the agent what to change. This case gives you the **vocabulary and intuition** you need to co-pilot effectively with an AI agent on any neural network or fine-tuning task.

## What you'll walk away knowing
- What a **tensor** is: shapes, dimensions, `unsqueeze`, `squeeze`, `reshape`.
- What **nn.Sequential** builds: layers stacked, data flowing through.
- What each step in the **training loop** does: forward → loss → backward → update.
- **What YOU need to know** vs what the agent handles — the co-pilot cheat sheet.

## The co-pilot cheat sheet

| You must own this (intuition) | The agent handles this (syntax) |
|---|---|
| **Shapes** — "(batch, features)" and why things don't fit | Exact API calls: `model.train()`, `loss.backward()` |
| **Loss** — is it going down? Stuck? Exploding? | Writing the training loop boilerplate |
| **Hyperparameters** — what learning rate / batch size / rank *control* | Default values, schedulers, optimizers |
| **Data format** — what goes in, what shape the model expects | DataLoader, tokenization, preprocessing |
| **Evaluation** — "low loss" ≠ "model works" | Metric code, plotting |
| **Architecture vocabulary** — Linear, activation, MLP, attention | Layer initialization, gradient math |

> **Rule of thumb:** if you can *describe* what you want in these terms, the agent can *build* it.
> If you can't describe a shape mismatch, neither can the agent fix it for you.

## Do it in this order (~25 min)
1. 🎬 Open **`animation.html`**:
   - Panel 1 — tensor shapes: see 1D/2D/3D, click unsqueeze/squeeze, watch shapes change.
   - Panel 2 — inside `nn.Sequential`: data flows through layers, see the shape transform.
   - Panel 3 — the training loop: step through forward → loss → backward → update.
2. 📓 Open **`tensor_nn_basics.ipynb`** — hands-on: create tensors, build a model, train it step by step.
3. ✅ Do **`challenge.md`**.

## Files
| File | What it is |
|------|------------|
| `animation.html` | Tensor shapes, network layers, training loop — interactive. |
| `THEORY.md` | Every term explained + curated links. |
| `tensor_nn_basics.ipynb` | Hands-on notebook: tensors → model → train. |
| `challenge.md` | Your turn. |

➡️ Next: Case 01 — Foundations of fine-tuning.
