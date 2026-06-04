# Case 01 · Foundations — What *is* fine-tuning?

**Level:** Medium (start here) · **Runs on:** CPU, ~5 seconds · **Prereq:** [`../../SETUP.md`](../../SETUP.md)

---

## What you'll walk away knowing
- Training = rolling downhill on a loss surface; what the **learning rate** does.
- The difference between **pre-training** and **fine-tuning** — *same hill, different starting point*.
- Why a **warm start** beats **from-scratch** on the same compute budget (you'll measure it).
- That fine-tuning changes weights in a **structured** way → the seed of LoRA (Case 02).
- The first glimpse of **catastrophic forgetting** (the enemy in your capstone).

## Do it in this order (≈20 min)
1. 🎬 Open **`animation.html`** in a browser. Play all three panels:
   - Panel 1 — drag the learning rate; watch tiny→slow, huge→bouncing.
   - Panel 2 — click ①②③ to feel pre-train → new task → fine-tune.
   - Panel 3 — toggle Full vs LoRA to see how *few* weights LoRA touches.
2. 🧪 Run the smoke test: `python train.py` — see the warm start win in numbers.
3. 📓 Open **`foundations.ipynb`** — run top to bottom. It plots the fit, animates fine-tuning, and shows a heatmap of which weights moved.
4. ✅ Do **`challenge.md`**.

## The one idea to remember
> Fine-tuning is **cheap adaptation from a good starting point**. You take a model that already
> "knows a lot," and with a *small* learning rate and *few* steps, you specialize it — instead of
> paying the full cost of learning from random weights again.

## Files
| File | What it is |
|------|------------|
| `animation.html` | Interactive intuition (no install). |
| `train.py` | Minimal, readable proof: warm start beats cold start. |
| `foundations.ipynb` | Full walkthrough with plots + animation + weight-change heatmap. |
| `challenge.md` | Your turn — a small open-ended exercise. |

➡️ Next: [`../02_lora_from_scratch/`](../02_lora_from_scratch/)
