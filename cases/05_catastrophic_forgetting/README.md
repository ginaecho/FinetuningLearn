# Case 05 · Catastrophic Forgetting — see it happen

**Level:** Pro-ish · **Runs on:** CPU, seconds · **Prereq:** Case 02 (LoRA)

---

## What you'll walk away knowing
- **How to make forgetting happen** — and recognize it in loss curves.
- **How to measure it** quantitatively: forgetting score = loss_after − loss_before.
- That **naive fine-tuning destroys old knowledge** even with tiny models.
- That **freezing the base** (LoRA) protects the base weights, but training the **same adapter** on a new task still overwrites the adapter's old knowledge.
- The first concrete motivation for replay, regularization, or modular adapters (Case 06).

## Do it in this order (≈20 min)
1. 🎬 Open **`animation.html`**:
   - Panel 1 — drag the slider to watch Task A's loss rise as Task B training progresses.
   - Panel 2 — compare the "forgetting thermometer" across strategies.
   - Panel 3 — see which weights moved during naive fine-tuning.
2. 🧪 Run `python catastrophic_forgetting.py` — see the numbers: pre-train A, naive B, naive C, LoRA comparison.
3. ✅ Do **`challenge.md`**.

## The one idea to remember
> Forgetting is not a bug — it's the **default behavior** of gradient descent on new data.
> Unless you explicitly add a mechanism to preserve old knowledge, the optimizer will happily
> overwrite it. The first step to solving it is being able to **measure** it.

## Files
| File | What it is |
|------|------------|
| `animation.html` | Interactive forgetting curves, thermometer, and weight drift. |
| `THEORY.md` | Vocabulary + deeper reading on forgetting metrics and literature. |
| `catastrophic_forgetting.py` | Make forgetting happen and measure it. |
| `challenge.md` | Your turn. |

➡️ Next: Case 06 — Continual Learning Toolkit.
