# Case 06 · Continual Learning Toolkit

**Level:** Pro · **Runs on:** CPU, seconds · **Prereq:** Case 05

---

## What you'll walk away knowing
- **Replay** — the strongest baseline. Mix old samples into new training.
- **EWC** — protect important weights with a Fisher-based penalty.
- **Per-task adapters** — zero forgetting, but need task ID at inference.
- How to **compare** methods using average loss across all tasks.
- When to use which: replay for general tasks, adapters for clearly separated domains, EWC when storage is limited.

## Do it in this order (≈25 min)
1. 🎬 Open **`animation.html`**:
   - Panel 1 — slide replay ratio, watch the batch composition change.
   - Panel 2 — toggle EWC, adjust lambda, see the spring pull weights back.
   - Panel 3 — click tasks to see how separate adapters stay independent.
2. 🧪 Run `python continual_learning.py` — four methods side by side.
3. ✅ Do **`challenge.md`**.

## The one idea to remember
> There is no single best method. Replay is the strongest and simplest baseline.
> EWC helps when you can't store old data. Adapters eliminate forgetting entirely
> but require knowing (or detecting) the task at inference time.
> Production systems often combine all three.

## Files
| File | What it is |
|------|------------|
| `animation.html` | Replay buffer, EWC spring, adapter library. |
| `THEORY.md` | Method details + curated papers. |
| `continual_learning.py` | Naive, replay, EWC, and per-task adapters compared. |
| `challenge.md` | Your turn. |

➡️ Next: Case 07 — Capstone.
