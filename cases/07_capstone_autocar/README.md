# Case 07 · Capstone: Streaming Self-Driving Updates

**Level:** Pro · **Runs on:** CPU, seconds · **Prereq:** Cases 05–06

---

## What you'll walk away knowing
- How to **architect a hybrid memory system** for streaming updates.
- **Retrieval** handles day-to-day surprises cheaply; **periodic replay** consolidates recurring patterns into weights.
- How to **trade off cost, accuracy, and forgetting** in a realistic scenario.
- That the "best" solution is rarely pure — it's a **balance** tuned to your data velocity and budget.

## Do it in this order (≈30 min)
1. 🎬 Open **`animation.html`**:
   - Panel 1 — watch the hybrid architecture flow: experiences → DB → retrieval + replay → LoRA.
   - Panel 2 — click to compare replay vs retrieval side by side.
   - Panel 3 — drag replay frequency to move the hybrid dot in cost-accuracy space.
2. 🧪 Run `python capstone_autocar.py` — three strategies compared on 4 driving conditions.
3. ✅ Do **`challenge.md`**.

## The one idea to remember
> Don't choose between retrieval and replay — build a **hybrid**.
> Retrieval for immediate, cheap, zero-forgetting updates.
> Periodic replay to compress recurring experiences into fast, generalizable weights.
> This mirrors human memory: hippocampus (episodic) → cortex (consolidated).

## Files
| File | What it is |
|------|------------|
| `animation.html` | Hybrid architecture, replay vs retrieval, trade-off space. |
| `THEORY.md` | Production architecture deep-dive + references. |
| `capstone_autocar.py` | Naive, retrieval-only, and hybrid strategies on a streaming task. |
| `challenge.md` | Your turn. |

➡️ You're at the end of the roadmap. Go back and tune any case, or apply these ideas to your own model.
