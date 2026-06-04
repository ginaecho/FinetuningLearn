# Case 02 · LoRA from scratch

**Level:** Medium · **Runs on:** CPU, seconds · **Prereq:** Case 01

---

## What you'll walk away knowing
- **Why** low-rank updates work: the change a task needs lives in a few directions.
- How to **implement LoRA yourself** in ~30 lines (`LoRALinear`).
- That LoRA **matches full fine-tuning** while training a tiny fraction of params (you'll measure it).
- The two practical tricks: **B=0 init** (safe start) and **merge** (free inference).
- The first real idea for fighting **catastrophic forgetting**: freeze the base, keep one adapter per task.

## Do it in this order (≈25 min)
1. 🎬 Open **`animation.html`**:
   - Panel 1 — drag rank `r`, watch `B·A` reconstruct the true update; see the param ratio drop.
   - Panel 2 — the wiring: frozen `W` + trainable `A,B`, dots flowing through both paths.
   - Panel 3 — slide matrix size to LLM scale; see LoRA fall to ~0.4% of params.
2. 🧪 Run `python lora.py` — full vs LoRA, side by side.
3. 📓 Open **`lora_from_scratch.ipynb`** — implement it, plus a **rank sweep** to find the sweet spot, adapter saving, and merging.
4. ✅ Do **`challenge.md`**.

## The one idea to remember
> You almost never need to move *all* the weights. Freeze the giant pretrained matrix and learn a small,
> low-rank **correction**. Same result, a fraction of the memory and compute — and the original knowledge
> stays intact in the frozen weights.

## The math, briefly
A full update to a `d×k` weight is `d·k` numbers. LoRA replaces it with `B (d×r)` and `A (r×k)` =
`r·(d+k)` numbers. For `d=k=4096, r=8`: 16.7M → 65K, about **0.4%**. The `α/r` scale just keeps the
update's magnitude stable as you change `r`.

## Files
| File | What it is |
|------|------------|
| `animation.html` | Low-rank reconstruction, layer wiring, and LLM-scale savings. Each panel has a **📖 Read the theory** expander + a **💬 Ask tutor** button. |
| `THEORY.md` | Every term explained + curated links (LoRA/QLoRA papers, PEFT docs) to go deeper. |
| `lora.py` | `LoRALinear` + full-vs-LoRA comparison you can run. |
| `lora_from_scratch.ipynb` | Build it, rank sweep, save adapters, merge for inference. |
| `challenge.md` | Your turn. |

➡️ Next: Case 03 — QLoRA on a real Llama (built when you're ready).
