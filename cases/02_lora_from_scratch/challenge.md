# 🧪 Case 02 — Your turn

Edit `lora_from_scratch.ipynb` / `lora.py`.

## Warm-ups
1. **Find the minimum rank.** From the rank sweep, what's the smallest `r` that still reaches a loss
   close to full fine-tuning? That's the cheapest adapter that "works" for this task.
2. **Alpha matters.** Fix `r=4` and try `alpha` ∈ {1, 4, 8, 32}. `scale = alpha/r` controls how strong the
   update is. Too small → underfits; too large → unstable. Find a good value.
3. **Where to inject?** Apply LoRA to only the **first** Linear, then only the **last**. Does the location
   matter for this task? (In transformers, LoRA is usually put on attention's q/v projections — a hint
   that *where* you adapt is a real design choice.)

## Build something ⭐
4. **A two-task adapter library.** Pretrain once on Task A. Then:
   - Train adapter **#1** on Task B (`phase=1.2`), save `A,B` to a dict.
   - Train a *fresh* adapter **#2** on Task C (`phase=-1.0`), save it.
   - Write a function `use_adapter(model, saved)` that loads a saved adapter into the frozen base and
     evaluates. Show the model does well on **whichever task's adapter is loaded**, using the *same*
     frozen base.
   - This is "modular fine-tuning" — your first concrete defense against forgetting. We scale it up in Case 06.

## Reflect
- LoRA keeps the base frozen, so it *can't* forget Task A's knowledge in the base. But the **adapter** is
  task-specific. If you instead kept training **one** adapter across A→B→C in sequence, would it forget?
  Write your hypothesis — Case 05 tests it.

> Done? Tell me you're ready and I'll build **Case 03 (QLoRA on a real Llama)** next.
