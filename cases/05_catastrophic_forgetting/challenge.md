# 🧪 Case 05 — Your turn

Edit `catastrophic_forgetting.py`.

## Warm-ups
1. **Learning rate matters.** Try lr ∈ {1e-4, 1e-3, 1e-2} for the B-training phase.
   Does a smaller lr reduce forgetting? Why or why not?
2. **Task similarity.** Change Task B's phase from 1.2 to 0.3 (much closer to A).
   Does the model forget less when tasks are similar?
3. **Count the drift.** Add code to compute the L2 norm of `W_after − W_before`
   for the first layer. Compare naive vs LoRA.

## Build something ⭐
4. **The forgetting curve.** After pre-training on A, fine-tune on B in small chunks
   (100 steps at a time). After each chunk, evaluate on A and B and append to a list.
   Plot `loss_A` and `loss_B` vs total steps. You should see the classic "forgetting curve."
   (Use matplotlib or just print the list.)

## Reflect
- You showed that LoRA's frozen base doesn't forget. But the adapter does.
  If you had 10 tasks and trained them all on the SAME adapter sequentially,
  what would happen to the earliest task's performance? Write your guess — Case 06 tests it.

> Done? Open `../06_continual_learning/`.
