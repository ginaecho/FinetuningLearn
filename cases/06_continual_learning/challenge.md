# 🧪 Case 06 — Your turn

Edit `continual_learning.py`.

## Warm-ups
1. **Replay ratio sweep.** Try buffer_frac ∈ {0.05, 0.15, 0.30, 0.50}.
   Plot average final loss vs replay ratio. Where is the elbow?
2. **EWC lambda sweep.** Try lam ∈ {100, 500, 1000, 5000}.
   Too low = no protection. Too high = can't learn B. Find the sweet spot.
3. **Adapter rank.** Try r ∈ {2, 4, 8, 16} for per-task adapters.
   Does higher rank help when tasks are very different (phases 0.0, 2.0, 4.0)?

## Build something ⭐
4. **A simple router.** Instead of needing task ID at test time, build a tiny
   k-nearest-neighbor router: given a test input, find which task's training data
   it is closest to, then load that task's adapter. Evaluate accuracy with the
   router vs with oracle task ID. How much does the router cost you?

## Reflect
- If you had unlimited storage and compute, would you still use adapters instead
  of a giant replay buffer? Why or why not?

> Done? Open `../07_capstone_autocar/`.
