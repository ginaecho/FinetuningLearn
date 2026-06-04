# 🧪 Case 07 — Your turn

Edit `capstone_autocar.py`.

## Warm-ups
1. **Memory size.** Change the memory subset size from 64 to 16 and then 128 per task.
   How does retrieval accuracy change? Is there a point of diminishing returns?
2. **Consolidation steps.** Change the LoRA replay training from 800 steps to 200 and then 2000.
   How does accuracy vs cost move on the trade-off plot?
3. **Nearest-neighbor k.** Try k ∈ {1, 3, 5, 10} for retrieval.
   Does more neighbors help or hurt?

## Build something ⭐
4. **Aging the memory.** Real experience DBs grow forever. Implement a simple "surprise filter":
   only add an experience to memory if its predicted loss (using current model) is above a threshold.
   This keeps memory small and focused on "hard" examples. Measure memory size vs accuracy.
5. **Two-tier LoRA.** Instead of one LoRA, keep a "fast LoRA" updated weekly on recent data
   and a "slow LoRA" updated quarterly on all data. At inference, add both: `y = base + slow + fast`.
   Does this beat the single-LoRA hybrid?

## Reflect
- You built a toy version of a production architecture. What is the biggest simplification
  in this code compared to a real self-driving system? (Think: safety certification, multimodal sensors,
  adversarial inputs, distribution shift detection.)

> Congratulations — you've completed the FinetuningLearn roadmap. 🎉
> Go back to any case, tune it deeper, or apply these ideas to your own model.
