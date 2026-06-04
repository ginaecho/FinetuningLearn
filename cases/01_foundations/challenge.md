# 🧪 Case 01 — Your turn

Small, open-ended experiments. Edit `foundations.ipynb` (or `train.py`) and observe. There's no single
"right" number — the goal is to build a *feel* and start forming your own questions.

## Warm-ups
1. **Learning rate sweep.** In `train()`, fine-tune Task B with `lr` ∈ {1e-4, 1e-3, 1e-2, 1e-1}.
   Plot final loss vs lr. Which is too small (slow)? Which is too big (unstable)? This is the single
   most important fine-tuning knob.
2. **How far can the task drift?** Change Task B's `phase` to 0.3, 1.2, then 3.0. Does the warm start
   still beat from-scratch when the new task is *very* different? Where does the advantage shrink?
3. **Budget.** Give the from-scratch model 5× the steps. How many steps does it need to match the
   fine-tuned model's loss? That ratio *is* the cost saving of fine-tuning.

## The real one (sets up Case 05) ⭐
4. **Catch forgetting in the act.** After fine-tuning the model on Task B, go back and measure its loss
   on **Task A** (`xa, ya`). Did adapting to B make it *worse* at A?
   - Add: `with torch.no_grad(): print('Task A loss after B-finetune:', nn.MSELoss()(ft(xa), ya).item())`
   - Compare to the Task A loss *before* fine-tuning.
   - **Write down the number.** That gap is **catastrophic forgetting**. In your self-driving capstone,
     "Task A" might be "highway driving" and "Task B" a new "snow dataset" — forgetting A is dangerous.

## Reflect (jot 2–3 sentences)
- If new data keeps arriving forever, what would happen to Task A's performance over many rounds?
- Name one cheap thing you could try to *reduce* forgetting. (Keep your guess — we test ideas in Case 06.)

> Done? Open `../02_lora_from_scratch/`.
