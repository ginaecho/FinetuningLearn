# 🧪 Case 03 — Your turn

Edit the notebook / script and observe. No single right answer — build a feel and form your own questions.

## Warm-ups
1. Change `lora_r` from 8 to 2 and 32. Re-run training. How does final loss change? How does training time change? Plot loss curves for all three on one chart.
2. Swap the base model to a different size — try `Qwen/Qwen2.5-1.5B-Instruct` or `meta-llama/Llama-3.2-1B-Instruct`. Watch the VRAM usage (Colab: `!nvidia-smi`). How close to the 16 GB limit do you get?

## Build something ⭐
3. Fine-tune on a *different* dataset — pick something you care about (your own Q&A pairs, a subset of `tatsu-lab/alpaca`, code instructions, etc.). Format it as the notebook shows, train, and compare generations before vs after. Does the model actually learn your task? How many examples does it take?

## Reflect (jot 2-3 sentences)
- You trained LoRA adapters on a 4-bit base. What happens to the knowledge the base model already had? Did you notice any degradation on unrelated prompts?
- If you saved two adapters (one per dataset), could you load each one separately on the same base? What would that buy you? (Keep your answer — Case 05 tests it.)

> Done? Open `../04_instruction_sft/`.
