# Cases

Work through these in order — each builds on the last and they all lead to the self-driving capstone.

| # | Folder | What you build | Status |
|---|--------|----------------|--------|
| 01 | [`01_foundations`](./01_foundations/) | Pre-train vs fine-tune; warm start beats cold start | ✅ |
| 02 | [`02_lora_from_scratch`](./02_lora_from_scratch/) | Implement LoRA yourself; match full tuning at ~1% cost | ✅ |
| 03 | `03_qlora_real_llm` | QLoRA on a real Llama via `peft` + 4-bit | 🔜 |
| 04 | `04_instruction_sft` | Instruction/SFT tuning + real evaluation | 🔜 |
| 05 | `05_catastrophic_forgetting` | Make forgetting happen and **measure** it | 🔜 |
| 06 | `06_continual_learning` | Replay · EWC · per-task adapters · distillation | 🔜 |
| 07 | `07_capstone_autocar` | Streaming self-driving updates: cost vs accuracy vs forgetting | 🔜 |

Each folder: `README.md` (start here) · `animation.html` (intuition) · `*.ipynb` (code) · `challenge.md` (your turn).

Built so far: **01 and 02**. Finish their challenges, then say the word and I'll build the next case.
