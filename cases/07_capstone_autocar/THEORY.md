# Case 07 · Theory & deeper reading

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **Hybrid memory** | Combining parametric (weights) and non-parametric (retrieval) memory. | Panel 1 flow diagram. |
| **Consolidation** | Periodic retraining that compresses experiences into weights. | Slow path in diagram. |
| **Experience memory** | External database of raw or embedded experiences. | Vector DB box. |
| **Data velocity** | How fast new data arrives. Determines replay frequency. | Panel 3 slider. |
| **Parametric memory** | Knowledge stored in model weights. Fast, compact, hard to update. | Tutorial §5. |
| **Non-parametric memory** | Knowledge stored in a database. Infinite, easy to update, search cost. | Tutorial §5. |

---

## The production architecture

For safety-critical embodied systems (self-driving, robotics), the recommended stack:

```
Foundation Model (frozen)
       +
Stable LoRA (core skills, replay-updated quarterly)
       +
Experience Memory (vector DB, real-time)
       +
Periodic Retraining (replay consolidation)
```

| Layer | Responsibility | Update Frequency |
|-------|---------------|------------------|
| **Foundation Model** | General world knowledge, language | Frozen |
| **Stable LoRA** | Core driving skills | Quarterly |
| **Experience Memory** | Rare accidents, local quirks, construction | Real-time |
| **Periodic Retraining** | Compress recurring patterns into weights | Monthly/Quarterly |

---

## Go deeper

**Papers on hybrid & memory systems**
- SuRe (ICLR 2026): surprise-driven prioritized replay + dual fast/slow LoRA with EMA merging.
  <https://openreview.net/pdf?id=IgZWU75BLL>
- I-LoRA: iterative merging of routing-tuned adapters.
  <https://openreview.net/pdf/79ecd09f7e097620b8d4d310dd0dc3082fa5ecdd.pdf>
- LLM Agent Memory survey (2026): RAG vs agent memory, hierarchies, retrieval vs parametric knowledge.
  <https://www.preprints.org/manuscript/202603.0359/v1>
- Real-Time Learning in LLMs 2026 (practical guide):
  <https://futureagi.com/blog/real-time-learning-in-large-language-models-llms/>

**Neuroscience connection**
- McClelland, McNaughton & O'Reilly (1995), *Why there are complementary learning systems*
  in the hippocampus and neocortex: <https://psych.stanford.edu/~jlm/papers/McClelland95.pdf>

> This case concludes the FinetuningLearn roadmap. You now have the vocabulary,
> intuition, and code to build continual learning systems for real models.
