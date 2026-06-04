"""
Case 05 · Catastrophic Forgetting — see it happen and measure it.
CPU, seconds, no extra installs.

Concept:
  Pre-train a small MLP on Task A.
  Fine-tune it on Task B (naive: train all weights).
  Measure: Task A's loss rises = forgetting.
  Compare with LoRA (base frozen) to see what stays safe and what doesn't.

Run: python cases/05_catastrophic_forgetting/catastrophic_forgetting.py
"""
import torch, torch.nn as nn, math, copy

torch.manual_seed(0)

def make_model():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1)
    )

def task_data(phase, n=256):
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)
    return x, torch.sin(x + phase)

def train(model, x, y, steps, lr):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()

def evaluate(model, tasks):
    lossf = nn.MSELoss()
    out = {}
    with torch.no_grad():
        for name, (x, y) in tasks.items():
            out[name] = lossf(model(x), y).item()
    return out

def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    """Minimal self-contained LoRA wrapper so this file needs no imports from other cases."""
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

def inject_lora(model, r=4, alpha=8):
    new = []
    for m in model:
        new.append(LoRALinear(m, r, alpha) if isinstance(m, nn.Linear) else m)
    return nn.Sequential(*new)


def main():
    print("\n=== Case 05: Catastrophic Forgetting ===\n")

    tasks = {
        "A (urban)":   task_data(0.0),
        "B (highway)": task_data(1.2),
        "C (rain)":    task_data(2.4),
    }

    # 1) Pre-train on A
    model = make_model()
    print("[1] Pre-train on Task A ...")
    train(model, *tasks["A (urban)"], steps=1500, lr=1e-2)
    base_perf = evaluate(model, tasks)
    print("    Performance after pre-training:")
    for k, v in base_perf.items():
        print(f"      {k}: loss = {v:.5f}")

    # 2) Naive: train on B, overwriting all weights
    naive = copy.deepcopy(model)
    print("\n[2] NAIVE fine-tune on Task B (all weights trainable)...")
    train(naive, *tasks["B (highway)"], steps=800, lr=1e-3)
    naive_perf = evaluate(naive, tasks)
    print("    Performance after fine-tuning on B:")
    for k, v in naive_perf.items():
        print(f"      {k}: loss = {v:.5f}")

    forget = naive_perf["A (urban)"] - base_perf["A (urban)"]
    print(f"\n    --> FORGETTING on A: +{forget:.5f}  (positive = A got worse)")

    # 3) Continue naive on C
    print("\n[3] Continue naive training on Task C...")
    train(naive, *tasks["C (rain)"], steps=800, lr=1e-3)
    naive_c = evaluate(naive, tasks)
    print("    Performance after fine-tuning on C:")
    for k, v in naive_c.items():
        print(f"      {k}: loss = {v:.5f}")

    f_a = naive_c["A (urban)"]   - base_perf["A (urban)"]
    f_b = naive_c["B (highway)"] - naive_perf["B (highway)"]
    print(f"\n    --> FORGETTING on A: +{f_a:.5f}")
    print(f"    --> FORGETTING on B: +{f_b:.5f}")

    # 4) LoRA: freeze base, train adapter on B, then continue on C
    lora = inject_lora(copy.deepcopy(model), r=4, alpha=8)
    print(f"\n[4] LoRA fine-tune on B (trainable params: {count_trainable(lora)})...")
    train(lora, *tasks["B (highway)"], steps=800, lr=1e-2)
    lora_b = evaluate(lora, tasks)
    print("    Performance after B (LoRA):")
    for k, v in lora_b.items():
        print(f"      {k}: loss = {v:.5f}")

    # Train SAME adapter on C — it will overwrite B's adapter!
    print("\n[5] Continue training the SAME LoRA adapter on C...")
    train(lora, *tasks["C (rain)"], steps=800, lr=1e-2)
    lora_c = evaluate(lora, tasks)
    print("    Performance after C (same adapter):")
    for k, v in lora_c.items():
        print(f"      {k}: loss = {v:.5f}")

    print(f"\n    --> FORGETTING on A (base frozen):  +{lora_c['A (urban)'] - base_perf['A (urban)']:.5f}")
    print(f"    --> FORGETTING on B (adapter lost): +{lora_c['B (highway)'] - lora_b['B (highway)']:.5f}")

    print("\n=== Takeaway ===")
    print("Naive fine-tuning destroys old knowledge. Even LoRA forgets if you keep")
    print("training the SAME adapter — the frozen base stays safe, but the adapter is overwritten.")
    print("Solutions: replay buffers, regularization, or SEPARATE adapters per task (Case 06).\n")


if __name__ == "__main__":
    main()
