"""
Case 02 · LoRA from scratch — the whole idea in one readable file (CPU, no extra installs).

Run:  python cases/02_lora_from_scratch/lora.py

LoRA (Low-Rank Adaptation) in one sentence:
    Freeze the pretrained weight W. Don't change it. Instead learn a tiny low-rank
    side-path  ΔW = (B @ A) * (alpha/r)  and use  W_eff = W + ΔW.
    Only A (r×in) and B (out×r) are trained -> a few % of the parameters.

This file:
  1. Builds a LoRALinear that wraps a frozen nn.Linear.
  2. Pre-trains a small MLP on Task A.
  3. Injects LoRA and fine-tunes ONLY the adapters on Task B.
  4. Compares trainable-parameter count and final loss vs full fine-tuning.
"""
import torch, torch.nn as nn, math, copy

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# The core: a Linear layer with a frozen weight + a trainable low-rank adapter.
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base                       # the PRETRAINED layer
        for p in self.base.parameters():       # FREEZE it
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.r, self.scale = r, alpha / r
        # Low-rank factors. A starts random-small, B starts at zero -> ΔW = 0 at init,
        # so the fine-tuned model begins exactly equal to the pretrained one.
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def make_model():
    return nn.Sequential(nn.Linear(1,64), nn.Tanh(),
                         nn.Linear(64,64), nn.Tanh(),
                         nn.Linear(64,1))

def inject_lora(model, r=4, alpha=8):
    """Replace every nn.Linear in a Sequential with a LoRALinear wrapper."""
    new = []
    for m in model:
        new.append(LoRALinear(m, r, alpha) if isinstance(m, nn.Linear) else m)
    return nn.Sequential(*new)

def task_data(phase, n=256):
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)
    return x, torch.sin(x + phase)

def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train(model, x, y, steps, lr):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
    return loss.item()


def main():
    print("\n=== Case 02: LoRA from scratch ===\n")

    # 1) Pre-train on Task A
    xa, ya = task_data(0.0)
    model = make_model()
    train(model, xa, ya, steps=1500, lr=1e-2)
    print(f"[1] Pretrained on Task A.  total params = {sum(p.numel() for p in model.parameters())}")

    # 2) New Task B. Approach A = FULL fine-tune (train everything).
    xb, yb = task_data(1.2)
    full = copy.deepcopy(model)
    full_loss = train(full, xb, yb, steps=400, lr=1e-3)
    print(f"\n[2] FULL fine-tune on Task B:")
    print(f"    trainable params = {count_trainable(full):5d}  ->  loss {full_loss:.5f}")

    # 3) Approach B = LoRA. Freeze W, train only adapters.
    lora = inject_lora(copy.deepcopy(model), r=4, alpha=8)
    lora_loss = train(lora, xb, yb, steps=400, lr=1e-2)   # adapters like a slightly higher lr
    print(f"\n[3] LoRA fine-tune on Task B (r=4):")
    print(f"    trainable params = {count_trainable(lora):5d}  ->  loss {lora_loss:.5f}")

    pct = 100 * count_trainable(lora) / count_trainable(full)
    print("\n=== Takeaway ===")
    print(f"    LoRA trained only {pct:.1f}% of the weights and reached loss {lora_loss:.5f}")
    print(f"    vs full fine-tune {full_loss:.5f}. Near-equal quality, a fraction of the cost.")
    print("    On a real LLM that 'fraction' is what lets you tune a 7B model on one GPU (Case 03).\n")

    # 4) Bonus: the frozen base is untouched -> you can keep MANY adapters, one per task.
    #    This is a first hint at how to fight catastrophic forgetting (Case 06).
    with torch.no_grad():
        base_unchanged = torch.equal(lora[0].base.weight, model[0].weight)
    print(f"    Frozen base weight unchanged by LoRA? {base_unchanged}  "
          f"(=> swap adapters per task without overwriting shared knowledge)\n")

if __name__ == "__main__":
    main()
