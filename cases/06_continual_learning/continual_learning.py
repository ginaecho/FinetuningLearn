"""
Case 06 · Continual Learning Toolkit — Replay, EWC, Per-Task Adapters.
CPU, seconds, no extra installs.

Compare four strategies on a 3-task sequence:
  1. Naive      — train on each task in order, overwrite everything.
  2. Replay     — mix old samples into every training batch.
  3. EWC        — penalize changes to weights that matter for old tasks.
  4. Adapters   — keep a separate LoRA adapter for each task.

We measure: final loss on ALL tasks (average = "continual accuracy").
Lower average loss = less forgetting + better new-task learning.

Run: python cases/06_continual_learning/continual_learning.py
"""
import torch, torch.nn as nn, math, copy

torch.manual_seed(0)

def make_model():
    return nn.Sequential(nn.Linear(1,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))

def task_data(phase, n=256):
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)
    return x, torch.sin(x + phase)

def train(model, x, y, steps, lr):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
    return loss.item()

def evaluate(model, tasks):
    lossf = nn.MSELoss()
    out = {}
    with torch.no_grad():
        for name, (x, y) in tasks.items():
            out[name] = lossf(model(x), y).item()
    return out

def avg_loss(perfs):
    return sum(perfs.values()) / len(perfs)


class LoRALinear(nn.Module):
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

def extract_adapter_state(model):
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

def load_adapter_state(model, state):
    sd = model.state_dict()
    for n, p in state.items():
        sd[n] = p
    model.load_state_dict(sd)


def run_naive(base_model, tasks, steps=600, lr=1e-3):
    """Train on A, then B, then C. Overwrite everything."""
    model = copy.deepcopy(base_model)
    for name, (x, y) in tasks.items():
        train(model, x, y, steps, lr)
    return evaluate(model, tasks)


def run_replay(base_model, tasks, buffer_frac=0.15, steps=600, lr=1e-3):
    """Keep a buffer of old samples; mix into training."""
    model = copy.deepcopy(base_model)
    buffer = []  # list of (x, y) tensors
    for name, (x, y) in tasks.items():
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        lossf = nn.MSELoss()
        for _ in range(steps):
            opt.zero_grad()
            loss = lossf(model(x), y)
            if buffer:
                # Concatenate all buffer data and subsample to current batch size
                rx = torch.cat([bx for bx, _ in buffer], dim=0)
                ry = torch.cat([by for _, by in buffer], dim=0)
                idx = torch.randperm(rx.size(0))[:x.size(0)]
                loss = loss + lossf(model(rx[idx]), ry[idx])
            loss.backward()
            opt.step()
        # Store subset of current task in buffer
        idx = torch.randperm(x.size(0))[:int(x.size(0) * buffer_frac)]
        buffer.append((x[idx].clone(), y[idx].clone()))
    return evaluate(model, tasks)


def run_ewc(base_model, tasks_list, lam=2000, steps=600, lr=1e-3):
    """
    Simplified EWC: penalize drift away from old-task parameters.

    In full EWC, Fisher information weights each parameter by its importance.
    In this toy demo the model often overfits Task A perfectly, giving near-zero
    Fisher. We therefore use a *uniform* importance baseline (0.01) so the penalty
    is stable and educational, while still showing the EWC concept clearly.
    """
    model = copy.deepcopy(base_model)
    old_params = {}

    for task_idx, (name, (x, y)) in enumerate(tasks_list):
        if task_idx > 0:
            # Save old parameters before learning the new task
            old_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        lossf = nn.MSELoss()
        for _ in range(steps):
            opt.zero_grad()
            loss = lossf(model(x), y)
            if task_idx > 0:
                ewc = 0
                for n, p in model.named_parameters():
                    if p.requires_grad and n in old_params:
                        # Uniform importance (real EWC would weight by Fisher)
                        ewc += ((p - old_params[n]) ** 2).sum()
                loss = loss + (lam / 2.0) * ewc * 0.01
            loss.backward()
            opt.step()

    return evaluate(model, dict(tasks_list))


def run_adapters(base_model, tasks, steps=600, lr=1e-2):
    """Train a separate LoRA adapter per task. Needs task ID at inference time."""
    adapters = {}
    for name, (x, y) in tasks.items():
        model = inject_lora(copy.deepcopy(base_model), r=4, alpha=8)
        train(model, x, y, steps, lr)
        adapters[name] = extract_adapter_state(model)

    # Evaluate each task with its own adapter
    out = {}
    for eval_name, (ex, ey) in tasks.items():
        model = inject_lora(copy.deepcopy(base_model), r=4, alpha=8)
        load_adapter_state(model, adapters[eval_name])
        out[eval_name] = nn.MSELoss()(model(ex), ey).item()
    return out


def main():
    print("\n=== Case 06: Continual Learning Toolkit ===\n")
    tasks = [
        ("A (urban)",   task_data(0.0)),
        ("B (highway)", task_data(1.2)),
        ("C (rain)",    task_data(2.4)),
    ]
    task_dict = dict(tasks)

    # Pre-train base on A
    base = make_model()
    print("[0] Pre-train base on Task A ...")
    train(base, tasks[0][1][0], tasks[0][1][1], steps=1500, lr=1e-2)

    print("\n[1] NAIVE sequential training")
    p = run_naive(base, task_dict)
    print(f"    Final losses: { {k: f'{v:.4f}' for k, v in p.items()} }")
    print(f"    Average loss: {avg_loss(p):.4f}")

    print("\n[2] REPLAY buffer (15% old data kept)")
    p = run_replay(base, task_dict, buffer_frac=0.15)
    print(f"    Final losses: { {k: f'{v:.4f}' for k, v in p.items()} }")
    print(f"    Average loss: {avg_loss(p):.4f}")

    print("\n[3] EWC (lambda=2000)")
    p = run_ewc(base, tasks, lam=2000)
    print(f"    Final losses: { {k: f'{v:.4f}' for k, v in p.items()} }")
    print(f"    Average loss: {avg_loss(p):.4f}")

    print("\n[4] Per-task LoRA ADAPTERS")
    p = run_adapters(base, task_dict)
    print(f"    Final losses: { {k: f'{v:.4f}' for k, v in p.items()} }")
    print(f"    Average loss: {avg_loss(p):.4f}")

    print("\n=== Takeaway ===")
    print("Naive forgets. Replay and EWC reduce forgetting by protecting old knowledge.")
    print("Per-task adapters eliminate forgetting entirely, but you need to know")
    print("which task you're facing at inference time (or build a router).")
    print("In production, you often combine all three (Case 07).\n")


if __name__ == "__main__":
    main()
