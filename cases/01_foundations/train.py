"""
Case 01 · Foundations — smoke test + the core idea, in plain PyTorch (CPU-friendly).

Run:  python cases/01_foundations/train.py

It demonstrates the *entire* point of fine-tuning with a tiny model you can read end to end:

  1. PRE-TRAIN a small MLP on "Task A" (a sine wave).
  2. A new "Task B" arrives (a shifted sine wave).
  3. FINE-TUNE the *same* weights on Task B with a small learning rate -> converges fast.
  4. Compare against training a fresh model from scratch on Task B -> needs many more steps.

No GPU, no downloads. ~5 seconds.
"""
import torch, torch.nn as nn, math, time

torch.manual_seed(0)

# ----- a tiny model: 1 -> 64 -> 64 -> 1 -----
def make_model():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )

def task_data(phase, n=256):
    """A 1-D regression task: y = sin(x + phase). 'phase' defines the task."""
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)
    y = torch.sin(x + phase)
    return x, y

def train(model, x, y, steps, lr, label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    print(f"  {label:<34} final loss = {loss.item():.5f}   ({steps} steps, lr={lr}, {time.time()-t0:.2f}s)")
    return loss.item()

def main():
    print("\n=== Case 01: pre-train -> new task -> fine-tune ===\n")

    # 1) PRE-TRAIN on Task A
    xa, ya = task_data(phase=0.0)
    model = make_model()
    print("[1] Pre-training on Task A  (sin(x)):")
    train(model, xa, ya, steps=1500, lr=1e-2, label="pretrained model on Task A")

    # 2) NEW TASK B arrives (shifted) -> fine-tune the SAME model, small LR, few steps
    xb, yb = task_data(phase=1.2)
    print("\n[2] New Task B arrives (sin(x + 1.2)). Fine-tune the pretrained model:")
    ft_loss = train(model, xb, yb, steps=300, lr=1e-3, label="FINE-TUNED (warm start)")

    # 3) BASELINE: a fresh model trained from scratch on Task B for the same few steps
    fresh = make_model()
    print("\n[3] Baseline: a FRESH model from scratch on Task B, same 300 steps:")
    scratch_loss = train(fresh, xb, yb, steps=300, lr=1e-3, label="FROM SCRATCH (cold start)")

    print("\n=== Takeaway ===")
    better = "LOWER (better)" if ft_loss < scratch_loss else "higher"
    print(f"  Fine-tuned loss is {better}: {ft_loss:.5f}  vs  from-scratch {scratch_loss:.5f}")
    print("  Same compute budget -> the warm start (fine-tuning) wins. That's the whole game.")
    print("  Next: cases/01_foundations/foundations.ipynb to SEE it, then Case 02 (LoRA).\n")

if __name__ == "__main__":
    main()
