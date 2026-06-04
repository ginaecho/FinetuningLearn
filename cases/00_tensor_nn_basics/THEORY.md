# Case 00 · Theory & deeper reading

A map from each thing in `animation.html` to the concept behind it and where to read more.

---

## The vocabulary, in order

| Term | Plain meaning | Where it shows up |
|------|---------------|-------------------|
| **Tensor** | A multi-dimensional array of numbers. A scalar is 0D, a vector is 1D, a matrix is 2D, and so on. PyTorch's basic data container. | Panel 1 |
| **Shape** | The size of each dimension. `(256, 1)` means 256 rows, 1 column. Shape mismatches are the #1 error in deep learning. | Panel 1 |
| **unsqueeze(dim)** | Insert a new dimension of size 1 at position `dim`. Turns `(256,)` into `(1, 256)` or `(256, 1)` depending on `dim`. | Panel 1 |
| **squeeze()** | Remove all dimensions of size 1. Turns `(1, 256, 1)` back into `(256,)`. | Panel 1 |
| **Batch dimension** | The first dimension, usually. A batch of 32 images of 28x28 pixels has shape `(32, 28, 28)`. Neural network layers expect a batch dimension. That's why you `unsqueeze`. | Panel 1 |
| **nn.Linear(in, out)** | A layer that does `y = Wx + b`. Takes `in` features, produces `out` features. The `W` matrix has shape `(out, in)`. | Panel 2 |
| **Activation function** | A non-linear function applied after a linear layer. Without it, stacking linear layers is just one big linear layer (pointless). Common: Tanh (squishes to [-1,1]), ReLU (zeroes negatives). | Panel 2 |
| **nn.Sequential** | Stack layers in order. Data flows through them top to bottom. Good for simple architectures. | Panel 2 |
| **Forward pass** | Run input through the network to get an output (prediction). | Panel 3 |
| **Loss function** | Measures how wrong the prediction is. MSE = average of (prediction - target)^2. Lower is better. | Panel 3 |
| **Backward pass** | Compute gradients: how much each weight contributed to the loss. PyTorch does this automatically with `loss.backward()`. | Panel 3 |
| **Optimizer** | Updates weights using gradients. Adam is the default choice — it adapts the learning rate per-parameter. `optimizer.step()` does the update. | Panel 3 |
| **Learning rate** | How big each weight update is. Too high → overshoots. Too low → learns too slowly. Typical: 1e-3 for training, 1e-4 to 2e-4 for fine-tuning. | Panel 3 |
| **Epoch** | One pass through the entire dataset. | Panel 3 |
| **Gradient** | The direction and magnitude to change a weight to reduce loss. Computed by backpropagation. | Panel 3 |

---

## The core idea

A neural network is just **matrix multiplications + non-linear functions**, trained by a loop that says "how wrong was I? adjust weights slightly in the right direction."

Everything in deep learning — transformers, LLMs, fine-tuning — is built on this loop. The architectures get fancier, but the training cycle stays the same: **forward → loss → backward → update**.

The reason you need to understand **shapes** is that every layer expects a specific input shape and produces a specific output shape. When these don't match, you get a `RuntimeError`. This is the most common bug, and the one an AI agent needs *your* help to diagnose — because you know what the data looks like.

---

## What you need to know to work with an AI agent

The agent is excellent at:
- Writing syntactically correct PyTorch / HuggingFace code
- Choosing reasonable default hyperparameters
- Setting up training loops, data loaders, evaluation
- Explaining error messages

The agent needs **you** for:
- **"What shape is my data?"** — you know if it's tabular, text, images, time series
- **"Is the loss behaving?"** — you watch the curve and say "it's stuck" or "it's oscillating"
- **"Did it actually learn?"** — you test it on real examples and judge the output
- **"What should I change?"** — you say "try a lower learning rate" or "use more data"

> Think of it like driving with GPS. The GPS (agent) knows every route and every turn.
> But you need to tell it where you want to go, and you need to look out the window to
> see if you've arrived.

---

## Go deeper

- **3Blue1Brown — Neural Networks** (video series): [youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  The best visual explanation of what neural networks do. Watch chapters 1-3.

- **PyTorch 60-min Blitz**: [pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
  Official tutorial. Covers tensors, autograd, and building a network.

- **Andrej Karpathy — "The spelled-out intro to neural networks"**: [youtube.com/watch?v=VMj-3S1tku0](https://www.youtube.com/watch?v=VMj-3S1tku0)
  Builds a neural network from scratch in Python. 2.5 hours but worth every minute.

- **PyTorch nn.Module docs**: [pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)
  Reference for all layers: Linear, Conv2d, LSTM, Transformer, etc.

- **Google ML Crash Course**: [developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)
  Broader ML context: features, labels, training, evaluation, overfitting.

<!-- ⭐ Promote: the tutor's "Promote kept → Markdown" button exports your kept explanations as a
     Markdown file. Paste them below this line to make them a permanent part of the tutorial. -->
