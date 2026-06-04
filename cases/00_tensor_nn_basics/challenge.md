# 🧪 Case 00 — Your turn

Edit the notebook / script and observe. No single right answer — build a feel and form your own questions.

## Warm-ups
1. Create a tensor of shape `(3, 4, 5)`. What does `x.unsqueeze(0).shape` give you? What about `x.unsqueeze(2).shape`? Predict first, then check. Draw the shapes on paper if it helps.
2. Build a model with `nn.Sequential` that has 3 hidden layers of size 128 instead of 64. Does it learn the sin(x) task faster? Slower? Why might more parameters not always help?

## Build something ⭐
3. Modify the training function to plot the loss curve *live* (using `matplotlib` with `plt.pause()`). Train the same model with learning rates 0.1, 0.001, and 0.00001. What happens to each loss curve? Can you find a learning rate that makes training diverge (loss goes UP)?

## Reflect (jot 2-3 sentences)
- When you got a shape mismatch error, what information did you need to fix it? How would you describe this to an AI agent?
- If someone asked you to "train a model to predict house prices from 5 features," what would you tell the agent? (Hint: think about input shape, output shape, loss function, and dataset format.)

> Done? Open `../01_foundations/`.
