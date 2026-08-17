"""
Example 2: Training a Simple Network on XOR

XOR is the classic test for neural networks because it cannot
be solved by a single linear layer — it requires a hidden layer
with non-linear activations.

Inputs → Labels:
  [0,0] → 0
  [0,1] → 1
  [1,0] → 1
  [1,1] → 0

If this trains correctly and loss approaches zero, the entire
NeuralForge stack is verified: Tensor, Linear, ReLU, MSELoss, Adam.
"""

import numpy as np
from neurein.tensor import Tensor
from neurein.nn.linear import Linear
from neurein.nn.activations import ReLU, Sigmoid
from neurein.loss.mse import MSELoss
from neurein.optim.adam import Adam

# XOR dataset
X = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float64))

# Network: 2 → 8 → 1
l1   = Linear(2, 8)
relu = ReLU()
l2   = Linear(8, 1)
sig  = Sigmoid()
loss_fn = MSELoss()

# Collect all parameters
params = l1.parameters() + l2.parameters()
opt = Adam(params, lr=0.01)

print("Training XOR network...")
print(f"{'Epoch':>6} | {'Loss':>10}")
print("-" * 20)

for epoch in range(1000):
    # Forward
    out = sig(l2(relu(l1(X))))
    loss = loss_fn(out, y)

    # Backward
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print(f"{epoch:>6} | {loss.data.item():>10.6f}")

print("\nFinal predictions:")
out = sig(l2(relu(l1(X))))
for i, (inp, pred, target) in enumerate(
    zip(X.data, out.data, y.data)
):
    print(f"  Input: {inp} → Predicted: {pred[0]:.3f} | Target: {target[0]}")
