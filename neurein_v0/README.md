# Neurein

A deep learning framework built entirely from scratch in Python and NumPy — no PyTorch, no autograd libraries, no shortcuts.

> **Neurein** — from *neuro* (neuron) + *ein* (one, in German). One neuron. Built from first principles.

> Status: 🚧 **Phase 1 — Core autograd engine complete**

---

## What This Is

Neurein reimplements the core machinery of frameworks like PyTorch — tensors, automatic differentiation, computation graphs, layers, optimizers, and loss functions — entirely from first principles.

It exists for one reason: to understand *why* deep learning works, not just *that* it works.

**Capstone goal:** rebuild the Conditional β-VAE from CSC 8851's HW1 using nothing but Neurein. No PyTorch. No autograd. If it trains and produces the same latent space visualizations as the original assignment, the framework is correct.

---

## Why This Exists

I served as a Teaching Assistant for CSC 8851 (Deep Learning) at Georgia State University for two semesters, grading assignments on VAEs, GANs, diffusion models, and graph neural networks.

After two semesters, I noticed a gap: I could recognize a *correct* implementation when I saw one, but I didn't have the mechanistic understanding of *why* it was correct. Neurein is how I closed that gap — by building every abstraction myself instead of relying on a library.

This project is also an unsolicited contribution back to the course. The goal: a tool a student could use in Week 1 — before ever touching PyTorch — to understand what `.backward()` actually does.

No one asked me to build this. That's the point.

---

## Quick Start

```python
import numpy as np
from neurein.tensor import Tensor
from neurein.nn.linear import Linear
from neurein.nn.activations import ReLU, Sigmoid
from neurein.loss.mse import MSELoss
from neurein.optim.adam import Adam

# Build a network
l1   = Linear(2, 8)
relu = ReLU()
l2   = Linear(8, 1)
sig  = Sigmoid()

# Forward pass
x    = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
out  = sig(l2(relu(l1(x))))

# Backward pass
loss_fn = MSELoss()
target  = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float64))
loss    = loss_fn(out, target)

opt = Adam(l1.parameters() + l2.parameters(), lr=0.05)
opt.zero_grad()
loss.backward()
opt.step()
```

---

## Project Structure

```
neurein/
├── neurein/
│   ├── engine.py           # Value class — scalar autograd
│   ├── tensor.py           # Tensor class — N-dimensional autograd
│   ├── nn/
│   │   ├── module.py       # base Module class
│   │   ├── linear.py       # Linear (fully connected) layer
│   │   ├── conv.py         # Conv2D layer
│   │   └── activations.py  # ReLU, Sigmoid, Tanh
│   ├── optim/
│   │   ├── sgd.py          # SGD optimizer
│   │   └── adam.py         # Adam optimizer
│   └── loss/
│       ├── mse.py          # Mean Squared Error
│       ├── bce.py          # Binary Cross Entropy
│       └── kl.py           # KL Divergence
├── tests/                  # pytest test suite
├── examples/               # worked examples
└── README.md
```

---

## Roadmap

| Phase | Weeks | Goal | Status |
|---|---|---|---|
| **1 — Core Engine** | 1–3 | Scalar autograd + Tensor class | ✅ Done |
| **2 — Building Blocks** | 4–6 | Linear, Conv2D, losses, optimizers | ✅ Done |
| **3 — Course Alignment** | 7–9 | Rebuild HW1 β-VAE end-to-end | 🚧 Next |
| **4 — Polish** | 10–12 | Full tests, docs, examples | ⬜ Pending |

---

## Verified Working

```
All engine tests passed.      ← scalar autograd, chain rule
All tensor tests passed.      ← N-dimensional autograd
All layer tests passed.       ← Linear, ReLU, 2-layer network
All optimizer tests passed.   ← SGD, Adam, zero_grad
```

XOR proof — trained using nothing but Neurein:
```
[0,0] → 0.005  (target 0)  ✓
[0,1] → 0.999  (target 1)  ✓
[1,0] → 0.999  (target 1)  ✓
[1,1] → 0.001  (target 0)  ✓
Final loss: 0.000007
```

---

## Design Principles

- **No autograd dependencies.** Only NumPy for array math.
- **Understand before implementing.** Every class was built only after being able to explain its mechanism in plain English.
- **Tested, not just working.** Every module has a test file.
- **Course-aligned scope.** Built to reproduce CSC 8851 assignments exactly.

---

## Reference

Architecture informed by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) — used as reference, not copied.

Course assignments from CSC 8851, Deep Learning, Georgia State University.

---

## Author

Built as part of a self-directed practical training program (SPT), Summer 2026.
GSU CS Master's student | Former TA, CSC 8851 Deep Learning
