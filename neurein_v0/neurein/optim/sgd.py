"""
sgd.py — Stochastic Gradient Descent optimizer

The simplest optimizer. At each step, every parameter moves a small
amount in the direction that reduces the loss:

    w = w - lr * gradient

'Stochastic' means we use a random mini-batch of data to estimate
the gradient at each step, rather than the entire dataset.

Optional: momentum accumulates a velocity vector to speed up
learning in consistent directions and dampen oscillations.
"""


class SGD:
    """
    Stochastic Gradient Descent.

    Args:
        parameters: list of Tensor parameters to optimize
        lr:         learning rate (step size)
        momentum:   momentum factor (default 0 = no momentum)
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        # Velocity terms for momentum
        self.velocities = [None] * len(parameters)

    def step(self):
        """Update all parameters using their computed gradients."""
        import numpy as np
        for i, p in enumerate(self.parameters):
            if self.momentum > 0:
                if self.velocities[i] is None:
                    self.velocities[i] = np.zeros_like(p.data)
                self.velocities[i] = (
                    self.momentum * self.velocities[i] + p.grad
                )
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """Reset all gradients to zero before the next forward pass."""
        for p in self.parameters:
            p.zero_grad()
