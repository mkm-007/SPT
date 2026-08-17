"""
adam.py — Adam optimizer (Adaptive Moment Estimation)

Adam improves on SGD by:
1. Keeping a running average of past gradients (like momentum)
2. Keeping a running average of squared gradients (adapts lr per parameter)
3. Bias correction to account for the cold start (first few steps)

Update rule:
    m = beta1 * m + (1 - beta1) * grad          # 1st moment (mean)
    v = beta2 * v + (1 - beta2) * grad^2        # 2nd moment (variance)
    m_hat = m / (1 - beta1^t)                   # bias-corrected
    v_hat = v / (1 - beta2^t)                   # bias-corrected
    w = w - lr * m_hat / (sqrt(v_hat) + eps)

This is the optimizer used in HW1 (AdamW) and HW2.
"""

import numpy as np


class Adam:
    """
    Adam optimizer.

    Args:
        parameters: list of Tensor parameters to optimize
        lr:         learning rate (default 1e-3)
        beta1:      decay rate for 1st moment (default 0.9)
        beta2:      decay rate for 2nd moment (default 0.999)
        eps:        small constant for numerical stability (default 1e-8)
        weight_decay: L2 regularization coefficient (default 0)
    """

    def __init__(self, parameters, lr=1e-3, beta1=0.9,
                 beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # step counter

        # Initialize moment estimates at zero
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """Update all parameters using Adam rule."""
        self.t += 1

        for i, p in enumerate(self.parameters):
            grad = p.grad

            # Weight decay (L2 regularization) — used in AdamW
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Update biased moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset all gradients to zero."""
        for p in self.parameters:
            p.zero_grad()
