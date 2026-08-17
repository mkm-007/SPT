"""
bce.py — Binary Cross Entropy loss

BCE = -mean( y * log(p) + (1-y) * log(1-p) )

Used for binary classification (outputs between 0 and 1).
Also used in VAE reconstruction loss (treating pixels as Bernoulli variables).

In HW1, the loss was: BCE_with_logits(x_hat, x) * 784
This is equivalent to BCE applied to sigmoid(x_hat) but numerically stable.
"""

from neurein.tensor import Tensor
import numpy as np
from neurein.nn.module import Module


class BCELoss(Module):
    """BCE loss. Expects predictions already passed through sigmoid."""

    def forward(self, prediction, target):
        # Clip to avoid log(0)
        p = prediction.data.clip(1e-7, 1 - 1e-7)
        from neurein.tensor import Tensor as T
        p_clipped = T(p, (prediction,), 'clip')

        def _backward_clip():
            prediction.grad += p_clipped.grad * (
                (prediction.data >= 1e-7) & (prediction.data <= 1 - 1e-7)
            ).astype(np.float64)

        p_clipped._backward = _backward_clip

        return -(target * p_clipped.log() + (Tensor(np.ones_like(target.data)) - target) * (Tensor(np.ones_like(p_clipped.data)) - p_clipped).log()).mean()


class BCEWithLogitsLoss(Module):
    """
    BCE loss with built-in sigmoid — numerically more stable.
    This is what HW1 uses: BCEWithLogits(x_hat, x) * 784
    """

    def forward(self, logits, target):
        # Numerically stable: log(1 + e^x) - x*y
        # equivalent to BCE(sigmoid(logits), target)
        loss_data = np.maximum(logits.data, 0) - logits.data * target.data + \
                    np.log(1 + np.exp(-np.abs(logits.data)))
        out = Tensor(loss_data, (logits, target), 'bce_with_logits')

        def _backward():
            sigmoid = 1.0 / (1.0 + np.exp(-logits.data))
            logits.grad += (sigmoid - target.data) * out.grad / logits.data.size
            target.grad += -logits.data * out.grad / logits.data.size

        out._backward = _backward
        return out.mean()
