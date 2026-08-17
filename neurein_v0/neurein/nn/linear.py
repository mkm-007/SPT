"""
linear.py — Fully connected (dense) layer

output = input @ weights.T + bias

Each output neuron is connected to every input neuron.
Weights and bias are learnable parameters updated by the optimizer.
"""

import numpy as np
from neurein.tensor import Tensor
from neurein.nn.module import Module


class Linear(Module):
    """
    A fully connected layer.

    Args:
        in_features:  number of input features
        out_features: number of output features
        bias:         whether to include a bias term (default True)
    """

    def __init__(self, in_features, out_features, bias=True):
        # Xavier initialization: keeps gradient variance stable across layers
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(out_features, in_features) * scale
        )
        self.use_bias = bias
        if bias:
            self.bias = Tensor(np.zeros((1, out_features)))

    def forward(self, x):
        """
        x:      (batch_size, in_features)
        weight: (out_features, in_features)
        output: (batch_size, out_features)
        """
        out = x @ self.weight.T
        if self.use_bias:
            out = out + self.bias
        return out

    def parameters(self):
        if self.use_bias:
            return [self.weight, self.bias]
        return [self.weight]
