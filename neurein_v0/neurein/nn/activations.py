"""
activations.py — Non-linear activation functions

Without activations, stacking linear layers is equivalent to one linear layer.
Activations introduce the non-linearity that lets networks learn complex patterns.
"""

from neurein.nn.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit: max(0, x)
    Sets all negative values to zero.
    Gradient is 1 for positive inputs, 0 for negative.
    """
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    """
    Sigmoid: 1 / (1 + e^-x)
    Squashes output to (0, 1). Used in binary classification output layers.
    Gradient: sigmoid(x) * (1 - sigmoid(x))
    """
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    """
    Hyperbolic tangent: (e^x - e^-x) / (e^x + e^-x)
    Squashes output to (-1, 1). Often used in hidden layers.
    Gradient: 1 - tanh(x)^2
    """
    def forward(self, x):
        return x.tanh()
