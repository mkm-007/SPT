"""
module.py — Base class for all layers

Every layer (Linear, Conv2D, ReLU, etc.) inherits from Module.
This gives every layer a consistent interface:
  .forward(x)      — compute the output
  .parameters()    — return all learnable weights and biases
  .zero_grad()     — reset all gradients to zero before each step
"""


class Module:
    """Base class for all Neurein layers."""

    def parameters(self):
        """Return all learnable parameters as a flat list."""
        return []

    def zero_grad(self):
        """Set all parameter gradients to zero."""
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        """Calling a layer runs its forward pass."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )
