"""
tensor.py — The Tensor class: N-dimensional automatic differentiation

Extends the scalar Value concept to full numpy arrays.
Every operation records what happened so gradients can flow backward.
"""

import numpy as np


def _unbroadcast(grad, original_shape):
    """
    When numpy broadcasts a smaller array to a larger one,
    the gradient needs to be summed back to the original shape.
    """
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(original_shape):
        if size == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """
    N-dimensional array with automatic differentiation.

    Each Tensor stores:
      .data      — the actual numpy array
      .grad      — accumulated gradient (same shape as data)
      ._backward — function that propagates grad to parents
      ._prev     — set of Tensors that created this one
    """

    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        """Transpose."""
        out = Tensor(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')\n{self.data}"

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        Matrix multiplication: C = A @ B
        dL/dA = dL/dC @ B.T
        dL/dB = A.T @ dL/dC
        """
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-np.array(other)))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.data / other, (self,), f'/{other}')

            def _backward():
                self.grad += out.grad / other

            out._backward = _backward
            return out
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(np.array(other)) + (-self)

    def __rtruediv__(self, other):
        return Tensor(np.array(other)) * self ** -1

    # ------------------------------------------------------------------ #
    # Reduction operations                                                 #
    # ------------------------------------------------------------------ #

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            grad = out.grad / n
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Activation operations                                                #
    # ------------------------------------------------------------------ #

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-8), (self,), 'log')

        def _backward():
            self.grad += (1.0 / (self.data + 1e-8)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Shape operations                                                     #
    # ------------------------------------------------------------------ #

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def flatten(self):
        return self.reshape(self.data.shape[0], -1)

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self):
        """Backpropagate from this tensor through the entire computation graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
