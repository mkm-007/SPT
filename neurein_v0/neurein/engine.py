"""
engine.py — The Value class: scalar automatic differentiation

This is the heart of Neurein. A Value wraps a single number
and silently builds a computation graph as you do math with it.
When you call .backward(), it walks that graph in reverse and
computes the gradient for every value that contributed to the output.

This is exactly what PyTorch does internally — just for scalars.
"""

import math


class Value:
    """
    A scalar value with automatic differentiation.

    Every time you do math with a Value, a new Value is created
    and the relationship between them is recorded. This forms a
    computation graph. Calling .backward() on the final output
    (usually the loss) sends gradients back through the entire graph.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)      # the actual number
        self.grad = 0.0              # gradient starts at zero
        self._backward = lambda: None  # no-op until an operation sets it
        self._prev = set(_children)  # the Values that created this one
        self._op = _op               # the operation that created this (for debugging)
        self.label = label           # optional name (for debugging)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------ #
    # Forward operations — each one also defines its own backward         #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # gradient accumulates with += because a value can be
            # used in multiple places in the graph
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float exponents for now"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x — the only function that is its own derivative
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, f"log undefined for non-positive value: {self.data}"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # d(log x)/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            # gradient is 1 if input was positive, 0 if it was negative
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh)/dx = 1 - tanh^2
            self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Python operator sugar — these let you write a + b, a - b, etc.     #
    # ------------------------------------------------------------------ #

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):   # handles: 2 + Value(3)
        return self + other

    def __rmul__(self, other):   # handles: 2 * Value(3)
        return self * other

    def __rsub__(self, other):   # handles: 2 - Value(3)
        return other + (-self)

    def __rtruediv__(self, other):  # handles: 2 / Value(3)
        return other * self ** -1

    # ------------------------------------------------------------------ #
    # Backpropagation                                                      #
    # ------------------------------------------------------------------ #

    def backward(self):
        """
        Compute gradients for every Value in the computation graph.

        Algorithm:
        1. Build a topological ordering of all Values in the graph
           (children always appear before parents)
        2. Set this Value's gradient to 1.0
           (the derivative of the loss w.r.t. itself is always 1)
        3. Walk the ordering in reverse, calling each Value's
           _backward() to propagate gradients to its children
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
