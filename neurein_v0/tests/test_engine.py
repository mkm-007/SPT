"""
Tests for the scalar Value autograd engine.

Every test here verifies that gradients computed by NeuralForge
match the analytically known correct values.
"""

import math
from neurein.engine import Value


def test_addition_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0


def test_addition_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    # d(a+b)/da = 1, d(a+b)/db = 1
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_multiplication_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    # d(a*b)/da = b = 3, d(a*b)/db = a = 2
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_power_backward():
    x = Value(3.0)
    y = x ** 2
    y.backward()
    # d(x^2)/dx = 2x = 6
    assert x.grad == 6.0


def test_chain_rule():
    # y = x^2, loss = y^2 → loss = x^4
    # d(loss)/dx = 4x^3 = 4*(2^3) = 32
    x = Value(2.0)
    y = x ** 2
    loss = y ** 2
    loss.backward()
    assert abs(x.grad - 32.0) < 1e-6


def test_relu_positive():
    x = Value(3.0)
    y = x.relu()
    y.backward()
    assert y.data == 3.0
    assert x.grad == 1.0


def test_relu_negative():
    x = Value(-3.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0


def test_sigmoid():
    x = Value(0.0)
    y = x.sigmoid()
    y.backward()
    assert abs(y.data - 0.5) < 1e-6
    # d(sigmoid(0))/dx = 0.5 * 0.5 = 0.25
    assert abs(x.grad - 0.25) < 1e-6


def test_neuron_forward():
    # Single neuron: output = relu(w1*x1 + w2*x2 + b)
    x1, x2 = Value(2.0), Value(3.0)
    w1, w2 = Value(0.5), Value(-0.3)
    b = Value(0.1)
    out = (w1 * x1 + w2 * x2 + b).relu()
    # 0.5*2 + (-0.3)*3 + 0.1 = 1.0 - 0.9 + 0.1 = 0.2
    assert abs(out.data - 0.2) < 1e-6


def test_gradient_accumulation():
    # A value used twice accumulates gradients from both paths
    x = Value(2.0)
    y = x + x   # equivalent to 2*x
    y.backward()
    assert x.grad == 2.0  # gradient from both uses of x sum up


if __name__ == '__main__':
    test_addition_forward()
    test_addition_backward()
    test_multiplication_backward()
    test_power_backward()
    test_chain_rule()
    test_relu_positive()
    test_relu_negative()
    test_sigmoid()
    test_neuron_forward()
    test_gradient_accumulation()
    print("All engine tests passed.")
