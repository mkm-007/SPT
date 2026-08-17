"""
Tests for neural network layers.
"""

import numpy as np
from neurein.tensor import Tensor
from neurein.nn.linear import Linear
from neurein.nn.activations import ReLU, Sigmoid


def test_linear_output_shape():
    layer = Linear(4, 8)
    x = Tensor(np.random.randn(16, 4))   # batch=16, in=4
    out = layer(x)
    assert out.shape == (16, 8)


def test_linear_backward():
    layer = Linear(3, 2, bias=False)
    x = Tensor(np.random.randn(5, 3))
    out = layer(x).sum()
    out.backward()
    assert layer.weight.grad.shape == layer.weight.shape


def test_relu_layer():
    relu = ReLU()
    x = Tensor(np.array([[-1.0, 2.0], [3.0, -4.0]]))
    out = relu(x)
    np.testing.assert_allclose(out.data, np.array([[0.0, 2.0], [3.0, 0.0]]))


def test_two_layer_network():
    # x → Linear(4,8) → ReLU → Linear(8,1)
    l1 = Linear(4, 8)
    l2 = Linear(8, 1)
    relu = ReLU()

    x = Tensor(np.random.randn(16, 4))
    out = l2(relu(l1(x)))
    assert out.shape == (16, 1)

    loss = out.mean()
    loss.backward()

    # All parameters should have gradients
    for p in l1.parameters() + l2.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.data.shape


if __name__ == '__main__':
    test_linear_output_shape()
    test_linear_backward()
    test_relu_layer()
    test_two_layer_network()
    print("All layer tests passed.")
