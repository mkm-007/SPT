"""
Tests for optimizers.
The key test: after one step, parameters must have actually changed.
"""

import numpy as np
from neurein.tensor import Tensor
from neurein.nn.linear import Linear
from neurein.nn.activations import ReLU
from neurein.optim.sgd import SGD
from neurein.optim.adam import Adam


def _make_network_and_loss():
    """Helper: build a small network and compute a loss."""
    layer = Linear(4, 2)
    x = Tensor(np.random.randn(8, 4))
    target = Tensor(np.random.randn(8, 2))
    out = layer(x)
    loss = ((out - target) ** 2).mean()
    return layer, loss


def test_sgd_updates_parameters():
    layer, loss = _make_network_and_loss()
    old_weight = layer.weight.data.copy()

    opt = SGD(layer.parameters(), lr=0.01)
    loss.backward()
    opt.step()

    assert not np.allclose(layer.weight.data, old_weight), \
        "SGD did not update weights"


def test_adam_updates_parameters():
    layer, loss = _make_network_and_loss()
    old_weight = layer.weight.data.copy()

    opt = Adam(layer.parameters(), lr=1e-3)
    loss.backward()
    opt.step()

    assert not np.allclose(layer.weight.data, old_weight), \
        "Adam did not update weights"


def test_zero_grad_resets():
    layer, loss = _make_network_and_loss()
    opt = SGD(layer.parameters(), lr=0.01)
    loss.backward()
    opt.zero_grad()

    for p in layer.parameters():
        np.testing.assert_allclose(p.grad, np.zeros_like(p.grad))


if __name__ == '__main__':
    test_sgd_updates_parameters()
    test_adam_updates_parameters()
    test_zero_grad_resets()
    print("All optimizer tests passed.")
