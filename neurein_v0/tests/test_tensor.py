"""
Tests for the Tensor class (N-dimensional autograd).
"""

import numpy as np
from neurein.tensor import Tensor


def test_add_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[1.0, 1.0], [1.0, 1.0]]))
    c = (a + b).sum()
    c.backward()
    np.testing.assert_allclose(a.grad, np.ones((2, 2)))


def test_matmul_backward():
    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))  # identity
    C = (A @ B).sum()
    C.backward()
    np.testing.assert_allclose(A.grad, np.ones((2, 2)))


def test_relu_backward():
    x = Tensor(np.array([[-1.0, 2.0], [3.0, -4.0]]))
    y = x.relu().sum()
    y.backward()
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(x.grad, expected)


def test_mean_backward():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = x.mean()
    y.backward()
    np.testing.assert_allclose(x.grad, np.full((2, 2), 0.25))


if __name__ == '__main__':
    test_add_backward()
    test_matmul_backward()
    test_relu_backward()
    test_mean_backward()
    print("All tensor tests passed.")
