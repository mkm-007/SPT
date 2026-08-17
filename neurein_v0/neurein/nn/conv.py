"""
conv.py — 2D Convolutional layer

A convolution slides a small filter (kernel) over the input
and computes dot products at every position. This lets the network
detect local patterns (edges, textures) regardless of their position.

Input:  (batch, in_channels, height, width)
Output: (batch, out_channels, out_height, out_width)
"""

import numpy as np
from neurein.tensor import Tensor
from neurein.nn.module import Module


class Conv2D(Module):
    """
    2D Convolution layer.

    Args:
        in_channels:  number of input channels
        out_channels: number of filters (output channels)
        kernel_size:  size of the sliding window (int or tuple)
        stride:       step size of the sliding window (default 1)
        padding:      zero-padding added to input borders (default 0)
        bias:         whether to include a bias term (default True)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Kaiming initialization for conv weights
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, *kernel_size) * scale
        )
        if bias:
            self.bias = Tensor(np.zeros(out_channels))

    def forward(self, x):
        """
        Forward pass using numpy for the convolution math.
        x: Tensor of shape (batch, in_channels, H, W)
        """
        batch, in_ch, H, W = x.data.shape
        kH, kW = self.kernel_size
        p, s = self.padding, self.stride

        # Pad input
        if p > 0:
            x_data = np.pad(x.data, ((0,0),(0,0),(p,p),(p,p)))
        else:
            x_data = x.data

        out_H = (H + 2*p - kH) // s + 1
        out_W = (W + 2*p - kW) // s + 1
        out_data = np.zeros((batch, self.out_channels, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                # Extract patch: (batch, in_ch, kH, kW)
                patch = x_data[:, :, i*s:i*s+kH, j*s:j*s+kW]
                # Each filter: (out_ch, in_ch, kH, kW)
                # Output at position: (batch, out_ch)
                out_data[:, :, i, j] = np.tensordot(
                    patch, self.weight.data, axes=([1,2,3],[1,2,3])
                )

        if self.use_bias:
            out_data += self.bias.data[None, :, None, None]

        out = Tensor(out_data, (x, self.weight) + ((self.bias,) if self.use_bias else ()), 'conv2d')

        def _backward():
            # Gradient with respect to input and weights
            w_data = self.weight.data
            grad_out = out.grad
            grad_x = np.zeros_like(x_data)
            grad_w = np.zeros_like(w_data)

            for i in range(out_H):
                for j in range(out_W):
                    patch = x_data[:, :, i*s:i*s+kH, j*s:j*s+kW]
                    g = grad_out[:, :, i, j]  # (batch, out_ch)
                    grad_w += np.tensordot(g, patch, axes=([0],[0]))
                    grad_x[:, :, i*s:i*s+kH, j*s:j*s+kW] += np.tensordot(
                        g, w_data, axes=([1],[0])
                    )

            if p > 0:
                x.grad += grad_x[:, :, p:-p, p:-p]
            else:
                x.grad += grad_x
            self.weight.grad += grad_w

            if self.use_bias:
                self.bias.grad += grad_out.sum(axis=(0, 2, 3))

        out._backward = _backward
        return out

    def parameters(self):
        if self.use_bias:
            return [self.weight, self.bias]
        return [self.weight]
