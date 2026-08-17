"""
Example 1: Scalar Autograd Engine

This shows the most fundamental thing Neurein does:
track operations on numbers and compute gradients automatically.
"""

from neurein.engine import Value

print("=" * 50)
print("Example 1: Scalar Autograd")
print("=" * 50)

# --- Basic gradient computation ---
print("\n--- Chain rule example ---")
x = Value(2.0, label='x')
w = Value(3.0, label='w')
y = x * w          # y = x*w = 6
loss = y ** 2      # loss = y^2 = 36

loss.backward()

print(f"x = {x.data}, w = {w.data}")
print(f"y = x*w = {y.data}")
print(f"loss = y^2 = {loss.data}")
print(f"d(loss)/dx = {x.grad}  (expected: 2*y*w = 2*6*3 = 36)")
print(f"d(loss)/dw = {w.grad}  (expected: 2*y*x = 2*6*2 = 24)")

# --- Single neuron ---
print("\n--- Single neuron ---")
x1 = Value(1.0, label='x1')
x2 = Value(2.0, label='x2')
w1 = Value(0.5, label='w1')
w2 = Value(-0.3, label='w2')
b  = Value(0.1, label='b')

output = (x1 * w1 + x2 * w2 + b).relu()
print(f"output = relu({x1.data}*{w1.data} + {x2.data}*{w2.data} + {b.data})")
print(f"output = {output.data:.4f}")

output.backward()
print(f"d(output)/dw1 = {w1.grad:.4f}")
print(f"d(output)/dw2 = {w2.grad:.4f}")
