"""
mse.py — Mean Squared Error loss

MSE = mean( (prediction - target)^2 )

Used for regression problems where output is a continuous value.
Why squared? Prevents positive and negative errors from cancelling,
and penalizes large errors more heavily than small ones.
"""

from neurein.nn.module import Module


class MSELoss(Module):
    def forward(self, prediction, target):
        diff = prediction - target
        return (diff ** 2).mean()
