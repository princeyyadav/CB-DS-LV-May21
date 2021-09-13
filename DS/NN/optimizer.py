import numpy as np

class GradientDescentOptimizer:

    def __init__(self, lr):
        self.lr = lr

    def minimize(self, w, grad):
        assert w.shape == grad.shape, f"Shape mismatch w shape {w.shape} != grad shape {grad.shape}"
        w = w-self.lr*grad
        return w