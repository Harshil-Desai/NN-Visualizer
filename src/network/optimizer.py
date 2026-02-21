import numpy as np
from .layers import Dense

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum

    def update(self, layers):
        for layer in layers:
            if isinstance(layer, Dense):
                # Update weights with momentum
                # v = m * v - lr * grad
                # w = w + v
                layer.weights_m = self.momentum * layer.weights_m - self.learning_rate * layer.weights_grad
                layer.weights += layer.weights_m
                
                # Update biases with momentum
                layer.bias_m = self.momentum * layer.bias_m - self.learning_rate * layer.bias_grad
                layer.bias += layer.bias_m
