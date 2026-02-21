import numpy as np

class NeuralNetwork:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.hooks = []

    def add(self, layer):
        self.layers.append(layer)

    def register_hook(self, hook_fn):
        """
        hook_fn: function(layer, activations, gradients)
        """
        self.hooks.append(hook_fn)

    def forward(self, input_data):
        current_data = input_data
        for layer in self.layers:
            current_data = layer.forward(current_data)
        return current_data

    def backward(self, output_gradient, learning_rate):
        current_gradient = output_gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, learning_rate)
        return current_gradient

    def train_step(self, x_batch, y_batch, loss_fn, optimizer):
        # 1. Forward pass
        y_pred = self.forward(x_batch)
        
        # 2. Compute loss
        loss_val = loss_fn.loss(y_batch, y_pred)
        
        # 3. Compute loss gradient
        grad = loss_fn.gradient(y_batch, y_pred)
        
        # 4. Backward pass
        self.backward(grad, optimizer.learning_rate)
        
        # 5. Optimize
        optimizer.update(self.layers)
        
        # 6. Call hooks for visualization
        self._trigger_hooks(x_batch, y_batch, y_pred, loss_val)
        
        return loss_val, y_pred

    def predict(self, input_data):
        return self.forward(input_data)

    def _trigger_hooks(self, x, y, y_pred, loss):
        for hook in self.hooks:
            hook(self, x, y, y_pred, loss)
