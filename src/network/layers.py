import numpy as np
from .activations import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    """
    Fully connected layer: y = Wx + b
    """
    def __init__(self, input_size, output_size, init_type='he'):
        super().__init__()
        self.weight_init(input_size, output_size, init_type)
        self.bias = np.zeros((1, output_size))
        
        # Gradients
        self.weights_grad = None
        self.bias_grad = None
        
        # For momentum
        self.weights_m = np.zeros_like(self.weights)
        self.bias_m = np.zeros_like(self.bias)

    def weight_init(self, input_size, output_size, init_type):
        if init_type == 'xavier':
            # Xavier/Glorot initialization for Sigmoid/Tanh
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            # He initialization for ReLU
            std = np.sqrt(2 / input_size)
            self.weights = np.random.normal(0, std, (input_size, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        # output_gradient is dL/dY
        # dL/dW = X.T * dL/dY
        # dL/db = sum(dL/dY)
        # dL/dX = dL/dY * W.T
        
        self.weights_grad = np.dot(self.input.T, output_gradient)
        self.bias_grad = np.sum(output_gradient, axis=0, keepdims=True)
        
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Note: Weights and biases are updated by the Optimizer class, 
        # but for simple SGD we could do it here. 
        # The user requested an Optimizer class, so we'll store gradients.
        
        return input_gradient

class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # dL/dX = dL/dY * f'(X)
        return output_gradient * self.activation_derivative(self.input)

class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(relu, relu_derivative)

class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derivative)

class Softmax(Layer):
    """
    Softmax is often treated specially with Cross-Entropy.
    For local gradient, dS_i/dx_j = S_i(kron_ij - S_j)
    """
    def forward(self, input_data):
        self.input = input_data
        self.output = softmax(input_data)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This implementation assumes the gradient comes from a source that 
        # hasn't already combined softmax and cross-entropy.
        # However, for efficiency, most frameworks do that.
        # dL/dx = sum_j (dL/dy_j * dy_j/dx_i)
        
        # Batch size handling
        n = self.output.shape[0]
        out_grad = np.zeros_like(self.input)
        
        for i in range(n):
            s = self.output[i].reshape(-1, 1)
            # Jacobian matrix for softmax: diag(s) - ss^T
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            out_grad[i] = np.dot(jacobian, output_gradient[i].reshape(-1, 1)).flatten()
            
        return out_grad
