import numpy as np

class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        raise NotImplementedError

class CrossEntropy(Loss):
    def loss(self, y_true, y_pred):
        """
        y_true: One-hot encoded labels (batch_size, num_classes)
        y_pred: Probabilities (softmax output) (batch_size, num_classes)
        """
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true, y_pred):
        """
        Combined gradient of Softmax + Cross-Entropy.
        
        When using Softmax activation followed by Cross-Entropy loss,
        the gradient simplifies to: (y_pred - y_true) / batch_size
        
        This is mathematically correct and numerically stable.
        
        If you have:
        - z = logits (input to softmax)
        - y_pred = softmax(z)
        - L = -sum(y_true * log(y_pred))
        
        Then: dL/dz = (y_pred - y_true) / batch_size
        
        This assumes the previous layer was Softmax.
        """
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size

class MSE(Loss):
    """Mean Squared Error - for regression tasks"""
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return 2 * (y_pred - y_true) / batch_size