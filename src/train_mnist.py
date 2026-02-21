import numpy as np
import pickle
import os
from network import NeuralNetwork, Dense, ReLU, Softmax, CrossEntropy, SGD
from data.loader import download_mnist, load_mnist, preprocess_data, get_batches

def compute_accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

def save_model(network, filepath):
    weights = []
    for layer in network.layers:
        if isinstance(layer, Dense):
            weights.append({'w': layer.weights, 'b': layer.bias})
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Model saved to {filepath}")

def train():
    # 1. Load Data
    download_mnist()
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    
    # 2. Define Network
    nn = NeuralNetwork([
        Dense(784, 128, init_type='he'),
        ReLU(),
        Dense(128, 64, init_type='he'),
        ReLU(),
        Dense(64, 10, init_type='xavier'),
        Softmax()
    ])
    
    loss_fn = CrossEntropy()
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    
    epochs = 5
    batch_size = 64
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        batches = 0
        
        for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
            loss, y_pred = nn.train_step(x_batch, y_batch, loss_fn, optimizer)
            
            acc = compute_accuracy(y_batch, y_pred)
            epoch_loss += loss
            epoch_acc += acc
            batches += 1
            
            if batches % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batches}: Loss = {loss:.4f}, Acc = {acc:.4f}")
        
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
        # Evaluate on test set
        test_pred = nn.predict(x_test[:1000]) # Sample for speed
        test_acc = compute_accuracy(y_test[:1000], test_pred)
        print(f"Test Accuracy: {test_acc:.4f}")

    # 3. Save Model
    os.makedirs('models', exist_ok=True)
    save_model(nn, 'models/mnist_model.pkl')

if __name__ == "__main__":
    train()
