import os
import gzip
import numpy as np
import requests

def download_mnist(data_dir='data'):
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"Downloading {f}...")
            r = requests.get(base_url + f, stream=True)
            with open(path, 'wb') as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            print(f"Completed {f}")

def load_mnist(data_dir='data'):
    def read_idx_images(path):
        with gzip.open(path, 'rb') as f:
            # Skip header (magic number, num_images, rows, cols)
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28)

    def read_idx_labels(path):
        with gzip.open(path, 'rb') as f:
            # Skip header (magic number, num_items)
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = read_idx_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = read_idx_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = read_idx_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = read_idx_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y, num_classes=10):
    # Normalize images to [0, 1]
    x = x.astype('float32') / 255.0
    
    # One-hot encode labels
    y_onehot = np.zeros((y.size, num_classes))
    y_onehot[np.arange(y.size), y] = 1
    
    return x, y_onehot

def get_batches(x, y, batch_size):
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:min(i + batch_size, n_samples)]
        yield x[batch_indices], y[batch_indices]

# Basic Data Augmentation
def add_noise(images, noise_factor=0.1):
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0)

def random_shift(images, max_shift=2):
    # This is a very simple shift implementation using np.roll
    # Real augmentation would use interpolation
    shifted_images = np.zeros_like(images)
    for i in range(images.shape[0]):
        img = images[i].reshape(28, 28)
        dx, dy = np.random.randint(-max_shift, max_shift + 1, 2)
        img = np.roll(img, dx, axis=0)
        img = np.roll(img, dy, axis=1)
        shifted_images[i] = img.flatten()
    return shifted_images
