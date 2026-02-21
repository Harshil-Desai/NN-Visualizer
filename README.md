# Neural Network 3D Visualizer

A custom neural network library built from scratch using NumPy, with a real-time 3D visualization interface powered by Three.js.

## Features
- **NN Core**: Modular implementation of Layers, Activations, Losses, and Optimizers.
- **3D Visualization**: Real-time rendering of network architecture, activations, and weight updates.
- **Gradient Flow**: Visual representation of backpropagation using particle systems.
- **MNIST Dataset**: Pre-configured training on the MNIST handwritten digit dataset.

## Structure
- `/src/network`: Python implementation of the neural network.
- `/src/visualizer`: Three.js frontend for visualization.
- `/src/data`: Data loading and preprocessing utilities.
- `/data`: Storage for datasets.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run training with visualization: `python src/train_mnist.py`
3. Open the visualizer in your browser (default: http://localhost:8000)

## Development
See `TUTORIAL.md` for mathematical details and implementation notes.
