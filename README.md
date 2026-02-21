# Neural Network Journey

**An interactive 3D neural network visualizer that trains on MNIST in the browser—with a NumPy backend and a Three.js frontend.**

---

## Overview

Neural Network Journey helps you understand how neural networks learn by training a real MNIST classifier and watching activations, gradients, and weights update in real time. The backend is a from-scratch NumPy implementation; the frontend renders the network in 3D and lets you draw digits to test predictions. It solves the problem of *seeing* what happens inside a network during training and inference.

**Key benefits:** no deep-learning framework required, runs locally or deploys to Render/Railway, and supports multiple users with a training lock and connection status.

---

## Key Features

- **Real-time 3D visualization** — Layers and neurons (Three.js) with activation-based coloring and optional gradient particles.
- **Backend training on MNIST** — NumPy-only neural network (Dense, ReLU, Softmax, CrossEntropy, SGD) with WebSocket streaming of stats and activations.
- **Draw & predict** — Draw digits in a 28×28-style canvas; preprocessing and inference run in the browser using weights synced after training.
- **Custom architecture** — Add/remove hidden layers and set neuron counts (input 784, output 10 fixed) via the Build Architecture modal.
- **Teaching mode** — Auto-pause at key moments (first forward/backward pass, high loss) with short explanations and animations.
- **Multi-user safe** — Single training lock so concurrent users get a “busy” response instead of conflicting runs.
- **Deployable** — Environment-based config, CORS, health checks, and optional static file serving for [Render](https://render.com) or [Railway](https://railway.app).

---

## Quick Start

```bash
# Clone and enter the project
cd NN-Visualizer

# Install dependencies
pip install -r requirements.txt

# Run the server (serves API + frontend)
cd src && uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. Wait for the connection indicator to turn green, then click **▶️** to start training. When training finishes, use **✏️ Draw Your Own Digit** to test predictions.

---

## Installation

### Requirements

| Requirement | Version / notes |
|-------------|------------------|
| Python      | 3.11 recommended (3.10+ supported) |
| NumPy       | 1.24.x (see `requirements.txt`)    |
| FastAPI     | 0.104.x                            |
| Browser     | Modern (WebSocket, ES6, Canvas)     |

### Local development

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional:** Copy env template and edit if needed:
   ```bash
   cp .env.example .env
   ```

4. **Run from project root:**
   ```bash
   cd src && uvicorn server:app --reload --port 8000
   ```
   Or from `src`:
   ```bash
   cd src
   uvicorn server:app --reload --port 8000
   ```

5. Open **http://localhost:8000**. The page is served by the same server; no separate frontend build is required.

### Production (Render)

- Connect your repo to [Render](https://render.com).
- Use the included `render.yaml` (build: `pip install -r requirements.txt`, start: `cd src && uvicorn server:app --host 0.0.0.0 --port $PORT`).
- Set `ENVIRONMENT=production` and `ALLOWED_ORIGINS` to your app URL(s) in the Render dashboard.
- Health check path: `/health`.

### Production (Railway)

- Use the provided `Procfile`: `web: cd src && uvicorn server:app --host 0.0.0.0 --port $PORT`.
- Set `PORT` and optionally `ENVIRONMENT` and `ALLOWED_ORIGINS` in Railway variables.

---

## Configuration

Configuration is driven by environment variables. For local dev, you can use a `.env` file (loaded via `python-dotenv` from the project root when the app runs from `src/`).

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port. |
| `ENVIRONMENT` | `development` | `development` or `production`. In production, CORS uses `ALLOWED_ORIGINS` instead of `*`. |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | Comma-separated origins for CORS in production. |

Example `.env`:

```env
PORT=8000
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

---

## Usage Examples

### 1. Start and stop training (HTTP)

```bash
# Start training
curl -X POST http://localhost:8000/start-training

# Stop training
curl -X POST http://localhost:8000/stop-training
```

### 2. Check health and status

```bash
# Health (for load balancers / Render)
curl http://localhost:8000/health

# API status (training flag, environment)
curl http://localhost:8000/api/status

# Training status (epoch, batch, loss, accuracy)
curl http://localhost:8000/status
```

### 3. Set learning rate and architecture

```bash
# Learning rate
curl -X POST "http://localhost:8000/set-learning-rate?lr=0.01"

# Architecture (must start with 784, end with 10)
curl -X POST http://localhost:8000/set-architecture \
  -H "Content-Type: application/json" \
  -d '{"layers": [784, 256, 128, 10]}'
```

### 4. WebSocket connection (browser or script)

The UI connects to `ws://localhost:8000/ws` (or `wss://...` on HTTPS). Message types from server:

- `connected` — initial handshake.
- `update` — batch updates (stats + activations).
- `pause_moment` — teaching pause (reason, message, activations).
- `training_complete` — final stats and weights.

### 5. Get weights after training

```bash
curl http://localhost:8000/get-weights
```

Returns JSON with layer types and weight/bias arrays for use in custom clients.

---

## API Reference

### HTTP endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}`. Use for health checks. |
| `GET` | `/api/status` | Returns `status`, `training_in_progress`, `environment`. |
| `GET` | `/status` | Returns `training`, `epoch`, `batch`, `loss`, `accuracy`. |
| `POST` | `/start-training` | Starts training. Returns `started`, `busy`, or `error`. |
| `POST` | `/stop-training` | Stops training. Returns `stopped`. |
| `POST` | `/set-learning-rate?lr=<float>` | Sets learning rate. |
| `POST` | `/set-batch-delay?ms=<int>` | Sets delay between batches (e.g. for “slow” mode). |
| `POST` | `/set-architecture` | Body: `{"layers": [784, ..., 10]}`. Updates architecture. |
| `GET` | `/get-weights` | Returns current network weights (or error if none). |

### WebSocket

- **Endpoint:** `/ws`
- **Server → client:** JSON messages with `type` one of `connected`, `update`, `pause_moment`, `training_complete`.

### Core Python modules

| Module | Purpose |
|--------|---------|
| `src/server` | FastAPI app, CORS, training lock, WebSocket manager, static mount. |
| `src/network` | `NeuralNetwork`, `Dense`, `ReLU`, `Softmax`, `CrossEntropy`, `SGD`. |
| `src/data.loader` | `download_mnist`, `load_mnist`, `preprocess_data`, `get_batches`. |

**Representative signatures:**

```python
# src/network/model.py
class NeuralNetwork:
    def __init__(self, layers=None)
    def forward(self, input_data)
    def backward(self, output_gradient, learning_rate)
    def train_step(self, x_batch, y_batch, loss_fn, optimizer)
    def predict(self, input_data)

# src/data/loader.py
def download_mnist(data_dir='data')
def load_mnist(data_dir='data')
def preprocess_data(x, y, num_classes=10)
def get_batches(x, y, batch_size)
```

---

## Contributing

Contributions are welcome. Please open an issue to discuss larger changes, and ensure the server runs with `cd src && uvicorn server:app --port 8000` and the main flows (train, draw, predict) still work.

If you add a `CONTRIBUTING.md`, link it here.

---

## License

This project does not currently include a LICENSE file. For reuse or distribution, add a license (e.g. MIT or Apache 2.0) to the repository.

---

## Support / Issues

- **Bugs and feature requests:** Open an issue in the project repository.
- **Learning path and internals:** See [TUTORIAL.md](TUTORIAL.md) for concepts, first run, and troubleshooting.
