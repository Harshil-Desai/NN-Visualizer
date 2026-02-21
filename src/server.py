import asyncio
import json
import os
import threading
import time
from threading import Lock

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from network import NeuralNetwork, Dense, ReLU, Softmax, CrossEntropy, SGD
from data.loader import download_mnist, load_mnist, preprocess_data, get_batches

# Environment
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

# Training lock for multi-user safety
training_lock = Lock()
training_in_progress = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else [o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

nn_state = {
    "training": False,
    "network": None,
    "epoch": 0,
    "batch": 0,
    "loss": 0,
    "accuracy": 0,
    "learning_rate": 0.01,
    "batch_delay": 0,
    "architecture": [784, 128, 64, 10]
}

def get_layer_activations(layer):
    if hasattr(layer, 'output') and layer.output is not None:
        avg_act = np.mean(layer.output, axis=0)
        return avg_act[:100].tolist()
    return []

def serialize_network(network):
    layers_data = []
    for layer in network.layers:
        if isinstance(layer, Dense):
            layers_data.append({
                'type': 'Dense',
                'weights': layer.weights.tolist(),
                'bias': layer.bias.tolist(),
                'inputSize': layer.weights.shape[0],
                'outputSize': layer.weights.shape[1]
            })
        elif isinstance(layer, ReLU):
            layers_data.append({'type': 'ReLU'})
        elif isinstance(layer, Softmax):
            layers_data.append({'type': 'Softmax'})
    return layers_data

def training_loop():
    global training_in_progress
    try:
        print("Starting training...")
        download_mnist()
        (x_train, y_train), (x_test, y_test) = load_mnist()
        x_train, y_train = preprocess_data(x_train, y_train)
        
        arch = nn_state["architecture"]
        nn_layers = []
        for i in range(len(arch) - 1):
            nn_layers.append(Dense(arch[i], arch[i+1], init_type='he' if i < len(arch)-2 else 'xavier'))
            if i < len(arch) - 2:
                nn_layers.append(ReLU())
            else:
                nn_layers.append(Softmax())
        nn = NeuralNetwork(nn_layers)
        
        nn_state["network"] = nn
        loss_fn = CrossEntropy()
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        
        epochs = 3
        batch_size = 64

        key_moments = {
            'first_forward': False,
            'first_backward': False,
            'loss_spike': False,
            'convergence': False
        }
        
        for epoch in range(epochs):
            if not nn_state["training"]: break
            
            nn_state["epoch"] = epoch + 1
            batches = 0
            
            for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
                if not nn_state["training"]: break

                optimizer.learning_rate = nn_state["learning_rate"] 
                
                loss, y_pred = nn.train_step(x_batch, y_batch, loss_fn, optimizer)
                predictions = np.argmax(y_pred, axis=1)
                labels = np.argmax(y_batch, axis=1)
                acc = np.mean(predictions == labels)
                
                batches += 1
                nn_state["batch"] = batches
                nn_state["loss"] = float(loss)
                nn_state["accuracy"] = float(acc)
                
                activations = [get_layer_activations(layer) for layer in nn.layers]
                
                # Only broadcast every 10 batches (reduces WebSocket spam on free tier); always send first batch
                if batches == 1 or batches % 10 == 0:
                    payload = {
                        "type": "update",
                        "stats": {
                            "epoch": epoch + 1,
                            "batch": batches,
                            "loss": float(loss),
                            "accuracy": float(acc)
                        },
                        "activations": activations
                    }
                    if manager.loop:
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast(json.dumps(payload)),
                            manager.loop
                        )

                # Batch delay for speed control
                if nn_state["batch_delay"] > 0:
                    time.sleep(nn_state["batch_delay"] / 1000)

                # Check for key teaching moments
                if batches == 1 and not key_moments['first_forward']:
                    # Pause after first forward pass
                    if manager.loop:
                        pause_payload = {
                            "type": "pause_moment",
                            "reason": "first_forward",
                            "message": "First forward pass complete! Watch how data flows through layers.",
                            "activations": activations
                        }
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast(json.dumps(pause_payload)),
                            manager.loop
                        )
                    key_moments['first_forward'] = True
                    time.sleep(3)  # Pause for 3 seconds
                
                if batches == 2 and not key_moments['first_backward']:
                    # Pause after first backward pass
                    if manager.loop:
                        pause_payload = {
                            "type": "pause_moment",
                            "reason": "first_backward",
                            "message": "First backpropagation complete! Weights have been updated.",
                            "stats": {
                                "epoch": epoch + 1,
                                "batch": batches,
                                "loss": float(loss),
                                "accuracy": float(acc)
                            }
                        }
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast(json.dumps(pause_payload)),
                            manager.loop
                        )
                    key_moments['first_backward'] = True
                    time.sleep(3)
                
                # Check for loss spike (something interesting)
                if batches > 10 and loss > 1.5 and not key_moments['loss_spike']:
                    if manager.loop:
                        pause_payload = {
                            "type": "pause_moment",
                            "reason": "high_loss",
                            "message": f"High loss detected ({loss:.3f})! The network is confused. Watch how it recovers.",
                            "stats": {
                                "epoch": epoch + 1,
                                "batch": batches,
                                "loss": float(loss),
                                "accuracy": float(acc)
                            }
                        }
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast(json.dumps(pause_payload)),
                            manager.loop
                        )
                    key_moments['loss_spike'] = True
                    time.sleep(2)
                
                if batches % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batches}, Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        # Training complete - send weights
        if nn_state["training"]:
            print("Training complete! Sending weights...")
            weights_data = serialize_network(nn)
            
            completion = {
                "type": "training_complete",
                "message": "Training completed!",
                "final_stats": {
                    "epochs": epochs,
                    "final_loss": float(nn_state["loss"]),
                    "final_accuracy": float(nn_state["accuracy"])
                },
                "weights": weights_data
            }
            
            if manager.loop:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(json.dumps(completion)),
                    manager.loop
                )
        
        # Release lock when training completes
        training_in_progress = False
        if training_lock.locked():
            training_lock.release()
        nn_state["training"] = False
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        training_in_progress = False
        if training_lock.locked():
            training_lock.release()
        nn_state["training"] = False

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/status")
async def api_status():
    return {
        "status": "healthy",
        "training_in_progress": training_in_progress,
        "environment": ENVIRONMENT
    }

@app.post("/start-training")
async def start_training():
    global training_in_progress, training_thread

    if not training_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Training already in progress. Please wait."}

    try:
        training_in_progress = True
        nn_state["training"] = True
        manager.loop = asyncio.get_event_loop()
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()
        return {"status": "started"}
    except Exception as e:
        training_lock.release()
        training_in_progress = False
        nn_state["training"] = False
        return {"status": "error", "message": str(e)}

@app.post("/stop-training")
async def stop_training():
    global training_in_progress
    nn_state["training"] = False
    training_in_progress = False
    if training_lock.locked():
        training_lock.release()
    return {"status": "stopped"}

@app.get("/status")
async def get_status():
    return {
        "training": nn_state["training"],
        "epoch": nn_state["epoch"],
        "batch": nn_state["batch"],
        "loss": nn_state["loss"],
        "accuracy": nn_state["accuracy"]
    }

@app.post("/set-learning-rate")
async def set_learning_rate(lr: float):
    nn_state["learning_rate"] = lr
    return {"status": "updated", "lr": lr}

@app.post("/set-batch-delay")
async def set_batch_delay(ms: int = 0):
    nn_state["batch_delay"] = max(0, ms)
    return {"status": "updated", "batch_delay": nn_state["batch_delay"]}

@app.post("/set-architecture")
async def set_architecture(body: dict):
    layers = body.get("layers", [784, 128, 64, 10])
    if len(layers) < 2 or layers[0] != 784 or layers[-1] != 10:
        return {"status": "error", "message": "Architecture must start with 784 and end with 10"}
    nn_state["architecture"] = layers
    # Rebuild network if not currently training
    if not nn_state["training"]:
        nn_layers = []
        for i in range(len(layers) - 1):
            nn_layers.append(Dense(layers[i], layers[i+1], init_type='he' if i < len(layers)-2 else 'xavier'))
            if i < len(layers) - 2:
                nn_layers.append(ReLU())
            else:
                nn_layers.append(Softmax())
        nn_state["network"] = NeuralNetwork(nn_layers)
    return {"status": "updated", "architecture": layers}

@app.get("/get-weights")
async def get_weights():
    if nn_state["network"] is None:
        return {"status": "error", "message": "No network available"}
    return {"status": "ok", "weights": serialize_network(nn_state["network"])}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print(f"Client connected. Total: {len(manager.active_connections)}")
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "status": "ready"
        }))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected. Remaining: {len(manager.active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

import os
static_dir = os.path.join(os.path.dirname(__file__), "visualizer")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="visualizer")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Neural Network Visualizer Server")
    print("=" * 60)
    print("Access at: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)