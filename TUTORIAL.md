# Neural Network Journey — Tutorial

This tutorial walks you through running and understanding the Neural Network Visualizer: from first concepts to training, drawing digits, and inspecting what the network learned.

---

## Prerequisites

Before you start, make sure you have:

- **Python 3.10+** (3.11 recommended), with `pip` available.
- A **modern browser** (Chrome, Firefox, Edge, Safari) with JavaScript and WebSockets enabled.
- **About 100 MB** free space for the MNIST dataset (downloaded on first training run).
- Basic familiarity with the command line (running commands in a terminal).

You do **not** need prior experience with neural networks or Three.js. The app teaches concepts as you go.

---

## Under the Hood (Reference)

This section is optional. It summarizes how the network and visualization work.

**Mathematics:** For each layer we compute \( z = W x + b \) and \( a = \sigma(z) \) (ReLU for hidden layers, Softmax for output). Backpropagation uses the chain rule: output error \( \delta_L \), then hidden errors \( \delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l) \), then weight gradients \( \partial L / \partial W_l = \delta_l \cdot a_{l-1}^T \).

**3D visualization:** Each neuron is a Three.js sphere. Color is HSL: low activation → blue, high → red. Connections are line segments; large layers are capped for performance. The Python training loop sends activation data over WebSockets; the frontend updates the scene on each batch (or every N batches in production).

**Data flow:** Python trains on MNIST and captures layer outputs → FastAPI broadcasts JSON over WebSockets → the browser updates the 3D view and stats. After training, the server sends weights; the client runs the same forward pass in JavaScript for the drawing pad.

---

## Learning Path

By the end of this tutorial you will:

1. Understand how the app is structured (backend vs frontend).
2. Run the server and complete a first training run.
3. Use the 3D view, stats, and teaching pauses to see what “forward pass” and “backpropagation” mean.
4. Draw your own digits and get predictions.
5. Change the network architecture and run again.
6. Know where to look when something goes wrong.

---

## Part 1: Basic Concepts

### What This App Does

The app has two main parts:

- **Backend (Python):** A neural network implemented from scratch in NumPy. It trains on the MNIST dataset (28×28 images of digits 0–9). There are no frameworks like PyTorch or TensorFlow—only NumPy, so you can see every step.
- **Frontend (JavaScript + Three.js):** A 3D visualization of the same network. Neurons are spheres; their color and brightness show *activation* (how “on” they are). The backend streams updates over WebSockets so the 3D view updates as training runs.

So: **Python trains the network; the browser shows it.**

### Key Ideas You’ll See

- **Forward pass:** Input (e.g. one image) is pushed through the layers. Each layer computes `output = activation(weights × input + bias)`. The last layer outputs 10 numbers (one per digit); the largest is the prediction.
- **Loss:** A single number measuring how wrong the predictions are (e.g. cross-entropy). Training tries to make this number smaller.
- **Backpropagation:** The “error” is sent backward through the layers. Each layer gets a signal that tells it how to change its weights and biases so the loss goes down.
- **Activations:** The values at each layer after the activation function. These are what you see in the 3D view (color/brightness of the spheres).

You don’t have to memorize this now. The UI will pause at “first forward” and “first backward” so you can connect these words to what you see on screen. For formulas and implementation details, see the **Under the Hood** section at the start of this tutorial.

---

## Part 2: Hello World — First Run

Goal: start the server, open the app, and run one training to completion.

### Step 1: Install and run the server

From the project root:

```bash
pip install -r requirements.txt
cd src
uvicorn server:app --reload --port 8000
```

You should see something like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Leave this terminal open. The server serves both the API and the frontend (HTML/JS/CSS).

### Step 2: Open the app

1. In your browser, go to **http://localhost:8000**.
2. Wait a moment. In the top context bar you should see:
   - A small **dot** next to the status text. It starts **red** (Disconnected).
   - After a few seconds it should turn **green** and the text should say **“Connected - ready to train”**.

If the dot stays red, check the terminal for errors and that nothing else is using port 8000. See **Troubleshooting** below.

### Step 3: Start training

1. Click the **▶️ (Play)** button in the control panel.
2. The status text should change to **“Training in progress...”** and the button to **⏸️ (Pause)**.
3. You may see short **“Auto-Paused”** overlays (e.g. “First forward pass”, “First backpropagation”). These are teaching moments; they auto-dismiss after a few seconds or you can click **Continue**.
4. In the right sidebar, **Training Stats** (Epoch, Batch, Loss, Accuracy) and the **loss chart** will update. The 3D network will show activations (colors) changing over time.
5. After a few minutes (default is 3 epochs), training finishes. You’ll see an alert like **“Training complete! Accuracy: XX%”** and the status will show the final accuracy.

**Expected:** Loss generally goes down, accuracy goes up. Exact numbers depend on random initialization and hardware.

### Step 4: Confirm you’re on track

- **Green dot** = connected.
- **Stats updating** = backend is sending updates.
- **3D neurons changing color** = activations are streaming.
- **Alert at the end** = training completed and weights were sent to the browser.

You’ve just run your first “hello world” training. Next, use the network with your own drawings.

---

## Part 3: Intermediate Features

### Draw a digit and get a prediction

1. After training has completed, click **✏️ Draw Your Own Digit** in the right **Tools** panel.
2. In the modal, draw a digit (0–9) on the black canvas with the mouse (or touch).
3. Click **Predict**.
4. The app runs the **same preprocessing** as the backend (grayscale, crop, resize to 28×28, center) and then runs a **forward pass** in the browser using the weights received at “training complete.”
5. You should see **Prediction: X** and a **Confidence** bar, plus small bars for each digit 0–9.

**Why it works:** When training finished, the server sent the trained weights to the client. The frontend keeps them in memory and runs inference locally, so you can try many digits without calling the server again.

**Common mistake:** Clicking **Predict** before training has completed. The app will show “Please complete training first!” Train once, then draw and predict.

### Change the network architecture

1. Click **🏗️ Build Architecture**.
2. You’ll see **Input: 784** and **Output: 10** fixed. You can add/remove **hidden layers** and set their **neuron counts** (e.g. 128, 64).
3. Add a layer with **+ Add Hidden Layer**, or remove one with the **×** next to a layer.
4. Click **Build Network**. Confirm the dialog (it will reset training progress).
5. The 3D view will rebuild with the new layer layout. Click **▶️** to train again with this architecture.

Larger networks (e.g. 784 → 256 → 128 → 10) have more parameters and may learn better but take longer and use more memory.

### Other tools

- **📊 Analyze Network:** After training, you can open analysis views (e.g. weight heatmaps for the first hidden layer, loss landscape). Explore these to see what the network learned.
- **🎨 Paint Weights Manually:** Advanced: paint weights for a chosen neuron and see how that changes its response to drawn digits.
- **⚖️ Compare: Backprop vs Random:** Side-by-side view comparing real training (backprop) with random updates; illustrates why learning requires gradient-based updates.

---

## Part 4: Advanced Usage

### Multiple tabs and “busy” state

Only one training run can execute at a time (single backend process). If you open two tabs:

- Tab 1 starts training → OK.
- Tab 2 clicks **▶️** → the server returns **“Training already in progress”** and the UI shows: **“Training is currently in progress by another user. Please wait and try again.”**

This is intentional so two users don’t corrupt the same training state.

### Speed and learning rate

- **Speed:** Use **Step**, **Slow**, **Normal**, or **Fast** to change how often updates are sent and whether there’s a delay between batches (useful for following along).
- **Learning rate:** Use the **Learning Rate** slider or the **Slow / Normal / Fast** presets. Higher values (e.g. 0.05) train faster but can be unstable; lower (e.g. 0.001) is smoother but slower.

### Running headless or from another machine

The server binds to `0.0.0.0`, so you can open the app from another device on your LAN:

```text
http://<your-computer-ip>:8000
```

Use **http** (not https) unless you put a reverse proxy in front. The WebSocket will use `ws://` automatically.

### Production and environment

For deployment (e.g. Render or Railway):

- Set **ENVIRONMENT=production** so CORS uses **ALLOWED_ORIGINS** instead of `*`.
- Set **ALLOWED_ORIGINS** to your public URL(s), e.g. `https://your-app.onrender.com`.
- Health checks should use **GET /health**.

See **README.md** for exact commands and config.

---

## Troubleshooting

### Connection indicator stays red

- **Cause:** Frontend can’t reach the WebSocket at `/ws`.
- **Checks:**
  1. Server is running (`cd src && uvicorn server:app --port 8000`).
  2. You’re using the same host/port in the browser (e.g. http://localhost:8000).
  3. No firewall or corporate proxy blocking WebSockets.
- **Try:** Reload the page; check the browser console (F12) for WebSocket errors.

### “Not connected to backend!” when clicking Play

- Same as above: the client is not connected to `/ws`. Fix the connection (red → green) first.

### “Training is currently in progress by another user”

- Another tab or user already started training. Wait for it to finish or stop it from that tab, then try again.

### Training seems stuck or very slow

- First run downloads MNIST; that can take a minute. Later runs start quickly.
- Default is 3 epochs; on slow or shared machines it can take a few minutes. Watch the **Batch** and **Epoch** numbers in the sidebar.
- If the process crashes, check the terminal for Python errors (e.g. out of memory). Try a smaller architecture (e.g. 784 → 64 → 10).

### “Please complete training first!” when predicting

- You must run at least one full training to completion before **Draw Your Own Digit** → **Predict** will work. The browser needs the weights from the server.

### Port 8000 already in use

- Run on another port: `uvicorn server:app --reload --port 8001` and open http://localhost:8001.
- Or stop the other process using 8000 (e.g. another instance of this app).

### Changes to frontend not showing

- Hard refresh: **Ctrl+Shift+R** (or **Cmd+Shift+R** on Mac). If you edited files under `src/visualizer/`, the server serves them as-is; no build step is required.

---

## Next Steps

- **README.md** — Quick start, installation, configuration, API reference, and deployment.
- **In-app help** — Use the **❓** button and the **Learn More** links for short explanations of backpropagation and concepts.
- **Code:**  
  - Backend: `src/server.py` (routes, WebSocket, training loop), `src/network/` (layers, loss, optimizer), `src/data/loader.py` (MNIST).  
  - Frontend: `src/visualizer/main-app.js` (BackendManager, NetworkVisualizer, MainApp), `src/visualizer/index.html`.
- **Under the Hood** (at the top of this file) — Math and 3D visualization summary.

Once you’re comfortable with one training run and drawing digits, try different architectures, learning rates, and the analysis tools to build intuition for how the network learns.
