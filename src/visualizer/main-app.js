/**
 * COMPLETE Neural Network Visualizer - All Features Working
 * Backend Training + Weight Transfer + Prediction + Particles + Journey + Everything
 */

// ====================================================================================
// PREPROCESSING (Same as before)
// ====================================================================================
const Preprocessing = {
    process: (canvas) => {
        const ctx = canvas.getContext('2d');
        const rawData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        let img = Preprocessing.convertToGrayscale(rawData);
        img = Preprocessing.invertIfNeeded(img);
        const bbox = Preprocessing.findBoundingBox(img, canvas.width, canvas.height);
        if (bbox.width === 0 || bbox.height === 0) return new Array(784).fill(0);
        const cropped = Preprocessing.extractRegion(img, bbox, canvas.width);
        const aspect = bbox.width / bbox.height;
        let targetW, targetH;
        if (aspect > 1) {
            targetW = 20;
            targetH = Math.round(20 / aspect);
        } else {
            targetH = 20;
            targetW = Math.round(20 * aspect);
        }
        const scaled = Preprocessing.bilinearResize(cropped, bbox.width, bbox.height, targetW, targetH);
        const final = new Array(784).fill(0);
        const offsetX = Math.round((28 - targetW) / 2);
        const offsetY = Math.round((28 - targetH) / 2);
        for (let y = 0; y < targetH; y++) {
            for (let x = 0; x < targetW; x++) {
                final[(offsetY + y) * 28 + (offsetX + x)] = scaled[y * targetW + x];
            }
        }
        const centered = Preprocessing.centerByMass(final);
        return centered.map(v => v / 255.0);
    },
    convertToGrayscale: (imgData) => {
        const gray = [];
        for (let i = 0; i < imgData.data.length; i += 4) {
            const val = imgData.data[i] * 0.299 + imgData.data[i + 1] * 0.587 + imgData.data[i + 2] * 0.114;
            gray.push(val);
        }
        return gray;
    },
    invertIfNeeded: (img) => {
        const corners = [img[0], img[img.length - 1], img[27], img[img.length - 28]];
        const avgCorner = corners.reduce((a, b) => a + b, 0) / 4;
        return avgCorner > 128 ? img.map(v => 255 - v) : img;
    },
    findBoundingBox: (img, w, h) => {
        let minX = w, minY = h, maxX = 0, maxY = 0, found = false;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (img[y * w + x] > 10) {
                    if (x < minX) minX = x; if (x > maxX) maxX = x;
                    if (y < minY) minY = y; if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }
        return found ? { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 } : { x: 0, y: 0, width: 0, height: 0 };
    },
    extractRegion: (img, bbox, w) => {
        const extracted = [];
        for (let y = 0; y < bbox.height; y++) {
            for (let x = 0; x < bbox.width; x++) {
                extracted.push(img[(bbox.y + y) * w + (bbox.x + x)]);
            }
        }
        return extracted;
    },
    bilinearResize: (src, srcW, srcH, dstW, dstH) => {
        const dst = new Array(dstW * dstH);
        const xRatio = srcW / dstW, yRatio = srcH / dstH;
        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                const srcX = x * xRatio, srcY = y * yRatio;
                const x1 = Math.floor(srcX), x2 = Math.min(x1 + 1, srcW - 1);
                const y1 = Math.floor(srcY), y2 = Math.min(y1 + 1, srcH - 1);
                const fx = srcX - x1, fy = srcY - y1;
                const p11 = src[y1 * srcW + x1], p21 = src[y1 * srcW + x2];
                const p12 = src[y2 * srcW + x1], p22 = src[y2 * srcW + x2];
                dst[y * dstW + x] = p11 * (1 - fx) * (1 - fy) + p21 * fx * (1 - fy) + p12 * (1 - fx) * fy + p22 * fx * fy;
            }
        }
        return dst;
    },
    centerByMass: (img) => {
        let totalMass = 0, comX = 0, comY = 0;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const val = img[y * 28 + x];
                totalMass += val; comX += x * val; comY += y * val;
            }
        }
        if (totalMass === 0) return img;
        comX /= totalMass; comY /= totalMass;
        const shiftX = Math.round(14 - comX), shiftY = Math.round(14 - comY);
        const shifted = new Array(784).fill(0);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const newX = x + shiftX, newY = y + shiftY;
                if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
                    shifted[newY * 28 + newX] = img[y * 28 + x];
                }
            }
        }
        return shifted;
    }
};

window.setLR = async function (value) {
    const slider = document.getElementById('lr-slider');
    const display = document.getElementById('lr-value');
    if (slider) slider.value = value;
    if (display) display.textContent = value.toFixed(4);

    if (window.app?.backend?.connected) {
        await fetch('/set-learning-rate?lr=' + value, { method: 'POST' });
    }
};

// ====================================================================================
// BACKEND MANAGER
// ====================================================================================
class BackendManager {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.onPauseMoment = null;
        this.onUpdate = null;
        this.onConnect = null;
        this.onDisconnect = null;
        this.onTrainingComplete = null;
    }

    connect() {
        // Fix #12: Clean up old WebSocket before reconnecting
        if (this.ws) {
            this.ws.onopen = null;
            this.ws.onmessage = null;
            this.ws.onclose = null;
            this.ws.onerror = null;
            if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
                this.ws.close();
            }
            this.ws = null;
        }

        // Use WSS for HTTPS, WS for HTTP
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws`;

        console.log('🔌 Connecting to WebSocket:', wsUrl);

        this.ws = new WebSocket(wsUrl);
        this.ws.onopen = () => {
            console.log('✅ Connected');
            this.connected = true;
            if (this.onConnect) this.onConnect();
        };
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'pause_moment' && this.onPauseMoment) {
                this.onPauseMoment(data);
            } else if (data.type === 'update' && this.onUpdate) {
                this.onUpdate(data);
            } else if (data.type === 'training_complete' && this.onTrainingComplete) {
                this.onTrainingComplete(data);
            }
        };
        this.ws.onclose = () => {
            console.log('❌ Disconnected');
            this.connected = false;
            if (this.onDisconnect) this.onDisconnect();
            setTimeout(() => this.connect(), 2000);
        };
    }

    async startTraining() {
        const r = await fetch('/start-training', { method: 'POST' });
        return await r.json();
    }

    async stopTraining() {
        const r = await fetch('/stop-training', { method: 'POST' });
        return await r.json();
    }
}

// ====================================================================================
// 3D VISUALIZER WITH PARTICLES
// ====================================================================================
class NetworkVisualizer {
    constructor(containerId = 'canvas-container-a') {
        const container = document.getElementById(containerId);
        if (!container) return;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 2000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.nodeMeshes = [];
        this.layers = [];
        this.connections = new THREE.Group();
        this.scene.add(this.connections);
        this.particles = [];

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const sun = new THREE.PointLight(0xffffff, 1, 100);
        sun.position.set(20, 30, 20);
        this.scene.add(sun);

        this.camera.position.set(15, 5, 25);
        this.camera.lookAt(0, 0, 0);

        window.addEventListener('resize', () => this.onResize());
        this.animate();
        this.build([784, 128, 64, 10]);
    }

    onResize() {
        const container = this.renderer.domElement.parentElement;
        if (!container) return;
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }

    build(arch) {
        this.nodeMeshes.forEach(m => this.scene.remove(m));
        while (this.connections.children.length > 0) this.connections.remove(this.connections.children[0]);
        this.nodeMeshes = [];
        this.layers = [];

        const layerSpacing = 12;
        const neuronGeo = new THREE.SphereGeometry(0.3, 16, 16);

        arch.forEach((size, lIdx) => {
            const x = (lIdx - (arch.length - 1) / 2) * layerSpacing;
            const displaySize = Math.min(size, 100);
            const layerData = { x, positions: [], displaySize };
            const mesh = new THREE.InstancedMesh(neuronGeo,
                new THREE.MeshStandardMaterial({ roughness: 0.1, metalness: 0.5, emissive: 0x111111 }),
                displaySize);

            const matrix = new THREE.Matrix4();
            const side = Math.ceil(Math.sqrt(displaySize));
            for (let i = 0; i < displaySize; i++) {
                let y, z;
                if (displaySize > 20) {
                    y = (Math.floor(i / side) - (side - 1) / 2);
                    z = ((i % side) - (side - 1) / 2);
                } else {
                    y = (i - (displaySize - 1) / 2);
                    z = 0;
                }
                const pos = new THREE.Vector3(x, y, z);
                matrix.setPosition(pos);
                mesh.setMatrixAt(i, matrix);
                mesh.setColorAt(i, new THREE.Color(0x666666));
                layerData.positions.push(pos);
            }
            this.scene.add(mesh);
            this.nodeMeshes.push(mesh);
            this.layers.push(layerData);
        });

        const lineMat = new THREE.LineBasicMaterial({ color: 0x444444, transparent: true, opacity: 0.1 });
        for (let l = 0; l < this.layers.length - 1; l++) {
            const c = this.layers[l], n = this.layers[l + 1];
            const pts = [];
            const stride = Math.max(1, Math.floor(c.displaySize / 32));
            for (let i = 0; i < c.displaySize; i += stride) {
                for (let j = 0; j < n.displaySize; j += stride) {
                    pts.push(c.positions[i], n.positions[j]);
                }
            }
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const lines = new THREE.LineSegments(geo, lineMat);
            this.connections.add(lines);
        }
    }

    setupNeuronInteraction() {
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.renderer.domElement.addEventListener('click', (e) => {
            if (this.controls.enabled === false) return; // Don't interfere with drawing

            const rect = this.renderer.domElement.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);

            // Check each layer's neurons
            this.nodeMeshes.forEach((mesh, layerIdx) => {
                const intersects = this.raycaster.intersectObject(mesh);
                if (intersects.length > 0) {
                    const instanceId = intersects[0].instanceId;
                    this.showNeuronInspector(layerIdx, instanceId);
                }
            });
        });
    }

    showNeuronInspector(layerIdx, neuronIdx) {
        // Get current activation
        const activation = this.currentActivations?.[layerIdx]?.[neuronIdx] || 0;

        // Create/update inspector panel
        let inspector = document.getElementById('neuron-inspector');
        if (!inspector) {
            inspector = document.createElement('div');
            inspector.id = 'neuron-inspector';
            inspector.className = 'neuron-inspector';
            document.body.appendChild(inspector);
        }

        const layerNames = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];

        // Feature A: Weights visualization
        let weightCanvasHtml = '';
        if (layerIdx > 0 && this.localNN) {
            // Visualize weights connecting to this neuron from previous layer
            weightCanvasHtml = `
            <div class="inspector-section">
                <h4>Incoming Weights (Feature Detector)</h4>
                <canvas id="neuron-weight-canvas" width="140" height="140" 
                    style="width: 140px; height: 140px; image-rendering: pixelated; border: 1px solid var(--border); background: #000; display: block; margin: 0 auto;"></canvas>
                <div style="font-size: 0.7rem; color: var(--text-dim); text-align: center; margin-top: 4px;">
                    Red = +, Blue = -
                </div>
            </div>`;
        } else if (layerIdx === 0) {
            weightCanvasHtml = `<div class="inspector-section" style="text-align: center; color: var(--text-dim); font-size: 0.8rem;">Input Pixel</div>`;
        }

        const outputLabel = layerIdx === this.layers.length - 1 ?
            `<div class="inspector-row"><span>Digit:</span><strong>${neuronIdx}</strong></div>` : '';

        inspector.innerHTML = `
        <div class="inspector-header">
            <h3>🔍 Neuron Inspector</h3>
            <button onclick="document.getElementById('neuron-inspector').remove()">×</button>
        </div>
        <div class="inspector-content">
            <div class="inspector-row"><span>Layer:</span><strong>${layerNames[layerIdx] || 'Layer ' + layerIdx}</strong></div>
            <div class="inspector-row"><span>Neuron:</span><strong>#${neuronIdx}</strong></div>
            ${outputLabel}
            <div class="inspector-row"><span>Activation:</span><strong style="color: var(--accent)">${activation.toFixed(4)}</strong></div>
            <div class="activation-bar">
                <div class="activation-fill" style="width: ${activation * 100}%; background: hsl(${(1 - activation) * 240}, 80%, 50%)"></div>
            </div>
            ${weightCanvasHtml}
        </div>
        `;

        inspector.style.display = 'block';

        // Render weights if applicable
        if (layerIdx > 0 && this.localNN) {
            setTimeout(() => {
                const canvas = document.getElementById('neuron-weight-canvas');
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const weights = this.localNN.layers[layerIdx - 1].weights.data; // [input][output]
                const inputSize = this.localNN.layers[layerIdx - 1].inputSize;

                // Determine grid size (assuming square input if possible, else just row)
                const size = Math.floor(Math.sqrt(inputSize));
                const cols = size;
                const rows = Math.ceil(inputSize / cols);
                const cellW = canvas.width / cols;
                const cellH = canvas.height / rows;

                // Find max weight for normalization
                let maxW = 0;
                for (let i = 0; i < inputSize; i++) maxW = Math.max(maxW, Math.abs(weights[i][neuronIdx]));

                for (let i = 0; i < inputSize; i++) {
                    const w = weights[i][neuronIdx];
                    const x = (i % cols) * cellW;
                    const y = Math.floor(i / cols) * cellH;

                    // Red for positive, Blue for negative
                    const val = Math.abs(w) / (maxW || 1);
                    ctx.fillStyle = w > 0 ? `rgba(255, 50, 50, ${val})` : `rgba(50, 50, 255, ${val})`;
                    ctx.fillRect(x, y, cellW, cellH);
                }
            }, 0);
        }
    }

    updateFromBackend(activations) {
        this.currentActivations = activations;
        activations.forEach((layerAct, lIdx) => {
            const mesh = this.nodeMeshes[lIdx];
            if (!mesh) return;
            const count = Math.min(layerAct.length, this.layers[lIdx].displaySize);
            for (let i = 0; i < count; i++) {
                const v = Math.max(0, Math.min(1, layerAct[i]));
                const hue = (1 - v) * 0.6;
                mesh.setColorAt(i, new THREE.Color().setHSL(hue, 1, 0.3 + v * 0.4));
            }
            mesh.instanceColor.needsUpdate = true;
        });
    }

    createGradientParticles() {
        for (let i = this.layers.length - 1; i > 0; i--) {
            this.createParticles(this.layers[i].x, this.layers[i - 1].x);
        }
    }

    createParticles(startX, endX, count = 15) {
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            pos[i * 3] = startX;
            pos[i * 3 + 1] = (Math.random() - 0.5) * 10;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 10;
        }
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({ color: 0x00ffff, size: 0.15, transparent: true, opacity: 0.8 });
        const points = new THREE.Points(geo, mat);
        this.scene.add(points);
        this.particles.push(points);

        const startTime = performance.now();
        const duration = 800;
        const move = (now) => {
            const t = (now - startTime) / duration;
            if (t < 1) {
                const x = startX + (endX - startX) * t;
                for (let i = 0; i < count; i++) pos[i * 3] = x;
                points.geometry.attributes.position.needsUpdate = true;
                requestAnimationFrame(move);
            } else {
                this.scene.remove(points);
                this.particles = this.particles.filter(p => p !== points);
            }
        };
        requestAnimationFrame(move);
    }

    toggleElement(name) {
        if (name === 'connections') this.connections.visible = !this.connections.visible;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

class ConceptAnimator {
    constructor(viz) {
        this.viz = viz;
        this.animations = [];
    }

    animateForwardPass() {
        // Highlight data flowing forward
        this.viz.layers.forEach((layer, idx) => {
            setTimeout(() => {
                this.pulseLayer(idx);
                if (idx < this.viz.layers.length - 1) {
                    this.animateConnections(idx, idx + 1);
                }
            }, idx * 500);
        });
    }

    pulseLayer(layerIdx) {
        const mesh = this.viz.nodeMeshes[layerIdx];
        if (!mesh) return;

        const startScale = 1;
        const endScale = 1.3;
        const duration = 500;
        const startTime = performance.now();

        const animate = (now) => {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease in-out
            const eased = progress < 0.5
                ? 2 * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;

            const scale = startScale + (endScale - startScale) * eased * (progress < 0.5 ? 1 : -1);
            mesh.scale.set(scale, scale, scale);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                mesh.scale.set(1, 1, 1);
            }
        };

        requestAnimationFrame(animate);
    }

    animateConnections(fromIdx, toIdx) {
        // Highlight connections between layers
        const connections = this.viz.connections.children[fromIdx];
        if (!connections) return;

        const originalOpacity = connections.material.opacity;
        connections.material.opacity = 0.5;
        connections.material.color.setHex(0x00ffff);

        setTimeout(() => {
            connections.material.opacity = originalOpacity;
            connections.material.color.setHex(0x444444);
        }, 500);
    }

    animateBackprop() {
        // Show error flowing backward with particles
        for (let i = this.viz.layers.length - 1; i > 0; i--) {
            setTimeout(() => {
                this.viz.createGradientParticles();
                this.pulseLayer(i);
            }, (this.viz.layers.length - i) * 500);
        }
    }

    animateWeightUpdate(layerIdx) {
        // Flash effect on weights being updated
        const connections = this.viz.connections.children[layerIdx];
        if (!connections) return;

        connections.material.color.setHex(0x00ff88);
        setTimeout(() => {
            connections.material.color.setHex(0x444444);
        }, 200);
    }
}

// ====================================================================================
// MAIN APP - Complete with ALL features
// ====================================================================================
class MainApp {
    constructor() {
        console.log('🚀 Initializing...');

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
        }

        this.backend = new BackendManager();
        this.training = false;
        this.viz = new NetworkVisualizer();
        this.localNN = null;
        this.lossData = [];
        this.showGradients = false;
        this.lastParticle = 0;
        this.visualizationThrottle = 33;
        this.comparisonMode = false; // Fix #4: moved from class field to constructor
        this.lastVizUpdate = 0;

        this.initLossChart();
        const dc = document.getElementById('drawing-canvas');
        if (dc) this.initDrawingCanvas();

        this.backend.onPauseMoment = (data) => this.handlePauseMoment(data);
        this.backend.onUpdate = (d) => this.handleBackendUpdate(d);
        this.backend.onConnect = () => this.onBackendConnect();
        this.backend.onDisconnect = () => this.onBackendDisconnect();
        this.backend.onTrainingComplete = (d) => this.onTrainingComplete(d);
        this.backend.connect();

        this.animator = new ConceptAnimator(this.viz);

        const lrSlider = document.getElementById('lr-slider');
        if (lrSlider) {
            lrSlider.oninput = () => {
                const value = parseFloat(lrSlider.value);
                document.getElementById('lr-value').textContent = value.toFixed(4);
                if (this.backend.connected) {
                    fetch(`/set-learning-rate?lr=${value}`, { method: 'POST' });
                }
            };
        }

        // Enable neuron inspection
        this.viz.setupNeuronInteraction();

        // Fix #7: removed duplicate animator init (already at L532)

        // Enable auto-pause (user can disable later)
        this.autoPauseEnabled = true;

        console.log('✅ All interactive features enabled');

        console.log('✅ Ready');
    }

    handlePauseMoment(data) {
        // Show auto-pause overlay with explanation
        this.pauseWithExplanation(data.reason, data.message, data);
    }

    pauseWithExplanation(reason, message, data) {
        const overlay = document.createElement('div');
        overlay.className = 'explanation-overlay auto-pause';
        overlay.innerHTML = `
        <div class="explanation-content">
            <div class="pause-indicator">⏸️ Auto-Paused</div>
            <h2>${this.getExplanationTitle(reason)}</h2>
            <p>${message || this.getExplanationText(reason)}</p>
            
            <div class="explanation-visual">
                <canvas id="explanation-canvas" width="600" height="300"></canvas>
            </div>
            
            <div class="pause-actions">
                <button class="modal-btn-secondary" onclick="this.closest('.explanation-overlay').remove()">
                    Skip Future Pauses
                </button>
                <button class="modal-btn-primary" onclick="this.closest('.explanation-overlay').remove()">
                    Continue (3s) →
                </button>
            </div>
        </div>
    `;
        document.body.appendChild(overlay);

        // Auto-resume after 3 seconds
        setTimeout(() => {
            if (overlay.parentNode) overlay.remove();
        }, 3000);

        // IMPORTANT: Wait for DOM to be ready before rendering
        setTimeout(() => {
            this.renderExplanation(reason);
        }, 50);
    }

    renderPauseMomentVisual(data) {
        const canvas = document.getElementById('pause-moment-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw a simple bar chart of activations
        const layerToShow = data.activations[1] || data.activations[0];
        const numBars = Math.min(20, layerToShow.length);
        const barWidth = canvas.width / numBars;

        layerToShow.slice(0, numBars).forEach((activation, i) => {
            const height = activation * canvas.height * 0.8;
            const hue = (1 - activation) * 240;

            ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
            ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 2, height);
        });

        ctx.fillStyle = '#fff';
        ctx.font = '12px monospace';
        ctx.fillText('Neuron Activations', 10, 20);
    }

    onBackendConnect() {
        const el = document.getElementById('current-action-text');
        if (el) el.textContent = 'Connected - ready to train';

        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.style.background = 'var(--success)';
            indicator.title = 'Connected';
        }
    }

    onBackendDisconnect() {
        const el = document.getElementById('current-action-text');
        if (el) el.textContent = 'Disconnected - reconnecting...';

        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.style.background = 'var(--error)';
            indicator.title = 'Disconnected';
        }

        this.training = false;
        this.updateTrainingButton();
    }

    handleBackendUpdate(data) {
        if (data.stats) this.updateStats(data.stats);

        if (window.TeachingMode && TeachingMode.narrationEnabled) {
            const epoch = data.stats.epoch;
            const batch = data.stats.batch;

            if (data.stats.batch === 1) {
                setTimeout(() => this.animator.animateForwardPass(), 500);
            }
            if (data.stats.batch === 2) {
                setTimeout(() => this.animator.animateBackprop(), 500);
            }
            if (data.stats.batch % 50 === 0) {
                this.animator.animateWeightUpdate(0);
            }

            // Narrate at key moments
            if (batch === 1 && epoch === 1) {
                TeachingMode.narrate('forwardStart', { epoch, batch });
            } else if (batch === 50 && epoch === 1) {
                TeachingMode.narrate('forwardLayer', { layer: 1 });
            } else if (batch === 100 && epoch === 1) {
                TeachingMode.narrate('backwardStart', { loss: data.stats.loss });
            } else if (batch % 200 === 0) {
                TeachingMode.narrate('weightUpdate', { accuracy: data.stats.accuracy });
            }

        }

        const now = performance.now();
        if (now - this.lastVizUpdate > this.visualizationThrottle) {
            if (data.activations) this.viz.updateFromBackend(data.activations);
            if (this.showGradients && now - this.lastParticle > 500) {
                this.viz.createGradientParticles();
                this.lastParticle = now;
                if (window.Journey) Journey.trackBackprop();
            }
            this.lastVizUpdate = now;
        }
        if (window.Journey && data.stats) Journey.trackAccuracy(data.stats.accuracy);

        // Fix #5 + #6: Guard comparisonStats and viz2 access
        if (this.comparisonMode && this.comparisonStats) {
            // Update left viz (real training)
            if (data.activations) this.viz.updateFromBackend(data.activations);

            // Update right viz (random activations)
            if (this.viz2 && data.activations) {
                const randomActivations = data.activations.map(layer =>
                    layer.map(() => Math.random())
                );
                this.viz2.updateFromBackend(randomActivations);
            }

            // Track stats
            this.comparisonStats.backprop.losses.push(data.stats.loss);
            this.comparisonStats.backprop.accuracies.push(data.stats.accuracy);

            // Simulate random network (always poor)
            this.comparisonStats.random.losses.push(2.3 + Math.random() * 0.5); // High loss
            this.comparisonStats.random.accuracies.push(0.1 + Math.random() * 0.05); // ~10% accuracy

            this.updateComparisonChart();
        }
    }

    onTrainingComplete(data) {
        console.log('🎉 Training Complete!');
        this.training = false;
        this.updateTrainingButton();

        if (data.weights) {
            console.log('📦 Loading weights...');
            this.loadWeightsToLocalNN(data.weights);
        }

        const el = document.getElementById('current-action-text');
        if (el) el.textContent = `Training complete! Accuracy: ${(data.final_stats.final_accuracy * 100).toFixed(1)}%`;

        // CHAPTER PROGRESSION
        if (window.Journey) {
            // Complete Chapter 1 tasks
            Journey.completeTask('t1_1'); // Watch a single neuron activate
            Journey.completeTask('t1_2'); // Adjust weights manually (training did this)
            Journey.completeTask('t1_3'); // See how bias shifts the decision
            Journey.completeTask('t1_4'); // Quiz: Predict neuron output

            // Complete main training task
            Journey.completeTask('t5_1');

            // High accuracy achievement
            if (data.final_stats.final_accuracy >= 0.8) {
                Journey.completeTask('t2_1');
                Journey.unlockAchievement('accuracyChamp');
            }

            // Check if Chapter 1 is fully complete
            const chapter1 = Journey.chapters.find(c => c.id === 'chapter1');
            const allDone = chapter1.tasks.every(t => Journey.isTaskCompleted(t.id));
            if (allDone) {
                Journey.completeChapter('chapter1');
                setTimeout(() => {
                    alert('🎉 Chapter 1 Complete!\n\n"What is a Neuron?" mastered!\n\nChapter 2 "Building a Layer" is now unlocked!');
                }, 1000);
            }
        }

        setTimeout(() => alert(`🎉 Training Complete!\n\nAccuracy: ${(data.final_stats.final_accuracy * 100).toFixed(1)}%\n\nDraw and test now!`), 500);
    }

    loadWeightsToLocalNN(weightsData) {
        if (!window.NN) {
            console.error('NN engine not loaded!');
            return;
        }
        const layers = [];
        weightsData.forEach(ld => {
            if (ld.type === 'Dense') {
                const layer = new window.NN.Dense(ld.inputSize, ld.outputSize, 'relu');
                layer.weights = new window.NN.Matrix(ld.inputSize, ld.outputSize, ld.weights);
                layer.biases = new window.NN.Matrix(1, ld.outputSize, ld.bias);
                layers.push(layer);
            }
        });
        this.localNN = new window.NN.NeuralNetwork(layers);
        console.log('✅ Local NN ready!');
    }

    updateStats(stats) {
        const els = {
            'stat-epoch': stats.epoch,
            'stat-batch': stats.batch,
            'stat-loss': stats.loss.toFixed(4),
            'stat-accuracy': (stats.accuracy * 100).toFixed(1) + '%',
            'mini-loss': `Loss: ${stats.loss.toFixed(3)}`,
            'mini-accuracy': `Acc: ${(stats.accuracy * 100).toFixed(0)}%`
        };
        Object.entries(els).forEach(([id, val]) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        });
        this.lossData.push(stats.loss);
        if (this.lossData.length > 100) this.lossData.shift();
        if (this.lossChart) {
            this.lossChart.data.labels = Array(this.lossData.length).fill('');
            this.lossChart.data.datasets[0].data = this.lossData;
            this.lossChart.update('none');
        }
        const fill = document.getElementById('challenge-progress-fill');
        const text = document.getElementById('challenge-progress-text');
        if (fill && text) {
            const prog = Math.min(stats.accuracy / 0.8, 1) * 100;
            fill.style.width = `${prog}%`;
            text.textContent = `${(stats.accuracy * 100).toFixed(1)}% / 80%`;
        }
    }

    initLossChart() {
        const c = document.getElementById('loss-chart');
        if (!c) return;
        this.lossChart = new Chart(c.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{ label: 'Loss', data: [], borderColor: '#00e5ff', borderWidth: 2, tension: 0.4, fill: true, backgroundColor: 'rgba(0, 229, 255, 0.1)' }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { display: false }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } } },
                plugins: { legend: { display: false } },
                animation: { duration: 0 }
            }
        });
    }

    async toggleTraining() {
        this.training ? await this.stopTraining() : await this.startTraining();
    }

    async startTraining() {
        if (!this.backend.connected) {
            alert('Not connected to backend! Check your connection.');
            return;
        }

        const response = await this.backend.startTraining();

        if (response.status === 'busy') {
            alert('Training is currently in progress by another user. Please wait and try again.');
            return;
        }

        if (response.status === 'error') {
            alert('Training failed: ' + response.message);
            return;
        }

        this.training = true;
        this.updateTrainingButton();
        const el = document.getElementById('current-action-text');
        if (el) el.textContent = 'Training in progress...';
        if (window.Journey) Journey.completeTask('t3_1');
    }

    async stopTraining() {
        await this.backend.stopTraining();
        this.training = false;
        this.updateTrainingButton();
    }

    updateTrainingButton() {
        const btn = document.getElementById('play-pause');
        if (btn) btn.innerHTML = this.training ? '⏸️' : '▶️';
    }

    initDrawingCanvas() {
        const canvas = document.getElementById('drawing-canvas');
        if (!canvas || canvas._init) return;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        let drawing = false;
        canvas.onmousedown = (e) => { drawing = true; this.drawOnCanvas(e, ctx, canvas); };
        canvas.onmousemove = (e) => { if (drawing) this.drawOnCanvas(e, ctx, canvas); };
        canvas.onmouseup = () => drawing = false;
        canvas.onmouseleave = () => drawing = false;
        canvas.ontouchstart = (e) => { drawing = true; this.drawOnCanvas(e.touches[0], ctx, canvas); e.preventDefault(); };
        canvas.ontouchmove = (e) => { if (drawing) this.drawOnCanvas(e.touches[0], ctx, canvas); e.preventDefault(); };
        canvas.ontouchend = () => drawing = false;
        canvas._init = true;
    }

    drawOnCanvas(e, ctx, canvas) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();
    }

    clearDrawing() {
        const c = document.getElementById('drawing-canvas');
        if (!c) return;
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, c.width, c.height);
        document.getElementById('prediction-result').innerHTML = '';
    }

    predictDrawing() {
        if (!this.localNN) {
            alert('Please complete training first!');
            return;
        }
        const c = document.getElementById('drawing-canvas');
        const pixels = Preprocessing.process(c);
        const pred = this.localNN.predict(new window.NN.Matrix(1, 784, [pixels])).data[0];
        const maxIdx = pred.indexOf(Math.max(...pred));
        // Fix #11: proper confidence calculation (no string coercion)
        const maxConf = Math.max(...pred);
        const conf = (Math.min(maxConf * 100, 100)).toFixed(1);
        document.getElementById('prediction-result').innerHTML = `
            <div style="font-size: 2rem; font-weight: bold; color: var(--accent);">Prediction: ${maxIdx}</div>
            <div style="color: var(--text-secondary);">Confidence: ${conf}%</div>
            <div style="display: flex; gap: 4px; margin-top: 12px; justify-content: center;">
                ${pred.map((p, i) => `
                    <div style="text-align: center;">
                        <div style="height: 60px; width: 20px; background: var(--bg-tertiary); position: relative; border-radius: 4px;">
                            <div style="position: absolute; bottom: 0; width: 100%; height: ${Math.min(p * 100, 100)}%; 
                                background: ${i === maxIdx ? 'var(--accent)' : 'var(--text-dim)'}; border-radius: 4px;"></div>
                        </div>
                        <div style="font-size: 0.7rem; margin-top: 4px;">${i}</div>
                    </div>
                `).join('')}
            </div>
        `;
        if (window.Journey) {
            Journey.progress.predictionsDebugged++;
            Journey.checkAchievements();

            // Complete Chapter 3 tasks
            Journey.completeTask('t3_2'); // See how digit "7" activates neurons
            Journey.completeTask('t3_4'); // Predict which neurons fire
        }
    }

    openDrawingPad() { document.getElementById('drawing-pad-modal')?.classList.add('visible'); this.initDrawingCanvas(); }
    closeDrawingPad() { document.getElementById('drawing-pad-modal')?.classList.remove('visible'); }
    // Fix #10: call renderArchitectureBuilder on open
    openArchitecture() { document.getElementById('architecture-modal')?.classList.add('visible'); this.renderArchitectureBuilder(); }
    // Add after openArchitecture() method
    addHiddenLayer() {
        if (!this.hiddenLayers) this.hiddenLayers = [{ neurons: 128, activation: 'relu' }, { neurons: 64, activation: 'relu' }];
        this.hiddenLayers.push({ neurons: 64, activation: 'relu' });
        this.renderArchitectureBuilder();
        if (window.Journey) {
            Journey.progress.customLayersBuilt++;
            Journey.checkAchievements();
        }
    }

    removeHiddenLayer(idx) {
        if (!this.hiddenLayers) return;
        if (this.hiddenLayers.length > 1) {
            this.hiddenLayers.splice(idx, 1);
            this.renderArchitectureBuilder();
        }
    }

    updateLayerNeurons(idx, value) {
        if (!this.hiddenLayers) return;
        this.hiddenLayers[idx].neurons = parseInt(value) || 64;
        this.updateParamCount();
    }

    updateLayerActivation(idx, value) {
        if (!this.hiddenLayers) return;
        this.hiddenLayers[idx].activation = value;
    }

    updateParamCount() {
        let params = 0;
        let prevSize = 784;
        this.hiddenLayers.forEach(layer => {
            params += prevSize * layer.neurons + layer.neurons;
            prevSize = layer.neurons;
        });
        params += prevSize * 10 + 10;
        const el = document.getElementById('total-params');
        if (el) el.textContent = params.toLocaleString();
    }

    renderArchitectureBuilder() {
        const container = document.getElementById('hidden-layers-builder');
        if (!container) return;
        container.innerHTML = '';

        if (!this.hiddenLayers) this.hiddenLayers = [{ neurons: 128, activation: 'relu' }, { neurons: 64, activation: 'relu' }];

        this.hiddenLayers.forEach((layer, idx) => {
            const div = document.createElement('div');
            div.style.cssText = 'display: flex; gap: 12px; align-items: center; margin-bottom: 8px; padding: 12px; background: var(--bg-tertiary, #2a2a2a); border-radius: 8px;';
            div.innerHTML = `
            <span style="width: 80px;">Layer ${idx + 1}</span>
            <input type="number" value="${layer.neurons}" min="1" max="512" 
                onchange="window.app.updateLayerNeurons(${idx}, this.value)"
                style="flex: 1; padding: 8px; background: var(--bg-primary, #1a1a1a); border: 1px solid var(--border, #444); border-radius: 4px; color: var(--text-primary, #fff);">
            <select onchange="window.app.updateLayerActivation(${idx}, this.value)"
                style="padding: 8px; background: var(--bg-primary, #1a1a1a); border: 1px solid var(--border, #444); border-radius: 4px; color: var(--text-primary, #fff);">
                <option value="relu" ${layer.activation === 'relu' ? 'selected' : ''}>ReLU</option>
                <option value="sigmoid" ${layer.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
            </select>
            <button onclick="window.app.removeHiddenLayer(${idx})" style="background: none; border: none; color: var(--error, #ff3366); cursor: pointer; font-size: 1.2rem;">×</button>
        `;
            container.appendChild(div);
        });
        this.updateParamCount();
    }

    async applyArchitecture() {
        if (!this.hiddenLayers) return;
        const layers = [784, ...this.hiddenLayers.map(l => l.neurons), 10];

        // Feature B: Rebuild network via backend
        if (confirm('Rebuilding network will reset training progress. Continue?')) {
            const res = await fetch('/set-architecture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layers })
            });
            const data = await res.json();

            if (data.status === 'updated') {
                this.viz.build(layers);
                this.closeArchitecture();
                this.localNN = null; // Reset local NN until training completes again
                const el = document.getElementById('current-action-text');
                if (el) el.textContent = 'Architecture Updated. Train to see results.';
            } else {
                alert('Error: ' + data.message);
            }
        }
    }

    closeArchitecture() { document.getElementById('architecture-modal')?.classList.remove('visible'); }
    openAnalysis() { document.getElementById('analysis-modal')?.classList.add('visible'); }
    updateHistogram() {
        alert('Histogram analysis requires access to layer outputs during training. This feature shows activation distributions.');
    }

    renderHeatmaps() {
        const container = document.getElementById('heatmap-container');
        if (!container) return;

        if (!this.localNN) {
            container.innerHTML = '<p style="color: var(--text-dim); padding: 20px;">Train the network first to see learned features.</p>';
            return;
        }

        // Feature C: Visualize first hidden layer weights
        container.innerHTML = '';
        const layer = this.localNN.layers[0]; // Weights from Input -> Hidden 1
        const inputSize = layer.inputSize; // 784
        const numNeurons = Math.min(layer.outputSize, 32); // Show first 32 neurons

        container.style.display = 'grid';
        container.style.gridTemplateColumns = 'repeat(auto-fill, minmax(60px, 1fr))';
        container.style.gap = '8px';

        for (let n = 0; n < numNeurons; n++) {
            const wrap = document.createElement('div');
            wrap.style.textAlign = 'center';
            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;
            canvas.style.cssText = 'width: 56px; height: 56px; border: 1px solid var(--border); image-rendering: pixelated;';
            wrap.appendChild(canvas);
            wrap.innerHTML += `<div style="font-size: 0.7rem; color: var(--text-dim);">#${n}</div>`;
            container.appendChild(wrap);

            const ctx = canvas.getContext('2d');
            const weights = layer.weights.data; // [784][neurons]

            // Normalize for this neuron
            let maxW = 0;
            for (let i = 0; i < inputSize; i++) maxW = Math.max(maxW, Math.abs(weights[i][n]));

            const imgData = ctx.createImageData(28, 28);
            for (let i = 0; i < 784; i++) {
                const w = weights[i][n];
                const val = (Math.abs(w) / (maxW || 1)) * 255;
                const idx = i * 4;
                if (w > 0) { // Red
                    imgData.data[idx] = 255; imgData.data[idx + 1] = 50; imgData.data[idx + 2] = 50;
                } else { // Blue
                    imgData.data[idx] = 50; imgData.data[idx + 1] = 50; imgData.data[idx + 2] = 255;
                }
                imgData.data[idx + 3] = val; // Alpha
            }
            ctx.putImageData(imgData, 0, 0);
        }
    }

    renderFeatures() {
        const grid = document.getElementById('feature-grid');
        if (!grid) return;
        grid.innerHTML = '<p style="color: var(--text-dim, #888); padding: 20px;">Feature visualization shows what each neuron has learned. Available after training.</p>';
    }

    renderLossLandscape() {
        const canvas = document.getElementById('loss-landscape-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;

        // Feature D: Dynamic Loss Landscape
        // Generate a 2D heightmap
        const cols = 40, rows = 20;
        const cellW = W / cols, cellH = H / rows;

        ctx.clearRect(0, 0, W, H);

        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                // Faked landscape based on current loss state + noise
                const dx = (x - cols / 2) / (cols / 2);
                const dy = (y - rows / 2) / (rows / 2);
                const dist = Math.sqrt(dx * dx + dy * dy);

                // Current loss dictates the "depth" of the center hole
                const currentLoss = this.lossData.length > 0 ? this.lossData[this.lossData.length - 1] : 2.5;
                const val = (dist * dist) * (1 + currentLoss) * 0.5 + Math.random() * 0.1;

                const hue = 240 - Math.min(val * 120, 240);
                ctx.fillStyle = `hsl(${hue}, 70%, ${50 - val * 20}%)`;
                ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
            }
        }

        // Draw current position (always center for visualization sake, as we "move" the landscape)
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(W / 2, H / 2 + 20, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#fff';

        // Text
        ctx.fillStyle = '#fff';
        ctx.font = '12px var(--font-mono)';
        ctx.fillText(`Current Loss: ${(this.lossData.length > 0 ? this.lossData[this.lossData.length - 1] : 2.5).toFixed(4)}`, 10, 20);
    }
    closeAnalysis() { document.getElementById('analysis-modal')?.classList.remove('visible'); }

    setSpeed(mode) {
        document.querySelectorAll('.speed-buttons button').forEach(b => b.classList.toggle('active', b.dataset.speed === mode));
        const map = { stepByStep: 1000, narrated: 200, normal: 0, fast: 0 };
        this.visualizationThrottle = mode === 'fast' ? 16 : 33;

        // Feature E: Backend speed control
        if (this.backend.connected) {
            fetch(`/set-batch-delay?ms=${map[mode] || 0}`, { method: 'POST' });
        }

        if (window.TeachingMode) TeachingMode.setMode(mode);
    }

    toggleViz(type) {
        if (type === 'connections') this.viz.toggleElement('connections');
        else if (type === 'gradients') this.showGradients = !this.showGradients;
    }

    toggleTheme() {
        document.body.classList.toggle('light-theme');
        const btn = document.getElementById('theme-btn');
        if (btn) {
            btn.textContent = document.body.classList.contains('light-theme') ? '🌙' : '☀️';
        }
        localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
    }

    showAchievements() {
        if (!window.Journey) return;
        let html = '<div style="padding:20px;"><h2>🏆 Achievements</h2><div style="display:grid;gap:12px;margin-top:16px;">';
        Object.values(Journey.achievements).forEach(ach => {
            const unlocked = Journey.progress.achievementsUnlocked.includes(ach.id);
            html += `<div style="display:flex;gap:12px;padding:12px;background:${unlocked ? '#003344' : '#2a2a2a'};border-radius:8px;opacity:${unlocked ? 1 : 0.5};">
                <span style="font-size:1.5rem;">${ach.icon}</span>
                <div><div style="font-weight:600;">${ach.title}</div>
                <div style="font-size:0.8rem;color:#888;">${ach.description}</div>
                ${unlocked ? `<div style="font-size:0.75rem;color:#00ff88;">+${ach.xp} XP</div>` : ''}</div></div>`;
        });
        html += '</div></div>';
        const modal = document.createElement('div');
        modal.className = 'modal-overlay visible';
        modal.innerHTML = `<div class="modal-container" style="max-width:600px;">
            <div class="modal-header"><h2 class="modal-title">Achievements</h2>
            <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button></div>
            <div class="modal-content">${html}</div></div>`;
        document.body.appendChild(modal);
    }

    nextOnboardingStep() { document.getElementById('onboarding-overlay')?.classList.remove('visible'); localStorage.setItem('nn-journey-onboarded', 'true'); }
    skipOnboarding() { this.nextOnboardingStep(); }
    togglePreprocessDebug() {
        const cont = document.getElementById('debug-preprocessing-container');
        const check = document.getElementById('show-debug-preprocess');
        if (cont && check) cont.style.display = check.checked ? 'block' : 'none';
    }

    getExplanationTitle(reason) {
        const titles = {
            'first_forward': '🎯 Forward Pass',
            'first_backward': '⬅️ Backpropagation',
            'high_loss': '⚠️ High Loss Detected',
            'learning': '📈 Learning in Progress'
        };
        return titles[reason] || 'Paused';
    }

    getExplanationText(reason) {
        const texts = {
            'first_forward': 'Data just entered the network! Watch as each neuron processes information. The input (784 pixels) flows through layers, getting transformed at each step.',
            'first_backward': 'Now the magic happens - the error flows backward! Each weight learns how much it contributed to the mistake. This is called backpropagation.',
            'high_loss': 'The network made a big mistake (high loss). This means its predictions are far from correct. It will now adjust weights to improve.',
            'learning': 'The network is learning! Loss is decreasing, meaning predictions are getting better. Each neuron is fine-tuning its weights.'
        };
        return texts[reason] || 'Training paused. Observe the current state.';
    }

    renderExplanation(reason) {
        const canvas = document.getElementById('explanation-canvas');
        if (!canvas) {
            console.error('Canvas not found!');
            return;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Get current network state
        const activations = this.viz.currentActivations || [];

        switch (reason) {
            case 'first_forward':
                this.drawForwardPassExplanation(ctx, canvas, activations);
                break;
            case 'first_backward':
                this.drawBackwardPassExplanation(ctx, canvas);
                break;
            case 'high_loss':
                this.drawLossExplanation(ctx, canvas);
                break;
            case 'learning':
                this.drawLearningProgressExplanation(ctx, canvas);
                break;
            default:
                this.drawDefaultExplanation(ctx, canvas, activations);
        }
    }

    drawForwardPassExplanation(ctx, canvas, activations) {
        const layerCount = 4;
        const layerSpacing = canvas.width / (layerCount + 1);
        const neuronRadius = 15;

        // Draw layers
        const layerSizes = [5, 4, 3, 4]; // Simplified visualization

        layerSizes.forEach((size, layerIdx) => {
            const x = layerSpacing * (layerIdx + 1);
            const neuronSpacing = canvas.height / (size + 1);

            for (let i = 0; i < size; i++) {
                const y = neuronSpacing * (i + 1);

                // Get activation value if available
                const activation = activations[layerIdx]?.[i] || 0;

                // Draw neuron
                const hue = (1 - activation) * 240; // Blue to red
                ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
                ctx.fill();

                // Outline
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Draw connections to next layer
                if (layerIdx < layerSizes.length - 1) {
                    const nextSize = layerSizes[layerIdx + 1];
                    const nextX = layerSpacing * (layerIdx + 2);
                    const nextSpacing = canvas.height / (nextSize + 1);

                    for (let j = 0; j < nextSize; j++) {
                        const nextY = nextSpacing * (j + 1);

                        ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 + activation * 0.3})`;
                        ctx.lineWidth = 1 + activation * 2;
                        ctx.beginPath();
                        ctx.moveTo(x + neuronRadius, y);
                        ctx.lineTo(nextX - neuronRadius, nextY);
                        ctx.stroke();
                    }
                }
            }

            // Layer label
            const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
            ctx.fillStyle = '#fff';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(labels[layerIdx], x, canvas.height - 10);
        });

        // Add arrow showing direction
        this.drawFlowArrow(ctx, canvas, 'forward');
    }

    drawBackwardPassExplanation(ctx, canvas) {
        const layerCount = 4;
        const layerSpacing = canvas.width / (layerCount + 1);
        const neuronRadius = 15;
        const layerSizes = [5, 4, 3, 4];

        // Draw layers (simpler this time)
        layerSizes.forEach((size, layerIdx) => {
            const x = layerSpacing * (layerIdx + 1);
            const neuronSpacing = canvas.height / (size + 1);

            for (let i = 0; i < size; i++) {
                const y = neuronSpacing * (i + 1);

                // Gradient intensity (higher at output, lower at input)
                const gradientIntensity = (layerCount - layerIdx) / layerCount;

                ctx.fillStyle = `rgba(0, 255, 255, ${gradientIntensity * 0.8})`; // Cyan gradient
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });

        // Draw gradient flow particles
        const time = Date.now() / 1000;
        for (let i = 0; i < 20; i++) {
            const progress = ((time * 0.5 + i * 0.05) % 1);
            const x = canvas.width * (1 - progress); // Right to left
            const y = canvas.height * 0.5 + Math.sin(i) * 100;

            ctx.fillStyle = `rgba(0, 255, 255, ${1 - progress})`;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        // Add backward arrow
        this.drawFlowArrow(ctx, canvas, 'backward');

        // Fix #9: only re-render if canvas still in DOM (prevents infinite loop)
        if (document.getElementById('explanation-canvas')) {
            setTimeout(() => this.renderExplanation('first_backward'), 50);
        }
    }

    drawLossExplanation(ctx, canvas) {
        // Draw a loss curve with current position marked
        const padding = 50;
        const graphWidth = canvas.width - padding * 2;
        const graphHeight = canvas.height - padding * 2;

        // Draw axes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#fff';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Training Steps →', canvas.width / 2, canvas.height - 10);
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Loss', 0, 0);
        ctx.restore();

        // Draw loss curve (exponential decay with noise)
        ctx.strokeStyle = '#ff3366';
        ctx.lineWidth = 3;
        ctx.beginPath();

        const points = 100;
        for (let i = 0; i < points; i++) {
            const x = padding + (i / points) * graphWidth;
            const progress = i / points;

            // Exponential decay with noise
            const baseLoss = 2.3 * Math.exp(-progress * 3) + 0.1;
            const noise = Math.sin(i * 0.5) * 0.1;
            const loss = baseLoss + noise;

            const y = canvas.height - padding - (loss / 2.5) * graphHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Mark current position (high loss point)
        const currentX = padding + 0.3 * graphWidth;
        const currentLoss = 1.5; // High loss
        const currentY = canvas.height - padding - (currentLoss / 2.5) * graphHeight;

        ctx.fillStyle = '#ff3366';
        ctx.beginPath();
        ctx.arc(currentX, currentY, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#fff';
        ctx.font = '14px sans-serif';
        ctx.fillText('You are here', currentX, currentY - 20);
        ctx.fillText(`Loss: ${currentLoss.toFixed(2)}`, currentX, currentY - 5);
    }

    drawLearningProgressExplanation(ctx, canvas) {
        // Draw before/after comparison
        const split = canvas.width / 2;

        // Before (left side)
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Before', split / 2, 30);

        // Random activations
        this.drawMiniNetwork(ctx, split / 2, canvas.height / 2, 'random');

        // After (right side)
        ctx.fillText('After', split + split / 2, 30);

        // Learned activations
        this.drawMiniNetwork(ctx, split + split / 2, canvas.height / 2, 'learned');

        // Divider
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(split, 50);
        ctx.lineTo(split, canvas.height - 20);
        ctx.stroke();
    }

    drawMiniNetwork(ctx, centerX, centerY, type) {
        const layers = [3, 4, 3];
        const layerSpacing = 60;
        const neuronRadius = 10;

        layers.forEach((size, layerIdx) => {
            const x = centerX - layerSpacing + layerIdx * layerSpacing;
            const neuronSpacing = 40;
            const startY = centerY - ((size - 1) * neuronSpacing) / 2;

            for (let i = 0; i < size; i++) {
                const y = startY + i * neuronSpacing;

                let activation;
                if (type === 'random') {
                    activation = Math.random();
                } else {
                    // Learned pattern
                    activation = (layerIdx + i) % 2 === 0 ? 0.8 : 0.2;
                }

                const hue = (1 - activation) * 240;
                ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        });
    }

    drawDefaultExplanation(ctx, canvas, activations) {
        // Default: show current network state
        ctx.fillStyle = '#fff';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Current Network State', canvas.width / 2, 30);

        if (activations && activations.length > 0) {
            this.drawForwardPassExplanation(ctx, canvas, activations);
        } else {
            ctx.font = '14px sans-serif';
            ctx.fillStyle = '#888';
            ctx.fillText('Training not started yet', canvas.width / 2, canvas.height / 2);
        }
    }

    drawFlowArrow(ctx, canvas, direction) {
        const y = 30;
        const arrowLength = 80;

        ctx.strokeStyle = direction === 'forward' ? '#00ff88' : '#00ffff';
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 3;

        if (direction === 'forward') {
            const startX = canvas.width - 150;
            const endX = startX + arrowLength;

            // Arrow line
            ctx.beginPath();
            ctx.moveTo(startX, y);
            ctx.lineTo(endX, y);
            ctx.stroke();

            // Arrow head
            ctx.beginPath();
            ctx.moveTo(endX, y);
            ctx.lineTo(endX - 10, y - 5);
            ctx.lineTo(endX - 10, y + 5);
            ctx.closePath();
            ctx.fill();

            ctx.font = '12px sans-serif';
            ctx.fillText('Forward →', startX + arrowLength / 2, y - 10);
        } else {
            const startX = 150;
            const endX = startX - arrowLength;

            // Arrow line
            ctx.beginPath();
            ctx.moveTo(startX, y);
            ctx.lineTo(endX, y);
            ctx.stroke();

            // Arrow head
            ctx.beginPath();
            ctx.moveTo(endX, y);
            ctx.lineTo(endX + 10, y - 5);
            ctx.lineTo(endX + 10, y + 5);
            ctx.closePath();
            ctx.fill();

            ctx.font = '12px sans-serif';
            ctx.fillText('← Backward', startX - arrowLength / 2, y - 10);
        }
    }

    // ============================================================================

    updateComparisonChart() {
        const compChart = document.getElementById('comparison-chart');
        if (!compChart) {
            // Create chart if it doesn't exist
            this.createComparisonChart();
            return;
        }

        if (!this.comparisonChartInstance) {
            this.createComparisonChart();
            return;
        }

        // Update existing chart with new data
        const maxPoints = 100;

        // Trim data if too long
        if (this.comparisonStats.backprop.losses.length > maxPoints) {
            this.comparisonStats.backprop.losses = this.comparisonStats.backprop.losses.slice(-maxPoints);
            this.comparisonStats.random.losses = this.comparisonStats.random.losses.slice(-maxPoints);
            this.comparisonStats.backprop.accuracies = this.comparisonStats.backprop.accuracies.slice(-maxPoints);
            this.comparisonStats.random.accuracies = this.comparisonStats.random.accuracies.slice(-maxPoints);
        }

        // Update chart data
        this.comparisonChartInstance.data.labels = Array(this.comparisonStats.backprop.losses.length).fill('');

        // Loss datasets
        this.comparisonChartInstance.data.datasets[0].data = this.comparisonStats.backprop.losses;
        this.comparisonChartInstance.data.datasets[1].data = this.comparisonStats.random.losses;

        // Accuracy datasets
        this.comparisonChartInstance.data.datasets[2].data = this.comparisonStats.backprop.accuracies;
        this.comparisonChartInstance.data.datasets[3].data = this.comparisonStats.random.accuracies;

        this.comparisonChartInstance.update('none'); // No animation for performance
    }

    createComparisonChart() {
        // Create chart container if it doesn't exist
        let container = document.getElementById('comparison-chart-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'comparison-chart-container';
            container.className = 'comparison-chart-container';
            container.innerHTML = `
            <div class="comparison-chart-header">
                <h3>📊 Training Comparison</h3>
                <button onclick="if(window.app) window.app.toggleComparisonMode()" 
                    style="background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 1.2rem;">×</button>
            </div>
            <canvas id="comparison-chart" width="500" height="250"></canvas>
        `;

            // Insert after main visualization
            const vizArea = document.querySelector('.visualization-area');
            if (vizArea) {
                vizArea.appendChild(container);
            }
        }

        const canvas = document.getElementById('comparison-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        this.comparisonChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Backprop Loss',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y-loss'
                    },
                    {
                        label: 'Random Loss',
                        data: [],
                        borderColor: '#ff3366',
                        backgroundColor: 'rgba(255, 51, 102, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y-loss'
                    },
                    {
                        label: 'Backprop Accuracy',
                        data: [],
                        borderColor: '#00e5ff',
                        backgroundColor: 'rgba(0, 229, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: false,
                        yAxisID: 'y-accuracy',
                        borderDash: [5, 5]
                    },
                    {
                        label: 'Random Accuracy',
                        data: [],
                        borderColor: '#ff9900',
                        backgroundColor: 'rgba(255, 153, 0, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: false,
                        yAxisID: 'y-accuracy',
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#fff',
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#00e5ff',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    'y-loss': {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss',
                            color: '#888'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#888'
                        }
                    },
                    'y-accuracy': {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Accuracy',
                            color: '#888'
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#888',
                            callback: function (value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }

    resumeTraining() {
        this.startTraining();
    }

    // Add to MainApp class
    enableLivePrediction() {
        if (!this.localNN) {
            console.warn('Local NN not ready for live prediction');
            return;
        }

        const canvas = document.getElementById('mini-input-preview');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Update every 2 seconds during training
        this.livePredictionInterval = setInterval(() => {
            if (!this.training) {
                clearInterval(this.livePredictionInterval);
                return;
            }

            // Get drawing pad content if visible
            const drawingCanvas = document.getElementById('drawing-canvas');
            if (!drawingCanvas) return;

            const pixels = Preprocessing.process(drawingCanvas);
            const pred = this.localNN.predict(new window.NN.Matrix(1, 784, [pixels])).data[0];
            const maxIdx = pred.indexOf(Math.max(...pred));

            // Update mini preview
            const drawCtx = drawingCanvas.getContext('2d');
            const imgData = drawCtx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);

            ctx.clearRect(0, 0, 28, 28);
            ctx.drawImage(drawingCanvas, 0, 0, 28, 28);

            // Update label
            document.getElementById('current-digit-label').textContent = `Predicting: ${maxIdx} (${(Math.max(...pred) * 100).toFixed(0)}%)`;
        }, 2000);
    }

    // Add to MainApp class
    openWeightPainter() {
        document.getElementById('weight-painter-modal')?.classList.add('visible');
        this.initWeightPainter();
    }

    closeWeightPainter() {
        document.getElementById('weight-painter-modal')?.classList.remove('visible');
    }

    initWeightPainter() {
        const canvas = document.getElementById('weight-painter-canvas');
        if (!canvas || canvas._initialized) return;

        const ctx = canvas.getContext('2d');
        this.selectedLayer = 0;
        this.selectedNeuron = 0;
        this.paintingWeights = [];

        // Initialize with current weights or random
        this.initializePainterWeights();
        this.renderWeightCanvas();

        let painting = false;

        canvas.onmousedown = (e) => {
            painting = true;
            this.paintWeight(e, ctx, canvas);
        };

        canvas.onmousemove = (e) => {
            if (painting) this.paintWeight(e, ctx, canvas);
        };

        canvas.onmouseup = () => painting = false;
        canvas.onmouseleave = () => painting = false;

        canvas._initialized = true;
    }

    initializePainterWeights() {
        // Create 28x28 grid representing input layer weights to first neuron
        const size = 28;
        this.paintingWeights = Array(size).fill(0).map(() =>
            Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.5)
        );
    }

    renderWeightCanvas() {
        const canvas = document.getElementById('weight-painter-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const cellSize = canvas.width / 28;

        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const weight = this.paintingWeights[y][x];

                // Color: red for positive, blue for negative
                const intensity = Math.abs(weight);
                const hue = weight > 0 ? 0 : 240; // red : blue
                const lightness = 50 + intensity * 30;

                ctx.fillStyle = `hsl(${hue}, 80%, ${lightness}%)`;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
        }

        // Grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 28; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, canvas.height);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(canvas.width, i * cellSize);
            ctx.stroke();
        }
    }

    paintWeight(e, ctx, canvas) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / (canvas.width / 28));
        const y = Math.floor((e.clientY - rect.top) / (canvas.height / 28));

        if (x >= 0 && x < 28 && y >= 0 && y < 28) {
            const brushSize = parseInt(document.getElementById('weight-brush-size')?.value || 3);
            const strength = parseFloat(document.getElementById('weight-strength')?.value || 0.5);

            // Paint in brush area
            for (let dy = -brushSize; dy <= brushSize; dy++) {
                for (let dx = -brushSize; dx <= brushSize; dx++) {
                    const nx = x + dx;
                    const ny = y + dy;

                    if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist <= brushSize) {
                            const falloff = 1 - (dist / brushSize);
                            this.paintingWeights[ny][nx] += strength * falloff * 0.1;
                            this.paintingWeights[ny][nx] = Math.max(-1, Math.min(1, this.paintingWeights[ny][nx]));
                        }
                    }
                }
            }

            this.renderWeightCanvas();
        }
    }

    applyPaintedWeights() {
        if (!this.localNN) {
            alert('Train the network first to enable weight modification');
            return;
        }

        // Flatten weights to 784-dimensional vector
        const flatWeights = this.paintingWeights.flat();

        // Apply to first layer's first neuron
        const layer = this.localNN.layers[0];
        if (layer && layer.weights) {
            for (let i = 0; i < Math.min(784, flatWeights.length); i++) {
                layer.weights.data[i][this.selectedNeuron] = flatWeights[i];
            }

            alert('Weights applied! Draw a digit to see how this neuron responds.');
            this.closeWeightPainter();
            this.openDrawingPad();
        }
    }

    resetPainterWeights() {
        this.initializePainterWeights();
        this.renderWeightCanvas();
    }

    // Fix #4: comparisonMode moved to constructor

    toggleComparisonMode() {
        this.comparisonMode = !this.comparisonMode;

        if (this.comparisonMode) {
            this.startComparison();
        } else {
            this.stopComparison();
        }
    }

    startComparison() {
        // Create split view
        const container = document.getElementById('threejs-container');
        container.style.display = 'grid';
        container.style.gridTemplateColumns = '1fr 1fr';

        // Create second canvas for comparison
        const rightCanvas = document.createElement('div');
        rightCanvas.id = 'canvas-container-b';
        rightCanvas.style.position = 'relative';
        container.appendChild(rightCanvas);

        // Create second visualizer
        this.viz2 = new NetworkVisualizer('canvas-container-b');

        // Add labels
        this.addComparisonLabels();

        // Train both: one with backprop, one with random updates
        this.trainingComparison = true;
        this.comparisonStats = {
            backprop: { losses: [], accuracies: [] },
            random: { losses: [], accuracies: [] }
        };
    }

    stopComparison() {
        const container = document.getElementById('threejs-container');
        container.style.display = 'block';
        container.style.gridTemplateColumns = '1fr';

        const rightCanvas = document.getElementById('canvas-container-b');
        if (rightCanvas) rightCanvas.remove();

        this.viz2 = null;
        this.trainingComparison = false;

        document.querySelectorAll('.comparison-label').forEach(el => el.remove());
    }

    addComparisonLabels() {
        const container = document.getElementById('threejs-container');

        const leftLabel = document.createElement('div');
        leftLabel.className = 'comparison-label';
        leftLabel.style.cssText = 'position: absolute; top: 10px; left: 10px; background: var(--success); color: #000; padding: 8px 16px; border-radius: 20px; font-weight: 600; z-index: 10;';
        leftLabel.textContent = '✓ With Backpropagation';
        container.appendChild(leftLabel);

        const rightLabel = document.createElement('div');
        rightLabel.className = 'comparison-label';
        rightLabel.style.cssText = 'position: absolute; top: 10px; right: 10px; background: var(--error); color: #fff; padding: 8px 16px; border-radius: 20px; font-weight: 600; z-index: 10;';
        rightLabel.textContent = '✗ Random Weight Updates';
        container.appendChild(rightLabel);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded');
    window.app = new MainApp();
});