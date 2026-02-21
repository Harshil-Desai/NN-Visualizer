/**
 * Neural Network Journey - Tooltip System
 * Context-aware tooltips with 3 difficulty levels
 */

// ============================================================================
// TOOLTIP CONTENT
// ============================================================================
const TOOLTIP_CONTENT = {
    // Neuron tooltips
    neuron: {
        beginner: 'A neuron is a tiny decision-maker. It receives inputs and produces an output.',
        intermediate: 'Computes: output = activation(Σ(input × weight) + bias)',
        advanced: 'Activation: {activation}. Pre-activation z = {preActivation}',
        interactive: true
    },

    // Weight/Connection tooltips
    weight: {
        beginner: 'This connection controls how much influence one neuron has on another.',
        intermediate: 'Weight: {value}. Red = positive (excitatory), Blue = negative (inhibitory)',
        advanced: 'Weight: {value}, Gradient: {gradient}. Update: Δw = -lr × gradient',
        interactive: false
    },

    // Layer tooltips
    inputLayer: {
        beginner: 'Input Layer: Each neuron represents one pixel of the image.',
        intermediate: '784 neurons for 28×28 image. Values normalized to [0,1].',
        advanced: 'Input shape: (batch_size, 784). No learnable parameters.'
    },
    hiddenLayer: {
        beginner: 'Hidden Layer: These neurons learn to detect patterns like edges and curves.',
        intermediate: 'Dense layer with {neurons} neurons and {activation} activation.',
        advanced: 'Parameters: {neurons}×{inputs} weights + {neurons} biases = {params} total'
    },
    outputLayer: {
        beginner: 'Output Layer: 10 neurons, one for each digit (0-9). Brightest = prediction.',
        intermediate: 'Softmax activation converts scores to probabilities summing to 1.',
        advanced: 'Softmax: P(y=k) = exp(zₖ) / Σexp(zᵢ). Cross-entropy loss used.'
    },

    // Control tooltips
    learningRate: {
        beginner: 'Learning Rate controls how big the weight changes are. Too high = unstable, too low = slow.',
        intermediate: 'Typical values: 0.001 to 0.1. Higher risks overshooting minima.',
        advanced: 'Hyperparameter α in w ← w - α∇L. Consider adaptive methods (Adam) for complex landscapes.',
        demo: 'learningRateDemo'
    },
    batchSize: {
        beginner: 'Batch Size: How many examples to process before updating weights.',
        intermediate: 'Trade-off: Larger = more stable gradients, smaller = faster iteration.',
        advanced: 'SGD uses batch=1, full-batch uses all data. Mini-batch (16-128) is common compromise.'
    },
    epochs: {
        beginner: 'Epoch: One complete pass through all training data.',
        intermediate: '10 epochs = network sees each digit 10 times. More epochs can lead to overfitting.',
        advanced: 'Monitor validation loss to detect overfitting. Use early stopping or regularization.'
    },

    // Metric tooltips
    loss: {
        beginner: 'Loss measures how wrong the prediction was. Goal: make this number as small as possible!',
        intermediate: 'Cross-entropy loss: L = -log(predicted probability of correct class)',
        advanced: 'L = -Σyₖlog(ŷₖ), gradient ∂L/∂z = ŷ - y for softmax-CE combination.'
    },
    accuracy: {
        beginner: 'Accuracy: What percentage of predictions are correct.',
        intermediate: 'Train accuracy can be misleading - always check test accuracy for true performance.',
        advanced: 'Accuracy = TP+TN / Total. Consider F1, precision, recall for imbalanced classes.'
    },

    // Visualization tooltips
    gradientParticles: {
        beginner: 'These particles show the "blame" flowing backward. Each weight learns from the error.',
        intermediate: 'Gradient magnitude shown by particle density. Brighter = larger gradient.',
        advanced: 'Visualizing ∂L/∂aˡ propagating via chain rule: ∂L/∂aˡ⁻¹ = Wˡᵀ × δˡ'
    },
    activationColor: {
        beginner: 'Bright = this neuron is "excited" and firing. Dark = this neuron is quiet.',
        intermediate: 'Color maps activation value. Blue (0) → Red (1) for normalized activations.',
        advanced: 'HSL color mapping: H = 0.6×(1-activation), S = 1, L = 0.3 + 0.4×activation'
    }
};

// ============================================================================
// TOOLTIP MANAGER
// ============================================================================
class TooltipManager {
    constructor() {
        this.currentTooltip = null;
        this.difficulty = 'beginner';
        this.pinned = [];
        this.history = [];
        this.maxHistory = 10;

        this.createTooltipElements();
        this.setupGlobalListeners();
    }

    createTooltipElements() {
        // Main tooltip container
        const tooltip = document.createElement('div');
        tooltip.id = 'journey-tooltip';
        tooltip.className = 'journey-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <span class="tooltip-title"></span>
                <button class="tooltip-pin" title="Pin this tooltip">📌</button>
                <button class="tooltip-close">×</button>
            </div>
            <div class="tooltip-body"></div>
            <div class="tooltip-footer">
                <button class="tooltip-learn-more">Learn More</button>
            </div>
        `;
        document.body.appendChild(tooltip);

        // Pinned tooltips container
        const pinned = document.createElement('div');
        pinned.id = 'pinned-tooltips';
        pinned.className = 'pinned-tooltips';
        document.body.appendChild(pinned);

        this.tooltipEl = tooltip;
        this.pinnedContainer = pinned;

        // Event listeners
        tooltip.querySelector('.tooltip-close').onclick = () => this.hide();
        tooltip.querySelector('.tooltip-pin').onclick = () => this.pinCurrent();
    }

    setupGlobalListeners() {
        document.addEventListener('click', (e) => {
            if (!this.tooltipEl.contains(e.target) &&
                !e.target.closest('[data-tooltip]')) {
                this.hide();
            }
        });
    }

    setDifficulty(level) {
        this.difficulty = level;
    }

    show(type, data = {}, targetElement = null) {
        const content = TOOLTIP_CONTENT[type];
        if (!content) return;

        // Get appropriate text for difficulty level
        let text = content[this.difficulty] || content.beginner;

        // Replace placeholders with actual data
        text = text.replace(/\{(\w+)\}/g, (match, key) => {
            return data[key] !== undefined ? data[key] : match;
        });

        // Update tooltip content
        this.tooltipEl.querySelector('.tooltip-title').textContent = this.formatTitle(type);
        this.tooltipEl.querySelector('.tooltip-body').innerHTML = `
            <p>${text}</p>
            ${content.interactive ? '<p class="tooltip-hint">Click for more details</p>' : ''}
        `;

        // Position tooltip
        if (targetElement) {
            this.positionNear(targetElement);
        } else if (data.x !== undefined && data.y !== undefined) {
            this.positionAt(data.x, data.y);
        }

        this.tooltipEl.classList.add('visible');
        this.currentTooltip = { type, data, content };

        // Update learn more button
        const learnMoreBtn = this.tooltipEl.querySelector('.tooltip-learn-more');
        if (content.demo) {
            learnMoreBtn.style.display = 'block';
            learnMoreBtn.onclick = () => this.openDemo(content.demo);
        } else {
            learnMoreBtn.style.display = 'none';
        }

        // Track for Journey system
        if (window.Journey) {
            window.Journey.trackTooltipView();
        }

        // Add to history
        this.addToHistory(type, text);
    }

    hide() {
        this.tooltipEl.classList.remove('visible');
        this.currentTooltip = null;
    }

    positionNear(element) {
        const rect = element.getBoundingClientRect();
        const tooltipRect = this.tooltipEl.getBoundingClientRect();

        let left = rect.right + 10;
        let top = rect.top;

        // Keep on screen
        if (left + tooltipRect.width > window.innerWidth) {
            left = rect.left - tooltipRect.width - 10;
        }
        if (top + tooltipRect.height > window.innerHeight) {
            top = window.innerHeight - tooltipRect.height - 10;
        }

        this.tooltipEl.style.left = `${left}px`;
        this.tooltipEl.style.top = `${top}px`;
    }

    positionAt(x, y) {
        this.tooltipEl.style.left = `${x + 15}px`;
        this.tooltipEl.style.top = `${y + 15}px`;
    }

    formatTitle(type) {
        return type.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
    }

    pinCurrent() {
        if (!this.currentTooltip) return;

        const pinnedEl = document.createElement('div');
        pinnedEl.className = 'pinned-tooltip';
        pinnedEl.innerHTML = `
            <div class="pinned-header">
                <span>${this.formatTitle(this.currentTooltip.type)}</span>
                <button class="pinned-close">×</button>
            </div>
            <div class="pinned-body">${this.tooltipEl.querySelector('.tooltip-body').innerHTML}</div>
        `;
        pinnedEl.querySelector('.pinned-close').onclick = () => pinnedEl.remove();

        this.pinnedContainer.appendChild(pinnedEl);
        this.pinned.push(this.currentTooltip.type);
        this.hide();
    }

    addToHistory(type, text) {
        this.history.unshift({ type, text, time: Date.now() });
        if (this.history.length > this.maxHistory) {
            this.history.pop();
        }
    }

    openDemo(demoName) {
        // Trigger demo modal
        if (window.Modals) {
            window.Modals.open(demoName);
        }
    }

    // Convenience methods for common tooltip triggers
    showForNeuron(layerIdx, neuronIdx, activation, x, y) {
        this.show('neuron', {
            layer: layerIdx,
            neuron: neuronIdx,
            activation: activation.toFixed(4),
            x, y
        });
    }

    showForWeight(fromLayer, toLayer, value, gradient, x, y) {
        this.show('weight', {
            fromLayer,
            toLayer,
            value: value.toFixed(4),
            gradient: gradient ? gradient.toFixed(6) : 'N/A',
            x, y
        });
    }

    showForControl(controlName, element) {
        this.show(controlName, {}, element);
    }
}

// ============================================================================
// MODAL SYSTEM
// ============================================================================
const MODAL_CONTENT = {
    learningRateDemo: {
        title: 'Learning Rate Explained',
        content: `
            <div class="modal-section">
                <h3>🎯 What is Learning Rate?</h3>
                <p>Learning rate controls how big each step is when updating weights.</p>
                
                <div class="demo-container">
                    <canvas id="lr-demo-canvas" width="400" height="200"></canvas>
                    <div class="demo-controls">
                        <label>Learning Rate: <input type="range" id="lr-demo-slider" min="0.001" max="1" step="0.01" value="0.1"></label>
                        <span id="lr-demo-value">0.1</span>
                        <button id="lr-demo-reset">Reset</button>
                    </div>
                </div>
                
                <h4>Try it!</h4>
                <ul>
                    <li><strong>Too small (0.001)</strong>: Watch the ball crawl slowly</li>
                    <li><strong>Just right (0.01-0.1)</strong>: Smooth descent to minimum</li>
                    <li><strong>Too large (0.5+)</strong>: Ball overshoots and oscillates!</li>
                </ul>
            </div>
        `,
        onOpen: (modal) => {
            // Initialize demo visualization
            const canvas = modal.querySelector('#lr-demo-canvas');
            const slider = modal.querySelector('#lr-demo-slider');
            const valueDisplay = modal.querySelector('#lr-demo-value');

            if (canvas && slider) {
                slider.oninput = () => {
                    valueDisplay.textContent = slider.value;
                };
            }
        }
    },

    whatIsBackprop: {
        title: 'How Does Backpropagation Work?',
        content: `
            <div class="modal-section">
                <h3>📞 The Telephone Game Analogy</h3>
                <p>Imagine a line of people playing telephone. The last person gets the wrong message.</p>
                <p>To fix it, you work <strong>BACKWARD</strong>, asking each person: "How much did YOU mess up the message?"</p>
                <p>That's backpropagation – working backward to find who's responsible for the error.</p>
            </div>
            
            <div class="modal-section">
                <h3>🔢 The Steps</h3>
                <ol>
                    <li><strong>Forward pass</strong>: Calculate prediction</li>
                    <li><strong>Calculate error</strong>: How wrong were we?</li>
                    <li><strong>Backward pass</strong>: For each weight, ask "If I changed you, how much would error decrease?"</li>
                    <li><strong>Update</strong>: Adjust weights in that direction</li>
                </ol>
            </div>
            
            <div class="modal-section">
                <h3>🎬 Watch It Happen</h3>
                <p>In the visualizer, cyan particles represent gradients flowing backward. Watch them carry "blame" from output to input!</p>
            </div>
        `
    },

    activationFunctions: {
        title: 'Why Activation Functions?',
        content: `
            <div class="modal-section">
                <h3>🤔 The Problem</h3>
                <p>Without activation functions, even a 100-layer network is just doing linear math.</p>
                <p><code>Linear + Linear + Linear = Still Linear</code></p>
                <p>The network can only draw straight lines to separate data!</p>
            </div>
            
            <div class="modal-section">
                <h3>💡 The Solution</h3>
                <p>Activation functions add <strong>non-linearity</strong>, letting the network learn curves and complex patterns.</p>
                
                <div class="function-comparison">
                    <div class="function-card">
                        <h4>ReLU</h4>
                        <code>max(0, x)</code>
                        <p>Simple & fast. Most popular for hidden layers.</p>
                    </div>
                    <div class="function-card">
                        <h4>Sigmoid</h4>
                        <code>1 / (1 + e^-x)</code>
                        <p>Squashes to [0,1]. Good for probabilities.</p>
                    </div>
                    <div class="function-card">
                        <h4>Softmax</h4>
                        <code>e^x / Σe^x</code>
                        <p>Outputs sum to 1. Perfect for classification.</p>
                    </div>
                </div>
            </div>
        `
    }
};

class ModalManager {
    constructor() {
        this.currentModal = null;
        this.createModalContainer();
    }

    createModalContainer() {
        const overlay = document.createElement('div');
        overlay.id = 'modal-overlay';
        overlay.className = 'modal-overlay';
        overlay.innerHTML = `
            <div class="modal-container">
                <div class="modal-header">
                    <h2 class="modal-title"></h2>
                    <button class="modal-close">×</button>
                </div>
                <div class="modal-content"></div>
                <div class="modal-footer">
                    <button class="modal-btn-secondary">Previous</button>
                    <button class="modal-btn-primary">Got it!</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);

        overlay.querySelector('.modal-close').onclick = () => this.close();
        overlay.querySelector('.modal-btn-primary').onclick = () => this.close();
        overlay.onclick = (e) => {
            if (e.target === overlay) this.close();
        };

        this.overlay = overlay;
        this.container = overlay.querySelector('.modal-container');
    }

    open(modalId) {
        const modal = MODAL_CONTENT[modalId];
        if (!modal) return;

        this.overlay.querySelector('.modal-title').textContent = modal.title;
        this.overlay.querySelector('.modal-content').innerHTML = modal.content;

        this.overlay.classList.add('visible');
        this.currentModal = modalId;

        if (modal.onOpen) {
            modal.onOpen(this.overlay);
        }
    }

    close() {
        this.overlay.classList.remove('visible');
        this.currentModal = null;
    }
}

// ============================================================================
// GLOBAL INSTANCES - Initialize after DOM ready
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    window.Tooltips = new TooltipManager();
    window.Modals = new ModalManager();
});
