/**
 * Global Logger Utility
 */
const Logger = {
    debugMode: false,
    log: (msg, ...args) => { if (Logger.debugMode) console.log(`[NN-DEBUG] ${msg}`, ...args); },
    error: (msg, ...args) => console.error(`[NN-ERROR] ${msg}`, ...args),
    warn: (msg, ...args) => console.warn(`[NN-WARN] ${msg}`, ...args)
};

// Matrix class is now imported from matrix.js

/**
 * Activations
 */
const Activations = {
    sigmoid: {
        fn: x => 1 / (1 + Math.exp(-x)),
        df: y => y * (1 - y) // Derivative of sigmoid given its output y
    },
    relu: {
        fn: x => Math.max(0, x),
        df: x => (x > 0 ? 1 : 0) // Derivative of relu given its input x
    },
    softmax: {
        fn: (matrix) => {
            // Stable Softmax implementation using the Max-Subtraction Trick.
            // Mathematical justification: softmax(x) = softmax(x - c)
            // By subtracting max(x) from all elements, we ensure the largest value is 0.
            // This prevents Math.exp(x) from overflowing to Infinity for large x.
            const result = new Matrix(matrix.rows, matrix.cols);
            for (let i = 0; i < matrix.rows; i++) {
                let max = -Infinity;
                for (let j = 0; j < matrix.cols; j++) {
                    if (matrix.data[i][j] > max) max = matrix.data[i][j];
                }
                let sum = 0;
                const row = [];
                for (let j = 0; j < matrix.cols; j++) {
                    // exp(x - max) is always <= 1, preventing overflow
                    row[j] = Math.exp(matrix.data[i][j] - max);
                    sum += row[j];
                }
                for (let j = 0; j < matrix.cols; j++) {
                    result.data[i][j] = row[j] / sum;
                }
            }
            return result;
        },
        // Softmax derivative is typically handled in Loss (Cross-Entropy)
    }
};

/**
 * Layers
 */
class Layer {
    constructor() {
        this.input = null;
        this.output = null;
    }
    forward() { throw new Error('Forward not implemented'); }
    backward() { throw new Error('Backward not implemented'); }
}

class Dense extends Layer {
    constructor(inputSize, outputSize, activation = 'relu') {
        super();
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        // Weight Initialization
        if (activation === 'relu') {
            this.weights = Matrix.heInit(inputSize, outputSize);
        } else {
            this.weights = Matrix.xavierInit(inputSize, outputSize);
        }

        this.biases = new Matrix(1, outputSize);
        this.activationName = activation;
        this.activation = Activations[activation];

        // Caches for gradients
        this.weightsGrad = null;
        this.biasesGrad = null;
    }

    forward(input) {
        this.input = input; // (batch, inputSize)

        // Z = XW + B
        this.z = Matrix.add(Matrix.multiply(input, this.weights), this.biases);

        // A = Act(Z)
        if (this.activationName === 'softmax') {
            this.output = this.activation.fn(this.z);
        } else {
            this.output = Matrix.map(this.z, this.activation.fn);
        }

        return this.output;
    }

    backward(gradOutput, learningRate) {
        // gradOutput is dL/dA

        let delta;
        if (this.activationName === 'softmax') {
            // Special case: Softmax + CrossEntropy is usually passed as dL/dZ = (A - Y)
            // But if we separate them:
            delta = gradOutput; // Assuming gradOutput is already dL/dZ for Softmax+CE
        } else {
            // dL/dZ = dL/dA * dA/dZ
            const dz = Matrix.map(this.z, this.activation.df);
            delta = Matrix.hadamard(gradOutput, dz);
        }

        // dL/dW = X^T * dL/dZ
        this.weightsGrad = Matrix.multiply(this.input.transpose(), delta);

        // dL/dB = sum(dL/dZ) across batch
        const biasG = new Matrix(1, this.outputSize);
        for (let i = 0; i < delta.rows; i++) {
            for (let j = 0; j < delta.cols; j++) {
                biasG.data[0][j] += delta.data[i][j];
            }
        }
        this.biasesGrad = biasG;

        // dL/dX_prev = dL/dZ * W^T
        const gradInput = Matrix.multiply(delta, this.weights.transpose());

        // SGD Update
        this.weights = Matrix.subtract(this.weights, Matrix.multiplyScalar(this.weightsGrad, learningRate));
        this.biases = Matrix.subtract(this.biases, Matrix.multiplyScalar(this.biasesGrad, learningRate));

        return gradInput;
    }

    toJSON() {
        return {
            type: 'Dense',
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            activation: this.activationName,
            weights: this.weights.data,
            biases: this.biases.data
        };
    }

    static fromJSON(data) {
        const layer = new Dense(data.inputSize, data.outputSize, data.activation);
        layer.weights = new Matrix(data.inputSize, data.outputSize, data.weights);
        layer.biases = new Matrix(1, data.outputSize, data.biases);
        return layer;
    }
}

/**
 * Dataset Helper Class
 */
class Dataset {
    constructor(images, labels, numClasses = 10) {
        this.images = images.map(img => img.map(p => p / 255.0)); // Normalize
        this.labels = labels;
        this.numClasses = numClasses;
        this.size = images.length;
    }

    getBatch(batchSize) {
        const batchX = new Matrix(batchSize, this.images[0].length);
        const batchY = new Matrix(batchSize, this.numClasses);

        for (let i = 0; i < batchSize; i++) {
            const idx = Math.floor(Math.random() * this.size);
            batchX.data[i] = [...this.images[idx]];

            // One-hot encode
            const label = this.labels[idx];
            batchY.data[i][label] = 1;
        }

        return { x: batchX, y: batchY };
    }

    getRandomSample() {
        const idx = Math.floor(Math.random() * this.size);
        return {
            x: new Matrix(1, this.images[0].length, [[...this.images[idx]]]),
            label: this.labels[idx]
        };
    }
}

/**
 * Neural Network Class (Updated with hooks)
 */
class NeuralNetwork {
    constructor(layers = []) {
        this.layers = layers;
        this.onStep = null; // Hook for visualization
    }

    add(layer) {
        this.layers.push(layer);
    }

    forward(input) {
        let current = input;
        for (const layer of this.layers) {
            current = layer.forward(current);
        }
        return current;
    }

    /**
     * Training Step: Forward + Backward
     */
    train(xBatch, yBatch, learningRate) {
        try {
            // 1. Forward
            const prediction = this.forward(xBatch);
            if (!Matrix.isValid(prediction)) throw new Error("NaN or Infinity detected in forward pass.");

            // 2. Initial Gradient (dL/dZ)
            const deltaL = Matrix.map(prediction, (val, i, j) => {
                return (val - yBatch.data[i][j]) / xBatch.rows;
            });

            // 3. Backward
            let grad = deltaL;
            const gradients = [];
            for (let i = this.layers.length - 1; i >= 0; i--) {
                const layerGrad = grad;
                grad = this.layers[i].backward(grad, learningRate);
                gradients.unshift(layerGrad);

                if (!Matrix.isValid(this.layers[i].weights)) {
                    throw new Error(`NaN detected in Layer ${i} weights. Try lowering learning rate.`);
                }
            }

            // Trigger visualization hook
            if (this.onStep) {
                this.onStep({
                    prediction,
                    activations: this.layers.map(l => l.output),
                    gradients: gradients,
                    weights: this.layers.map(l => l.weights)
                });
            }

            return prediction;
        } catch (e) {
            Logger.error("Training Error: " + e.message);
            throw e; // Bubble up to UI
        }
    }

    predict(input) {
        return this.forward(input);
    }

    toJSON() {
        return {
            layers: this.layers.map(l => l.toJSON())
        };
    }

    static fromJSON(data) {
        const nn = new NeuralNetwork();
        nn.layers = data.layers.map(lData => {
            if (lData.type === 'Dense') {
                return Dense.fromJSON(lData);
            }
            throw new Error('Unknown layer type: ' + lData.type);
        });
        return nn;
    }
}

// Export for browser
window.NN = { Matrix, NeuralNetwork, Dense, Activations, Dataset, Logger };
