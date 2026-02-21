/**
 * Neural Vision - Custom Matrix Library
 * A robust, zero-dependency linear algebra library for neural networks.
 * Optimized for clarity and error reporting.
 */

class Matrix {
    /**
     * Create a new Matrix.
     * @param {number} rows - Number of rows.
     * @param {number} cols - Number of columns.
     * @param {number[][]} [data] - Optional 2D array of data. If provided, checks dimensions.
     */
    constructor(rows, cols, data = null) {
        if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1) {
            throw new Error(`Invalid Matrix dimensions: ${rows}x${cols}. Rows and Cols must be positive integers.`);
        }
        this.rows = rows;
        this.cols = cols;

        if (data) {
            if (!Array.isArray(data) || data.length !== rows || (rows > 0 && data[0].length !== cols)) {
                throw new Error(`Matrix data mismatch. Expected ${rows}x${cols}, but got ${data.length}x${data.length > 0 ? data[0].length : 'NaN'}`);
            }
            this.data = data;
        } else {
            this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
        }
    }

    /**
     * Create a Matrix from a 2D array.
     * @param {number[][]} arr 
     * @returns {Matrix}
     */
    static fromArray(arr) {
        if (!Array.isArray(arr) || !Array.isArray(arr[0])) {
            throw new Error('fromArray expected a 2D array.');
        }
        return new Matrix(arr.length, arr[0].length, arr);
    }

    /**
     * Matrix Multiplication (Dot Product): A x B
     * @param {Matrix} a 
     * @param {Matrix} b 
     * @returns {Matrix} Result of A * B
     */
    static multiply(a, b) {
        if (a.cols !== b.rows) {
            throw new Error(`Dimension Mismatch in multiply: Cannot multiply ${a.rows}x${a.cols} and ${b.rows}x${b.cols}. Inner dimensions must match (${a.cols} != ${b.rows}).`);
        }

        // Naive O(n^3) implementation - sufficient for basic MNIST
        // Future optimization: Cache locality improvements or WebGPU
        const result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.cols; j++) {
                let sum = 0;
                // Unrolling loop slightly for small gain? No, keeping simple for now.
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    /**
     * Element-wise Addition with support for Broadcasting vectors.
     * If B is a row vector (1 x cols), it adds B to every row of A.
     * @param {Matrix} a 
     * @param {Matrix} b 
     */
    static add(a, b) {
        const result = new Matrix(a.rows, a.cols);

        if (a.rows === b.rows && a.cols === b.cols) {
            // Standard element-wise add
            for (let i = 0; i < a.rows; i++) {
                for (let j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] + b.data[i][j];
                }
            }
        } else if (b.rows === 1 && b.cols === a.cols) {
            // Broadcasting: Add row vector `b` to every row of `a`
            // Specific use case: Adding biases to weights output
            for (let i = 0; i < a.rows; i++) {
                for (let j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] + b.data[0][j];
                }
            }
        } else {
            throw new Error(`Dimension Mismatch in add: Cannot add ${a.rows}x${a.cols} and ${b.rows}x${b.cols} (Broadcasting only supported for 1xN vectors).`);
        }
        return result;
    }

    /**
     * Element-wise Subtraction (A - B)
     */
    static subtract(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error(`Dimension Mismatch in subtract: ${a.rows}x${a.cols} vs ${b.rows}x${b.cols}`);
        }
        const result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    /**
     * Hadamard Product (Element-wise Multiplication) using static map pattern
     */
    static hadamard(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error(`Dimension Mismatch in hadamard: ${a.rows}x${a.cols} vs ${b.rows}x${b.cols}`);
        }
        return Matrix.map(a, (val, i, j) => val * b.data[i][j]);
    }

    /**
     * Scalar Multiplication
     */
    static multiplyScalar(m, scalar) {
        return Matrix.map(m, val => val * scalar);
    }

    /**
     * Transpose the matrix (swap rows and columns)
     */
    transpose() {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    /**
     * Functional map: apply `fn` to every element, returning a new Matrix.
     * @param {Matrix} m 
     * @param {function} fn (value, row, col) => newValue
     */
    static map(m, fn) {
        const result = new Matrix(m.rows, m.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                result.data[i][j] = fn(m.data[i][j], i, j);
            }
        }
        return result;
    }

    /**
     * Returns a flattened array of the data
     */
    toArray() {
        return this.data.flat();
    }

    /**
     * Check for NaNs or Infinities.
     */
    static isValid(matrix) {
        if (!matrix || !matrix.data) return false;
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                if (!Number.isFinite(matrix.data[i][j])) return false;
            }
        }
        return true;
    }

    /**
     * Standard Normal Distribution (Gaussian)
     * Box-Muller transform
     */
    static randn() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    /**
     * He Initialization (ReLU)
     */
    static heInit(rows, cols) {
        const std = Math.sqrt(2 / rows);
        const m = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                m.data[i][j] = Matrix.randn() * std;
            }
        }
        return m;
    }

    /**
     * Xavier/Glorot Initialization (Sigmoid/Tanh)
     */
    static xavierInit(rows, cols) {
        const std = Math.sqrt(2 / (rows + cols));
        const m = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                m.data[i][j] = Matrix.randn() * std;
            }
        }
        return m;
    }
}

// Export to window for browser usage without modules
window.Matrix = Matrix;
