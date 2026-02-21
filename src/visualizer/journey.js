/**
 * Neural Network Journey - Learning System
 * Manages chapters, progress, achievements, and gamification
 */

// ============================================================================
// CHAPTER DEFINITIONS
// ============================================================================
const CHAPTERS = [
    {
        id: 'chapter1',
        title: 'What is a Neuron?',
        icon: '🧠',
        duration: '5-10 min',
        concepts: ['neuron', 'activation', 'weights', 'bias'],
        description: 'Learn the fundamental building block of neural networks.',
        tasks: [
            { id: 't1_1', text: 'Watch a single neuron activate', type: 'observe' },
            { id: 't1_2', text: 'Adjust weights manually', type: 'interact' },
            { id: 't1_3', text: 'See how bias shifts the decision', type: 'observe' },
            { id: 't1_4', text: 'Quiz: Predict neuron output', type: 'quiz' }
        ],
        unlocks: ['chapter2']
    },
    {
        id: 'chapter2',
        title: 'Building a Layer',
        icon: '🏗️',
        duration: '10-15 min',
        concepts: ['layer', 'dense', 'matrix', 'activationFunctions'],
        description: 'Connect neurons together to form powerful layers.',
        tasks: [
            { id: 't2_1', text: 'Connect 3 neurons to form a layer', type: 'interact' },
            { id: 't2_2', text: 'Understand activation functions', type: 'learn' },
            { id: 't2_3', text: 'Compare ReLU vs Sigmoid', type: 'compare' },
            { id: 't2_4', text: 'Design your own 2-layer network', type: 'build' }
        ],
        unlocks: ['chapter3'],
        requires: ['chapter1']
    },
    {
        id: 'chapter3',
        title: 'The Forward Pass',
        icon: '➡️',
        duration: '15-20 min',
        concepts: ['forward', 'normalization', 'softmax', 'prediction'],
        description: 'Trace data as it flows through the network.',
        tasks: [
            { id: 't3_1', text: 'Trace data in slow motion', type: 'observe' },
            { id: 't3_2', text: 'See how digit "7" activates neurons', type: 'observe' },
            { id: 't3_3', text: 'Understand softmax probabilities', type: 'learn' },
            { id: 't3_4', text: 'Predict which neurons fire for "3"', type: 'quiz' }
        ],
        unlocks: ['chapter4'],
        requires: ['chapter2']
    },
    {
        id: 'chapter4',
        title: 'Backpropagation',
        icon: '⬅️',
        duration: '20-30 min',
        concepts: ['loss', 'gradient', 'backprop', 'chainRule'],
        description: 'The magic of learning - how errors teach the network.',
        tasks: [
            { id: 't4_1', text: 'See what happens on a mistake', type: 'observe' },
            { id: 't4_2', text: 'Watch gradients flow backward', type: 'observe' },
            { id: 't4_3', text: 'Understand blame assignment', type: 'learn' },
            { id: 't4_4', text: 'Adjust learning rate interactively', type: 'interact' }
        ],
        unlocks: ['chapter5'],
        requires: ['chapter3']
    },
    {
        id: 'chapter5',
        title: 'Training End-to-End',
        icon: '🎓',
        duration: '30+ min',
        concepts: ['epochs', 'batches', 'overfitting', 'generalization'],
        description: 'Put it all together and train a real network.',
        tasks: [
            { id: 't5_1', text: 'Train on full MNIST dataset', type: 'train' },
            { id: 't5_2', text: 'Experiment with architectures', type: 'build' },
            { id: 't5_3', text: 'Compare train vs test accuracy', type: 'analyze' },
            { id: 't5_4', text: 'Final project: Build your network', type: 'project' }
        ],
        requires: ['chapter4']
    }
];

// ============================================================================
// ACHIEVEMENTS
// ============================================================================
const ACHIEVEMENTS = {
    firstSteps: {
        id: 'firstSteps',
        title: '👶 First Steps',
        description: 'Completed Chapter 1',
        icon: '👶',
        xp: 50,
        condition: (progress) => progress.chaptersCompleted.includes('chapter1')
    },
    layerMaster: {
        id: 'layerMaster',
        title: '🏗️ Layer Master',
        description: 'Built a custom network with 3+ layers',
        icon: '🏗️',
        xp: 100,
        condition: (progress) => progress.customLayersBuilt >= 3
    },
    gradientWatcher: {
        id: 'gradientWatcher',
        title: '🌊 Gradient Watcher',
        description: 'Watched 100 backpropagation cycles',
        icon: '🌊',
        xp: 75,
        condition: (progress) => progress.backpropWatched >= 100
    },
    accuracyChamp: {
        id: 'accuracyChamp',
        title: '🏆 Accuracy Champion',
        description: 'Achieved 95%+ accuracy',
        icon: '🏆',
        xp: 150,
        condition: (progress) => progress.maxAccuracy >= 0.95
    },
    speedRunner: {
        id: 'speedRunner',
        title: '⚡ Speed Runner',
        description: 'Reached 90% accuracy in under 60 seconds',
        icon: '⚡',
        xp: 200,
        condition: (progress) => progress.fastestTo90 && progress.fastestTo90 < 60000
    },
    curious: {
        id: 'curious',
        title: '🔍 Curious Mind',
        description: 'Viewed 20+ tooltips',
        icon: '🔍',
        xp: 30,
        condition: (progress) => progress.tooltipsViewed >= 20
    },
    debugger: {
        id: 'debugger',
        title: '🐛 Debugging Detective',
        description: 'Used prediction debugger 10 times',
        icon: '🐛',
        xp: 60,
        condition: (progress) => progress.predictionsDebugged >= 10
    }
};

// ============================================================================
// GLOSSARY
// ============================================================================
const GLOSSARY = {
    neuron: {
        term: 'Neuron',
        beginner: 'A tiny decision-maker that receives inputs and produces an output.',
        intermediate: 'A computational unit that computes a weighted sum of inputs, adds a bias, and applies an activation function.',
        advanced: 'Implements f(Σwᵢxᵢ + b) where w are weights, x are inputs, b is bias, and f is the activation function.'
    },
    weight: {
        term: 'Weight',
        beginner: 'A number that controls how important each input is.',
        intermediate: 'A learnable parameter that scales the input signal. Positive weights amplify, negative weights inhibit.',
        advanced: 'Element of the weight matrix W. Updated via gradient descent: w ← w - α∂L/∂w'
    },
    bias: {
        term: 'Bias',
        beginner: 'A number that shifts the neuron\'s output up or down.',
        intermediate: 'An offset added before the activation function, allowing the neuron to activate even with zero input.',
        advanced: 'Learnable parameter b in z = Wx + b. Provides translation invariance to the decision boundary.'
    },
    activation: {
        term: 'Activation',
        beginner: 'How "excited" or "active" a neuron is (0 = off, 1 = fully on).',
        intermediate: 'The output of a neuron after applying the activation function to the weighted sum.',
        advanced: 'a = σ(z) where σ is the activation function (ReLU, sigmoid, etc.) and z is the pre-activation.'
    },
    gradient: {
        term: 'Gradient',
        beginner: 'A signal that tells weights which way to change to reduce errors.',
        intermediate: 'The partial derivative of the loss with respect to a parameter, indicating the direction of steepest increase.',
        advanced: '∇L = [∂L/∂w₁, ∂L/∂w₂, ...]. Computed via backpropagation using the chain rule.'
    },
    backprop: {
        term: 'Backpropagation',
        beginner: 'The process of sending error signals backward through the network to teach it.',
        intermediate: 'An algorithm that computes gradients layer by layer, from output to input, using the chain rule.',
        advanced: 'Implements ∂L/∂wˡ = ∂L/∂aˡ × ∂aˡ/∂zˡ × ∂zˡ/∂wˡ recursively through layers.'
    },
    loss: {
        term: 'Loss',
        beginner: 'A number measuring how wrong the network\'s prediction was. Lower is better.',
        intermediate: 'A function comparing predictions to true labels. Cross-entropy for classification, MSE for regression.',
        advanced: 'L(ŷ,y) = -Σyᵢlog(ŷᵢ) for cross-entropy, or L = (1/n)Σ(ŷᵢ-yᵢ)² for MSE.'
    },
    epoch: {
        term: 'Epoch',
        beginner: 'One complete pass through all training examples.',
        intermediate: 'Training typically requires multiple epochs for the network to learn patterns.',
        advanced: 'One epoch = dataset_size / batch_size gradient updates. Multiple epochs risk overfitting.'
    }
};

// ============================================================================
// JOURNEY MANAGER CLASS
// ============================================================================
class JourneyManager {
    constructor() {
        this.chapters = CHAPTERS;
        this.achievements = ACHIEVEMENTS;
        this.glossary = GLOSSARY;
        this.difficulty = 'beginner'; // beginner, intermediate, advanced
        this.progress = this.loadProgress();
        this.listeners = [];
    }

    // Progress Management
    loadProgress() {
        const saved = localStorage.getItem('nn-journey-progress');
        if (saved) {
            return JSON.parse(saved);
        }
        return {
            currentChapter: 'chapter1',
            chaptersCompleted: [],
            tasksCompleted: [],
            xp: 0,
            achievementsUnlocked: [],
            tooltipsViewed: 0,
            backpropWatched: 0,
            predictionsDebugged: 0,
            maxAccuracy: 0,
            customLayersBuilt: 0,
            fastestTo90: null,
            startTime: null
        };
    }

    saveProgress() {
        localStorage.setItem('nn-journey-progress', JSON.stringify(this.progress));
        this.notifyListeners();
    }

    resetProgress() {
        localStorage.removeItem('nn-journey-progress');
        this.progress = this.loadProgress();
        this.notifyListeners();
    }

    // Chapter Navigation
    getCurrentChapter() {
        return this.chapters.find(c => c.id === this.progress.currentChapter);
    }

    setCurrentChapter(chapterId) {
        const chapter = this.chapters.find(c => c.id === chapterId);
        if (chapter && this.isChapterUnlocked(chapterId)) {
            this.progress.currentChapter = chapterId;
            this.saveProgress();
            return true;
        }
        return false;
    }

    isChapterUnlocked(chapterId) {
        const chapter = this.chapters.find(c => c.id === chapterId);
        if (!chapter.requires) return true;
        return chapter.requires.every(req => this.progress.chaptersCompleted.includes(req));
    }

    isChapterCompleted(chapterId) {
        return this.progress.chaptersCompleted.includes(chapterId);
    }

    completeChapter(chapterId) {
        if (!this.progress.chaptersCompleted.includes(chapterId)) {
            this.progress.chaptersCompleted.push(chapterId);
            this.addXP(100);
            this.checkAchievements();
            this.saveProgress();
        }
    }

    // Task Management
    completeTask(taskId) {
        if (!this.progress.tasksCompleted.includes(taskId)) {
            this.progress.tasksCompleted.push(taskId);
            this.addXP(25);

            // Check if all tasks in current chapter are done
            const chapter = this.getCurrentChapter();
            const allDone = chapter.tasks.every(t =>
                this.progress.tasksCompleted.includes(t.id)
            );
            if (allDone) {
                this.completeChapter(chapter.id);
            }
            this.saveProgress();
        }
    }

    isTaskCompleted(taskId) {
        return this.progress.tasksCompleted.includes(taskId);
    }

    // XP & Achievements
    addXP(amount) {
        this.progress.xp += amount;
        this.saveProgress();
    }

    checkAchievements() {
        Object.values(this.achievements).forEach(achievement => {
            if (!this.progress.achievementsUnlocked.includes(achievement.id)) {
                if (achievement.condition(this.progress)) {
                    this.unlockAchievement(achievement.id);
                }
            }
        });
    }

    unlockAchievement(achievementId) {
        if (!this.progress.achievementsUnlocked.includes(achievementId)) {
            this.progress.achievementsUnlocked.push(achievementId);
            const achievement = this.achievements[achievementId];
            this.addXP(achievement.xp);
            this.showAchievementNotification(achievement);
            this.saveProgress();
        }
    }

    showAchievementNotification(achievement) {
        const notification = document.createElement('div');
        notification.className = 'achievement-notification';
        notification.innerHTML = `
            <div class="achievement-icon">${achievement.icon}</div>
            <div class="achievement-info">
                <div class="achievement-title">${achievement.title}</div>
                <div class="achievement-desc">${achievement.description}</div>
                <div class="achievement-xp">+${achievement.xp} XP</div>
            </div>
        `;
        document.body.appendChild(notification);
        setTimeout(() => notification.classList.add('show'), 100);
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 500);
        }, 3000);
    }

    // Stats Tracking
    trackTooltipView() {
        this.progress.tooltipsViewed++;
        this.checkAchievements();
        this.saveProgress();
    }

    trackBackprop() {
        this.progress.backpropWatched++;
        this.checkAchievements();
        this.saveProgress();
    }

    trackAccuracy(accuracy) {
        if (accuracy > this.progress.maxAccuracy) {
            this.progress.maxAccuracy = accuracy;
            this.checkAchievements();
            this.saveProgress();
        }
    }

    // Difficulty
    setDifficulty(level) {
        this.difficulty = level;
        localStorage.setItem('nn-journey-difficulty', level);
        this.notifyListeners();
    }

    getDifficulty() {
        return localStorage.getItem('nn-journey-difficulty') || 'beginner';
    }

    // Glossary
    getDefinition(term) {
        const entry = this.glossary[term];
        if (!entry) return null;
        return {
            term: entry.term,
            definition: entry[this.getDifficulty()] || entry.beginner
        };
    }

    // Event System
    addListener(callback) {
        this.listeners.push(callback);
    }

    notifyListeners() {
        this.listeners.forEach(cb => cb(this.progress));
    }

    // UI Helpers
    getProgressPercentage() {
        const totalTasks = this.chapters.reduce((sum, c) => sum + c.tasks.length, 0);
        return Math.round((this.progress.tasksCompleted.length / totalTasks) * 100);
    }

    getChapterProgress(chapterId) {
        const chapter = this.chapters.find(c => c.id === chapterId);
        const completed = chapter.tasks.filter(t =>
            this.progress.tasksCompleted.includes(t.id)
        ).length;
        return Math.round((completed / chapter.tasks.length) * 100);
    }
}

// ============================================================================
// TEACHING MODE MANAGER
// ============================================================================
class TeachingModeManager {
    constructor() {
        this.mode = 'normal'; // normal, narrated, stepByStep, superSlow
        this.paused = false;
        this.currentStep = null;
        this.narrationEnabled = true;
        this.listeners = [];
    }

    setMode(mode) {
        this.mode = mode;
        this.notifyListeners('modeChange', mode);
    }

    getSpeedMultiplier() {
        switch (this.mode) {
            case 'superSlow': return 0.05;
            case 'narrated': return 0.3;
            case 'stepByStep': return 0; // Manual stepping
            case 'fast': return 3;
            default: return 1;
        }
    }

    isStepMode() {
        return this.mode === 'stepByStep';
    }

    pause() {
        this.paused = true;
        this.notifyListeners('pause');
    }

    resume() {
        this.paused = false;
        this.notifyListeners('resume');
    }

    stepForward() {
        this.notifyListeners('stepForward');
    }

    stepBackward() {
        this.notifyListeners('stepBackward');
    }

    // Narration
    narrate(step, data) {
        if (!this.narrationEnabled) return;

        const narrations = {
            forwardStart: 'Data is entering the network. Watch as it flows forward through each layer.',
            forwardLayer: `Layer ${data?.layer || ''} is processing. Each neuron computes its weighted sum and applies activation.`,
            forwardEnd: 'Forward pass complete! The network has made its prediction.',
            lossCalc: `Loss calculated: ${data?.loss?.toFixed(4) || '?'}. This measures how wrong the prediction was.`,
            backwardStart: 'Now the error flows backward. Watch the cyan particles carry gradient information.',
            backwardLayer: `Gradients reaching Layer ${data?.layer || ''}. Each weight learns how much it contributed to the error.`,
            backwardEnd: 'Backpropagation complete! Weights have been updated.',
            weightUpdate: 'Weights updated by a small amount in the direction that reduces error.'
        };

        const text = narrations[step] || '';
        this.showNarration(text);
        this.notifyListeners('narration', { step, text, data });
    }

    showNarration(text) {
        let el = document.getElementById('teaching-narration');
        if (!el) {
            el = document.createElement('div');
            el.id = 'teaching-narration';
            el.className = 'teaching-narration';
            document.body.appendChild(el);
        }
        el.textContent = text;
        el.classList.add('show');
        setTimeout(() => el.classList.remove('show'), 4000);
    }

    addListener(callback) {
        this.listeners.push(callback);
    }

    notifyListeners(event, data) {
        this.listeners.forEach(cb => cb(event, data));
    }
}

// ============================================================================
// GLOBAL INSTANCES
// ============================================================================
window.Journey = new JourneyManager();
window.TeachingMode = new TeachingModeManager();
