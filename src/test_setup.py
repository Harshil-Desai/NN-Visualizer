#!/usr/bin/env python3
"""
Test script to verify Neural Network Visualizer setup
Run this before starting the full application
"""

import sys
import os

def test_imports():
    """Test all required Python imports"""
    print("Testing Python imports...")
    try:
        import fastapi
        print("  ✅ fastapi")
    except ImportError:
        print("  ❌ fastapi - Install with: pip install fastapi --break-system-packages")
        return False
    
    try:
        import uvicorn
        print("  ✅ uvicorn")
    except ImportError:
        print("  ❌ uvicorn - Install with: pip install uvicorn --break-system-packages")
        return False
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - Install with: pip install numpy --break-system-packages")
        return False
    
    try:
        import requests
        print("  ✅ requests")
    except ImportError:
        print("  ❌ requests - Install with: pip install requests --break-system-packages")
        return False
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'src/server.py',
        'src/network/__init__.py',
        'src/network/model.py',
        'src/network/layers.py',
        'src/network/loss.py',
        'src/network/optimizer.py',
        'src/network/activations.py',
        'src/data/loader.py',
        'src/visualizer/index.html',
        'src/visualizer/main-app.js',
        'src/visualizer/matrix.js',
        'src/visualizer/nn-engine.js',
        'src/visualizer/journey.js',
        'src/visualizer/styles.css'
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath} - MISSING!")
            all_exist = False
    
    return all_exist

def test_network_module():
    """Test if network module can be imported and used"""
    print("\nTesting network module...")
    
    try:
        from network import NeuralNetwork, Dense, ReLU, Softmax, CrossEntropy, SGD
        print("  ✅ All network classes imported successfully")
        
        # Try creating a simple network
        nn = NeuralNetwork([
            Dense(10, 5, init_type='he'),
            ReLU(),
            Dense(5, 2, init_type='xavier'),
            Softmax()
        ])
        print("  ✅ Network instantiation successful")
        
        # Try a forward pass
        import numpy as np
        x = np.random.randn(2, 10)
        y = np.array([[1, 0], [0, 1]])
        
        output = nn.forward(x)
        print(f"  ✅ Forward pass successful (output shape: {output.shape})")
        
        # Try a training step
        loss_fn = CrossEntropy()
        optimizer = SGD(learning_rate=0.01)
        loss, pred = nn.train_step(x, y, loss_fn, optimizer)
        print(f"  ✅ Training step successful (loss: {loss:.4f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """Test if data loader works"""
    print("\nTesting data loader...")
    
    try:
        from data.loader import download_mnist, load_mnist, preprocess_data
        print("  ✅ Data loader imports successful")
        
        # Note: Don't actually download in test, just check function exists
        print("  ℹ️  MNIST download function available (not testing actual download)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_server_config():
    """Test server configuration"""
    print("\nTesting server configuration...")
    
    try:
        # Check if we can import server
        import importlib.util
        spec = importlib.util.spec_from_file_location("server", "server.py")
        if spec is None:
            print("  ❌ Cannot load server.py")
            return False
        
        print("  ✅ server.py can be loaded")
        
        # Check static files directory
        if os.path.exists('src/visualizer'):
            print("  ✅ Visualizer directory exists")
        else:
            print("  ❌ src/visualizer directory missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Neural Network Visualizer - Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_file_structure()
    all_passed &= test_network_module()
    all_passed &= test_data_loader()
    all_passed &= test_server_config()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nYou can now start the server with:")
        print("  python server.py")
        print("\nThen open browser at:")
        print("  http://localhost:8000")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before starting the server.")
        print("Refer to FIX_GUIDE.md for detailed instructions.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())