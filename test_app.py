#!/usr/bin/env python3
"""
Test script for Plant Disease Detection App
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from pathlib import Path

def create_dummy_model():
    """Create a dummy model for testing if no trained model exists"""
    
    # Create a simple model with the same architecture
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensemble Model
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([xception_output, densenet_output])
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def test_model_creation():
    """Test if model can be created successfully"""
    
    print("Testing model creation...")
    try:
        model = create_dummy_model()
        print(f"[OK] Model created successfully with {model.count_params():,} parameters")
        return model
    except Exception as e:
        print(f"[ERROR] Error creating model: {e}")
        return None

def test_image_processing():
    """Test image processing functions"""
    
    print("\nTesting image processing...")
    try:
        from utils import clean_image, get_prediction, make_results
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (256, 256), color='green')
        
        # Test clean_image function
        processed_image = clean_image(dummy_image)
        assert processed_image.shape == (1, 512, 512, 3), f"Expected shape (1, 512, 512, 3), got {processed_image.shape}"
        print("[OK] Image cleaning function works correctly")
        
        return processed_image
        
    except Exception as e:
        print(f"[ERROR] Error in image processing: {e}")
        return None

def test_prediction():
    """Test prediction pipeline"""
    
    print("\nTesting prediction pipeline...")
    try:
        from utils import get_prediction, make_results
        
        # Create dummy model and image
        model = create_dummy_model()
        dummy_image = Image.new('RGB', (256, 256), color='green')
        
        # Process image
        from utils import clean_image
        processed_image = clean_image(dummy_image)
        
        # Get prediction
        predictions, predictions_arr = get_prediction(model, processed_image)
        
        # Validate prediction format
        assert predictions.shape == (1, 4), f"Expected predictions shape (1, 4), got {predictions.shape}"
        assert isinstance(predictions_arr, (int, np.integer)), f"Expected int, got {type(predictions_arr)}"
        assert 0 <= predictions_arr <= 3, f"Expected prediction in range [0, 3], got {predictions_arr}"
        
        print("[OK] Prediction function works correctly")
        
        # Test result formatting
        result = make_results(predictions, predictions_arr)
        assert 'status' in result and 'prediction' in result, "Result should contain 'status' and 'prediction' keys"
        
        print("[OK] Result formatting works correctly")
        print(f"  Sample result: {result}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error in prediction pipeline: {e}")
        return False

def test_streamlit_imports():
    """Test if all required packages can be imported"""
    
    print("\nTesting package imports...")
    
    packages = [
        ('streamlit', 'st'),
        ('tensorflow', 'tf'),
        ('PIL', 'Image'),
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib.pyplot', 'plt')
    ]
    
    failed_imports = []
    
    for package, alias in packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"[OK] {package}")
        except ImportError as e:
            print(f"[ERROR] {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nMissing packages: {', '.join(failed_imports)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    else:
        print("[OK] All packages imported successfully")
        return True

def create_test_image():
    """Create a test image for the app"""
    
    print("\nCreating test image...")
    
    # Create a simple leaf-like image
    img = Image.new('RGB', (512, 512), color=(34, 139, 34))  # Forest green
    
    # Add some patterns to make it look more like a leaf
    pixels = img.load()
    for i in range(512):
        for j in range(512):
            # Add some variation
            if (i + j) % 20 < 10:
                pixels[i, j] = (50, 155, 50)
    
    # Save test image
    test_image_path = Path('test_leaf.jpg')
    img.save(test_image_path)
    print(f"[OK] Test image saved as {test_image_path}")
    
    return test_image_path

def main():
    """Run all tests"""
    
    print("Plant Disease Detection - System Test")
    print("=" * 50)
    
    # Test 1: Package imports
    if not test_streamlit_imports():
        print("\n[FAILED] Package import test failed!")
        return
    
    # Test 2: Model creation
    model = test_model_creation()
    if model is None:
        print("\n[FAILED] Model creation test failed!")
        return
    
    # Test 3: Image processing
    processed_image = test_image_processing()
    if processed_image is None:
        print("\n[FAILED] Image processing test failed!")
        return
    
    # Test 4: Prediction pipeline
    if not test_prediction():
        print("\n[FAILED] Prediction pipeline test failed!")
        return
    
    # Test 5: Create test image
    test_image_path = create_test_image()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All tests passed successfully!")
    print("\nNext steps:")
    print("1. Run 'python prepare_data.py' to prepare your dataset")
    print("2. Run 'python train_model.py' to train the model")
    print("3. Run 'streamlit run app.py' to start the web application")
    print(f"4. Use {test_image_path} to test the web app")
    
    # Save dummy model for testing
    if not Path('model.h5').exists():
        print("\nSaving dummy model for testing...")
        model.save('model.h5')
        print("[OK] Dummy model saved as 'model.h5'")
        print("  (Replace this with your trained model)")

if __name__ == "__main__":
    main()