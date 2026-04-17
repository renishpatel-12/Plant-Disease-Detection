#!/usr/bin/env python3
"""
Check Python version and TensorFlow compatibility
"""

import sys
import subprocess

def check_python_version():
    """Check current Python version"""
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("[OK] Python version is compatible with TensorFlow")
        return True
    elif version.major == 3 and version.minor > 11:
        print("[ERROR] Python version is too new for TensorFlow")
        print("   TensorFlow supports Python 3.8-3.11")
        return False
    else:
        print("[ERROR] Python version is too old for TensorFlow")
        return False

def check_tensorflow():
    """Check if TensorFlow can be imported"""
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} is installed")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] GPU available: {len(gpus)} device(s)")
        else:
            print("[INFO] No GPU detected, using CPU")
        
        return True
    except ImportError:
        print("[ERROR] TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"[ERROR] TensorFlow error: {e}")
        return False

def check_other_packages():
    """Check other required packages"""
    packages = ['streamlit', 'PIL', 'numpy', 'pandas', 'matplotlib', 'sklearn']
    missing = []
    
    for package in packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"[OK] Pillow (PIL) is installed")
            elif package == 'sklearn':
                import sklearn
                print(f"[OK] scikit-learn is installed")
            else:
                exec(f"import {package}")
                print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[ERROR] {package} is not installed")
            missing.append(package)
    
    return len(missing) == 0

def suggest_solutions():
    """Suggest solutions based on findings"""
    print("\n" + "="*50)
    print("SOLUTIONS:")
    print("="*50)
    
    version = sys.version_info
    
    if version.major == 3 and version.minor > 11:
        print("[FIX] Your Python is too new. Options:")
        print("   1. Install Python 3.11 from https://www.python.org/downloads/")
        print("   2. Run: setup_python311.bat")
        print("   3. Use conda: conda create -n plant_disease python=3.11")
        
    elif version.major == 3 and 8 <= version.minor <= 11:
        print("[FIX] Python version is good. Install packages:")
        print("   pip install -r requirements.txt")
        
    else:
        print("[FIX] Install Python 3.11:")
        print("   Download from https://www.python.org/downloads/")

def main():
    """Main function"""
    print("Plant Disease Detection - System Check")
    print("="*50)
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check TensorFlow
    tf_ok = check_tensorflow()
    print()
    
    # Check other packages
    packages_ok = check_other_packages()
    print()
    
    # Overall status
    if python_ok and tf_ok and packages_ok:
        print("[SUCCESS] All checks passed! You can run the app:")
        print("   streamlit run app.py")
    else:
        suggest_solutions()

if __name__ == "__main__":
    main()