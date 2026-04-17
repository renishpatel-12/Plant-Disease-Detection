#!/usr/bin/env python3
"""
Plant Disease Detection - Easy Run Script
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required packages"""
    if not Path('requirements.txt').exists():
        print("✗ requirements.txt not found!")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing requirements")

def test_system():
    """Test if everything is working"""
    return run_command("python test_app.py", "Testing system")

def prepare_data():
    """Prepare sample data"""
    return run_command("python prepare_data.py", "Preparing sample data")

def train_model():
    """Train the model"""
    return run_command("python train_model.py", "Training model")

def run_app():
    """Run the Streamlit app"""
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your browser automatically.")
    print("Press Ctrl+C to stop the app.")
    
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running app: {e}")

def main():
    """Main function with menu"""
    
    print("🌱 Plant Disease Detection - Easy Setup & Run")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Install requirements")
        print("2. Test system")
        print("3. Prepare sample data")
        print("4. Train model")
        print("5. Run web app")
        print("6. Full setup (1→2→3→5)")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            install_requirements()
            
        elif choice == '2':
            test_system()
            
        elif choice == '3':
            prepare_data()
            
        elif choice == '4':
            if not Path('dataset').exists():
                print("⚠️  No dataset found. Running prepare_data.py first...")
                if prepare_data():
                    train_model()
            else:
                train_model()
                
        elif choice == '5':
            if not Path('model.h5').exists():
                print("⚠️  No model found. Creating dummy model for testing...")
                if test_system():
                    run_app()
            else:
                run_app()
                
        elif choice == '6':
            print("\n🔄 Running full setup...")
            
            # Step 1: Install requirements
            if not install_requirements():
                print("❌ Setup failed at requirements installation")
                continue
                
            # Step 2: Test system
            if not test_system():
                print("❌ Setup failed at system test")
                continue
                
            # Step 3: Prepare data
            if not prepare_data():
                print("❌ Setup failed at data preparation")
                continue
                
            # Step 4: Run app
            print("\n✅ Setup completed successfully!")
            input("Press Enter to start the web app...")
            run_app()
            
        elif choice == '7':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main()