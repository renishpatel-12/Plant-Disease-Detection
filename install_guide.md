# TensorFlow Installation Guide for Plant Disease Detection

## Problem: Python 3.14 Compatibility

Your current Python version (3.14) is too new for TensorFlow. TensorFlow currently supports Python 3.8-3.11.

## Solutions:

### Option 1: Install Python 3.11 (Recommended)

1. **Download Python 3.11**:
   - Go to https://www.python.org/downloads/
   - Download Python 3.11.x (latest 3.11 version)
   - Install it alongside your current Python

2. **Create a virtual environment**:
   ```bash
   # Using Python 3.11
   py -3.11 -m venv plant_disease_env
   
   # Activate the environment
   plant_disease_env\Scripts\activate
   
   # Install requirements
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

### Option 2: Use Conda (Alternative)

1. **Install Miniconda**:
   - Download from https://docs.conda.io/en/latest/miniconda.html

2. **Create environment with Python 3.11**:
   ```bash
   conda create -n plant_disease python=3.11
   conda activate plant_disease
   pip install -r requirements.txt
   ```

### Option 3: Use Docker (Advanced)

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build and run**:
   ```bash
   docker build -t plant-disease .
   docker run -p 8501:8501 plant-disease
   ```

## Quick Test

After installing Python 3.11, test if TensorFlow works:

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## Updated Requirements for Python 3.11

```
streamlit>=1.28.0
tensorflow>=2.13.0
Pillow>=9.5.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## Why This Happens

- TensorFlow releases are tied to specific Python versions
- Python 3.14 was released recently (October 2024)
- TensorFlow team needs time to add support for new Python versions
- Usually takes 3-6 months after Python release

## Next Steps

1. Install Python 3.11 using Option 1
2. Run `python test_app.py` to verify everything works
3. Run `streamlit run app.py` to start the web application

Need help? Check the troubleshooting section in README.md