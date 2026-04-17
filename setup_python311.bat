@echo off
echo Plant Disease Detection - Python 3.11 Setup
echo ==========================================

echo.
echo This script will help you set up Python 3.11 for TensorFlow compatibility.
echo.

echo Step 1: Checking if Python 3.11 is available...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 not found!
    echo.
    echo Please install Python 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Python 3.11 found!
py -3.11 --version

echo.
echo Step 2: Creating virtual environment...
py -3.11 -m venv plant_disease_env
if errorlevel 1 (
    echo Failed to create virtual environment!
    pause
    exit /b 1
)

echo Virtual environment created successfully!

echo.
echo Step 3: Activating virtual environment...
call plant_disease_env\Scripts\activate.bat

echo.
echo Step 4: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 5: Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo Step 6: Testing installation...
python test_app.py
if errorlevel 1 (
    echo Installation test failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To use the app:
echo 1. Activate the environment: plant_disease_env\Scripts\activate
echo 2. Run the app: streamlit run app.py
echo.
echo The virtual environment is now active.
echo You can start the app by running: streamlit run app.py
echo.

pause