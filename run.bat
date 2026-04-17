@echo off
echo Plant Disease Detection - Windows Quick Start
echo =============================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found!
echo.

echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing requirements!
    pause
    exit /b 1
)

echo.
echo Testing system...
python test_app.py
if errorlevel 1 (
    echo System test failed!
    pause
    exit /b 1
)

echo.
echo Starting web application...
echo The app will open in your browser automatically.
echo Press Ctrl+C to stop the app.
echo.

streamlit run app.py

pause