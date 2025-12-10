@echo off
echo ========================================
echo Traffic Accidents Analysis - First Time Setup
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
echo Python found!
echo.

echo Step 2: Installing required packages...
echo This may take a few minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    echo Try running: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
echo Packages installed successfully!
echo.

echo Step 3: Verifying installation...
python -c "import pandas, numpy, sklearn, matplotlib, streamlit, jupyter; print('All packages verified!')"
if errorlevel 1 (
    echo ERROR: Some packages failed to import
    echo Please check the error message above
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Run "jupyter notebook traffic_analysis.ipynb" to train models
echo 2. Run "streamlit run app.py" to launch the web application
echo.
echo Or simply double-click START_PROJECT.bat
echo.
pause
