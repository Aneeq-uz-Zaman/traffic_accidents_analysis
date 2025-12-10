@echo off
echo ========================================
echo Traffic Accidents Analysis - Quick Start
echo ========================================
echo.

REM Check if models exist
if not exist "best_model.pkl" (
    echo WARNING: Model files not found!
    echo You need to train the models first.
    echo.
    echo Opening Jupyter Notebook to train models...
    echo After training, close the notebook and run this file again.
    echo.
    pause
    jupyter notebook traffic_analysis.ipynb
    exit
)

echo Models found! Starting web application...
echo.
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run app.py
