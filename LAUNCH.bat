@echo off
title Fake News Detection - One-Click Launch
color 0A

echo.
echo  ========================================
echo   🔍 FAKE NEWS DETECTION SYSTEM
echo  ========================================
echo.
echo  Starting your AI-powered news analyzer...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo    Download from: https://python.org
    pause
    exit /b 1
)

echo ✅ Python found!

REM Install dependencies if needed
if not exist "models\" (
    echo 📦 Installing dependencies...
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
    
    echo 📥 Downloading AI models...
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('punkt_tab', quiet=True)"
)

echo 🚀 Launching application...
echo.
echo  ========================================
echo   🌐 ACCESS YOUR APP AT:
echo   
echo   💻 Local:    http://localhost:5000
echo   📱 Network:  http://%COMPUTERNAME%:5000
echo   
echo   Press Ctrl+C to stop the server
echo  ========================================
echo.

REM Start the application
python app.py

pause
