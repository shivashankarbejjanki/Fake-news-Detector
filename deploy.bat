@echo off
echo ========================================
echo   Fake News Detection - One-Click Deploy
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [3/4] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

echo [4/4] Starting the application...
echo.
echo ========================================
echo   Application will start in 3 seconds...
echo   Access at: http://localhost:5000
echo ========================================
timeout /t 3 /nobreak > nul

python app.py
