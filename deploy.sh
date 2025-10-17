#!/bin/bash

echo "========================================"
echo "  Fake News Detection - One-Click Deploy"
echo "========================================"
echo

echo "[1/4] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "[2/4] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "[3/4] Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

echo "[4/4] Starting the application..."
echo
echo "========================================"
echo "  Application will start in 3 seconds..."
echo "  Access at: http://localhost:5000"
echo "========================================"
sleep 3

python3 app.py
