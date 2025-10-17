"""
Setup script for Fake News Detection System
Run this script to set up the project environment and train initial models.
"""

import os
import sys
import subprocess
import nltk
from pathlib import Path

def setup_project():
    """
    Set up the project environment and train initial models.
    """
    print("🚀 Setting up Fake News Detection System...")
    print("=" * 50)
    
    # Create necessary directories
    directories = ['models', 'results', 'plots', 'data', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️ NLTK download failed: {e}")
    
    # Train initial models
    print("\n🤖 Training initial models...")
    try:
        subprocess.run([sys.executable, 'train_models.py'], check=True)
        print("✅ Models trained successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Model training failed: {e}")
        print("You can train models later by running: python train_models.py")
    
    print("\n🎉 Setup completed!")
    print("\n🚀 To start the application, run:")
    print("   python app.py")
    print("\n🌐 Then visit: http://127.0.0.1:5000")

if __name__ == "__main__":
    setup_project()
