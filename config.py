"""
Configuration file for Fake News Detection System
"""

import os

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG') or True
    
    # Model Configuration
    MAX_FEATURES = 10000
    MAX_TEXT_LENGTH = 10000
    MIN_TEXT_LENGTH = 10
    
    # File Paths
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    PLOTS_DIR = 'plots'
    
    # ML Parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # BERT Configuration (Optional)
    BERT_MODEL_NAME = 'distilbert-base-uncased'
    BERT_MAX_LENGTH = 512
