"""
Flask Web Application for Fake News Detection
This module provides a web interface for real-time fake news detection.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from ml_models import FakeNewsClassifier
from ensemble_predictor import EnsemblePredictor
from advanced_analysis import TextAnalyzer

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Global variables for models and preprocessor
preprocessor = None
classifier = None
models_loaded = False

class FakeNewsPredictor:
    """
    A wrapper class for making predictions using trained models.
    """
    
    def __init__(self):
        self.preprocessor = None
        self.models = {}
        self.model_names = ['logistic_regression', 'random_forest', 'naive_bayes']
        self.is_initialized = False
        self.ensemble_predictor = None
        self.text_analyzer = TextAnalyzer()
    
    def initialize(self):
        """Initialize the preprocessor and train models if needed."""
        try:
            print("Initializing Fake News Predictor...")
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor(vectorizer_type='tfidf', max_features=5000)
            
            # Load sample data and train models if no saved models exist
            models_dir = 'models'
            if not os.path.exists(models_dir) or not any(f.endswith('.joblib') for f in os.listdir(models_dir)):
                print("No saved models found. Training new models...")
                self._train_models()
            else:
                print("Loading saved models...")
                self._load_models()
            
            self.is_initialized = True
            print("Fake News Predictor initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing predictor: {str(e)}")
            self.is_initialized = False
    
    def _train_models(self):
        """Train models using sample data."""
        # Load and preprocess data
        df = self.preprocessor.load_data(use_sample=True)
        df_processed = self.preprocessor.preprocess_dataset(df)
        X = self.preprocessor.extract_features(df_processed)
        y = df_processed['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Train models
        classifier = FakeNewsClassifier()
        classifier.train_models(X_train, y_train)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        classifier.save_models()
        
        # Store trained models
        self.models = classifier.trained_models
        
        # Initialize ensemble predictor
        self.ensemble_predictor = EnsemblePredictor(self.models)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')
    
    def _load_models(self):
        """Load saved models from disk."""
        models_dir = 'models'
        
        # Load preprocessor
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        
        # Load models
        for model_name in self.model_names:
            model_files = [f for f in os.listdir(models_dir) if f.startswith(model_name) and f.endswith('.joblib')]
            if model_files:
                # Load the most recent model
                model_files.sort(reverse=True)
                model_path = os.path.join(models_dir, model_files[0])
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} from {model_path}")
        
        # Initialize ensemble predictor after loading models
        if self.models:
            self.ensemble_predictor = EnsemblePredictor(self.models)
    
    def predict(self, text, model_name='logistic_regression'):
        """
        Make prediction for given text.
        
        Args:
            text (str): Input text to classify
            model_name (str): Model to use for prediction
            
        Returns:
            dict: Prediction results
        """
        if not self.is_initialized:
            return {'error': 'Predictor not initialized'}
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not available'}
        
        try:
            # Preprocess text
            X_processed = self.preprocessor.process_single_text(text)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X_processed)[0]
            
            # Get confidence score
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed)[0]
                confidence = max(probabilities)
                fake_prob = probabilities[0]
                real_prob = probabilities[1]
            else:
                confidence = None
                fake_prob = None
                real_prob = None
            
            # Prepare result
            result = {
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'prediction_numeric': int(prediction),
                'confidence': float(confidence) if confidence else None,
                'fake_probability': float(fake_prob) if fake_prob else None,
                'real_probability': float(real_prob) if real_prob else None,
                'model_used': model_name
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def get_all_predictions(self, text):
        """
        Get predictions from all available models.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Predictions from all models
        """
        results = {}
        
        for model_name in self.models.keys():
            results[model_name] = self.predict(text, model_name)
        
        return results

# Initialize predictor
predictor = FakeNewsPredictor()

@app.before_request
def initialize_app():
    """Initialize the application before first request."""
    if not hasattr(app, '_initialized'):
        predictor.initialize()
        app._initialized = True

@app.route('/')
def index():
    """Home page with input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        text = request.form.get('news_text', '').strip()
        model_choice = request.form.get('model_choice', 'logistic_regression')
        
        if not text:
            flash('Please enter some news text to analyze.', 'error')
            return redirect(url_for('index'))
        
        if len(text) < 10:
            flash('Please enter a longer text for better analysis.', 'warning')
            return redirect(url_for('index'))
        
        # Make prediction
        if model_choice == 'all':
            results = predictor.get_all_predictions(text)
        else:
            result = predictor.predict(text, model_choice)
            results = {model_choice: result}
        
        # Add ensemble prediction if multiple models
        ensemble_result = None
        if len(results) > 1 and predictor.ensemble_predictor:
            try:
                X_processed = predictor.preprocessor.process_single_text(text)
                ensemble_result = predictor.ensemble_predictor.predict_ensemble(X_processed)
            except Exception as e:
                print(f"Ensemble prediction failed: {e}")
        
        # Add advanced text analysis
        text_analysis = predictor.text_analyzer.analyze_text_features(text)
        
        return render_template('results.html', 
                             text=text, 
                             results=results, 
                             ensemble_result=ensemble_result,
                             text_analysis=text_analysis,
                             model_choice=model_choice)
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        model_name = data.get('model', 'logistic_regression')
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = predictor.predict(text, model_name)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')

@app.route('/api/models')
def api_models():
    """API endpoint to get available models."""
    if predictor.is_initialized:
        return jsonify({
            'models': list(predictor.models.keys()),
            'status': 'ready'
        })
    else:
        return jsonify({
            'models': [],
            'status': 'not_initialized'
        })

@app.route('/manifest.json')
def manifest():
    """Serve PWA manifest."""
    return app.send_static_file('manifest.json')

@app.route('/sw.js')
def service_worker():
    """Serve service worker."""
    response = app.send_static_file('sw.js')
    response.headers['Content-Type'] = 'application/javascript'
    return response

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for comprehensive analysis."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        include_ensemble = data.get('include_ensemble', True)
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get all model predictions
        results = predictor.get_all_predictions(text)
        
        # Add ensemble prediction
        ensemble_result = None
        if include_ensemble and predictor.ensemble_predictor:
            try:
                X_processed = predictor.preprocessor.process_single_text(text)
                ensemble_result = predictor.ensemble_predictor.predict_ensemble(X_processed)
            except Exception as e:
                ensemble_result = {'error': str(e)}
        
        # Add text analysis
        text_analysis = predictor.text_analyzer.analyze_text_features(text)
        
        response = {
            'individual_predictions': results,
            'ensemble_prediction': ensemble_result,
            'text_analysis': {
                'credibility_score': text_analysis['credibility_score']['score'],
                'fake_indicators_count': len(text_analysis['fake_indicators_found']),
                'real_indicators_count': len(text_analysis['real_indicators_found']),
                'readability': text_analysis['readability'],
                'text_statistics': text_analysis['text_statistics']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Initialize predictor
    predictor.initialize()
    
    print("Starting Flask application...")
    print("Access the application at: http://127.0.0.1:5000")
    
    # Run the application
    app.run(debug=False, host='0.0.0.0', port=5000)
