"""
Streamlit version of the Fake News Detection App
For easy deployment on Streamlit Cloud
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_preprocessing import DataPreprocessor
    from ml_models import FakeNewsClassifier
    from advanced_analysis import TextAnalyzer
    from ensemble_predictor import EnsemblePredictor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-news {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .fake-news {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Initialize and cache the models."""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(vectorizer_type='tfidf', max_features=5000)
        
        # Load sample data and train models
        df = preprocessor.load_data(use_sample=True)
        df_processed = preprocessor.preprocess_dataset(df)
        X = preprocessor.extract_features(df_processed)
        y = df_processed['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Train models
        classifier = FakeNewsClassifier()
        classifier.train_models(X_train, y_train)
        
        # Initialize ensemble and text analyzer
        ensemble_predictor = EnsemblePredictor(classifier.trained_models)
        text_analyzer = TextAnalyzer()
        
        return preprocessor, classifier, ensemble_predictor, text_analyzer
        
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None, None, None

def get_confidence_class(confidence):
    """Get CSS class for confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Analysis", "About", "API Demo"])
    
    if page == "Home":
        show_home_page()
    elif page == "Analysis":
        show_analysis_page()
    elif page == "About":
        show_about_page()
    elif page == "API Demo":
        show_api_demo()

def show_home_page():
    """Show the home page with basic analysis."""
    
    st.markdown("## 📰 Analyze News Articles")
    st.markdown("Enter a news article below to check if it might be fake or real.")
    
    # Initialize models
    with st.spinner("Initializing models... This may take a moment."):
        preprocessor, classifier, ensemble_predictor, text_analyzer = initialize_models()
    
    if preprocessor is None:
        st.error("Failed to initialize models. Please refresh the page.")
        return
    
    # Text input
    text_input = st.text_area(
        "Enter news article text:",
        height=200,
        placeholder="Paste or type your news article here..."
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Choose analysis method:",
        ["All Models (Recommended)", "Logistic Regression", "Random Forest", "Naive Bayes"]
    )
    
    # Sample texts
    st.markdown("### 📝 Try Sample Texts")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📰 Sample Real News"):
            st.session_state.sample_text = "Scientists at MIT have developed a new solar panel technology that increases efficiency by 40%. The research, published in Nature Energy, shows promising results for renewable energy applications. The team used advanced materials science to create panels that can capture more sunlight and convert it more efficiently than previous generations of solar technology."
    
    with col2:
        if st.button("⚠️ Sample Fake News"):
            st.session_state.sample_text = "SHOCKING: Doctors discover this one weird trick that cures all diseases instantly! Big Pharma doesn't want you to know about this amazing secret that will change your life FOREVER! ACT NOW before it's too late!"
    
    # Use session state for text input
    if 'sample_text' in st.session_state:
        text_input = st.text_area(
            "Enter news article text:",
            value=st.session_state.sample_text,
            height=200,
            placeholder="Paste or type your news article here..."
        )
    
    # Analysis button
    if st.button("🔍 Analyze Article", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        if len(text_input.strip()) < 10:
            st.warning("Please enter at least 10 characters for analysis.")
            return
        
        # Perform analysis
        with st.spinner("Analyzing..."):
            analyze_text(text_input, model_choice, preprocessor, classifier, ensemble_predictor, text_analyzer)

def analyze_text(text, model_choice, preprocessor, classifier, ensemble_predictor, text_analyzer):
    """Analyze the given text and display results."""
    
    try:
        # Process text
        X_processed = preprocessor.process_single_text(text)
        
        # Get predictions
        if model_choice == "All Models (Recommended)":
            results = {}
            for model_name in classifier.trained_models.keys():
                prediction, confidence = classifier.predict_single(model_name, X_processed)
                results[model_name] = {
                    'prediction': 'Real' if prediction == 1 else 'Fake',
                    'confidence': confidence
                }
            
            # Ensemble prediction
            ensemble_result = ensemble_predictor.predict_ensemble(X_processed)
            
        else:
            model_map = {
                "Logistic Regression": "logistic_regression",
                "Random Forest": "random_forest", 
                "Naive Bayes": "naive_bayes"
            }
            model_name = model_map[model_choice]
            prediction, confidence = classifier.predict_single(model_name, X_processed)
            results = {
                model_name: {
                    'prediction': 'Real' if prediction == 1 else 'Fake',
                    'confidence': confidence
                }
            }
            ensemble_result = None
        
        # Text analysis
        text_analysis = text_analyzer.analyze_text_features(text)
        
        # Display results
        display_results(text, results, ensemble_result, text_analysis)
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")

def display_results(text, results, ensemble_result, text_analysis):
    """Display analysis results."""
    
    st.markdown("## 📊 Analysis Results")
    
    # Show input text
    with st.expander("📄 Analyzed Text", expanded=False):
        st.text(text)
    
    # Individual model results
    if len(results) > 1:
        st.markdown("### 🤖 Individual Model Predictions")
        
        cols = st.columns(len(results))
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx]:
                prediction = result['prediction']
                confidence = result.get('confidence', 0)
                
                # Determine styling
                box_class = "real-news" if prediction == "Real" else "fake-news"
                conf_class = get_confidence_class(confidence)
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h4>{model_name.replace('_', ' ').title()}</h4>
                    <p><strong>Prediction:</strong> {prediction}</p>
                    <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Ensemble result
    if ensemble_result:
        st.markdown("### 🎯 Ensemble Prediction (Combined Models)")
        
        prediction = ensemble_result['ensemble_prediction']
        confidence = ensemble_result['ensemble_confidence']
        consensus = ensemble_result['consensus_strength']
        
        box_class = "real-news" if prediction == "Real" else "fake-news"
        conf_class = get_confidence_class(confidence)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h3>Final Prediction: {prediction}</h3>
                <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence:.1%}</span></p>
                <p><strong>Consensus:</strong> {consensus.title()}</p>
                <p><strong>Recommendation:</strong> {ensemble_result['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Probability chart
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ['Fake', 'Real']
            sizes = [ensemble_result['fake_probability'], ensemble_result['real_probability']]
            colors = ['#ff6b6b', '#51cf66']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Probability Breakdown')
            st.pyplot(fig)
    
    # Text analysis
    st.markdown("### 🔍 Advanced Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Credibility Metrics")
        credibility_score = text_analysis['credibility_score']['score']
        
        # Progress bar for credibility
        st.metric("Credibility Score", f"{credibility_score:.2f}/1.00")
        st.progress(credibility_score)
        
        st.metric("Fake Indicators Found", len(text_analysis['fake_indicators_found']))
        st.metric("Real Indicators Found", len(text_analysis['real_indicators_found']))
    
    with col2:
        st.markdown("#### 📝 Text Statistics")
        stats = text_analysis['text_statistics']
        
        st.metric("Word Count", stats['word_count'])
        st.metric("Readability", text_analysis['readability']['complexity'].title())
        st.metric("Exclamation Marks", stats['exclamation_count'])
    
    # Detailed indicators
    if text_analysis['fake_indicators_found'] or text_analysis['real_indicators_found']:
        with st.expander("🔍 Detailed Indicators Analysis", expanded=False):
            
            if text_analysis['fake_indicators_found']:
                st.markdown("**⚠️ Fake News Indicators Detected:**")
                for word, category in text_analysis['fake_indicators_found']:
                    st.write(f"• '{word}' ({category})")
            
            if text_analysis['real_indicators_found']:
                st.markdown("**✅ Real News Indicators Detected:**")
                for word, category in text_analysis['real_indicators_found']:
                    st.write(f"• '{word}' ({category})")

def show_analysis_page():
    """Show detailed analysis page."""
    st.markdown("## 📊 Detailed Analysis")
    st.markdown("This page provides comprehensive analysis tools and explanations.")
    
    # Add analysis tools here
    st.info("Advanced analysis tools coming soon!")

def show_about_page():
    """Show about page."""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This Fake News Detection System uses advanced machine learning algorithms to analyze text content 
    and identify potentially misleading or false information.
    
    ### 🤖 Machine Learning Models
    - **Logistic Regression**: Linear classifier, excellent for text classification
    - **Random Forest**: Ensemble method, robust to overfitting
    - **Multinomial Naive Bayes**: Probabilistic classifier, fast and effective
    
    ### 🔍 Advanced Features
    - **Ensemble Prediction**: Combines multiple models for better accuracy
    - **Text Analysis**: Linguistic pattern detection
    - **Confidence Scoring**: Reliability assessment
    - **Real-time Processing**: Instant analysis
    
    ### ⚠️ Important Disclaimer
    This system is designed for educational and research purposes. Always verify important 
    information through multiple trusted sources. No automated system is 100% accurate.
    """)

def show_api_demo():
    """Show API demonstration."""
    st.markdown("## 🔌 API Demo")
    
    st.markdown("""
    ### API Endpoints
    
    **POST /api/predict**
    ```json
    {
        "text": "Your news article text",
        "model": "logistic_regression"
    }
    ```
    
    **POST /api/analyze**
    ```json
    {
        "text": "Your news article text",
        "include_ensemble": true
    }
    ```
    """)
    
    st.code("""
import requests

# Basic prediction
response = requests.post('http://localhost:5000/api/predict', 
                        json={'text': 'Your news text here'})
result = response.json()

# Comprehensive analysis
response = requests.post('http://localhost:5000/api/analyze', 
                        json={'text': 'Your news text here'})
result = response.json()
    """, language='python')

if __name__ == "__main__":
    main()
