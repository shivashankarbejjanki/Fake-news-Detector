"""
Main Training Script for Fake News Detection Models
This script trains all models and saves them for use in the web application.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from ml_models import FakeNewsClassifier

def main():
    """
    Main training pipeline for fake news detection models.
    """
    print("=" * 60)
    print("🚀 FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Step 1: Initialize Data Preprocessor
    print("\n📊 Step 1: Initializing Data Preprocessor")
    print("-" * 40)
    
    preprocessor = DataPreprocessor(
        vectorizer_type='tfidf',
        max_features=10000,
        test_size=0.2,
        random_state=42
    )
    
    # Step 2: Load and Preprocess Data
    print("\n📁 Step 2: Loading and Preprocessing Data")
    print("-" * 40)
    
    # Load data (using sample data for demonstration)
    df = preprocessor.load_data(use_sample=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    # Preprocess the data
    df_processed = preprocessor.preprocess_dataset(df)
    
    # Step 3: Feature Extraction
    print("\n🔧 Step 3: Feature Extraction")
    print("-" * 40)
    
    X = preprocessor.extract_features(df_processed)
    y = df_processed['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    
    # Step 4: Split Data
    print("\n✂️ Step 4: Splitting Data")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Initialize and Train Models
    print("\n🤖 Step 5: Training Machine Learning Models")
    print("-" * 40)
    
    classifier = FakeNewsClassifier(random_state=42)
    
    # Train models with cross-validation
    classifier.train_models(X_train, y_train, cv_folds=5)
    
    # Step 6: Hyperparameter Tuning (Optional)
    print("\n⚙️ Step 6: Hyperparameter Tuning")
    print("-" * 40)
    
    try:
        classifier.hyperparameter_tuning(X_train, y_train, cv_folds=3)
        print("✅ Hyperparameter tuning completed successfully!")
    except Exception as e:
        print(f"⚠️ Hyperparameter tuning failed: {e}")
        print("Continuing with default parameters...")
    
    # Step 7: Model Evaluation
    print("\n📈 Step 7: Evaluating Models")
    print("-" * 40)
    
    results = classifier.evaluate_models(X_test, y_test)
    
    # Step 8: Generate Visualizations
    print("\n📊 Step 8: Generating Visualizations")
    print("-" * 40)
    
    try:
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot confusion matrices
        classifier.plot_confusion_matrices(figsize=(15, 5))
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✅ Confusion matrices saved to plots/confusion_matrices.png")
        
        # Plot ROC curves
        classifier.plot_roc_curves(X_test, y_test, figsize=(10, 8))
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        print("✅ ROC curves saved to plots/roc_curves.png")
        
    except Exception as e:
        print(f"⚠️ Visualization generation failed: {e}")
    
    # Step 9: Model Comparison
    print("\n🏆 Step 9: Model Performance Comparison")
    print("-" * 40)
    
    comparison_df = classifier.get_model_comparison()
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("✅ Model comparison saved to results/model_comparison.csv")
    
    # Step 10: Feature Importance Analysis
    print("\n🔍 Step 10: Feature Importance Analysis")
    print("-" * 40)
    
    try:
        feature_names = preprocessor.get_feature_names()
        
        for model_name in ['logistic_regression', 'random_forest']:
            importance_df = classifier.get_feature_importance(
                model_name, feature_names, top_n=20
            )
            
            if importance_df is not None:
                print(f"\n📋 Top 10 features for {model_name.replace('_', ' ').title()}:")
                print(importance_df.head(10).to_string(index=False))
                
                # Save feature importance
                importance_df.to_csv(
                    f'results/{model_name}_feature_importance.csv', 
                    index=False
                )
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                top_features = importance_df.head(15)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Importance Score')
                plt.title(f'Top 15 Features - {model_name.replace("_", " ").title()}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'plots/{model_name}_feature_importance.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Feature importance plot saved to plots/{model_name}_feature_importance.png")
    
    except Exception as e:
        print(f"⚠️ Feature importance analysis failed: {e}")
    
    # Step 11: Save Models and Preprocessor
    print("\n💾 Step 11: Saving Models")
    print("-" * 40)
    
    try:
        # Save ML models
        classifier.save_models('models')
        
        # Save preprocessor
        import joblib
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        print("✅ Preprocessor saved to models/preprocessor.joblib")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'feature_matrix_shape': X.shape,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'vectorizer_type': preprocessor.vectorizer_type,
            'max_features': preprocessor.max_features,
            'models_trained': list(classifier.trained_models.keys()),
            'best_model': comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        }
        
        import json
        with open('models/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Training metadata saved to models/training_metadata.json")
        
    except Exception as e:
        print(f"⚠️ Model saving failed: {e}")
    
    # Step 12: Test Sample Predictions
    print("\n🧪 Step 12: Testing Sample Predictions")
    print("-" * 40)
    
    sample_texts = [
        "Scientists at MIT have developed a new renewable energy technology that could revolutionize solar power generation.",
        "SHOCKING: Doctors discover this one weird trick that cures all diseases instantly! Big Pharma doesn't want you to know!",
        "The local government announced new infrastructure projects to improve public transportation in the city.",
        "BREAKING: Aliens confirmed to be living among us, government officials reveal in exclusive interview!"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📝 Sample {i}: {text[:80]}...")
        
        try:
            # Process text
            X_sample = preprocessor.process_single_text(text)
            
            # Get predictions from all models
            for model_name in classifier.trained_models.keys():
                prediction, confidence = classifier.predict_single(model_name, X_sample)
                label = "Real" if prediction == 1 else "Fake"
                conf_str = f"{confidence:.2%}" if confidence else "N/A"
                
                print(f"  {model_name.replace('_', ' ').title()}: {label} (Confidence: {conf_str})")
                
        except Exception as e:
            print(f"  ⚠️ Prediction failed: {e}")
    
    # Step 13: Generate Training Report
    print("\n📋 Step 13: Generating Training Report")
    print("-" * 40)
    
    try:
        report = generate_training_report(
            comparison_df, results, metadata, sample_texts
        )
        
        with open('results/training_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Training report saved to results/training_report.txt")
        
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📁 Generated Files:")
    print("  📊 Models: models/")
    print("  📈 Plots: plots/")
    print("  📋 Results: results/")
    
    print("\n🚀 Next Steps:")
    print("  1. Run the web application: python app.py")
    print("  2. Access at: http://127.0.0.1:5000")
    print("  3. Test with your own news articles!")
    
    print("\n💡 Tips:")
    print("  • Check results/model_comparison.csv for detailed metrics")
    print("  • View plots/ directory for visualizations")
    print("  • Read results/training_report.txt for full analysis")

def generate_training_report(comparison_df, results, metadata, sample_texts):
    """
    Generate a comprehensive training report.
    
    Args:
        comparison_df: Model comparison dataframe
        results: Evaluation results
        metadata: Training metadata
        sample_texts: Sample texts used for testing
        
    Returns:
        str: Formatted training report
    """
    report = f"""
FAKE NEWS DETECTION - TRAINING REPORT
=====================================

Training Date: {metadata['training_date']}
Dataset Shape: {metadata['dataset_shape']}
Feature Matrix Shape: {metadata['feature_matrix_shape']}

CONFIGURATION
=============
Vectorizer Type: {metadata['vectorizer_type']}
Max Features: {metadata['max_features']}
Training Samples: {metadata['train_samples']}
Test Samples: {metadata['test_samples']}

MODEL PERFORMANCE
=================
{comparison_df.to_string(index=False)}

Best Performing Model: {metadata['best_model']}

DETAILED RESULTS
================
"""
    
    for model_name, result in results.items():
        report += f"\n{model_name.replace('_', ' ').title()}:\n"
        report += f"  Accuracy: {result['accuracy']:.4f}\n"
        report += f"  Precision: {result['precision']:.4f}\n"
        report += f"  Recall: {result['recall']:.4f}\n"
        report += f"  F1-Score: {result['f1_score']:.4f}\n"
        if result['roc_auc']:
            report += f"  ROC AUC: {result['roc_auc']:.4f}\n"
        report += f"  Confusion Matrix:\n{result['confusion_matrix']}\n"
    
    report += f"""
SAMPLE PREDICTIONS
==================
The following sample texts were used to test the trained models:

"""
    
    for i, text in enumerate(sample_texts, 1):
        report += f"{i}. {text}\n\n"
    
    report += """
FILES GENERATED
===============
• models/ - Trained machine learning models
• plots/ - Visualization files (confusion matrices, ROC curves, feature importance)
• results/ - Performance metrics and analysis files

USAGE INSTRUCTIONS
==================
1. Start the web application:
   python app.py

2. Access the application at:
   http://127.0.0.1:5000

3. Enter news text and select a model for analysis

4. Use the API endpoints for programmatic access:
   POST /api/predict - Make predictions
   GET /api/models - Get available models

NOTES
=====
• This system is designed for educational and research purposes
• Always verify important information through multiple trusted sources
• Model performance may vary with different datasets
• Regular retraining is recommended as fake news patterns evolve

Report generated automatically by the Fake News Detection Training Pipeline.
"""
    
    return report

if __name__ == "__main__":
    main()
