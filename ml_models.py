"""
Machine Learning Models Module for Fake News Detection
This module implements various ML algorithms for fake news classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FakeNewsClassifier:
    """
    A comprehensive machine learning classifier for fake news detection.
    Supports multiple algorithms and provides detailed evaluation metrics.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the FakeNewsClassifier.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        
        # Initialize models with default parameters
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all machine learning models with default parameters."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            )
        }
    
    def train_models(self, X_train, y_train, cv_folds=5):
        """
        Train all models and perform cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
        """
        print("=== Training Machine Learning Models ===\n")
        
        for model_name, model in self.models.items():
            print(f"Training {model_name.replace('_', ' ').title()}...")
            
            # Train the model
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("-" * 50)
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results for all models
        """
        print("=== Evaluating Models on Test Data ===\n")
        
        results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"Evaluating {model_name.replace('_', ' ').title()}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # ROC AUC score (if probability predictions available)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"ROC AUC: {roc_auc:.4f}")
            print("-" * 50)
        
        self.evaluation_results = results
        return results
    
    def plot_confusion_matrices(self, figsize=(15, 5)):
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize (tuple): Figure size for the plot
        """
        if not self.evaluation_results:
            print("No evaluation results available. Please run evaluate_models first.")
            return
        
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[idx],
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real']
            )
            
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test, figsize=(10, 8)):
        """
        Plot ROC curves for models that support probability predictions.
        
        Args:
            X_test: Test features
            y_test: Test labels
            figsize (tuple): Figure size for the plot
        """
        plt.figure(figsize=figsize)
        
        for model_name, model in self.trained_models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(
                    fpr, tpr, 
                    label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})',
                    linewidth=2
                )
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Fake News Detection Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('models/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_model_comparison(self):
        """
        Get a comparison table of all model performances.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        if not self.evaluation_results:
            print("No evaluation results available. Please run evaluate_models first.")
            return None
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results['roc_auc'] if results['roc_auc'] else 'N/A'
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison.round(4)
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """
        Perform hyperparameter tuning for all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
        """
        print("=== Hyperparameter Tuning ===\n")
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        }
        
        tuned_models = {}
        
        for model_name, model in self.models.items():
            if model_name in param_grids:
                print(f"Tuning {model_name.replace('_', ' ').title()}...")
                
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name],
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                tuned_models[model_name] = grid_search.best_estimator_
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
                print("-" * 50)
        
        # Update trained models with tuned versions
        self.trained_models.update(tuned_models)
        print("Hyperparameter tuning completed!\n")
    
    def save_models(self, models_dir='models'):
        """
        Save all trained models to disk.
        
        Args:
            models_dir (str): Directory to save models
        """
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.trained_models.items():
            filename = f"{models_dir}/{model_name}_{timestamp}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            object: Loaded model
        """
        return joblib.load(model_path)
    
    def predict_single(self, model_name, X_single):
        """
        Make prediction for a single sample.
        
        Args:
            model_name (str): Name of the model to use
            X_single: Single sample features
            
        Returns:
            tuple: (prediction, probability)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.trained_models[model_name]
        prediction = model.predict(X_single)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_single)[0]
            confidence = max(probability)
        else:
            confidence = None
        
        return prediction, confidence
    
    def get_feature_importance(self, model_name, feature_names=None, top_n=20):
        """
        Get feature importance for models that support it.
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = abs(model.coef_[0])
        else:
            print(f"Model {model_name} does not support feature importance.")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df

def main():
    """
    Demonstration of the machine learning models.
    """
    print("=== Fake News Detection - ML Models Demo ===\n")
    
    # This is a demonstration - in practice, you would load preprocessed data
    from data_preprocessing import DataPreprocessor
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor(vectorizer_type='tfidf', max_features=1000)
    df = preprocessor.load_data(use_sample=True)
    
    # Preprocess data
    df_processed = preprocessor.preprocess_dataset(df)
    X = preprocessor.extract_features(df_processed)
    y = df_processed['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Initialize classifier
    classifier = FakeNewsClassifier()
    
    # Train models
    classifier.train_models(X_train, y_train)
    
    # Evaluate models
    results = classifier.evaluate_models(X_test, y_test)
    
    # Display comparison
    comparison = classifier.get_model_comparison()
    print("\n=== Model Comparison ===")
    print(comparison)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Plot results
    classifier.plot_confusion_matrices()
    classifier.plot_roc_curves(X_test, y_test)
    
    # Save models
    classifier.save_models()
    
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    main()
