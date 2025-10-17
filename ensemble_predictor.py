"""
Ensemble Predictor for Enhanced Fake News Detection
Combines multiple models for better confidence and accuracy.
"""

import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for better accuracy.
    """
    
    def __init__(self, models_dict, weights=None):
        """
        Initialize ensemble predictor.
        
        Args:
            models_dict (dict): Dictionary of trained models
            weights (dict): Optional weights for each model
        """
        self.models = models_dict
        self.weights = weights or {name: 1.0 for name in models_dict.keys()}
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.5
        }
    
    def predict_ensemble(self, X_sample):
        """
        Make ensemble prediction using all models.
        
        Args:
            X_sample: Preprocessed text sample
            
        Returns:
            dict: Ensemble prediction results
        """
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            pred = model.predict(X_sample)[0]
            predictions[model_name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_sample)[0]
                probabilities[model_name] = {
                    'fake_prob': proba[0],
                    'real_prob': proba[1],
                    'confidence': max(proba)
                }
        
        # Calculate ensemble prediction
        ensemble_result = self._calculate_ensemble(predictions, probabilities)
        
        return ensemble_result
    
    def _calculate_ensemble(self, predictions, probabilities):
        """
        Calculate ensemble prediction using weighted voting.
        
        Args:
            predictions (dict): Individual model predictions
            probabilities (dict): Individual model probabilities
            
        Returns:
            dict: Ensemble results
        """
        # Weighted voting
        weighted_votes = {'fake': 0, 'real': 0}
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0)
            
            # Adjust weight by confidence if available
            if model_name in probabilities:
                confidence = probabilities[model_name]['confidence']
                weight *= confidence
            
            if pred == 0:  # Fake
                weighted_votes['fake'] += weight
            else:  # Real
                weighted_votes['real'] += weight
            
            total_weight += weight
        
        # Normalize votes
        fake_percentage = weighted_votes['fake'] / total_weight
        real_percentage = weighted_votes['real'] / total_weight
        
        # Final prediction
        final_prediction = 1 if real_percentage > fake_percentage else 0
        final_confidence = max(fake_percentage, real_percentage)
        
        # Consensus analysis
        consensus_info = self._analyze_consensus(predictions)
        
        return {
            'ensemble_prediction': 'Real' if final_prediction == 1 else 'Fake',
            'ensemble_confidence': final_confidence,
            'fake_probability': fake_percentage,
            'real_probability': real_percentage,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'consensus_strength': consensus_info['strength'],
            'agreement_level': consensus_info['agreement'],
            'confidence_level': self._get_confidence_level(final_confidence),
            'recommendation': self._get_recommendation(final_confidence, consensus_info)
        }
    
    def _analyze_consensus(self, predictions):
        """
        Analyze consensus among models.
        
        Args:
            predictions (dict): Individual model predictions
            
        Returns:
            dict: Consensus analysis
        """
        pred_values = list(predictions.values())
        pred_counts = Counter(pred_values)
        
        total_models = len(pred_values)
        max_agreement = max(pred_counts.values())
        
        agreement_percentage = max_agreement / total_models
        
        if agreement_percentage == 1.0:
            strength = "unanimous"
        elif agreement_percentage >= 0.75:
            strength = "strong"
        elif agreement_percentage >= 0.6:
            strength = "moderate"
        else:
            strength = "weak"
        
        return {
            'strength': strength,
            'agreement': agreement_percentage,
            'breakdown': dict(pred_counts)
        }
    
    def _get_confidence_level(self, confidence):
        """Get confidence level category."""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommendation(self, confidence, consensus_info):
        """
        Get recommendation based on confidence and consensus.
        
        Args:
            confidence (float): Final confidence score
            consensus_info (dict): Consensus analysis
            
        Returns:
            str: Recommendation text
        """
        if confidence >= 0.8 and consensus_info['strength'] in ['unanimous', 'strong']:
            return "High confidence prediction. Result is reliable."
        
        elif confidence >= 0.6 and consensus_info['strength'] in ['strong', 'moderate']:
            return "Moderate confidence. Consider additional verification."
        
        elif consensus_info['strength'] == 'weak':
            return "Models disagree significantly. Verify through multiple trusted sources."
        
        else:
            return "Low confidence prediction. Exercise caution and seek additional verification."

def demonstrate_ensemble():
    """
    Demonstrate ensemble prediction capabilities.
    """
    print("🤖 Ensemble Prediction Demo")
    print("=" * 40)
    
    # Simulate model predictions for demonstration
    sample_predictions = {
        'logistic_regression': 1,  # Real
        'random_forest': 0,        # Fake  
        'naive_bayes': 1           # Real
    }
    
    sample_probabilities = {
        'logistic_regression': {'fake_prob': 0.35, 'real_prob': 0.65, 'confidence': 0.65},
        'random_forest': {'fake_prob': 0.58, 'real_prob': 0.42, 'confidence': 0.58},
        'naive_bayes': {'fake_prob': 0.25, 'real_prob': 0.75, 'confidence': 0.75}
    }
    
    # Create mock models dict for demonstration
    class MockModel:
        def __init__(self, pred, proba):
            self.pred = pred
            self.proba = proba
        
        def predict(self, X):
            return [self.pred]
        
        def predict_proba(self, X):
            return [self.proba]
    
    models_dict = {
        'logistic_regression': MockModel(1, [0.35, 0.65]),
        'random_forest': MockModel(0, [0.58, 0.42]),
        'naive_bayes': MockModel(1, [0.25, 0.75])
    }
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(models_dict)
    
    # Simulate prediction
    result = ensemble._calculate_ensemble(sample_predictions, sample_probabilities)
    
    print(f"Individual Predictions:")
    for model, pred in sample_predictions.items():
        label = "Real" if pred == 1 else "Fake"
        conf = sample_probabilities[model]['confidence']
        print(f"  {model}: {label} ({conf:.1%} confidence)")
    
    print(f"\nEnsemble Result:")
    print(f"  Prediction: {result['ensemble_prediction']}")
    print(f"  Confidence: {result['ensemble_confidence']:.1%}")
    print(f"  Consensus: {result['consensus_strength']}")
    print(f"  Recommendation: {result['recommendation']}")

if __name__ == "__main__":
    demonstrate_ensemble()
