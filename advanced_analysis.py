"""
Advanced Analysis Tools for Fake News Detection
Provides detailed text analysis and confidence improvement techniques.
"""

import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

class TextAnalyzer:
    """
    Advanced text analyzer for fake news detection insights.
    """
    
    def __init__(self):
        # Fake news indicators (common patterns)
        self.fake_indicators = {
            'sensational_words': [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'secret',
                'hidden', 'exposed', 'revealed', 'exclusive', 'breaking',
                'urgent', 'must see', 'you won\'t believe', 'doctors hate',
                'scientists baffled', 'miracle', 'instant', 'guaranteed'
            ],
            'emotional_appeals': [
                'outraged', 'furious', 'devastated', 'terrified', 'panicked',
                'shocked', 'disgusted', 'betrayed', 'abandoned', 'forgotten'
            ],
            'conspiracy_terms': [
                'cover up', 'conspiracy', 'they don\'t want you to know',
                'hidden agenda', 'secret society', 'illuminati', 'deep state',
                'mainstream media', 'big pharma', 'government lies'
            ],
            'urgency_phrases': [
                'act now', 'limited time', 'before it\'s too late', 'urgent',
                'immediate action', 'don\'t wait', 'time is running out'
            ]
        }
        
        # Real news indicators
        self.real_indicators = {
            'credible_sources': [
                'according to', 'research shows', 'study finds', 'data indicates',
                'experts say', 'published in', 'peer reviewed', 'university',
                'institute', 'journal', 'professor', 'researcher'
            ],
            'factual_language': [
                'approximately', 'estimated', 'preliminary', 'ongoing',
                'investigation', 'analysis', 'evidence', 'confirmed',
                'verified', 'documented', 'reported', 'observed'
            ],
            'attribution': [
                'spokesperson said', 'official statement', 'press release',
                'interview', 'quoted', 'attributed to', 'source'
            ]
        }
    
    def analyze_text_features(self, text):
        """
        Analyze text for various indicators of fake/real news.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Analysis results
        """
        text_lower = text.lower()
        
        # Count indicators
        fake_score = 0
        real_score = 0
        
        fake_found = []
        real_found = []
        
        # Check fake indicators
        for category, words in self.fake_indicators.items():
            found_words = [word for word in words if word in text_lower]
            if found_words:
                fake_found.extend([(word, category) for word in found_words])
                fake_score += len(found_words) * 2  # Weight fake indicators more
        
        # Check real indicators
        for category, words in self.real_indicators.items():
            found_words = [word for word in words if word in text_lower]
            if found_words:
                real_found.extend([(word, category) for word in found_words])
                real_score += len(found_words)
        
        # Additional analysis
        analysis = {
            'fake_indicators_found': fake_found,
            'real_indicators_found': real_found,
            'fake_score': fake_score,
            'real_score': real_score,
            'text_statistics': self._get_text_statistics(text),
            'readability': self._calculate_readability(text),
            'sentiment_indicators': self._analyze_sentiment_indicators(text),
            'credibility_score': self._calculate_credibility_score(fake_score, real_score, text)
        }
        
        return analysis
    
    def _get_text_statistics(self, text):
        """Get basic text statistics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def _calculate_readability(self, text):
        """Calculate simple readability metrics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return {'flesch_score': 0, 'complexity': 'unknown'}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = np.mean([self._count_syllables(word) for word in words])
        
        # Simplified Flesch Reading Ease
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        if flesch_score >= 60:
            complexity = 'easy'
        elif flesch_score >= 30:
            complexity = 'moderate'
        else:
            complexity = 'difficult'
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'complexity': complexity,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables': avg_syllables
        }
    
    def _count_syllables(self, word):
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _analyze_sentiment_indicators(self, text):
        """Analyze sentiment-related indicators."""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'shocking']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_ratio': (positive_count - negative_count) / max(1, positive_count + negative_count)
        }
    
    def _calculate_credibility_score(self, fake_score, real_score, text):
        """Calculate overall credibility score."""
        # Base score from indicators
        indicator_score = (real_score - fake_score) / max(1, real_score + fake_score)
        
        # Adjust based on text characteristics
        stats = self._get_text_statistics(text)
        
        # Penalize excessive caps and exclamations
        caps_penalty = min(0.3, stats['caps_ratio'] * 2)
        exclamation_penalty = min(0.2, stats['exclamation_count'] / max(1, stats['word_count']) * 10)
        
        # Bonus for balanced sentence structure
        sentence_bonus = 0.1 if 10 <= stats['avg_sentence_length'] <= 25 else 0
        
        final_score = indicator_score - caps_penalty - exclamation_penalty + sentence_bonus
        
        # Normalize to 0-1 range
        credibility_score = (final_score + 1) / 2
        
        return {
            'score': max(0, min(1, credibility_score)),
            'indicator_contribution': indicator_score,
            'caps_penalty': caps_penalty,
            'exclamation_penalty': exclamation_penalty,
            'sentence_bonus': sentence_bonus
        }
    
    def generate_analysis_report(self, text, ml_predictions=None):
        """
        Generate comprehensive analysis report.
        
        Args:
            text (str): Text to analyze
            ml_predictions (dict): Optional ML model predictions
            
        Returns:
            str: Formatted analysis report
        """
        analysis = self.analyze_text_features(text)
        
        report = f"""
ADVANCED TEXT ANALYSIS REPORT
============================

TEXT STATISTICS:
- Word Count: {analysis['text_statistics']['word_count']}
- Sentence Count: {analysis['text_statistics']['sentence_count']}
- Average Words per Sentence: {analysis['text_statistics']['avg_sentence_length']:.1f}
- Exclamation Marks: {analysis['text_statistics']['exclamation_count']}
- Capital Letters Ratio: {analysis['text_statistics']['caps_ratio']:.1%}

READABILITY:
- Flesch Score: {analysis['readability']['flesch_score']:.1f}
- Complexity: {analysis['readability']['complexity'].title()}

CREDIBILITY INDICATORS:
- Fake News Indicators Found: {len(analysis['fake_indicators_found'])}
- Real News Indicators Found: {len(analysis['real_indicators_found'])}
- Credibility Score: {analysis['credibility_score']['score']:.2f}/1.00

FAKE NEWS INDICATORS DETECTED:
"""
        
        if analysis['fake_indicators_found']:
            for word, category in analysis['fake_indicators_found']:
                report += f"  • '{word}' ({category})\n"
        else:
            report += "  None detected\n"
        
        report += "\nREAL NEWS INDICATORS DETECTED:\n"
        
        if analysis['real_indicators_found']:
            for word, category in analysis['real_indicators_found']:
                report += f"  • '{word}' ({category})\n"
        else:
            report += "  None detected\n"
        
        # Add ML predictions if provided
        if ml_predictions:
            report += f"\nMACHINE LEARNING PREDICTIONS:\n"
            for model, result in ml_predictions.items():
                if isinstance(result, dict) and 'prediction' in result:
                    pred = result['prediction']
                    conf = result.get('confidence', 'N/A')
                    report += f"  • {model}: {pred} (Confidence: {conf})\n"
        
        # Overall assessment
        credibility = analysis['credibility_score']['score']
        if credibility >= 0.7:
            assessment = "HIGH CREDIBILITY - Text shows strong indicators of legitimate news"
        elif credibility >= 0.5:
            assessment = "MODERATE CREDIBILITY - Mixed indicators, verify through additional sources"
        else:
            assessment = "LOW CREDIBILITY - Text shows patterns often associated with misinformation"
        
        report += f"\nOVERALL ASSESSMENT:\n{assessment}\n"
        
        return report

def create_confidence_visualization(predictions_data):
    """
    Create visualization for model confidence comparison.
    
    Args:
        predictions_data (dict): Dictionary with model predictions and confidence scores
    """
    models = list(predictions_data.keys())
    confidences = [predictions_data[model].get('confidence', 0) for model in models]
    predictions = [predictions_data[model].get('prediction', 'Unknown') for model in models]
    
    # Create color map
    colors = ['red' if pred == 'Fake' else 'green' for pred in predictions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, confidences, color=colors, alpha=0.7)
    
    # Add confidence threshold lines
    plt.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.5, label='High Confidence (80%)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence (60%)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Low Confidence (50%)')
    
    plt.xlabel('Models')
    plt.ylabel('Confidence Score')
    plt.title('Model Confidence Comparison')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add value labels on bars
    for bar, conf, pred in zip(bars, confidences, predictions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.1%}\n({pred})', ha='center', va='bottom')
    
    plt.savefig('plots/confidence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Demonstrate advanced analysis capabilities.
    """
    print("Advanced Text Analysis Demo")
    print("=" * 40)
    
    # Sample texts for analysis
    fake_sample = """
    SHOCKING DISCOVERY! Scientists are BAFFLED by this one weird trick that 
    doctors DON'T WANT YOU TO KNOW! This AMAZING secret will change your life 
    FOREVER! Big Pharma is trying to HIDE this incredible breakthrough! 
    ACT NOW before it's too late!!!
    """
    
    real_sample = """
    According to a study published in the Journal of Medical Research, 
    researchers at Stanford University have identified a potential new 
    treatment approach. The preliminary findings, based on a controlled 
    trial involving 200 participants, suggest promising results. 
    Dr. Smith, the lead researcher, emphasized that further investigation 
    is needed to confirm these initial observations.
    """
    
    analyzer = TextAnalyzer()
    
    print("\nFAKE NEWS SAMPLE ANALYSIS:")
    print("-" * 30)
    fake_analysis = analyzer.analyze_text_features(fake_sample)
    print(f"Credibility Score: {fake_analysis['credibility_score']['score']:.2f}")
    print(f"Fake Indicators: {len(fake_analysis['fake_indicators_found'])}")
    print(f"Real Indicators: {len(fake_analysis['real_indicators_found'])}")
    
    print("\nREAL NEWS SAMPLE ANALYSIS:")
    print("-" * 30)
    real_analysis = analyzer.analyze_text_features(real_sample)
    print(f"Credibility Score: {real_analysis['credibility_score']['score']:.2f}")
    print(f"Fake Indicators: {len(real_analysis['fake_indicators_found'])}")
    print(f"Real Indicators: {len(real_analysis['real_indicators_found'])}")
    
    # Generate full report
    print("\nDETAILED REPORT FOR FAKE SAMPLE:")
    print(analyzer.generate_analysis_report(fake_sample))

if __name__ == "__main__":
    main()
