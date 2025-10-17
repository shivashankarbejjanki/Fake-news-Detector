"""
Desktop GUI Application for Fake News Detection
Uses tkinter for a native desktop experience
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
import os
import threading
import webbrowser
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_preprocessing import DataPreprocessor
    from ml_models import FakeNewsClassifier
    from advanced_analysis import TextAnalyzer
    from ensemble_predictor import EnsemblePredictor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class FakeNewsDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Fake News Detection System")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize models
        self.preprocessor = None
        self.classifier = None
        self.ensemble_predictor = None
        self.text_analyzer = TextAnalyzer()
        self.models_loaded = False
        
        self.setup_ui()
        self.load_models_async()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔍 Fake News Detection System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input section
        ttk.Label(main_frame, text="Enter news article:", 
                 font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # Text input
        self.text_input = scrolledtext.ScrolledText(main_frame, height=10, width=70, 
                                                   wrap=tk.WORD, font=('Arial', 10))
        self.text_input.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                            pady=(0, 10))
        
        # Sample buttons frame
        sample_frame = ttk.Frame(main_frame)
        sample_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Button(sample_frame, text="📰 Sample Real News", 
                  command=self.load_real_sample).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(sample_frame, text="⚠️ Sample Fake News", 
                  command=self.load_fake_sample).pack(side=tk.LEFT)
        
        # Model selection
        ttk.Label(main_frame, text="Select Model:", 
                 font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        self.model_var = tk.StringVar(value="All Models")
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var, 
                                  values=["All Models", "Logistic Regression", 
                                         "Random Forest", "Naive Bayes"], 
                                  state="readonly", width=20)
        model_combo.grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        
        # Analyze button
        self.analyze_btn = ttk.Button(main_frame, text="🔍 Analyze Article", 
                                     command=self.analyze_text, style='Accent.TButton')
        self.analyze_btn.grid(row=5, column=1, padx=(10, 0), pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=70, 
                                                     wrap=tk.WORD, font=('Courier', 9))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Loading models... Please wait.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Menu bar
        self.setup_menu()
    
    def setup_menu(self):
        """Set up the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear Text", command=self.clear_text)
        file_menu.add_command(label="Clear Results", command=self.clear_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Open Web Version", command=self.open_web_version)
        tools_menu.add_command(label="View Confidence Guide", command=self.show_confidence_guide)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def load_models_async(self):
        """Load models in a separate thread."""
        def load_models():
            try:
                self.status_var.set("Initializing preprocessor...")
                self.preprocessor = DataPreprocessor(vectorizer_type='tfidf', max_features=5000)
                
                self.status_var.set("Loading sample data...")
                df = self.preprocessor.load_data(use_sample=True)
                df_processed = self.preprocessor.preprocess_dataset(df)
                X = self.preprocessor.extract_features(df_processed)
                y = df_processed['label'].values
                
                self.status_var.set("Training models...")
                X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
                
                self.classifier = FakeNewsClassifier()
                self.classifier.train_models(X_train, y_train)
                
                self.status_var.set("Initializing ensemble predictor...")
                self.ensemble_predictor = EnsemblePredictor(self.classifier.trained_models)
                
                self.models_loaded = True
                self.status_var.set("✅ Ready! Enter text and click Analyze.")
                self.analyze_btn.config(state='normal')
                
            except Exception as e:
                self.status_var.set(f"❌ Error loading models: {str(e)}")
                messagebox.showerror("Error", f"Failed to load models: {str(e)}")
        
        # Disable analyze button initially
        self.analyze_btn.config(state='disabled')
        
        # Start loading in background
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()
    
    def load_real_sample(self):
        """Load a sample real news article."""
        sample_text = """Scientists at MIT have developed a new solar panel technology that increases efficiency by 40%. The research, published in Nature Energy, shows promising results for renewable energy applications. The team used advanced materials science to create panels that can capture more sunlight and convert it more efficiently than previous generations of solar technology. Dr. Sarah Johnson, the lead researcher, emphasized that further testing is needed to confirm the long-term viability of the technology."""
        
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(1.0, sample_text)
    
    def load_fake_sample(self):
        """Load a sample fake news article."""
        sample_text = """SHOCKING: Doctors discover this one weird trick that cures all diseases instantly! Big Pharma doesn't want you to know about this amazing secret that will change your life FOREVER! This incredible breakthrough has been HIDDEN from the public for years, but now the truth is finally revealed! ACT NOW before it's too late and this information gets BANNED!"""
        
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(1.0, sample_text)
    
    def analyze_text(self):
        """Analyze the input text."""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait.")
            return
        
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        
        if len(text) < 10:
            messagebox.showwarning("Warning", "Please enter at least 10 characters for analysis.")
            return
        
        # Start analysis in background
        self.progress.start()
        self.analyze_btn.config(state='disabled')
        self.status_var.set("Analyzing text...")
        
        def analyze():
            try:
                # Process text
                X_processed = self.preprocessor.process_single_text(text)
                
                # Get predictions
                results = {}
                model_choice = self.model_var.get()
                
                if model_choice == "All Models":
                    for model_name in self.classifier.trained_models.keys():
                        prediction, confidence = self.classifier.predict_single(model_name, X_processed)
                        results[model_name] = {
                            'prediction': 'Real' if prediction == 1 else 'Fake',
                            'confidence': confidence
                        }
                    
                    # Ensemble prediction
                    ensemble_result = self.ensemble_predictor.predict_ensemble(X_processed)
                else:
                    model_map = {
                        "Logistic Regression": "logistic_regression",
                        "Random Forest": "random_forest",
                        "Naive Bayes": "naive_bayes"
                    }
                    model_name = model_map[model_choice]
                    prediction, confidence = self.classifier.predict_single(model_name, X_processed)
                    results[model_name] = {
                        'prediction': 'Real' if prediction == 1 else 'Fake',
                        'confidence': confidence
                    }
                    ensemble_result = None
                
                # Text analysis
                text_analysis = self.text_analyzer.analyze_text_features(text)
                
                # Display results
                self.display_results(text, results, ensemble_result, text_analysis)
                
                self.progress.stop()
                self.analyze_btn.config(state='normal')
                self.status_var.set("✅ Analysis complete!")
                
            except Exception as e:
                self.progress.stop()
                self.analyze_btn.config(state='normal')
                self.status_var.set(f"❌ Analysis failed: {str(e)}")
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def display_results(self, text, results, ensemble_result, text_analysis):
        """Display analysis results."""
        self.results_text.delete(1.0, tk.END)
        
        report = f"""
FAKE NEWS DETECTION ANALYSIS REPORT
{'='*50}
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT TEXT:
{'-'*20}
{text[:200]}{'...' if len(text) > 200 else ''}

INDIVIDUAL MODEL PREDICTIONS:
{'-'*30}
"""
        
        for model_name, result in results.items():
            prediction = result['prediction']
            confidence = result.get('confidence', 0)
            status_icon = "✅" if prediction == "Real" else "❌"
            
            report += f"{status_icon} {model_name.replace('_', ' ').title()}:\n"
            report += f"   Prediction: {prediction}\n"
            report += f"   Confidence: {confidence:.1%}\n\n"
        
        if ensemble_result:
            prediction = ensemble_result['ensemble_prediction']
            confidence = ensemble_result['ensemble_confidence']
            consensus = ensemble_result['consensus_strength']
            status_icon = "✅" if prediction == "Real" else "❌"
            
            report += f"""ENSEMBLE PREDICTION (RECOMMENDED):
{'-'*35}
{status_icon} Final Prediction: {prediction}
   Confidence: {confidence:.1%}
   Consensus: {consensus.title()}
   Recommendation: {ensemble_result['recommendation']}

"""
        
        # Text analysis
        credibility_score = text_analysis['credibility_score']['score']
        fake_indicators = len(text_analysis['fake_indicators_found'])
        real_indicators = len(text_analysis['real_indicators_found'])
        
        report += f"""ADVANCED TEXT ANALYSIS:
{'-'*25}
📊 Credibility Score: {credibility_score:.2f}/1.00
⚠️  Fake Indicators Found: {fake_indicators}
✅ Real Indicators Found: {real_indicators}
📝 Word Count: {text_analysis['text_statistics']['word_count']}
📖 Readability: {text_analysis['readability']['complexity'].title()}

"""
        
        if text_analysis['fake_indicators_found']:
            report += "FAKE NEWS INDICATORS DETECTED:\n"
            for word, category in text_analysis['fake_indicators_found']:
                report += f"  • '{word}' ({category})\n"
            report += "\n"
        
        if text_analysis['real_indicators_found']:
            report += "REAL NEWS INDICATORS DETECTED:\n"
            for word, category in text_analysis['real_indicators_found']:
                report += f"  • '{word}' ({category})\n"
            report += "\n"
        
        # Overall assessment
        if ensemble_result:
            confidence = ensemble_result['ensemble_confidence']
        else:
            confidence = list(results.values())[0].get('confidence', 0)
        
        if confidence >= 0.8:
            assessment = "🟢 HIGH CONFIDENCE - Result is reliable"
        elif confidence >= 0.6:
            assessment = "🟡 MEDIUM CONFIDENCE - Consider additional verification"
        else:
            assessment = "🔴 LOW CONFIDENCE - Verify through multiple trusted sources"
        
        report += f"""OVERALL ASSESSMENT:
{'-'*20}
{assessment}

DISCLAIMER:
This system is for educational purposes. Always verify important 
information through multiple trusted sources.
"""
        
        self.results_text.insert(1.0, report)
    
    def clear_text(self):
        """Clear the input text."""
        self.text_input.delete(1.0, tk.END)
    
    def clear_results(self):
        """Clear the results."""
        self.results_text.delete(1.0, tk.END)
    
    def open_web_version(self):
        """Open the web version in browser."""
        webbrowser.open('http://127.0.0.1:5000')
    
    def show_confidence_guide(self):
        """Show confidence interpretation guide."""
        guide_text = """
CONFIDENCE LEVEL GUIDE:

🟢 HIGH CONFIDENCE (80%+):
- Model is very certain about its prediction
- Result is reliable, but still verify for important decisions

🟡 MEDIUM CONFIDENCE (60-80%):
- Model has moderate certainty
- Consider additional verification

🔴 LOW CONFIDENCE (Below 60%):
- Model is uncertain, borderline case
- High caution required - verify through multiple sources

ENSEMBLE CONSENSUS:
- Unanimous: All models agree
- Strong: 75%+ models agree
- Moderate: 60-75% models agree
- Weak: Less than 60% agreement

Always use critical thinking and verify important information!
"""
        messagebox.showinfo("Confidence Guide", guide_text)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
🔍 Fake News Detection System

Version: 1.0
Built with: Python, scikit-learn, NLTK, tkinter

Features:
• Multiple ML algorithms (Logistic Regression, Random Forest, Naive Bayes)
• Ensemble prediction for better accuracy
• Advanced text analysis
• Confidence scoring
• Real-time processing

Disclaimer:
This system is designed for educational and research purposes.
Always verify important information through multiple trusted sources.

© 2024 - Built with AI and Machine Learning
"""
        messagebox.showinfo("About", about_text)

def main():
    """Main function to run the desktop application."""
    root = tk.Tk()
    
    # Set up styling
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configure custom styles
    style.configure('Accent.TButton', foreground='white', background='#007bff')
    
    app = FakeNewsDetectorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user.")

if __name__ == "__main__":
    main()
