"""
BERT-based Fake News Detection Model (Bonus Implementation)
This module implements advanced fake news detection using pre-trained BERT models.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class BERTFakeNewsClassifier:
    """
    BERT-based classifier for fake news detection using Hugging Face transformers.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=512, num_labels=2):
        """
        Initialize the BERT classifier.
        
        Args:
            model_name (str): Pre-trained model name from Hugging Face
            max_length (int): Maximum sequence length
            num_labels (int): Number of classification labels
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
    
    def initialize_model(self, from_pretrained=True):
        """
        Initialize the tokenizer and model.
        
        Args:
            from_pretrained (bool): Whether to load pre-trained weights
        """
        print("Initializing BERT model...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if from_pretrained:
            # Load pre-trained model for sequence classification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification"
            )
        else:
            # Initialize model with random weights
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_labels
            self.model = AutoModelForSequenceClassification.from_config(config)
        
        self.model.to(self.device)
        print("Model initialized successfully!")
    
    def tokenize_data(self, texts, labels=None):
        """
        Tokenize text data for BERT input.
        
        Args:
            texts (list): List of text strings
            labels (list, optional): List of labels
            
        Returns:
            Dataset: Tokenized dataset
        """
        print("Tokenizing data...")
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        if labels is not None:
            dataset_dict['labels'] = torch.tensor(labels, dtype=torch.long)
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset
    
    def prepare_data(self, df, text_column='text', label_column='label', test_size=0.2):
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Test set proportion
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        print("Preparing data for BERT training...")
        
        # Extract texts and labels
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Tokenize datasets
        train_dataset = self.tokenize_data(train_texts, train_labels)
        val_dataset = self.tokenize_data(val_texts, val_labels)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics for training.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            dict: Computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, val_dataset, output_dir='./bert_model', 
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train the BERT model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Output directory for model
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
        """
        print("Starting BERT model training...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            save_total_limit=2,
            report_to=None,  # Disable wandb logging
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model training completed and saved to {output_dir}")
    
    def evaluate(self, test_dataset):
        """
        Evaluate the trained model.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            dict: Evaluation results
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        print("Evaluating BERT model...")
        results = self.trainer.evaluate(test_dataset)
        
        print("Evaluation Results:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        
        return results
    
    def predict(self, texts):
        """
        Make predictions on new texts.
        
        Args:
            texts (list or str): Text(s) to classify
            
        Returns:
            dict: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please initialize the model first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        results = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'labels': ['Fake' if p == 0 else 'Real' for p in predictions],
            'confidence': [max(prob) for prob in probabilities]
        }
        
        return results
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to saved model
        """
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "Model not initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Approximate size in MB
        }

class SimpleBERTClassifier:
    """
    Simplified BERT classifier for quick inference without full training setup.
    """
    
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize simplified BERT classifier.
        
        Args:
            model_name (str): Pre-trained model name
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model for zero-shot classification
        from transformers import pipeline
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",  # Alternative: use a general sentiment model
                device=0 if torch.cuda.is_available() else -1
            )
            print("Loaded pre-trained classification model")
        except Exception as e:
            print(f"Could not load pre-trained model: {e}")
            self.classifier = None
    
    def predict(self, text):
        """
        Make prediction using pre-trained model.
        
        Args:
            text (str): Text to classify
            
        Returns:
            dict: Prediction results
        """
        if self.classifier is None:
            return {"error": "Model not available"}
        
        try:
            result = self.classifier(text)
            
            # Convert to our format
            prediction = {
                "prediction": "Real" if result[0]['label'] == 'POSITIVE' else "Fake",
                "confidence": result[0]['score'],
                "model_used": "BERT (Pre-trained)"
            }
            
            return prediction
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

def main():
    """
    Demonstration of BERT-based fake news detection.
    """
    print("=== BERT Fake News Detection Demo ===\n")
    
    # Check if we have sample data
    try:
        from data_preprocessing import DataPreprocessor
        
        # Load sample data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(use_sample=True)
        
        print(f"Loaded {len(df)} samples for BERT training")
        
        # Initialize BERT classifier
        bert_classifier = BERTFakeNewsClassifier(
            model_name='distilbert-base-uncased',
            max_length=256  # Smaller for demo
        )
        
        # Initialize model
        bert_classifier.initialize_model()
        
        # Get model info
        model_info = bert_classifier.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        # Prepare data (small sample for demo)
        train_dataset, val_dataset = bert_classifier.prepare_data(df, test_size=0.3)
        
        # For demo purposes, we'll skip actual training (it takes time)
        print("\n[DEMO] Skipping actual training - would take several minutes")
        print("In production, you would run: bert_classifier.train(train_dataset, val_dataset)")
        
        # Demo prediction with simplified classifier
        print("\n=== Testing Simplified BERT Classifier ===")
        simple_bert = SimpleBERTClassifier()
        
        sample_texts = [
            "Scientists discover new renewable energy breakthrough",
            "SHOCKING: Aliens confirmed to be living among us!"
        ]
        
        for text in sample_texts:
            result = simple_bert.predict(text)
            print(f"\nText: {text}")
            print(f"Result: {result}")
        
    except ImportError as e:
        print(f"Could not import dependencies: {e}")
        print("Make sure to install: pip install transformers torch datasets")
    
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    main()
