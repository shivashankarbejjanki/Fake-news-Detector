"""
Data Preprocessing Module for Fake News Detection
This module handles data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for fake news detection.
    Handles text cleaning, tokenization, and feature extraction.
    """
    
    def __init__(self, vectorizer_type='tfidf', max_features=10000, test_size=0.2, random_state=42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
            max_features (int): Maximum number of features for vectorization
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
        # Initialize vectorizer based on type
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2)
            )
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    def load_data(self, file_path=None, use_sample=True):
        """
        Load dataset from file or create sample data for demonstration.
        
        Args:
            file_path (str): Path to the dataset file
            use_sample (bool): Whether to use sample data if no file provided
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path and pd.io.common.file_exists(file_path):
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
        elif use_sample:
            print("Creating sample dataset for demonstration...")
            # Create sample data for demonstration
            sample_data = {
                'title': [
                    "Scientists Discover New Species of Butterfly in Amazon",
                    "SHOCKING: Aliens Found Living Among Us, Government Confirms",
                    "Local School Wins National Science Competition",
                    "Miracle Cure for All Diseases Found in Kitchen Spice",
                    "New Study Shows Benefits of Regular Exercise",
                    "BREAKING: Time Travel Invented by Teenager in Garage",
                    "Climate Change Report Shows Concerning Trends",
                    "Celebrity Spotted Eating Normal Food Like Regular Person",
                    "Economic Growth Continues for Third Quarter",
                    "Doctors Hate This One Simple Trick to Lose Weight",
                    "University Researchers Develop New Solar Panel Technology",
                    "EXCLUSIVE: Bigfoot Caught on Camera Shopping at Mall",
                    "New Traffic Laws Take Effect Next Month",
                    "Ancient Aliens Built the Pyramids, Expert Claims",
                    "Local Restaurant Wins Award for Community Service"
                ],
                'text': [
                    "Researchers from the University of São Paulo have identified a new species of butterfly in the Amazon rainforest. The discovery was published in the Journal of Lepidoptera Research after extensive field studies.",
                    "Government sources allegedly confirm that extraterrestrial beings have been living among humans for decades. This shocking revelation comes from unnamed officials who claim to have inside knowledge.",
                    "Jefferson Middle School's science team took first place in the National Science Bowl competition, beating teams from across the country with their knowledge of physics and chemistry.",
                    "A common kitchen spice has been found to cure every known disease according to a study that definitely exists. Doctors are amazed by this one simple ingredient that Big Pharma doesn't want you to know about.",
                    "A comprehensive study published in the New England Journal of Medicine confirms that regular physical exercise contributes to improved cardiovascular health and mental wellbeing.",
                    "A 16-year-old from suburban Ohio claims to have invented a working time machine in his family's garage. The device allegedly uses quantum mechanics and household items.",
                    "The latest climate report from the Intergovernmental Panel on Climate Change shows accelerating trends in global temperature rise and sea level increase.",
                    "Popular celebrity was photographed eating a sandwich at a local deli, proving that famous people also consume food for sustenance like ordinary humans.",
                    "The national economy shows continued growth for the third consecutive quarter, with unemployment rates declining and consumer confidence rising.",
                    "This amazing weight loss secret that doctors don't want you to know will help you lose 50 pounds in one week without diet or exercise. Click here to learn more!",
                    "Engineers at MIT have developed a new type of solar panel that is 40% more efficient than current technology, potentially revolutionizing renewable energy.",
                    "Exclusive footage shows a large, hairy humanoid creature purchasing groceries at a shopping mall in Oregon. Cryptozoology experts are calling it definitive proof.",
                    "New traffic regulations regarding speed limits in school zones will be implemented starting next month, with increased fines for violations during school hours.",
                    "According to self-proclaimed expert Dr. Giorgio Tsoukalos, ancient extraterrestrial visitors were responsible for constructing the Egyptian pyramids using advanced alien technology.",
                    "Maria's Family Restaurant received the Community Champion Award for their work providing free meals to local families in need during the pandemic."
                ],
                'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
            }
            df = pd.DataFrame(sample_data)
        else:
            raise FileNotFoundError("No dataset file provided and sample data disabled.")
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenize text and apply stemming.
        
        Args:
            text (str): Text to tokenize and stem
            
        Returns:
            str: Processed text
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        processed_tokens = [
            self.stemmer.stem(token) for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def preprocess_dataset(self, df, text_column='text', label_column='label'):
        """
        Preprocess the entire dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("Starting data preprocessing...")
        
        # Handle missing values
        df = df.dropna(subset=[text_column, label_column])
        
        # Combine title and text if both exist
        if 'title' in df.columns and text_column == 'text':
            df['combined_text'] = df['title'].fillna('') + ' ' + df[text_column].fillna('')
            text_column = 'combined_text'
        
        # Clean text
        print("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Advanced preprocessing (tokenization and stemming)
        print("Applying tokenization and stemming...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_stem)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"Preprocessing completed. Final dataset shape: {df.shape}")
        return df
    
    def extract_features(self, df, text_column='processed_text', fit_vectorizer=True):
        """
        Extract features using TF-IDF or Count Vectorizer.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            text_column (str): Name of the processed text column
            fit_vectorizer (bool): Whether to fit the vectorizer
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        print(f"Extracting features using {self.vectorizer_type} vectorizer...")
        
        if fit_vectorizer:
            features = self.vectorizer.fit_transform(df[text_column])
        else:
            features = self.vectorizer.transform(df[text_column])
        
        print(f"Feature extraction completed. Shape: {features.shape}")
        return features
    
    def split_data(self, X, y):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"Splitting data with test size: {self.test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """
        Get feature names from the vectorizer.
        
        Returns:
            list: Feature names
        """
        if self.vectorizer is not None:
            return self.vectorizer.get_feature_names_out()
        return None
    
    def process_single_text(self, text):
        """
        Process a single text for prediction.
        
        Args:
            text (str): Input text
            
        Returns:
            scipy.sparse matrix: Processed feature vector
        """
        # Clean and preprocess the text
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_stem(cleaned)
        
        # Transform using fitted vectorizer
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please fit the vectorizer first.")
        
        return self.vectorizer.transform([processed])

def main():
    """
    Demonstration of the data preprocessing pipeline.
    """
    print("=== Fake News Detection - Data Preprocessing Demo ===\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(vectorizer_type='tfidf', max_features=5000)
    
    # Load sample data
    df = preprocessor.load_data(use_sample=True)
    
    # Display basic information
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Preprocess the dataset
    df_processed = preprocessor.preprocess_dataset(df)
    
    # Extract features
    X = preprocessor.extract_features(df_processed)
    y = df_processed['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test single text processing
    sample_text = "This is a sample news article about recent scientific discoveries."
    processed_sample = preprocessor.process_single_text(sample_text)
    print(f"\nSample text processing:")
    print(f"Original: {sample_text}")
    print(f"Processed shape: {processed_sample.shape}")

if __name__ == "__main__":
    main()
