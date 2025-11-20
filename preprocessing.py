"""
Text preprocessing module
Handles text cleaning, tokenization, padding, and embeddings
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import pickle
import os

# Try to import nltk and download required data
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using basic tokenization.")

import config


class TextPreprocessor:
    """Class to handle text preprocessing"""
    
    def __init__(self, 
                 max_vocab_size: int = config.MAX_VOCAB_SIZE,
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH,
                 use_stopwords: bool = False,
                 language: str = 'french'):
        """
        Initialize TextPreprocessor
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_sequence_length: Maximum sequence length for padding
            use_stopwords: Whether to remove stopwords
            language: Language for stopwords ('french', 'english')
        """
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_stopwords = use_stopwords
        self.language = language
        
        # Initialize vocabulary
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        
        # Load stopwords if needed
        if use_stopwords and NLTK_AVAILABLE:
            self.stopwords = set(stopwords.words(language))
        else:
            self.stopwords = set()
    
    def clean_text(self, text: str, lowercase: bool = True, 
                   remove_punctuation: bool = True,
                   remove_numbers: bool = False,
                   remove_extra_spaces: bool = True) -> str:
        """
        Clean text by removing unwanted characters
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_extra_spaces: Remove extra whitespace
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (for social media text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text, language=self.language)
        else:
            # Simple whitespace tokenization
            tokens = text.split()
        
        # Remove stopwords if specified
        if self.use_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from list of texts
        
        Args:
            texts: List of text strings
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        # Sort by frequency and keep top max_vocab_size words
        sorted_words = sorted(self.word_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        # Build word to index mapping
        for idx, (word, count) in enumerate(sorted_words, start=2):  # Start at 2 (after PAD and UNK)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        print(f"Most common words: {sorted_words[:10]}")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers
        
        Args:
            texts: List of text strings
            
        Returns:
            List of integer sequences
        """
        sequences = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            
            # Convert tokens to indices
            sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                       for token in tokens]
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], 
                     maxlen: Optional[int] = None,
                     padding: str = 'post',
                     truncating: str = 'post') -> np.ndarray:
        """
        Pad sequences to the same length
        
        Args:
            sequences: List of integer sequences
            maxlen: Maximum length (uses self.max_sequence_length if None)
            padding: 'pre' or 'post' padding
            truncating: 'pre' or 'post' truncating
            
        Returns:
            Padded numpy array
        """
        if maxlen is None:
            maxlen = self.max_sequence_length
        
        padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            # Truncate if necessary
            if len(seq) > maxlen:
                if truncating == 'post':
                    seq = seq[:maxlen]
                else:  # 'pre'
                    seq = seq[-maxlen:]
            
            # Pad if necessary
            if len(seq) < maxlen:
                if padding == 'post':
                    padded[i, :len(seq)] = seq
                else:  # 'pre'
                    padded[i, -len(seq):] = seq
            else:
                padded[i] = seq
        
        return padded
    
    def preprocess_data(self, texts: List[str], 
                       build_vocab: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            texts: List of text strings
            build_vocab: Whether to build vocabulary (only for training data)
            
        Returns:
            Padded sequences as numpy array
        """
        if build_vocab:
            self.build_vocabulary(texts)
        
        sequences = self.texts_to_sequences(texts)
        padded_sequences = self.pad_sequences(sequences)
        
        return padded_sequences
    
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor configuration and vocabulary
        
        Args:
            filepath: Path to save the preprocessor
        """
        preprocessor_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_counts': self.word_counts,
            'max_vocab_size': self.max_vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'use_stopwords': self.use_stopwords,
            'language': self.language
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'TextPreprocessor':
        """
        Load preprocessor from file
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            TextPreprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        preprocessor = cls(
            max_vocab_size=preprocessor_data['max_vocab_size'],
            max_sequence_length=preprocessor_data['max_sequence_length'],
            use_stopwords=preprocessor_data['use_stopwords'],
            language=preprocessor_data['language']
        )
        
        preprocessor.word_to_idx = preprocessor_data['word_to_idx']
        preprocessor.idx_to_word = preprocessor_data['idx_to_word']
        preprocessor.word_counts = preprocessor_data['word_counts']
        
        print(f"Preprocessor loaded from {filepath}")
        print(f"Vocabulary size: {len(preprocessor.word_to_idx)}")
        
        return preprocessor


class EmbeddingLoader:
    """Class to load pre-trained word embeddings"""
    
    @staticmethod
    def load_glove(filepath: str, word_to_idx: Dict[str, int], 
                   embedding_dim: int) -> np.ndarray:
        """
        Load GloVe embeddings
        
        Args:
            filepath: Path to GloVe file
            word_to_idx: Word to index mapping
            embedding_dim: Dimension of embeddings
            
        Returns:
            Embedding matrix
        """
        print(f"Loading GloVe embeddings from {filepath}...")
        
        embeddings_index = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        print(f"Found {len(embeddings_index)} word vectors.")
        
        # Create embedding matrix
        vocab_size = len(word_to_idx)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        found = 0
        for word, idx in word_to_idx.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
                found += 1
            else:
                # Initialize with random values for unknown words
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
        print(f"Found embeddings for {found}/{vocab_size} words ({found/vocab_size*100:.1f}%)")
        
        return embedding_matrix
    
    @staticmethod
    def load_word2vec(filepath: str, word_to_idx: Dict[str, int]) -> np.ndarray:
        """
        Load Word2Vec embeddings using gensim
        
        Args:
            filepath: Path to Word2Vec file
            word_to_idx: Word to index mapping
            
        Returns:
            Embedding matrix
        """
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError("gensim is required for Word2Vec. Install with: pip install gensim")
        
        print(f"Loading Word2Vec embeddings from {filepath}...")
        
        # Load Word2Vec model
        word_vectors = KeyedVectors.load_word2vec_format(filepath, binary=True)
        embedding_dim = word_vectors.vector_size
        
        # Create embedding matrix
        vocab_size = len(word_to_idx)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        found = 0
        for word, idx in word_to_idx.items():
            if word in word_vectors:
                embedding_matrix[idx] = word_vectors[word]
                found += 1
            else:
                # Initialize with random values for unknown words
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
        print(f"Found embeddings for {found}/{vocab_size} words ({found/vocab_size*100:.1f}%)")
        
        return embedding_matrix


if __name__ == "__main__":
    # Example usage
    print("TextPreprocessor Example")
    
    # Sample texts
    sample_texts = [
        "Ce film est VRAIMENT magnifique! J'ai adoré chaque minute.",
        "Très déçu par ce produit... Qualité médiocre et prix excessif.",
        "C'est ok, rien de spécial. Ça fait le travail.",
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        max_vocab_size=1000,
        max_sequence_length=50,
        use_stopwords=False
    )
    
    # Test cleaning
    print("\nCleaning examples:")
    for text in sample_texts[:2]:
        cleaned = preprocessor.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}\n")
    
    # Build vocabulary and preprocess
    padded = preprocessor.preprocess_data(sample_texts, build_vocab=True)
    
    print(f"\nPadded sequences shape: {padded.shape}")
    print(f"First sequence: {padded[0][:20]}")  # Show first 20 tokens
    
    # Save and load
    save_path = os.path.join(config.MODELS_DIR, 'preprocessor.pkl')
    preprocessor.save_preprocessor(save_path)
    
    loaded_preprocessor = TextPreprocessor.load_preprocessor(save_path)
    print("\nPreprocessor successfully saved and loaded!")
