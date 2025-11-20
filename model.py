"""
Model architectures for sentiment analysis
Includes LSTM, GRU, and BERT-based models
"""

import numpy as np
from typing import Optional, Tuple
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.models import Sequential
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

import config


class SentimentLSTM:
    """LSTM-based sentiment analysis model"""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 lstm_units: int = config.LSTM_UNITS,
                 num_classes: int = config.NUM_CLASSES,
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH,
                 dropout_rate: float = config.DROPOUT_RATE,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True):
        """
        Initialize LSTM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            num_classes: Number of output classes
            max_sequence_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
            embedding_matrix: Pre-trained embedding matrix (optional)
            trainable_embeddings: Whether to train embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        self.model = None
        
        self._build_model(embedding_matrix, trainable_embeddings)
    
    def _build_model(self, embedding_matrix: Optional[np.ndarray], 
                     trainable_embeddings: bool):
        """Build the LSTM model architecture"""
        
        model = Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                weights=[embedding_matrix] if embedding_matrix is not None else None,
                trainable=trainable_embeddings,
                name='embedding'
            ),
            
            # Spatial Dropout for embeddings
            layers.SpatialDropout1D(self.dropout_rate),
            
            # Bidirectional LSTM
            layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True),
                name='bidirectional_lstm_1'
            ),
            layers.Dropout(self.dropout_rate),
            
            # Second LSTM layer
            layers.Bidirectional(
                layers.LSTM(self.lstm_units // 2, return_sequences=False),
                name='bidirectional_lstm_2'
            ),
            layers.Dropout(self.dropout_rate),
            
            # Dense layers
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        print("LSTM model built successfully")
        
    def compile_model(self, learning_rate: float = config.LEARNING_RATE):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled")
    
    def get_model(self) -> Model:
        """Return the Keras model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class SentimentGRU:
    """GRU-based sentiment analysis model"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 gru_units: int = config.GRU_UNITS,
                 num_classes: int = config.NUM_CLASSES,
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH,
                 dropout_rate: float = config.DROPOUT_RATE,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True):
        """
        Initialize GRU model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            gru_units: Number of GRU units
            num_classes: Number of output classes
            max_sequence_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
            embedding_matrix: Pre-trained embedding matrix (optional)
            trainable_embeddings: Whether to train embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        self.model = None
        
        self._build_model(embedding_matrix, trainable_embeddings)
    
    def _build_model(self, embedding_matrix: Optional[np.ndarray],
                     trainable_embeddings: bool):
        """Build the GRU model architecture"""
        
        model = Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                weights=[embedding_matrix] if embedding_matrix is not None else None,
                trainable=trainable_embeddings,
                name='embedding'
            ),
            
            # Spatial Dropout for embeddings
            layers.SpatialDropout1D(self.dropout_rate),
            
            # Bidirectional GRU
            layers.Bidirectional(
                layers.GRU(self.gru_units, return_sequences=True),
                name='bidirectional_gru_1'
            ),
            layers.Dropout(self.dropout_rate),
            
            # Second GRU layer
            layers.Bidirectional(
                layers.GRU(self.gru_units // 2, return_sequences=False),
                name='bidirectional_gru_2'
            ),
            layers.Dropout(self.dropout_rate),
            
            # Dense layers
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        print("GRU model built successfully")
    
    def compile_model(self, learning_rate: float = config.LEARNING_RATE):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled")
    
    def get_model(self) -> Model:
        """Return the Keras model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class SentimentCNN:
    """CNN-based sentiment analysis model"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 num_classes: int = config.NUM_CLASSES,
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH,
                 dropout_rate: float = config.DROPOUT_RATE,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True,
                 filter_sizes: list = [3, 4, 5],
                 num_filters: int = 128):
        """
        Initialize CNN model (inspired by Kim's CNN for sentence classification)
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            max_sequence_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
            embedding_matrix: Pre-trained embedding matrix (optional)
            trainable_embeddings: Whether to train embeddings
            filter_sizes: List of filter sizes for conv layers
            num_filters: Number of filters per filter size
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.model = None
        
        self._build_model(embedding_matrix, trainable_embeddings)
    
    def _build_model(self, embedding_matrix: Optional[np.ndarray],
                     trainable_embeddings: bool):
        """Build the CNN model architecture"""
        
        # Input layer
        inputs = layers.Input(shape=(self.max_sequence_length,), name='input')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            trainable=trainable_embeddings,
            name='embedding'
        )(inputs)
        
        # Dropout on embeddings
        embedding = layers.SpatialDropout1D(self.dropout_rate)(embedding)
        
        # Multiple parallel convolutional layers with different filter sizes
        conv_outputs = []
        for filter_size in self.filter_sizes:
            conv = layers.Conv1D(
                filters=self.num_filters,
                kernel_size=filter_size,
                activation='relu',
                name=f'conv_{filter_size}'
            )(embedding)
            pool = layers.GlobalMaxPooling1D(name=f'pool_{filter_size}')(conv)
            conv_outputs.append(pool)
        
        # Concatenate all conv outputs
        concatenated = layers.Concatenate(name='concatenate')(conv_outputs)
        
        # Dropout
        dropout = layers.Dropout(self.dropout_rate)(concatenated)
        
        # Dense layer
        dense = layers.Dense(64, activation='relu', name='dense')(dropout)
        dropout2 = layers.Dropout(self.dropout_rate)(dense)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(dropout2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='sentiment_cnn')
        print("CNN model built successfully")
    
    def compile_model(self, learning_rate: float = config.LEARNING_RATE):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled")
    
    def get_model(self) -> Model:
        """Return the Keras model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class SentimentBERT:
    """BERT-based sentiment analysis model using transformers"""
    
    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH,
                 model_name: str = 'bert-base-multilingual-cased',
                 dropout_rate: float = config.DROPOUT_RATE):
        """
        Initialize BERT model
        
        Args:
            num_classes: Number of output classes
            max_sequence_length: Maximum sequence length
            model_name: Name of pre-trained BERT model
            dropout_rate: Dropout rate
        """
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.model = None
        self.tokenizer = None
        
        try:
            from transformers import TFBertModel, BertTokenizer
            self.TFBertModel = TFBertModel
            self.BertTokenizer = BertTokenizer
            self._build_model()
        except ImportError:
            print("Warning: transformers library not available. Install with: pip install transformers")
    
    def _build_model(self):
        """Build BERT-based model"""
        
        # Load BERT tokenizer
        self.tokenizer = self.BertTokenizer.from_pretrained(self.model_name)
        
        # Input layers
        input_ids = layers.Input(shape=(self.max_sequence_length,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(self.max_sequence_length,), dtype=tf.int32, name='attention_mask')
        
        # BERT layer
        bert_model = self.TFBertModel.from_pretrained(self.model_name)
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = bert_output.last_hidden_state[:, 0, :]
        
        # Dropout
        dropout = layers.Dropout(self.dropout_rate)(cls_output)
        
        # Dense layer
        dense = layers.Dense(128, activation='relu', name='dense')(dropout)
        dropout2 = layers.Dropout(self.dropout_rate)(dense)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(dropout2)
        
        # Create model
        self.model = Model(inputs=[input_ids, attention_mask], outputs=outputs, name='sentiment_bert')
        print(f"BERT model ({self.model_name}) built successfully")
    
    def compile_model(self, learning_rate: float = 2e-5):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate (typically lower for BERT)
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("BERT model compiled")
    
    def tokenize_texts(self, texts: list) -> dict:
        """
        Tokenize texts using BERT tokenizer
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def get_model(self) -> Model:
        """Return the Keras model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


def create_model(model_type: str = 'lstm', **kwargs) -> Tuple[object, Model]:
    """
    Factory function to create sentiment analysis models
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'cnn', 'bert')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Tuple of (model_wrapper, keras_model)
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        model_wrapper = SentimentLSTM(**kwargs)
    elif model_type == 'gru':
        model_wrapper = SentimentGRU(**kwargs)
    elif model_type == 'cnn':
        model_wrapper = SentimentCNN(**kwargs)
    elif model_type == 'bert':
        model_wrapper = SentimentBERT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'lstm', 'gru', 'cnn', 'bert'")
    
    model_wrapper.compile_model()
    
    return model_wrapper, model_wrapper.get_model()


if __name__ == "__main__":
    # Example usage
    print("Creating sample models...\n")
    
    # LSTM model
    print("=" * 50)
    print("LSTM Model")
    print("=" * 50)
    lstm_wrapper, lstm_model = create_model(
        model_type='lstm',
        vocab_size=10000,
        embedding_dim=100,
        lstm_units=128,
        num_classes=3
    )
    lstm_wrapper.summary()
    
    # GRU model
    print("\n" + "=" * 50)
    print("GRU Model")
    print("=" * 50)
    gru_wrapper, gru_model = create_model(
        model_type='gru',
        vocab_size=10000,
        embedding_dim=100,
        gru_units=128,
        num_classes=3
    )
    gru_wrapper.summary()
    
    # CNN model
    print("\n" + "=" * 50)
    print("CNN Model")
    print("=" * 50)
    cnn_wrapper, cnn_model = create_model(
        model_type='cnn',
        vocab_size=10000,
        embedding_dim=100,
        num_classes=3
    )
    cnn_wrapper.summary()
