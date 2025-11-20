"""
Model architectures for sentiment analysis - PyTorch version
Compatible with Python 3.12+
"""

import numpy as np
from typing import Optional, Tuple
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

import config


class SentimentLSTM_PyTorch(nn.Module):
    """LSTM-based sentiment analysis model using PyTorch"""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 lstm_units: int = config.LSTM_UNITS,
                 num_classes: int = config.NUM_CLASSES,
                 dropout_rate: float = config.DROPOUT_RATE,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True):
        super(SentimentLSTM_PyTorch, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True, bidirectional=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # Spatial dropout on embeddings
        x = x.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)
        x = self.spatial_dropout(x)
        x = x.squeeze(1)  # (batch, seq_len, embedding_dim)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        # Take the last output
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


class SentimentGRU_PyTorch(nn.Module):
    """GRU-based sentiment analysis model using PyTorch"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 gru_units: int = config.GRU_UNITS,
                 num_classes: int = config.NUM_CLASSES,
                 dropout_rate: float = config.DROPOUT_RATE,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True):
        super(SentimentGRU_PyTorch, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Bidirectional GRU layers
        self.gru1 = nn.GRU(embedding_dim, gru_units, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(gru_units * 2, gru_units // 2, batch_first=True, bidirectional=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(gru_units, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # Spatial dropout on embeddings
        x = x.unsqueeze(1)
        x = self.spatial_dropout(x)
        x = x.squeeze(1)
        
        # GRU layers
        x, _ = self.gru1(x)
        x = self.dropout(x)
        
        x, _ = self.gru2(x)
        x = self.dropout(x)
        
        # Take the last output
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


class SentimentCNN_PyTorch(nn.Module):
    """CNN-based sentiment analysis model using PyTorch"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 num_classes: int = config.NUM_CLASSES,
                 dropout_rate: float = config.DROPOUT_RATE,
                 embedding_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = True,
                 filter_sizes: list = [3, 4, 5],
                 num_filters: int = 128):
        super(SentimentCNN_PyTorch, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # Spatial dropout on embeddings
        x = x.unsqueeze(1)
        x = self.spatial_dropout(x)
        x = x.squeeze(1)
        
        # Transpose for conv1d: (batch, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_results = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len - kernel_size + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_results.append(pooled.squeeze(2))  # (batch, num_filters)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_results, dim=1)  # (batch, len(filter_sizes) * num_filters)
        
        # Dropout
        x = self.dropout(x)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


def get_device():
    """Get the best available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    else:
        return torch.device('cpu')


def create_model_pytorch(model_type: str = 'lstm', **kwargs):
    """
    Factory function to create PyTorch sentiment analysis models
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'cnn')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        PyTorch model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        model = SentimentLSTM_PyTorch(**kwargs)
    elif model_type == 'gru':
        model = SentimentGRU_PyTorch(**kwargs)
    elif model_type == 'cnn':
        model = SentimentCNN_PyTorch(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'lstm', 'gru', 'cnn'")
    
    # Move model to appropriate device
    device = get_device()
    model = model.to(device)
    
    print(f"Model created and moved to {device}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating sample PyTorch models...\n")
    
    # Check device
    device = get_device()
    print(f"Using device: {device}\n")
    
    # LSTM model
    print("=" * 50)
    print("LSTM Model (PyTorch)")
    print("=" * 50)
    lstm_model = create_model_pytorch(
        model_type='lstm',
        vocab_size=10000,
        embedding_dim=100,
        lstm_units=128,
        num_classes=3
    )
    print(lstm_model)
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randint(0, 10000, (2, 128)).to(device)
    output = lstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}\n")
    
    # GRU model
    print("=" * 50)
    print("GRU Model (PyTorch)")
    print("=" * 50)
    gru_model = create_model_pytorch(
        model_type='gru',
        vocab_size=10000,
        embedding_dim=100,
        gru_units=128,
        num_classes=3
    )
    print(f"Parameters: {sum(p.numel() for p in gru_model.parameters()):,}\n")
    
    # CNN model
    print("=" * 50)
    print("CNN Model (PyTorch)")
    print("=" * 50)
    cnn_model = create_model_pytorch(
        model_type='cnn',
        vocab_size=10000,
        embedding_dim=100,
        num_classes=3
    )
    print(f"Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
