"""
Configuration file for sentiment analysis project
"""

import os

# Paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Text preprocessing
MAX_SEQUENCE_LENGTH = 128
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 100

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

# LSTM/GRU parameters
LSTM_UNITS = 128
GRU_UNITS = 128

# Classes
SENTIMENT_CLASSES = ['négatif', 'neutre', 'positif']
NUM_CLASSES = len(SENTIMENT_CLASSES)

# Random seed for reproducibility
RANDOM_SEED = 42

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
