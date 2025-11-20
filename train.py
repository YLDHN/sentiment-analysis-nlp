"""
Training module for sentiment analysis models
Handles model training with validation and callbacks
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
        TensorBoard, CSVLogger
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

import config
from preprocessing import TextPreprocessor
from model import create_model


class ModelTrainer:
    """Class to handle model training"""
    
    def __init__(self, 
                 model_type: str = 'lstm',
                 model_name: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """
        Initialize ModelTrainer
        
        Args:
            model_type: Type of model ('lstm', 'gru', 'cnn', 'bert')
            model_name: Name for saving the model
            experiment_name: Name for the experiment (for logging)
        """
        self.model_type = model_type
        self.model_name = model_name or f"{model_type}_sentiment"
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.model_wrapper = None
        self.model = None
        self.preprocessor = None
        self.history = None
        
        # Create directories
        self.model_dir = os.path.join(config.MODELS_DIR, self.model_name)
        self.log_dir = os.path.join(config.LOGS_DIR, self.experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def prepare_data(self, 
                    train_texts: list, 
                    train_labels: np.ndarray,
                    val_texts: list,
                    val_labels: np.ndarray,
                    preprocessor: Optional[TextPreprocessor] = None) -> Tuple:
        """
        Prepare data for training
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            preprocessor: Pre-configured preprocessor (optional)
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, preprocessor)
        """
        if preprocessor is None:
            # Create and fit preprocessor on training data
            preprocessor = TextPreprocessor(
                max_vocab_size=config.MAX_VOCAB_SIZE,
                max_sequence_length=config.MAX_SEQUENCE_LENGTH
            )
            X_train = preprocessor.preprocess_data(train_texts, build_vocab=True)
            
            # Save preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            preprocessor.save_preprocessor(preprocessor_path)
        else:
            # Use existing preprocessor
            X_train = preprocessor.preprocess_data(train_texts, build_vocab=False)
        
        # Preprocess validation data
        X_val = preprocessor.preprocess_data(val_texts, build_vocab=False)
        
        self.preprocessor = preprocessor
        
        return X_train, train_labels, X_val, val_labels, preprocessor
    
    def build_model(self, 
                   vocab_size: int,
                   num_classes: int = config.NUM_CLASSES,
                   embedding_matrix: Optional[np.ndarray] = None,
                   **kwargs):
        """
        Build the model
        
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            embedding_matrix: Pre-trained embedding matrix (optional)
            **kwargs: Additional model parameters
        """
        self.model_wrapper, self.model = create_model(
            model_type=self.model_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_matrix=embedding_matrix,
            **kwargs
        )
        
        print(f"\nModel summary:")
        self.model_wrapper.summary()
    
    def get_callbacks(self, 
                     patience: int = 5,
                     monitor: str = 'val_loss',
                     mode: str = 'min') -> list:
        """
        Create training callbacks
        
        Args:
            patience: Patience for early stopping
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitored metric
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = os.path.join(self.model_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_path = os.path.join(self.log_dir, 'training_log.csv')
        csv_logger = CSVLogger(csv_path)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = config.EPOCHS,
             batch_size: int = config.BATCH_SIZE,
             callbacks: Optional[list] = None,
             class_weights: Optional[dict] = None) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks (optional)
            class_weights: Class weights for imbalanced data (optional)
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Callbacks: {len(callbacks)}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model (optional)
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, f'{self.model_name}_final.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Plot loss
            ax2.plot(self.history.history['loss'], label='Train Loss')
            ax2.plot(self.history.history['val_loss'], label='Val Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Model Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.log_dir, 'training_history.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
            
            plt.close()
        except ImportError:
            print("matplotlib not available for plotting")
    
    def compute_class_weights(self, y_train: np.ndarray) -> dict:
        """
        Compute class weights for imbalanced datasets
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print(f"\nClass weights: {class_weights}")
        
        return class_weights


def train_from_dataframes(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         model_type: str = 'lstm',
                         use_class_weights: bool = True,
                         **kwargs):
    """
    Convenience function to train from DataFrames
    
    Args:
        train_df: Training DataFrame with 'text' and 'label_encoded' columns
        val_df: Validation DataFrame with 'text' and 'label_encoded' columns
        model_type: Type of model to train
        use_class_weights: Whether to use class weights
        **kwargs: Additional training parameters
        
    Returns:
        ModelTrainer instance
    """
    # Initialize trainer
    trainer = ModelTrainer(model_type=model_type)
    
    # Prepare data
    X_train, y_train, X_val, y_val, preprocessor = trainer.prepare_data(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label_encoded'].values,
        val_texts=val_df['text'].tolist(),
        val_labels=val_df['label_encoded'].values
    )
    
    # Build model
    vocab_size = len(preprocessor.word_to_idx)
    trainer.build_model(vocab_size=vocab_size)
    
    # Compute class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = trainer.compute_class_weights(y_train)
    
    # Train
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        class_weights=class_weights,
        **kwargs
    )
    
    # Save model
    trainer.save_model()
    
    # Plot history
    trainer.plot_history()
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Training Example")
    
    # Load data
    from data_loader import DataLoader
    
    loader = DataLoader()
    
    # Load processed data
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'val.csv'))
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Train model
    trainer = train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        model_type='lstm',
        epochs=10,
        batch_size=32
    )
    
    print("\nTraining completed successfully!")
