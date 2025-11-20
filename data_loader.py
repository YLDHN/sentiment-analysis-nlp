"""
Data loading and splitting module
Handles loading datasets from various formats and splitting into train/val/test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os
import json

import config


class DataLoader:
    """Class to handle data loading and splitting"""
    
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Initialize DataLoader
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_csv(self, filepath: str, text_column: str = 'text', 
                 label_column: str = 'sentiment') -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        df = pd.read_csv(filepath)
        
        # Rename columns to standardized names
        df = df.rename(columns={text_column: 'text', label_column: 'label'})
        
        # Keep only text and label columns
        df = df[['text', 'label']]
        
        # Remove any NaN values
        df = df.dropna()
        
        print(f"Loaded {len(df)} samples from {filepath}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def load_json(self, filepath: str, text_key: str = 'text', 
                  label_key: str = 'sentiment') -> pd.DataFrame:
        """
        Load data from JSON file
        
        Args:
            filepath: Path to JSON file
            text_key: Key for text in JSON
            label_key: Key for label in JSON
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df = df.rename(columns={text_key: 'text', label_key: 'label'})
        df = df[['text', 'label']]
        df = df.dropna()
        
        print(f"Loaded {len(df)} samples from {filepath}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def load_txt(self, filepath: str, delimiter: str = '\t') -> pd.DataFrame:
        """
        Load data from TXT file with delimiter
        Expects format: text<delimiter>label
        
        Args:
            filepath: Path to TXT file
            delimiter: Delimiter between text and label
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        texts = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(delimiter)
                    if len(parts) >= 2:
                        texts.append(delimiter.join(parts[:-1]))
                        labels.append(parts[-1])
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        df = df.dropna()
        
        print(f"Loaded {len(df)} samples from {filepath}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def encode_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
        """
        Encode string labels to integers
        
        Args:
            df: DataFrame with 'label' column
            
        Returns:
            Tuple of (DataFrame with encoded labels, label_to_id dict, id_to_label dict)
        """
        unique_labels = sorted(df['label'].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        df['label_encoded'] = df['label'].map(label_to_id)
        
        print(f"\nLabel encoding:")
        for label, idx in label_to_id.items():
            print(f"  {label} -> {idx}")
        
        return df, label_to_id, id_to_label
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = config.TRAIN_RATIO,
                   val_ratio: float = config.VAL_RATIO,
                   test_ratio: float = config.TEST_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_ratio,
            random_state=self.random_seed,
            stratify=df['label_encoded'] if 'label_encoded' in df.columns else None
        )
        
        # Second split: separate train and validation
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_ratio,
            random_state=self.random_seed,
            stratify=train_val_df['label_encoded'] if 'label_encoded' in train_val_df.columns else None
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame, output_dir: str = config.PROCESSED_DATA_DIR):
        """
        Save data splits to CSV files
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        print(f"\nSaved splits to {output_dir}")
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 1000):
        """
        Create a sample dataset for testing (French sentiment data)
        
        Args:
            output_path: Path to save the sample dataset
            num_samples: Number of samples to generate
        """
        np.random.seed(self.random_seed)
        
        # Sample French sentences with sentiments
        positive_samples = [
            "Ce film est absolument magnifique, j'ai adoré !",
            "Quelle excellente expérience, je recommande vivement.",
            "Produit de très haute qualité, très satisfait de mon achat.",
            "Le service était impeccable, personnel très aimable.",
            "Un restaurant fantastique avec une cuisine délicieuse.",
            "Je suis très content de cette acquisition, parfait !",
            "Superbe performance, bravo aux artistes !",
            "Une journée merveilleuse, tout était parfait.",
            "Excellent rapport qualité-prix, je reviendrai.",
            "C'est exactement ce que je cherchais, merci beaucoup !",
        ]
        
        negative_samples = [
            "Très déçu par ce produit, qualité médiocre.",
            "Service catastrophique, je ne reviendrai jamais.",
            "Film ennuyeux et prévisible, une perte de temps.",
            "Mauvaise expérience, personnel désagréable.",
            "Produit défectueux, demande de remboursement.",
            "Restaurant horrible, nourriture immangeable.",
            "Je regrette vraiment cet achat, nul.",
            "Performance décevante, pas à la hauteur.",
            "Prix exorbitant pour une qualité médiocre.",
            "À éviter absolument, très mauvais.",
        ]
        
        neutral_samples = [
            "Le produit correspond à la description.",
            "C'était correct, sans plus.",
            "Expérience normale, rien de spécial.",
            "Le film est ok, pas extraordinaire.",
            "Service standard, conforme aux attentes.",
            "Qualité acceptable pour le prix.",
            "Ça fait le travail, c'est fonctionnel.",
            "Résultat moyen, pourrait être mieux.",
            "C'est un produit basique, rien de remarquable.",
            "Ni bon ni mauvais, dans la moyenne.",
        ]
        
        # Generate samples
        texts = []
        labels = []
        
        samples_per_class = num_samples // 3
        
        for _ in range(samples_per_class):
            texts.append(np.random.choice(positive_samples))
            labels.append('positif')
            
            texts.append(np.random.choice(negative_samples))
            labels.append('négatif')
            
            texts.append(np.random.choice(neutral_samples))
            labels.append('neutre')
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'sentiment': labels})
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"Created sample dataset with {len(df)} samples at {output_path}")
        print(f"Label distribution:\n{df['sentiment'].value_counts()}")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Create a sample dataset for demonstration
    sample_path = os.path.join(config.RAW_DATA_DIR, 'sample_data.csv')
    loader.create_sample_dataset(sample_path, num_samples=1500)
    
    # Load the data
    df = loader.load_csv(sample_path, text_column='text', label_column='sentiment')
    
    # Encode labels
    df, label_to_id, id_to_label = loader.encode_labels(df)
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
