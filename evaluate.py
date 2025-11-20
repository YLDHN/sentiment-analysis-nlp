"""
Evaluation module for sentiment analysis models
Computes metrics and generates visualizations
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, List, Tuple

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

import config
from preprocessing import TextPreprocessor


class ModelEvaluator:
    """Class to handle model evaluation"""
    
    def __init__(self, 
                 model_path: str,
                 preprocessor_path: str,
                 label_names: List[str] = config.SENTIMENT_CLASSES):
        """
        Initialize ModelEvaluator
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            label_names: List of label names
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.label_names = label_names
        
        # Load model and preprocessor
        self.model = self.load_model(model_path)
        self.preprocessor = TextPreprocessor.load_preprocessor(preprocessor_path)
        
        self.predictions = None
        self.true_labels = None
    
    def load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model loading")
        
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        # Preprocess texts
        X = self.preprocessor.preprocess_data(texts, build_vocab=False)
        
        # Get predictions
        probabilities = self.model.predict(X, verbose=0)
        predicted_labels = np.argmax(probabilities, axis=1)
        
        return predicted_labels, probabilities
    
    def evaluate(self, 
                test_texts: List[str], 
                test_labels: np.ndarray,
                verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_texts: List of test texts
            test_labels: Array of true labels
            verbose: Whether to print results
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for evaluation metrics")
        
        # Make predictions
        predicted_labels, probabilities = self.predict(test_texts)
        
        # Store for later use
        self.predictions = predicted_labels
        self.true_labels = test_labels
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(test_labels, predicted_labels)
        metrics['precision_macro'] = precision_score(test_labels, predicted_labels, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(test_labels, predicted_labels, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(test_labels, predicted_labels, average='macro', zero_division=0)
        
        # Weighted metrics (better for imbalanced datasets)
        metrics['precision_weighted'] = precision_score(test_labels, predicted_labels, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(test_labels, predicted_labels, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(test_labels, predicted_labels, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(test_labels, predicted_labels, average=None, zero_division=0)
        recall_per_class = recall_score(test_labels, predicted_labels, average=None, zero_division=0)
        f1_per_class = f1_score(test_labels, predicted_labels, average=None, zero_division=0)
        
        for i, label_name in enumerate(self.label_names):
            if i < len(precision_per_class):
                metrics[f'precision_{label_name}'] = precision_per_class[i]
                metrics[f'recall_{label_name}'] = recall_per_class[i]
                metrics[f'f1_{label_name}'] = f1_per_class[i]
        
        if verbose:
            self.print_metrics(metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
        
        print(f"\nWeighted Metrics:")
        print(f"  Precision:          {metrics['precision_weighted']:.4f}")
        print(f"  Recall:             {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:           {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for label_name in self.label_names:
            if f'f1_{label_name}' in metrics:
                print(f"  {label_name.capitalize()}:")
                print(f"    Precision: {metrics[f'precision_{label_name}']:.4f}")
                print(f"    Recall:    {metrics[f'recall_{label_name}']:.4f}")
                print(f"    F1-Score:  {metrics[f'f1_{label_name}']:.4f}")
        
        print("=" * 60)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix
        
        Returns:
            Confusion matrix
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate() first.")
        
        return confusion_matrix(self.true_labels, self.predictions)
    
    def print_confusion_matrix(self):
        """Print confusion matrix"""
        cm = self.get_confusion_matrix()
        
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        
        # Header
        print("\n{:>12}".format(""), end="")
        for label in self.label_names:
            print(f"{label:>12}", end="")
        print()
        
        # Rows
        for i, label in enumerate(self.label_names):
            print(f"{label:>12}", end="")
            for j in range(len(self.label_names)):
                if i < len(cm) and j < len(cm[i]):
                    print(f"{cm[i][j]:>12}", end="")
            print()
        
        print("=" * 60)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """
        Plot confusion matrix as heatmap
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = self.get_confusion_matrix()
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.label_names,
                yticklabels=self.label_names,
                cbar_kws={'label': 'Count'}
            )
            
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib and seaborn are required for plotting")
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report
        
        Returns:
            Classification report as string
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate() first.")
        
        report = classification_report(
            self.true_labels, 
            self.predictions,
            target_names=self.label_names,
            digits=4
        )
        
        return report
    
    def print_classification_report(self):
        """Print classification report"""
        report = self.get_classification_report()
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60)
    
    def analyze_errors(self, 
                      test_texts: List[str], 
                      test_labels: np.ndarray,
                      num_examples: int = 5) -> pd.DataFrame:
        """
        Analyze misclassified examples
        
        Args:
            test_texts: List of test texts
            test_labels: Array of true labels
            num_examples: Number of examples to show per error type
            
        Returns:
            DataFrame with error analysis
        """
        if self.predictions is None:
            self.predictions, _ = self.predict(test_texts)
        
        # Find misclassified examples
        errors = []
        
        for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, self.predictions)):
            if true_label != pred_label:
                errors.append({
                    'text': text,
                    'true_label': self.label_names[true_label] if true_label < len(self.label_names) else f"Unknown_{true_label}",
                    'predicted_label': self.label_names[pred_label] if pred_label < len(self.label_names) else f"Unknown_{pred_label}",
                    'index': i
                })
        
        errors_df = pd.DataFrame(errors)
        
        print(f"\n" + "=" * 60)
        print(f"ERROR ANALYSIS")
        print("=" * 60)
        print(f"Total errors: {len(errors)} / {len(test_labels)} ({len(errors)/len(test_labels)*100:.2f}%)")
        
        if len(errors) > 0:
            print(f"\nShowing up to {num_examples} examples per error type:\n")
            
            # Group by error type
            for true_label in self.label_names:
                for pred_label in self.label_names:
                    if true_label != pred_label:
                        subset = errors_df[
                            (errors_df['true_label'] == true_label) & 
                            (errors_df['predicted_label'] == pred_label)
                        ]
                        
                        if len(subset) > 0:
                            print(f"\n{true_label.upper()} → {pred_label.upper()} ({len(subset)} cases):")
                            for idx, row in subset.head(num_examples).iterrows():
                                print(f"  - \"{row['text'][:100]}...\"")
        
        print("=" * 60)
        
        return errors_df
    
    def save_metrics(self, metrics: Dict[str, float], filepath: str):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save the metrics
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {filepath}")
    
    def generate_report(self, 
                       test_texts: List[str],
                       test_labels: np.ndarray,
                       output_dir: str):
        """
        Generate comprehensive evaluation report
        
        Args:
            test_texts: List of test texts
            test_labels: Array of true labels
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating evaluation report...")
        
        # Evaluate
        metrics = self.evaluate(test_texts, test_labels, verbose=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        self.save_metrics(metrics, metrics_path)
        
        # Print confusion matrix
        self.print_confusion_matrix()
        
        # Plot confusion matrix
        cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(cm_plot_path)
        
        # Print classification report
        self.print_classification_report()
        
        # Save classification report
        report = self.get_classification_report()
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")
        
        # Analyze errors
        errors_df = self.analyze_errors(test_texts, test_labels)
        errors_path = os.path.join(output_dir, 'errors.csv')
        errors_df.to_csv(errors_path, index=False)
        print(f"Error analysis saved to {errors_path}")
        
        print(f"\nEvaluation report generated in {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Evaluation Example")
    
    # Paths
    model_path = os.path.join(config.MODELS_DIR, 'lstm_sentiment', 'best_model.h5')
    preprocessor_path = os.path.join(config.MODELS_DIR, 'lstm_sentiment', 'preprocessor.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")
    elif not os.path.exists(preprocessor_path):
        print(f"Preprocessor not found at {preprocessor_path}")
    else:
        # Load test data
        test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'test.csv'))
        
        print(f"Test samples: {len(test_df)}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            label_names=config.SENTIMENT_CLASSES
        )
        
        # Generate report
        output_dir = os.path.join(config.LOGS_DIR, 'evaluation_report')
        evaluator.generate_report(
            test_texts=test_df['text'].tolist(),
            test_labels=test_df['label_encoded'].values,
            output_dir=output_dir
        )
