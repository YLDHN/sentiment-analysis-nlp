"""
Prediction module for sentiment analysis
Provides interface to predict sentiment of new texts
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Optional

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

import config
from preprocessing import TextPreprocessor


class SentimentPredictor:
    """Class to handle sentiment prediction on new texts"""
    
    def __init__(self,
                 model_path: str,
                 preprocessor_path: str,
                 label_names: List[str] = config.SENTIMENT_CLASSES,
                 threshold: float = 0.5):
        """
        Initialize SentimentPredictor
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            label_names: List of label names
            threshold: Confidence threshold for predictions
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.label_names = label_names
        self.threshold = threshold
        
        # Load model and preprocessor
        print("Loading model and preprocessor...")
        self.model = self._load_model(model_path)
        self.preprocessor = TextPreprocessor.load_preprocessor(preprocessor_path)
        print("Ready for predictions!")
    
    def _load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for predictions")
        
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess text
        X = self.preprocessor.preprocess_data([text], build_vocab=False)
        
        # Get prediction
        probabilities = self.model.predict(X, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Build result
        result = {
            'text': text,
            'sentiment': self.label_names[predicted_class],
            'confidence': float(confidence),
            'is_confident': confidence >= self.threshold
        }
        
        if return_probabilities:
            result['probabilities'] = {
                label: float(prob) 
                for label, prob in zip(self.label_names, probabilities)
            }
        
        return result
    
    def predict_batch(self, 
                     texts: List[str], 
                     return_probabilities: bool = False) -> List[Dict]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess texts
        X = self.preprocessor.preprocess_data(texts, build_vocab=False)
        
        # Get predictions
        probabilities = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        # Build results
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'sentiment': self.label_names[predicted_classes[i]],
                'confidence': float(confidences[i]),
                'is_confident': confidences[i] >= self.threshold
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    label: float(prob) 
                    for label, prob in zip(self.label_names, probabilities[i])
                }
            
            results.append(result)
        
        return results
    
    def predict_with_explanation(self, text: str) -> Dict:
        """
        Predict sentiment with detailed explanation
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction and explanation
        """
        result = self.predict_single(text, return_probabilities=True)
        
        # Add explanation
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize(cleaned_text)
        
        result['cleaned_text'] = cleaned_text
        result['num_tokens'] = len(tokens)
        result['tokens'] = tokens[:20]  # Show first 20 tokens
        
        # Confidence interpretation
        confidence = result['confidence']
        if confidence >= 0.9:
            result['confidence_level'] = 'très élevée'
        elif confidence >= 0.7:
            result['confidence_level'] = 'élevée'
        elif confidence >= 0.5:
            result['confidence_level'] = 'moyenne'
        else:
            result['confidence_level'] = 'faible'
        
        return result
    
    def interactive_predict(self):
        """
        Interactive prediction mode
        User can input texts and get predictions in real-time
        """
        print("\n" + "=" * 60)
        print("MODE PRÉDICTION INTERACTIVE")
        print("=" * 60)
        print("Entrez du texte pour analyser le sentiment.")
        print("Tapez 'quit' ou 'q' pour quitter.")
        print("=" * 60 + "\n")
        
        while True:
            try:
                # Get user input
                text = input("\nTexte: ").strip()
                
                # Check for exit command
                if text.lower() in ['quit', 'q', 'exit']:
                    print("Au revoir!")
                    break
                
                # Skip empty input
                if not text:
                    continue
                
                # Make prediction
                result = self.predict_with_explanation(text)
                
                # Display results
                print("\n" + "-" * 60)
                print(f"Sentiment: {result['sentiment'].upper()}")
                print(f"Confiance: {result['confidence']:.2%} ({result['confidence_level']})")
                print(f"\nProbabilités:")
                for label, prob in result['probabilities'].items():
                    bar_length = int(prob * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    print(f"  {label:10} [{bar}] {prob:.2%}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nInterrompu. Au revoir!")
                break
            except Exception as e:
                print(f"Erreur: {e}")
    
    def predict_from_file(self, 
                         input_file: str,
                         output_file: Optional[str] = None,
                         text_column: str = 'text') -> List[Dict]:
        """
        Predict sentiments for texts in a file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save predictions (optional)
            text_column: Name of the column containing text
            
        Returns:
            List of prediction results
        """
        import pandas as pd
        
        # Load data
        df = pd.read_csv(input_file)
        texts = df[text_column].tolist()
        
        print(f"Predicting sentiments for {len(texts)} texts...")
        
        # Make predictions
        results = self.predict_batch(texts, return_probabilities=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Combine with original data
        output_df = df.copy()
        output_df['predicted_sentiment'] = results_df['sentiment']
        output_df['confidence'] = results_df['confidence']
        
        # Add probability columns
        for label in self.label_names:
            output_df[f'prob_{label}'] = results_df['probabilities'].apply(lambda x: x[label])
        
        # Save if output file specified
        if output_file:
            output_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(output_df['predicted_sentiment'].value_counts())
        print(f"\nAverage confidence: {output_df['confidence'].mean():.2%}")
        
        return results
    
    def analyze_sentiment_distribution(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment distribution in a collection of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with distribution statistics
        """
        results = self.predict_batch(texts)
        
        # Count sentiments
        sentiment_counts = {}
        for label in self.label_names:
            sentiment_counts[label] = sum(1 for r in results if r['sentiment'] == label)
        
        # Calculate statistics
        total = len(texts)
        distribution = {
            'total_texts': total,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': {
                label: count / total * 100 
                for label, count in sentiment_counts.items()
            },
            'average_confidence': np.mean([r['confidence'] for r in results]),
            'confident_predictions': sum(1 for r in results if r['is_confident']),
            'confident_percentage': sum(1 for r in results if r['is_confident']) / total * 100
        }
        
        return distribution
    
    def print_distribution(self, distribution: Dict):
        """
        Print sentiment distribution analysis
        
        Args:
            distribution: Distribution dictionary from analyze_sentiment_distribution
        """
        print("\n" + "=" * 60)
        print("ANALYSE DE DISTRIBUTION DES SENTIMENTS")
        print("=" * 60)
        
        print(f"\nTotal de textes analysés: {distribution['total_texts']}")
        
        print(f"\nDistribution des sentiments:")
        for label in self.label_names:
            count = distribution['sentiment_counts'][label]
            percentage = distribution['sentiment_percentages'][label]
            print(f"  {label.capitalize():10} : {count:5} ({percentage:5.1f}%)")
        
        print(f"\nConfiance moyenne: {distribution['average_confidence']:.2%}")
        print(f"Prédictions confiantes: {distribution['confident_predictions']} / {distribution['total_texts']} "
              f"({distribution['confident_percentage']:.1f}%)")
        
        print("=" * 60)


def quick_predict(text: str,
                 model_path: Optional[str] = None,
                 preprocessor_path: Optional[str] = None) -> str:
    """
    Quick prediction function for single text
    
    Args:
        text: Input text
        model_path: Path to model (uses default if None)
        preprocessor_path: Path to preprocessor (uses default if None)
        
    Returns:
        Predicted sentiment
    """
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'lstm_sentiment', 'best_model.h5')
    if preprocessor_path is None:
        preprocessor_path = os.path.join(config.MODELS_DIR, 'lstm_sentiment', 'preprocessor.pkl')
    
    predictor = SentimentPredictor(model_path, preprocessor_path)
    result = predictor.predict_single(text)
    
    return result['sentiment']


if __name__ == "__main__":
    # Example usage
    print("Sentiment Prediction Example")
    
    # Paths
    model_path = os.path.join(config.MODELS_DIR, 'lstm_sentiment', 'best_model.h5')
    preprocessor_path = os.path.join(config.MODELS_DIR, 'lstm_sentiment', 'preprocessor.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        print("Please train a model first using train.py")
    elif not os.path.exists(preprocessor_path):
        print(f"\nPreprocessor not found at {preprocessor_path}")
    else:
        # Initialize predictor
        predictor = SentimentPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        
        # Test predictions
        test_texts = [
            "Ce film est absolument magnifique, j'ai adoré chaque moment !",
            "Très déçu par ce produit, qualité médiocre pour le prix.",
            "C'est correct, rien d'extraordinaire mais ça fait le travail.",
        ]
        
        print("\n" + "=" * 60)
        print("EXEMPLES DE PRÉDICTIONS")
        print("=" * 60)
        
        for text in test_texts:
            result = predictor.predict_with_explanation(text)
            print(f"\nTexte: \"{text}\"")
            print(f"Sentiment: {result['sentiment'].upper()} (confiance: {result['confidence']:.2%})")
            print(f"Probabilités: {result['probabilities']}")
        
        # Interactive mode
        print("\n\nLancement du mode interactif...")
        predictor.interactive_predict()
