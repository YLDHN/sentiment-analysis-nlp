"""
Main script for sentiment analysis pipeline
Provides CLI interface for all operations
"""

import argparse
import os
import pandas as pd

import config
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from train import train_from_dataframes, ModelTrainer
from evaluate import ModelEvaluator
from predict import SentimentPredictor


def prepare_data(args):
    """Prepare and split data"""
    print("\n" + "=" * 60)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    loader = DataLoader()
    
    # Load data based on file type
    file_ext = os.path.splitext(args.input_file)[1].lower()
    
    if file_ext == '.csv':
        df = loader.load_csv(args.input_file, args.text_column, args.label_column)
    elif file_ext == '.json':
        df = loader.load_json(args.input_file, args.text_column, args.label_column)
    elif file_ext == '.txt':
        df = loader.load_txt(args.input_file)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    # Encode labels
    df, label_to_id, id_to_label = loader.encode_labels(df)
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
    
    print("\n✓ Données préparées avec succès!")


def create_sample_data(args):
    """Create sample dataset"""
    print("\n" + "=" * 60)
    print("CRÉATION D'UN DATASET D'EXEMPLE")
    print("=" * 60)
    
    loader = DataLoader()
    output_path = os.path.join(config.RAW_DATA_DIR, 'sample_data.csv')
    
    loader.create_sample_dataset(output_path, num_samples=args.num_samples)
    
    print(f"\n✓ Dataset d'exemple créé: {output_path}")


def train_model(args):
    """Train sentiment analysis model"""
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'val.csv'))
    
    print(f"Train: {len(train_df)} échantillons")
    print(f"Val:   {len(val_df)} échantillons")
    
    # Train model
    trainer = train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        model_type=args.model_type,
        use_class_weights=args.use_class_weights,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n✓ Entraînement terminé avec succès!")


def evaluate_model(args):
    """Evaluate trained model"""
    print("\n" + "=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)
    
    # Determine paths
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(config.MODELS_DIR, f'{args.model_type}_sentiment', 'best_model.h5')
    
    if args.preprocessor_path:
        preprocessor_path = args.preprocessor_path
    else:
        preprocessor_path = os.path.join(config.MODELS_DIR, f'{args.model_type}_sentiment', 'preprocessor.pkl')
    
    # Load test data
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'test.csv'))
    
    print(f"Test: {len(test_df)} échantillons")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        label_names=config.SENTIMENT_CLASSES
    )
    
    # Generate report
    output_dir = args.output_dir or os.path.join(config.LOGS_DIR, 'evaluation_report')
    evaluator.generate_report(
        test_texts=test_df['text'].tolist(),
        test_labels=test_df['label_encoded'].values,
        output_dir=output_dir
    )
    
    print(f"\n✓ Évaluation terminée! Rapport sauvegardé dans {output_dir}")


def predict_text(args):
    """Predict sentiment for text"""
    print("\n" + "=" * 60)
    print("PRÉDICTION DE SENTIMENT")
    print("=" * 60)
    
    # Determine paths
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(config.MODELS_DIR, f'{args.model_type}_sentiment', 'best_model.h5')
    
    if args.preprocessor_path:
        preprocessor_path = args.preprocessor_path
    else:
        preprocessor_path = os.path.join(config.MODELS_DIR, f'{args.model_type}_sentiment', 'preprocessor.pkl')
    
    # Initialize predictor
    predictor = SentimentPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    
    # Interactive mode
    if args.interactive:
        predictor.interactive_predict()
    
    # Predict from file
    elif args.input_file:
        predictor.predict_from_file(
            input_file=args.input_file,
            output_file=args.output_file,
            text_column=args.text_column
        )
    
    # Predict single text
    elif args.text:
        result = predictor.predict_with_explanation(args.text)
        
        print(f"\nTexte: \"{args.text}\"")
        print(f"\nSentiment: {result['sentiment'].upper()}")
        print(f"Confiance: {result['confidence']:.2%} ({result['confidence_level']})")
        print(f"\nProbabilités:")
        for label, prob in result['probabilities'].items():
            print(f"  {label:10} : {prob:.2%}")
    
    else:
        print("Veuillez fournir --text, --input-file, ou --interactive")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Analyse de Sentiment (NLP)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  
  # Créer un dataset d'exemple
  python main.py create-sample --num-samples 1500
  
  # Préparer les données
  python main.py prepare --input-file data/raw/sample_data.csv
  
  # Entraîner un modèle LSTM
  python main.py train --model-type lstm --epochs 10
  
  # Évaluer le modèle
  python main.py evaluate --model-type lstm
  
  # Prédire le sentiment d'un texte
  python main.py predict --text "Ce film est excellent!"
  
  # Mode interactif
  python main.py predict --interactive
  
  # Prédire à partir d'un fichier
  python main.py predict --input-file data.csv --output-file predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Create sample data command
    sample_parser = subparsers.add_parser('create-sample', help='Créer un dataset d\'exemple')
    sample_parser.add_argument('--num-samples', type=int, default=1500,
                              help='Nombre d\'échantillons à générer')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Préparer et diviser les données')
    prepare_parser.add_argument('--input-file', type=str, required=True,
                               help='Chemin vers le fichier de données')
    prepare_parser.add_argument('--text-column', type=str, default='text',
                               help='Nom de la colonne contenant le texte')
    prepare_parser.add_argument('--label-column', type=str, default='sentiment',
                               help='Nom de la colonne contenant les labels')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Entraîner un modèle')
    train_parser.add_argument('--model-type', type=str, default='lstm',
                             choices=['lstm', 'gru', 'cnn', 'bert'],
                             help='Type de modèle')
    train_parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                             help='Nombre d\'époques')
    train_parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                             help='Taille du batch')
    train_parser.add_argument('--use-class-weights', action='store_true', default=True,
                             help='Utiliser des poids de classe pour les données déséquilibrées')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Évaluer un modèle')
    eval_parser.add_argument('--model-type', type=str, default='lstm',
                            help='Type de modèle')
    eval_parser.add_argument('--model-path', type=str,
                            help='Chemin vers le modèle sauvegardé')
    eval_parser.add_argument('--preprocessor-path', type=str,
                            help='Chemin vers le preprocessor sauvegardé')
    eval_parser.add_argument('--output-dir', type=str,
                            help='Dossier de sortie pour le rapport')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Prédire le sentiment')
    predict_parser.add_argument('--model-type', type=str, default='lstm',
                               help='Type de modèle')
    predict_parser.add_argument('--model-path', type=str,
                               help='Chemin vers le modèle sauvegardé')
    predict_parser.add_argument('--preprocessor-path', type=str,
                               help='Chemin vers le preprocessor sauvegardé')
    predict_parser.add_argument('--text', type=str,
                               help='Texte à analyser')
    predict_parser.add_argument('--input-file', type=str,
                               help='Fichier CSV contenant les textes')
    predict_parser.add_argument('--output-file', type=str,
                               help='Fichier pour sauvegarder les prédictions')
    predict_parser.add_argument('--text-column', type=str, default='text',
                               help='Nom de la colonne contenant le texte')
    predict_parser.add_argument('--interactive', action='store_true',
                               help='Mode interactif')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'create-sample':
        create_sample_data(args)
    elif args.command == 'prepare':
        prepare_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'predict':
        predict_text(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
