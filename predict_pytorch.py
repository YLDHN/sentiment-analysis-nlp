#!/usr/bin/env python3
"""
Script de Prédiction PyTorch pour Analyse de Sentiment
========================================================

Utilise le modèle PyTorch entraîné pour prédire le sentiment de textes.

Utilisation:
    python predict_pytorch.py "Votre texte ici"
    python predict_pytorch.py --interactive
    python predict_pytorch.py --file textes.txt

Auteur: Projet NLP
Date: Novembre 2025
"""

import torch
import pickle
import argparse
import sys
import os
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from model_pytorch import SentimentLSTM_PyTorch
import re


class SentimentPredictorPyTorch:
    """Prédit le sentiment avec un modèle PyTorch"""
    
    def __init__(self, model_path='models/pytorch_lstm/best_model.pth',
                 vocab_path='models/pytorch_lstm/vocabulary.pkl'):
        """
        Initialise le prédicteur
        
        Args:
            model_path: Chemin vers le modèle .pth
            vocab_path: Chemin vers le vocabulaire .pkl
        """
        self.device = self._get_device()
        self.vocab_data = self._load_vocabulary(vocab_path)
        self.model = self._load_model(model_path)
        
        print(f"✅ Modèle chargé sur {self.device}")
        print(f"✅ Vocabulaire: {len(self.vocab_data['word_to_idx'])} mots")
    
    def _get_device(self):
        """Détecte le device disponible"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_vocabulary(self, vocab_path):
        """Charge le vocabulaire"""
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulaire introuvable: {vocab_path}")
        
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        return vocab_data
    
    def _load_model(self, model_path):
        """Charge le modèle PyTorch"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Créer le modèle
        vocab_size = len(self.vocab_data['word_to_idx'])
        model = SentimentLSTM_PyTorch(
            vocab_size=vocab_size,
            embedding_dim=128,
            lstm_units=128,
            num_classes=3,
            dropout_rate=0.5
        )
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def clean_text(self, text):
        """Nettoie le texte"""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            text = re.sub(r'[^a-zàâäçéèêëïîôùûüÿœæ0-9\s\']', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""
    
    def tokenize(self, text):
        """Tokenise le texte"""
        return text.split()
    
    def text_to_sequence(self, text):
        """Convertit un texte en séquence d'indices"""
        max_len = self.vocab_data['max_seq_len']
        word_to_idx = self.vocab_data['word_to_idx']
        
        # Nettoyer et tokeniser
        cleaned = self.clean_text(text)
        words = self.tokenize(cleaned)
        
        # Convertir en indices
        sequence = [word_to_idx.get(word, word_to_idx.get("<UNK>", 1)) for word in words]
        
        # Padding
        if len(sequence) < max_len:
            sequence = sequence + [0] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        
        return sequence
    
    def predict(self, text):
        """
        Prédit le sentiment d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            dict: {
                'text': texte original,
                'cleaned': texte nettoyé,
                'prediction': label prédit,
                'probabilities': dict des probabilités par classe
            }
        """
        # Préparer le texte
        cleaned = self.clean_text(text)
        sequence = self.text_to_sequence(text)
        tensor = torch.LongTensor([sequence]).to(self.device)
        
        # Prédire
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        # Mapper les résultats
        idx_to_label = self.vocab_data.get('idx_to_label', {0: 'neutre', 1: 'négatif', 2: 'positif'})
        predicted_label = idx_to_label[predicted.item()]
        
        # Emojis
        emoji_map = {
            'neutre': '😐',
            'négatif': '😞',
            'positif': '😊'
        }
        
        result = {
            'text': text,
            'cleaned': cleaned,
            'prediction': predicted_label,
            'prediction_emoji': f"{predicted_label.capitalize()} {emoji_map.get(predicted_label, '')}",
            'probabilities': {
                'Neutre': probabilities[0][0].item(),
                'Négatif': probabilities[0][1].item(),
                'Positif': probabilities[0][2].item()
            },
            'confidence': probabilities[0][predicted].item()
        }
        
        return result
    
    def predict_batch(self, texts):
        """Prédit le sentiment de plusieurs textes"""
        return [self.predict(text) for text in texts]
    
    def interactive(self):
        """Mode interactif"""
        print("\n" + "="*70)
        print("🔮 MODE INTERACTIF - PRÉDICTION DE SENTIMENT")
        print("="*70)
        print("\nEntrez des textes à analyser (ou 'quit' pour quitter)\n")
        
        while True:
            try:
                text = input("📝 Texte: ").strip()
                
                if not text:
                    continue
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Au revoir!")
                    break
                
                result = self.predict(text)
                self._print_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
    
    def _print_result(self, result):
        """Affiche un résultat de prédiction"""
        print("\n" + "-"*70)
        print(f"📝 Texte: \"{result['text']}\"")
        if result['text'] != result['cleaned']:
            print(f"🧹 Nettoyé: \"{result['cleaned']}\"")
        print(f"\n➜ Prédiction: {result['prediction_emoji']}")
        print(f"   Confiance: {result['confidence']:.2%}")
        print(f"\n📊 Probabilités:")
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 40)
            print(f"   {label:8s} {prob:6.2%} {bar}")
        print("-"*70)


def main():
    parser = argparse.ArgumentParser(
        description='Prédiction de sentiment avec PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Mode interactif
  python predict_pytorch.py --interactive
  
  # Texte unique
  python predict_pytorch.py "Ce film est génial!"
  
  # Plusieurs textes
  python predict_pytorch.py "Texte 1" "Texte 2" "Texte 3"
  
  # Depuis un fichier
  python predict_pytorch.py --file textes.txt
        """
    )
    
    parser.add_argument('texts', nargs='*', help='Texte(s) à analyser')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Mode interactif')
    parser.add_argument('-f', '--file', help='Fichier contenant les textes (un par ligne)')
    parser.add_argument('-m', '--model', default='models/pytorch_lstm/best_model.pth',
                       help='Chemin vers le modèle')
    parser.add_argument('-v', '--vocab', default='models/pytorch_lstm/vocabulary.pkl',
                       help='Chemin vers le vocabulaire')
    
    args = parser.parse_args()
    
    # Vérifier les fichiers
    if not os.path.exists(args.model):
        print(f"❌ Erreur: Modèle introuvable: {args.model}")
        print(f"\n💡 Entraînez d'abord le modèle avec:")
        print(f"   python train_pytorch_simple.py")
        sys.exit(1)
    
    if not os.path.exists(args.vocab):
        print(f"❌ Erreur: Vocabulaire introuvable: {args.vocab}")
        sys.exit(1)
    
    # Charger le prédicteur
    print("\n🔄 Chargement du modèle...\n")
    predictor = SentimentPredictorPyTorch(model_path=args.model, vocab_path=args.vocab)
    
    # Mode interactif
    if args.interactive:
        predictor.interactive()
        return
    
    # Depuis fichier
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ Erreur: Fichier introuvable: {args.file}")
            sys.exit(1)
        
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"\n📂 Analyse de {len(texts)} textes depuis {args.file}\n")
        print("="*70)
        
        for i, text in enumerate(texts, 1):
            result = predictor.predict(text)
            print(f"\n{i}. {result['prediction_emoji']} ({result['confidence']:.0%}) - \"{text[:60]}...\"" if len(text) > 60 else f"\n{i}. {result['prediction_emoji']} ({result['confidence']:.0%}) - \"{text}\"")
        
        print("\n" + "="*70)
        return
    
    # Textes en arguments
    if args.texts:
        print("\n" + "="*70)
        print(f"🔮 PRÉDICTIONS ({len(args.texts)} texte{'s' if len(args.texts) > 1 else ''})")
        print("="*70)
        
        for text in args.texts:
            result = predictor.predict(text)
            predictor._print_result(result)
        
        return
    
    # Aucun argument = aide
    parser.print_help()
    print("\n💡 Astuce: Utilisez --interactive pour le mode interactif")


if __name__ == "__main__":
    main()
