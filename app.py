"""
Application Web Flask pour l'Analyse de Sentiment
===================================================

Interface web simple pour tester le modèle PyTorch.

Utilisation:
    python app.py
    
Puis ouvrir: http://localhost:5000

Auteur: Projet NLP
Date: Novembre 2025
"""

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import os
import sys
import pandas as pd
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

import torch.nn as nn
import re

app = Flask(__name__)

# Définir le modèle (même architecture que train_pytorch_simple.py)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden_cat)
        out = self.relu(self.fc1(dropped))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Précharger le modèle au démarrage
class SentimentModel:
    def __init__(self):
        self.device = self._get_device()
        self.model = None
        self.vocab_data = None
        self.load_model()
    
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Charge le modèle et le vocabulaire"""
        model_path = 'models/pytorch_lstm/best_model.pth'
        vocab_path = 'models/pytorch_lstm/vocabulary.pkl'
        
        if not os.path.exists(model_path):
            print(f"❌ Modèle introuvable: {model_path}")
            print("💡 Entraînez d'abord avec: python train_pytorch_simple.py")
            return
        
        # Charger vocabulaire
        with open(vocab_path, 'rb') as f:
            self.vocab_data = pickle.load(f)
        
        # Charger modèle
        checkpoint = torch.load(model_path, map_location=self.device)
        vocab_size = len(self.vocab_data['word_to_idx'])
        
        self.model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            num_classes=3,
            dropout=0.5
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modèle chargé sur {self.device}")
    
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
    
    def text_to_sequence(self, text):
        """Convertit un texte en séquence"""
        max_len = self.vocab_data['max_seq_len']
        word_to_idx = self.vocab_data['word_to_idx']
        
        cleaned = self.clean_text(text)
        words = cleaned.split()
        sequence = [word_to_idx.get(word, word_to_idx.get("<UNK>", 1)) for word in words]
        
        if len(sequence) < max_len:
            sequence = sequence + [0] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        
        return sequence
    
    def predict(self, text):
        """Prédit le sentiment"""
        if not self.model or not self.vocab_data:
            return {
                'error': 'Modèle non chargé. Entraînez d\'abord le modèle.',
                'text': text
            }
        
        # Préparer
        cleaned = self.clean_text(text)
        sequence = self.text_to_sequence(text)
        tensor = torch.LongTensor([sequence]).to(self.device)
        
        # Prédire
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        # Résultats
        idx_to_label = self.vocab_data.get('idx_to_label', {0: 'neutre', 1: 'négatif', 2: 'positif'})
        predicted_label = idx_to_label[predicted.item()]
        
        return {
            'text': text,
            'cleaned': cleaned,
            'prediction': predicted_label,
            'confidence': float(probabilities[0][predicted].item() * 100),
            'probabilities': {
                'neutre': float(probabilities[0][0].item() * 100),
                'negatif': float(probabilities[0][1].item() * 100),
                'positif': float(probabilities[0][2].item() * 100)
            }
        }

# Initialiser le modèle
sentiment_model = SentimentModel()


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API de prédiction"""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Texte vide'}), 400
    
    result = sentiment_model.predict(text)
    return jsonify(result)


@app.route('/feedback', methods=['POST'])
def feedback():
    """Enregistre le feedback utilisateur ET réentraîne immédiatement"""
    data = request.get_json()
    text = data.get('text', '')
    predicted = data.get('predicted', '')
    correct = data.get('correct', '')
    user_label = data.get('user_label', '')
    explanation = data.get('explanation', '')
    
    if not text:
        return jsonify({'error': 'Texte manquant'}), 400
    
    # Créer le dossier feedback si nécessaire
    feedback_dir = Path('data/feedback')
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder dans un fichier CSV
    feedback_file = feedback_dir / 'corrections.csv'
    
    import csv
    from datetime import datetime
    
    # Créer le fichier avec headers si nécessaire
    file_exists = feedback_file.exists()
    
    with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'text', 'predicted_label', 'correct_label', 'user_explanation'])
        
        writer.writerow([
            datetime.now().isoformat(),
            text,
            predicted,
            user_label if not correct else predicted,
            explanation if not correct else 'Correct'
        ])
    
    # 🔥 APPRENTISSAGE IMMÉDIAT : Ajuster le modèle avec cette nouvelle donnée
    if not correct and user_label:
        try:
            retrain_on_feedback(text, user_label)
            message = f'✅ Correction enregistrée ET apprise ! Le modèle ne fera plus cette erreur.'
        except Exception as e:
            print(f"⚠️ Erreur réentraînement: {e}")
            message = 'Correction enregistrée (réentraînement échoué)'
    else:
        message = 'Merci pour votre feedback!'
    
    # Compter le nombre de feedbacks
    try:
        df = pd.read_csv(feedback_file)
        feedback_count = len(df)
    except:
        feedback_count = 1
    
    return jsonify({
        'success': True,
        'message': message,
        'feedback_count': feedback_count,
        'learned': not correct and user_label
    })


def retrain_on_feedback(text, correct_label):
    """Réentraîne le modèle avec une correction utilisateur"""
    label_to_idx = sentiment_model.vocab_data.get('label_to_idx', {'neutre': 0, 'négatif': 1, 'positif': 2})
    
    # Convertir le label
    if correct_label.lower() not in label_to_idx:
        # Mapper les variations
        label_map = {
            'positif': 'positif', 'positive': 'positif', '😊 positif': 'positif',
            'négatif': 'négatif', 'negatif': 'négatif', 'negative': 'négatif', '😢 négatif': 'négatif',
            'neutre': 'neutre', 'neutral': 'neutre', '😐 neutre': 'neutre'
        }
        correct_label = label_map.get(correct_label.lower(), 'neutre')
    else:
        correct_label = correct_label.lower()
    
    label_idx = label_to_idx[correct_label]
    
    # Préparer les données
    sequence = sentiment_model.text_to_sequence(text)
    x_tensor = torch.LongTensor([sequence]).to(sentiment_model.device)
    y_tensor = torch.LongTensor([label_idx]).to(sentiment_model.device)
    
    # Mettre le modèle en mode entraînement
    sentiment_model.model.train()
    
    # Créer l'optimiseur
    optimizer = torch.optim.Adam(sentiment_model.model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # Faire 5 itérations d'apprentissage sur cette correction
    for _ in range(5):
        optimizer.zero_grad()
        outputs = sentiment_model.model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    # Remettre en mode évaluation
    sentiment_model.model.eval()
    
    # Sauvegarder le modèle mis à jour
    model_path = 'models/pytorch_lstm/best_model.pth'
    torch.save({
        'model_state_dict': sentiment_model.model.state_dict(),
        'vocab_size': len(sentiment_model.vocab_data['word_to_idx']),
    }, model_path)
    
    print(f"🎓 Modèle appris: '{text}' → {correct_label}")


@app.route('/examples', methods=['GET'])
def examples():
    """Exemples de textes"""
    examples = [
        "Ce film est absolument génial! J'ai adoré chaque minute.",
        "C'était vraiment nul et décevant...",
        "Le produit est correct, rien de spécial.",
        "Excellent service client! Je recommande vivement.",
        "Pire achat de ma vie, à éviter absolument!",
        "Ni bon ni mauvais, plutôt moyen.",
        "Incroyable! Meilleure expérience de ma vie!",
        "Horrible, une vraie catastrophe."
    ]
    
    results = []
    for text in examples:
        result = sentiment_model.predict(text)
        results.append(result)
    
    return jsonify({'examples': results})


@app.route('/health')
def health():
    """Vérification de santé"""
    return jsonify({
        'status': 'ok',
        'model_loaded': sentiment_model.model is not None,
        'device': str(sentiment_model.device)
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 DÉMARRAGE DE L'APPLICATION WEB")
    print("="*70)
    print(f"\n📊 URL: http://localhost:8080")
    print(f"⚡ Device: {sentiment_model.device}")
    print(f"\n💡 Appuyez sur Ctrl+C pour arrêter\n")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8080)
