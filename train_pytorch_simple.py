#!/usr/bin/env python3
"""
Script d'Entraînement PyTorch pour Analyse de Sentiment
==========================================================

Ce script montre comment entraîner un modèle LSTM avec PyTorch
pour l'analyse de sentiment en français.

Utilisation:
    python train_pytorch_simple.py

Prérequis:
    - Données préparées dans data/processed/
    - PyTorch installé
    - GPU Apple Silicon (MPS) recommandé

Auteur: Projet NLP
Date: Novembre 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Configuration
os.makedirs('models/pytorch_lstm', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ========================================
# 1. CONFIGURATION
# ========================================

print("=" * 60)
print("🎯 ENTRAÎNEMENT PYTORCH - ANALYSE DE SENTIMENT")
print("=" * 60)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n🚀 Device: {device}")

# Hyperparamètres
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_SEQ_LEN = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 4

print(f"\n📊 Hyperparamètres:")
print(f"  - Embedding dim: {EMBEDDING_DIM}")
print(f"  - Hidden dim: {HIDDEN_DIM}")
print(f"  - Nombre de couches: {NUM_LAYERS}")
print(f"  - Dropout: {DROPOUT}")
print(f"  - Max sequence length: {MAX_SEQ_LEN}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Époques: {EPOCHS}")


# ========================================
# 2. FONCTIONS DE PRÉTRAITEMENT
# ========================================

def clean_text(text):
    """Nettoie le texte"""
    if isinstance(text, str):
        # Minuscules
        text = text.lower()
        # Supprimer URLs
        text = re.sub(r'http\S+', '', text)
        # Supprimer mentions
        text = re.sub(r'@\w+', '', text)
        # Supprimer hashtags (garder le texte)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Garder seulement lettres, chiffres, espaces et apostrophes
        text = re.sub(r'[^a-zàâäçéèêëïîôùûüÿœæ0-9\s\']', ' ', text)
        # Supprimer espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def tokenize(text):
    """Tokenise le texte"""
    return text.split()

def build_vocab(texts, max_vocab=10000):
    """Construit le vocabulaire"""
    word_counts = Counter()
    for text in texts:
        words = tokenize(clean_text(text))
        word_counts.update(words)
    
    # Mots les plus fréquents
    most_common = word_counts.most_common(max_vocab - 2)
    
    # Créer dictionnaires
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, 2):
        word_to_idx[word] = idx
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word, most_common

def texts_to_sequences(texts, word_to_idx, max_len):
    """Convertit textes en séquences d'indices"""
    sequences = []
    for text in texts:
        words = tokenize(clean_text(text))
        sequence = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
        
        # Padding
        if len(sequence) < max_len:
            sequence = sequence + [0] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        
        sequences.append(sequence)
    
    return np.array(sequences)


# ========================================
# 3. MODÈLE LSTM
# ========================================

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 pour bidirectionnel
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concaténer derniers états forward et backward
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Dropout
        dropped = self.dropout(hidden_cat)
        
        # Couches denses
        out = self.relu(self.fc1(dropped))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# ========================================
# 4. CHARGEMENT DES DONNÉES
# ========================================

print(f"\n" + "=" * 60)
print("📂 CHARGEMENT DES DONNÉES")
print("=" * 60)

# Charger les DataFrames
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

print(f"  - Train: {len(train_df)} échantillons")
print(f"  - Validation: {len(val_df)} échantillons")
print(f"  - Test: {len(test_df)} échantillons")

# Distribution des classes
print(f"\n📊 Distribution des classes (train):")
print(train_df['label'].value_counts())


# ========================================
# 5. CONSTRUCTION DU VOCABULAIRE
# ========================================

print(f"\n" + "=" * 60)
print("📖 CONSTRUCTION DU VOCABULAIRE")
print("=" * 60)

word_to_idx, idx_to_word, most_common = build_vocab(train_df['text'].tolist(), max_vocab=10000)

print(f"  - Taille du vocabulaire: {len(word_to_idx)}")
print(f"\n  📝 Mots les plus fréquents:")
for word, count in most_common[:10]:
    print(f"     - {word}: {count}")


# ========================================
# 6. PRÉPARATION DES DONNÉES
# ========================================

print(f"\n" + "=" * 60)
print("🔧 PRÉPARATION DES SÉQUENCES")
print("=" * 60)

# Convertir textes en séquences
X_train = texts_to_sequences(train_df['text'].tolist(), word_to_idx, MAX_SEQ_LEN)
X_val = texts_to_sequences(val_df['text'].tolist(), word_to_idx, MAX_SEQ_LEN)
X_test = texts_to_sequences(test_df['text'].tolist(), word_to_idx, MAX_SEQ_LEN)

# Labels - Encoder les labels texte en entiers
label_to_idx = {'neutre': 0, 'négatif': 1, 'positif': 2}
y_train = np.array([label_to_idx[label] for label in train_df['label'].values])
y_val = np.array([label_to_idx[label] for label in val_df['label'].values])
y_test = np.array([label_to_idx[label] for label in test_df['label'].values])

print(f"  - X_train shape: {X_train.shape}")
print(f"  - y_train shape: {y_train.shape}")

# Créer tensors
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.LongTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"  - Nombre de batches (train): {len(train_loader)}")
print(f"  - Nombre de batches (val): {len(val_loader)}")


# ========================================
# 7. INITIALISATION DU MODÈLE
# ========================================

print(f"\n" + "=" * 60)
print("🧠 INITIALISATION DU MODÈLE")
print("=" * 60)

num_classes = len(train_df['label'].unique())
vocab_size = len(word_to_idx)

model = SentimentLSTM(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print(f"  - Nombre de classes: {num_classes}")
print(f"  - Taille du vocabulaire: {vocab_size}")

# Compter les paramètres
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  📊 Paramètres du modèle:")
print(f"     - Total: {total_params:,}")
print(f"     - Entraînables: {trainable_params:,}")

# Loss et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ========================================
# 8. FONCTIONS D'ENTRAÎNEMENT
# ========================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Entraîne le modèle pendant une époque"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métriques
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Évalue le modèle"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ========================================
# 9. ENTRAÎNEMENT
# ========================================

print(f"\n" + "=" * 60)
print("🚀 ENTRAÎNEMENT")
print("=" * 60)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0

for epoch in range(EPOCHS):
    print(f"\n📍 Époque {epoch+1}/{EPOCHS}")
    print("-" * 60)
    
    # Entraînement
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # Sauvegarder historique
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Afficher résultats
    print(f"  📊 Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
    print(f"  📊 Val   - Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
    
    # Sauvegarder meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
        }, 'models/pytorch_lstm/best_model.pth')
        print(f"  ✅ Meilleur modèle sauvegardé! (val_acc: {val_acc:.2f}%)")


# ========================================
# 10. ÉVALUATION FINALE
# ========================================

print(f"\n" + "=" * 60)
print("📈 ÉVALUATION SUR L'ENSEMBLE DE TEST")
print("=" * 60)

# Charger le meilleur modèle
checkpoint = torch.load('models/pytorch_lstm/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"\n  🎯 Test Loss: {test_loss:.4f}")
print(f"  🎯 Test Accuracy: {test_acc:.2f}%")

# Prédictions détaillées
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

# Rapport de classification
print(f"\n📊 Rapport de Classification:")
print(classification_report(all_labels, all_preds, target_names=['neutre', 'négatif', 'positif']))

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
print(f"\n📊 Matrice de Confusion:")
print(cm)


# ========================================
# 11. VISUALISATION
# ========================================

print(f"\n" + "=" * 60)
print("📊 GÉNÉRATION DES GRAPHIQUES")
print("=" * 60)

# Créer figure avec subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graphique 1: Loss
axes[0].plot(history['train_loss'], label='Train', marker='o')
axes[0].plot(history['val_loss'], label='Validation', marker='s')
axes[0].set_title('📉 Évolution de la Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique 2: Accuracy
axes[1].plot(history['train_acc'], label='Train', marker='o')
axes[1].plot(history['val_acc'], label='Validation', marker='s')
axes[1].set_title('📈 Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Graphique 3: Matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Neutre', 'Négatif', 'Positif'],
            yticklabels=['Neutre', 'Négatif', 'Positif'])
axes[2].set_title('🎯 Matrice de Confusion', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Vraie classe')
axes[2].set_xlabel('Classe prédite')

plt.tight_layout()
plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
print("  ✅ Graphique sauvegardé: results/training_results.png")


# ========================================
# 12. FONCTION DE PRÉDICTION
# ========================================

def predict_sentiment(text, model, word_to_idx, device, max_len=100):
    """Prédit le sentiment d'un texte"""
    model.eval()
    
    # Prétraiter
    sequence = texts_to_sequences([text], word_to_idx, max_len)
    tensor = torch.LongTensor(sequence).to(device)
    
    # Prédire
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    label_map = {0: 'Neutre 😐', 1: 'Négatif 😞', 2: 'Positif 😊'}
    
    result = {
        'prediction': label_map[predicted.item()],
        'probabilities': {
            'Neutre': probabilities[0][0].item(),
            'Négatif': probabilities[0][1].item(),
            'Positif': probabilities[0][2].item()
        }
    }
    
    return result

# Exemples de prédiction
print(f"\n" + "=" * 60)
print("🔮 EXEMPLES DE PRÉDICTIONS")
print("=" * 60)

test_texts = [
    "Ce film est absolument génial! J'ai adoré chaque minute.",
    "C'était vraiment nul et décevant...",
    "Le produit est correct, rien de spécial.",
    "Excellent service client! Je recommande vivement.",
    "Pire achat de ma vie, à éviter absolument!"
]

for text in test_texts:
    result = predict_sentiment(text, model, word_to_idx, device)
    print(f"\n📝 Texte: \"{text}\"")
    print(f"   ➜ Prédiction: {result['prediction']}")
    print(f"   ➜ Scores:")
    for label, prob in result['probabilities'].items():
        print(f"      - {label}: {prob:.2%}")

print(f"\n" + "=" * 60)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
print(f"\n📁 Fichiers générés:")
print(f"  - Modèle: models/pytorch_lstm/best_model.pth")
print(f"  - Graphiques: results/training_results.png")
print(f"\n🎉 Vous pouvez maintenant utiliser le modèle pour prédire le sentiment de nouveaux textes!")
