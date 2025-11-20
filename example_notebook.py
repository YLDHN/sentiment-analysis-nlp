"""
Exemple de notebook Jupyter pour l'analyse de sentiment
Ce notebook montre comment utiliser le projet étape par étape
"""

# %% [markdown]
# # Analyse de Sentiment avec NLP
# 
# Ce notebook démontre l'utilisation complète du système d'analyse de sentiment.
# 
# ## Table des matières
# 1. Configuration et imports
# 2. Chargement et préparation des données
# 3. Prétraitement du texte
# 4. Construction et entraînement du modèle
# 5. Évaluation
# 6. Prédictions

# %% [markdown]
# ## 1. Configuration et imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modules du projet
import config
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from model import create_model
from train import ModelTrainer
from evaluate import ModelEvaluator
from predict import SentimentPredictor

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Imports réussis")

# %% [markdown]
# ## 2. Chargement et préparation des données

# %%
# Créer un dataset d'exemple
loader = DataLoader()
sample_path = 'data/raw/sample_data.csv'
loader.create_sample_dataset(sample_path, num_samples=1000)

# Charger les données
df = loader.load_csv(sample_path, text_column='text', label_column='sentiment')

# Afficher quelques exemples
print("\nExemples de données:")
print(df.head(10))

# Distribution des classes
print("\nDistribution des sentiments:")
print(df['sentiment'].value_counts())

# Visualiser la distribution
plt.figure(figsize=(10, 5))
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Distribution des sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Nombre d\'échantillons')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Encoder et diviser les données

# %%
# Encoder les labels
df, label_to_id, id_to_label = loader.encode_labels(df)

print("Encodage des labels:")
print(label_to_id)

# Diviser les données
train_df, val_df, test_df = loader.split_data(df)

# Sauvegarder
loader.save_splits(train_df, val_df, test_df)

print(f"\nTrain: {len(train_df)} échantillons")
print(f"Val:   {len(val_df)} échantillons")
print(f"Test:  {len(test_df)} échantillons")

# %% [markdown]
# ## 4. Prétraitement du texte

# %%
# Créer le preprocesseur
preprocessor = TextPreprocessor(
    max_vocab_size=5000,
    max_sequence_length=100,
    use_stopwords=False
)

# Exemple de nettoyage
sample_text = "Ce FILM est VRAIMENT magnifique!!! J'ai adoré #cinema"
cleaned = preprocessor.clean_text(sample_text)
print(f"Original: {sample_text}")
print(f"Nettoyé:  {cleaned}")

# Tokenization
tokens = preprocessor.tokenize(cleaned)
print(f"Tokens:   {tokens}")

# %% [markdown]
# ## 5. Construire et entraîner le modèle

# %%
# Préparer les données d'entraînement
X_train = preprocessor.preprocess_data(train_df['text'].tolist(), build_vocab=True)
X_val = preprocessor.preprocess_data(val_df['text'].tolist(), build_vocab=False)

y_train = train_df['label_encoded'].values
y_val = val_df['label_encoded'].values

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")

# Créer le modèle
vocab_size = len(preprocessor.word_to_idx)
print(f"\nTaille du vocabulaire: {vocab_size}")

model_wrapper, model = create_model(
    model_type='lstm',
    vocab_size=vocab_size,
    embedding_dim=100,
    lstm_units=64,
    num_classes=3
)

# Afficher l'architecture
model.summary()

# %% [markdown]
# ## 6. Entraîner le modèle

# %%
# Créer le trainer
trainer = ModelTrainer(model_type='lstm', model_name='lstm_notebook')
trainer.model = model
trainer.preprocessor = preprocessor

# Entraîner
history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=5,
    batch_size=32
)

# Visualiser l'historique d'entraînement
trainer.plot_history()

# %% [markdown]
# ## 7. Évaluer le modèle

# %%
# Préparer les données de test
X_test = preprocessor.preprocess_data(test_df['text'].tolist(), build_vocab=False)
y_test = test_df['label_encoded'].values

# Évaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Prédictions
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Métriques détaillées
from sklearn.metrics import classification_report, confusion_matrix

print("\nRapport de classification:")
print(classification_report(y_test, predicted_classes, 
                          target_names=config.SENTIMENT_CLASSES))

# Matrice de confusion
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.SENTIMENT_CLASSES,
            yticklabels=config.SENTIMENT_CLASSES)
plt.title('Matrice de confusion')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Prédictions sur nouveaux textes

# %%
# Sauvegarder le modèle et le preprocesseur
import os
os.makedirs('models/lstm_notebook', exist_ok=True)
model.save('models/lstm_notebook/model.h5')
preprocessor.save_preprocessor('models/lstm_notebook/preprocessor.pkl')

# Créer un predictor
predictor = SentimentPredictor(
    model_path='models/lstm_notebook/model.h5',
    preprocessor_path='models/lstm_notebook/preprocessor.pkl'
)

# Exemples de prédictions
test_texts = [
    "Ce restaurant est absolument fantastique! La nourriture est délicieuse.",
    "Très déçu, service horrible et plats froids.",
    "C'est correct, rien d'exceptionnel.",
    "J'adore ce produit, meilleur achat de l'année!",
    "Qualité médiocre, je ne recommande pas du tout."
]

print("\n" + "="*70)
print("PRÉDICTIONS SUR NOUVEAUX TEXTES")
print("="*70)

for text in test_texts:
    result = predictor.predict_single(text, return_probabilities=True)
    
    print(f"\nTexte: \"{text}\"")
    print(f"Sentiment: {result['sentiment'].upper()} (confiance: {result['confidence']:.2%})")
    print(f"Probabilités: {result['probabilities']}")

# %% [markdown]
# ## 9. Analyse de distribution

# %%
# Analyser la distribution sur un ensemble de textes
distribution = predictor.analyze_sentiment_distribution(test_texts)

# Visualiser
labels = list(distribution['sentiment_counts'].keys())
counts = list(distribution['sentiment_counts'].values())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(labels, counts, color=['red', 'gray', 'green'])
plt.title('Distribution des sentiments')
plt.ylabel('Nombre de textes')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['red', 'gray', 'green'])
plt.title('Proportion des sentiments')

plt.tight_layout()
plt.show()

print(f"\nConfiance moyenne: {distribution['average_confidence']:.2%}")

# %% [markdown]
# ## 10. Conclusion
# 
# Ce notebook a démontré le pipeline complet :
# - Chargement et préparation des données
# - Prétraitement du texte
# - Construction et entraînement d'un modèle LSTM
# - Évaluation avec métriques détaillées
# - Prédictions sur nouveaux textes
# 
# Vous pouvez maintenant :
# - Tester avec vos propres données
# - Essayer différentes architectures (GRU, CNN, BERT)
# - Ajuster les hyperparamètres
# - Utiliser des embeddings pré-entraînés

# %%
print("\n✓ Notebook terminé avec succès!")
