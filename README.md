# 🎯 Analyse de Sentiment (NLP)

Système d'analyse de sentiment en français basé sur PyTorch (LSTM bidirectionnel).

## 🌟 Fonctionnalités

- ✅ **Analyse de sentiment** : positif / négatif / neutre
- 🌐 **Interface web** : interface intuitive avec Flask
- 💾 **Sauvegarde des corrections** : historique dans `data/feedback/corrections.csv`
- ⚡ **Accélération GPU** : support MPS (Apple Silicon), CUDA, ou CPU

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate  # macOS/Linux
# OU
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install torch flask pandas numpy scikit-learn
```

### 2. Lancement

```bash
# Tout-en-un : vérifie, entraîne si besoin, et lance
python3 start.py
```

L'application sera disponible sur : **http://localhost:8080**

## 📋 Utilisation

### Interface Web

1. **Analyser du texte**
   - Tapez votre texte dans le champ
   - Cliquez sur "Analyser"
   - Obtenez le sentiment + niveau de confiance

2. **Corriger les erreurs**
   - Si la prédiction est fausse, cliquez "❌ Non, c'est faux"
   - Sélectionnez le bon sentiment
   - Ajoutez une explication (optionnel)
   - Cliquez "Envoyer la correction"
   - La correction est enregistrée dans `data/feedback/corrections.csv`

3. **Réentraîner avec les corrections**
   ```bash
   python3 prepare_data.py
   python3 train_pytorch_simple.py
   ```

### Exemples de Textes

```
Positifs :
- "Ce film est génial ! J'ai adoré"
- "Excellent produit, je recommande"
- "Tu es vraiment sympa"

Négatifs :
- "C'est nul et décevant"
- "Horrible expérience"
- "Je déteste ce service"

Neutres :
- "Le produit est correct"
- "C'est comme d'habitude"
- "Rien de spécial"
```

## 🏗️ Architecture

### Structure du Projet

```
sentiment-analysis-nlp/
├── app.py                          # Application Flask principale
├── start.py                        # Script de lancement
├── prepare_data.py                 # Préparation train/val/test
├── train_pytorch_simple.py         # Entraînement du modèle
├── requirements.txt                # Dépendances Python
├── templates/
│   └── index.html                  # Interface web
├── models/
│   └── pytorch_lstm/
│       ├── best_model.pth          # Modèle entraîné
│       └── vocabulary.pkl          # Vocabulaire
└── data/
    ├── raw/
    │   └── sample_data.csv         # Données d'entraînement brutes
    ├── processed/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── feedback/
        └── corrections.csv         # Corrections utilisateur
```

### Modèle PyTorch

- **Architecture** : LSTM bidirectionnel (2 couches)
- **Embedding** : 128 dimensions
- **Hidden units** : 128 (× 2 pour bidirectionnel)
- **Classes** : 3 (positif, négatif, neutre)
- **Dropout** : 0.3
- **Optimiseur** : Adam (lr=0.001)
- **Loss** : CrossEntropyLoss

## 🛠️ Commandes Utiles

### Entraînement Manuel

```bash
# Préparer les données puis entraîner
python3 prepare_data.py
python3 train_pytorch_simple.py
```

### Réinitialiser le Modèle

```bash
# Supprimer le modèle actuel
rm -rf models/pytorch_lstm/

# Réentraîner depuis zéro
python3 prepare_data.py
python3 train_pytorch_simple.py
```

### Tester l'API directement

```bash
# Prédiction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Ce film est génial"}'

# Santé
curl http://localhost:8080/health
```

## 📊 Données

### Format des Données d'Entraînement

`data/raw/sample_data.csv` :
```csv
text,sentiment
"Ce film est excellent !",positif
"Vraiment nul",négatif
"C'est correct",neutre
```

### Format des Corrections

`data/feedback/corrections.csv` :
```csv
timestamp,text,predicted_label,correct_label,user_explanation
2025-11-20T16:29:06,tu es beau,négatif,positif,C'est un compliment
```

## 🔧 Configuration

### Environnement Python

- **Version minimale** : Python 3.9
- **Recommandé** : Python 3.11+

### Accélération GPU

Le système détecte automatiquement :
- **Apple Silicon** : MPS (Metal Performance Shaders)
- **NVIDIA** : CUDA
- **Autre** : CPU (fallback)

### Port de l'Application

Par défaut : **8080**

Pour changer, modifiez la dernière ligne de `app.py` :
```python
app.run(debug=True, host='0.0.0.0', port=VOTRE_PORT)
```

## 📈 Performances

### Résultats d'Entraînement

- **Validation accuracy** : ~79%
- **Test accuracy** : ~73%
- **Temps d'entraînement** : ~2 minutes (MPS, 30 epochs)
- **Vocabulaire** : ~311 mots
- **Paramètres** : ~691 000

## 🐛 Résolution de Problèmes

### Le modèle fait des erreurs

Le vocabulaire couvre les expressions courantes françaises. Pour améliorer :

1. Utilisez le système de feedback pour enregistrer les corrections
2. Relancez l'entraînement :
   ```bash
   python3 prepare_data.py && python3 train_pytorch_simple.py
   ```

### "Modèle non trouvé"

```bash
python3 prepare_data.py && python3 train_pytorch_simple.py
```

### "Module not found"

```bash
pip install torch flask pandas numpy scikit-learn
```

### Port 8080 déjà utilisé

```bash
# Tuer le processus sur le port 8080
lsof -ti:8080 | xargs kill -9
```

## 📝 Licence

Projet éducatif - Libre d'utilisation

## 🎓 Crédits

- **Framework** : PyTorch
- **Web** : Flask
- **Data** : pandas, NumPy
- **ML** : scikit-learn

---

Pour démarrer : `python3 start.py` 🚀
