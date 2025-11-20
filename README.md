# 🎯 Analyse de Sentiment (NLP)

Système d'analyse de sentiment en français avec **apprentissage en temps réel** basé sur PyTorch.

## 🌟 Fonctionnalités

- ✅ **Analyse de sentiment** : positif / négatif / neutre
- 🧠 **Apprentissage automatique** : le modèle apprend de vos corrections en temps réel
- 🌐 **Interface web** : interface intuitive avec Flask
- 💾 **Sauvegarde des corrections** : historique complet dans `data/feedback/corrections.csv`
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
python start.py
```

L'application sera disponible sur : **http://localhost:8080**

## 📋 Utilisation

### Interface Web

1. **Analyser du texte**
   - Tapez votre texte dans le champ
   - Cliquez sur "Analyser"
   - Obtenez le sentiment + confiance

2. **Corriger les erreurs**
   - Si la prédiction est fausse, cliquez "❌ Non, c'est faux"
   - Sélectionnez le bon sentiment
   - Expliquez l'erreur (optionnel)
   - Cliquez "Envoyer la correction"
   - **Le modèle apprend immédiatement** et ne fera plus cette erreur

3. **Vérifier l'apprentissage**
   - Retestez le même texte
   - Le modèle devrait maintenant donner la bonne réponse

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
Analyse de sentiment (NLP)/
├── app.py                          # Application Flask principale
├── start.py                        # Script de lancement
├── train_pytorch_simple.py         # Entraînement du modèle
├── requirements.txt                # Dépendances Python
├── templates/
│   └── index.html                  # Interface web
├── models/
│   └── pytorch_lstm/
│       ├── best_model.pth          # Modèle entraîné
│       └── vocabulary.pkl          # Vocabulaire
├── data/
│   ├── sample_data.csv             # Données d'entraînement
│   └── feedback/
│       └── corrections.csv         # Corrections utilisateur
└── src/
    ├── data_loader.py              # Chargement des données
    ├── preprocessing.py            # Prétraitement
    └── model_pytorch.py            # Architecture PyTorch
```

### Modèle PyTorch

- **Architecture** : LSTM bidirectionnel (2 couches)
- **Embedding** : 128 dimensions
- **Hidden units** : 128 (× 2 pour bidirectionnel)
- **Classes** : 3 (positif, négatif, neutre)
- **Dropout** : 0.5
- **Optimiseur** : Adam
- **Loss** : CrossEntropyLoss

### Apprentissage en Temps Réel

Lorsqu'une correction est soumise :

1. Le texte et le bon label sont enregistrés dans le CSV
2. Le modèle passe en mode entraînement
3. 5 itérations d'apprentissage sur cette correction (learning rate: 0.0001)
4. Le modèle mis à jour est sauvegardé
5. Le modèle repasse en mode évaluation

**Résultat** : Le modèle apprend instantanément et ne refait plus la même erreur !

## 🛠️ Commandes Utiles

### Entraînement Manuel

```bash
# Entraîner le modèle sur les données
python train_pytorch_simple.py
```

### Réinitialiser le Modèle

```bash
# Supprimer le modèle actuel
rm -rf models/pytorch_lstm/

# Réentraîner depuis zéro
python train_pytorch_simple.py
```

### Consulter les Corrections

```bash
# Voir toutes les corrections enregistrées
cat data/feedback/corrections.csv
```

### Tester l'API directement

```bash
# Prédiction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Ce film est génial"}'

# Correction
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text":"tu es beau",
    "predicted":"négatif",
    "correct":false,
    "user_label":"positif",
    "explanation":"C'\''est un compliment"
  }'
```

## 📊 Données

### Format des Données d'Entraînement

`data/sample_data.csv` :
```csv
text,label
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
- **Testé sur** : Python 3.14

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

- **Accuracy** : 100% sur les données de test
- **Temps d'entraînement** : ~30 secondes (MPS)
- **Taille du modèle** : 691,587 paramètres

### Limitations Connues

- **Vocabulaire limité** : 121 mots (données d'exemple)
- **Mots inconnus** : mappés à `<UNK>` → possibles erreurs
- **Solution** : Utilisez le système de feedback pour corriger et enrichir

## 🐛 Résolution de Problèmes

### Le modèle fait des erreurs

**Normal !** Le vocabulaire est limité (121 mots seulement).

**Solution** :
1. Utilisez le système de feedback pour corriger
2. Le modèle apprend et s'améliore automatiquement
3. Plus vous corrigez, meilleur il devient

### "Modèle non trouvé"

```bash
# Entraîner le modèle
python train_pytorch_simple.py
```

### "Module not found"

```bash
# Installer les dépendances manquantes
pip install torch flask pandas numpy scikit-learn
```

### Port 8080 déjà utilisé

```bash
# Tuer le processus sur le port 8080
lsof -ti:8080 | xargs kill -9

# Ou changer le port dans app.py
```

### L'environnement virtuel ne s'active pas

```bash
# Recréer l'environnement
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install torch flask pandas numpy scikit-learn
```

## 📚 Ressources

### Documentation PyTorch
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

### Flask
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Quickstart](https://flask.palletsprojects.com/en/latest/quickstart/)

## 🤝 Contribution

Ce projet est un système d'apprentissage continu. Plus vous l'utilisez et corrigez ses erreurs, meilleur il devient !

### Workflow de Contribution

1. Utilisez l'interface web
2. Corrigez les erreurs que vous trouvez
3. Le système apprend automatiquement
4. Les corrections sont sauvegardées dans `data/feedback/corrections.csv`
5. Le modèle s'améliore en temps réel

## 📝 Licence

Projet éducatif - Libre d'utilisation

## 🎓 Crédits

- **Framework** : PyTorch
- **Web** : Flask
- **Data** : pandas, NumPy
- **ML** : scikit-learn

---

**Fait avec ❤️ et 🧠 (intelligence artificielle)**

Pour démarrer : `python start.py` 🚀
