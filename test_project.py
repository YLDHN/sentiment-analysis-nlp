#!/usr/bin/env python3
"""
Test rapide du projet Analyse de Sentiment
Vérifie que tous les composants fonctionnent correctement
"""

import sys
import os

def test_imports():
    """Teste l'importation de tous les modules"""
    print("=" * 60)
    print("TEST 1: Importation des modules")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - MPS disponible: {torch.backends.mps.is_available()}")
        print(f"  - CUDA disponible: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import nltk
        print(f"✓ NLTK {nltk.__version__}")
    except ImportError as e:
        print(f"✗ NLTK: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    print("\n✓ Toutes les bibliothèques sont importées avec succès!\n")
    return True


def test_project_modules():
    """Teste l'importation des modules du projet"""
    print("=" * 60)
    print("TEST 2: Modules du projet")
    print("=" * 60)
    
    modules = [
        'config',
        'data_loader',
        'preprocessing',
        'model_pytorch',
        'train',
        'evaluate',
        'predict',
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            return False
    
    print("\n✓ Tous les modules du projet sont importables!\n")
    return True


def test_pytorch_models():
    """Teste la création des modèles PyTorch"""
    print("=" * 60)
    print("TEST 3: Modèles PyTorch")
    print("=" * 60)
    
    try:
        from model_pytorch import (
            SentimentLSTM_PyTorch, 
            SentimentGRU_PyTorch, 
            SentimentCNN_PyTorch,
            get_device
        )
        import torch
        
        device = get_device()
        print(f"✓ Device détecté: {device}")
        
        # Test LSTM
        model_lstm = SentimentLSTM_PyTorch(vocab_size=1000, embedding_dim=64)
        print(f"✓ SentimentLSTM_PyTorch créé")
        
        # Test GRU
        model_gru = SentimentGRU_PyTorch(vocab_size=1000, embedding_dim=64)
        print(f"✓ SentimentGRU_PyTorch créé")
        
        # Test CNN
        model_cnn = SentimentCNN_PyTorch(vocab_size=1000, embedding_dim=64)
        print(f"✓ SentimentCNN_PyTorch créé")
        
        # Test forward pass
        batch_size = 4
        seq_len = 32
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            output_lstm = model_lstm(dummy_input)
            output_gru = model_gru(dummy_input)
            output_cnn = model_cnn(dummy_input)
        
        print(f"✓ Forward pass LSTM: {output_lstm.shape}")
        print(f"✓ Forward pass GRU: {output_gru.shape}")
        print(f"✓ Forward pass CNN: {output_cnn.shape}")
        
        print("\n✓ Tous les modèles PyTorch fonctionnent!\n")
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_pipeline():
    """Teste le pipeline de données"""
    print("=" * 60)
    print("TEST 4: Pipeline de données")
    print("=" * 60)
    
    try:
        from data_loader import DataLoader
        from preprocessing import TextPreprocessor
        
        # Test DataLoader
        loader = DataLoader()
        print("✓ DataLoader créé")
        
        # Test création dataset exemple (temporaire)
        import tempfile
        import pandas as pd
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        loader.create_sample_dataset(output_path=temp_path, num_samples=100)
        df = pd.read_csv(temp_path)
        print(f"✓ Dataset d'exemple créé: {df.shape}")
        print(f"  - Colonnes: {list(df.columns)}")
        print(f"  - Distribution: {df['sentiment'].value_counts().to_dict()}")
        
        # Nettoyer le fichier temporaire
        import os as os_module
        os_module.unlink(temp_path)
        
        # Test TextPreprocessor
        preprocessor = TextPreprocessor()
        print("✓ TextPreprocessor créé")
        
        # Test nettoyage
        text = "Ceci est un TEST! #sentiment @user https://example.com 😊"
        cleaned = preprocessor.clean_text(text)
        print(f"✓ Nettoyage: '{text}' -> '{cleaned}'")
        
        # Test tokenization
        tokens = preprocessor.tokenize(cleaned)
        print(f"✓ Tokenization: {len(tokens)} tokens")
        
        # Test vocabulaire
        all_texts = df['text'].tolist()
        preprocessor.build_vocabulary(all_texts)
        print(f"✓ Vocabulaire construit avec succès")
        
        print("\n✓ Pipeline de données fonctionne!\n")
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directories():
    """Vérifie que les répertoires nécessaires existent"""
    print("=" * 60)
    print("TEST 5: Structure de répertoires")
    print("=" * 60)
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'results',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (manquant)")
            all_exist = False
    
    if all_exist:
        print("\n✓ Tous les répertoires existent!\n")
    else:
        print("\n⚠ Certains répertoires sont manquants (seront créés automatiquement)\n")
    
    return True


def main():
    """Exécute tous les tests"""
    print("\n" + "=" * 60)
    print("🧪 TEST DU PROJET ANALYSE DE SENTIMENT")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports des bibliothèques", test_imports()))
    
    # Test 2: Modules projet
    results.append(("Modules du projet", test_project_modules()))
    
    # Test 3: Modèles PyTorch
    results.append(("Modèles PyTorch", test_pytorch_models()))
    
    # Test 4: Pipeline de données
    results.append(("Pipeline de données", test_data_pipeline()))
    
    # Test 5: Répertoires
    results.append(("Structure de répertoires", test_directories()))
    
    # Résumé
    print("=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\n{total_passed}/{total_tests} tests réussis")
    
    if total_passed == total_tests:
        print("\n🎉 Tous les tests sont passés! Le projet est prêt à l'emploi.\n")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) échoué(s). Vérifiez les erreurs ci-dessus.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
