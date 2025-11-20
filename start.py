#!/usr/bin/env python3
"""
Script de Lancement - Analyse de Sentiment (NLP)
=================================================

Lance l'application web Flask avec apprentissage en temps réel.

Usage:
    python start.py
    
Puis ouvrir: http://localhost:8080

Auteur: Projet NLP
Date: Novembre 2025
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ requis")
        sys.exit(1)
    print("✅ Version compatible\n")

def check_venv():
    """Vérifie si l'environnement virtuel est activé"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  Environnement virtuel non activé")
        print("\n💡 Activation recommandée:")
        print("   macOS/Linux: source venv/bin/activate")
        print("   Windows:     venv\\Scripts\\activate\n")
        response = input("Continuer quand même? (o/N): ")
        if response.lower() != 'o':
            sys.exit(0)
    else:
        print("✅ Environnement virtuel activé\n")

def check_dependencies():
    """Vérifie les dépendances"""
    required = ['torch', 'flask', 'pandas', 'numpy', 'scikit-learn']
    missing = []
    
    print("📦 Vérification des dépendances:")
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (manquant)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Packages manquants: {', '.join(missing)}")
        print("\n💡 Installation:")
        print(f"   pip install {' '.join(missing)}\n")
        sys.exit(1)
    print()

def check_model():
    """Vérifie si le modèle est entraîné"""
    model_path = Path('models/pytorch_lstm/best_model.pth')
    vocab_path = Path('models/pytorch_lstm/vocabulary.pkl')
    
    print("🤖 Vérification du modèle:")
    if not model_path.exists() or not vocab_path.exists():
        print("   ❌ Modèle non trouvé")
        print("\n💡 Entraînement requis:")
        print("   python train_pytorch_simple.py\n")
        
        response = input("Lancer l'entraînement maintenant? (o/N): ")
        if response.lower() == 'o':
            print("\n🚀 Lancement de l'entraînement...\n")
            subprocess.run([sys.executable, 'train_pytorch_simple.py'])
            print()
        else:
            sys.exit(0)
    else:
        print("   ✅ Modèle prêt\n")

def start_app():
    """Lance l'application Flask"""
    print("="*70)
    print("🚀 LANCEMENT DE L'APPLICATION")
    print("="*70)
    print("\n📊 Interface web: http://localhost:8080")
    print("🧠 Apprentissage: EN TEMPS RÉEL")
    print("💾 Corrections: data/feedback/corrections.csv")
    print("\n💡 Ctrl+C pour arrêter\n")
    print("="*70 + "\n")
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n✋ Application arrêtée\n")
        sys.exit(0)

def main():
    """Point d'entrée principal"""
    print("\n" + "="*70)
    print("🎯 ANALYSE DE SENTIMENT (NLP) - DÉMARRAGE")
    print("="*70 + "\n")
    
    check_python_version()
    check_venv()
    check_dependencies()
    check_model()
    start_app()

if __name__ == '__main__':
    main()
