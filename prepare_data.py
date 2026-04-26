#!/usr/bin/env python3
"""
Prépare les données brutes en splits train/val/test.
Usage: python prepare_data.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/raw/sample_data.csv')

# Renommer la colonne sentiment -> label si nécessaire
if 'sentiment' in df.columns:
    df = df.rename(columns={'sentiment': 'label'})

df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.strip() != '']

# Encoder les labels
label_to_idx = {'neutre': 0, 'négatif': 1, 'positif': 2}
df = df[df['label'].isin(label_to_idx.keys())]
df['label_encoded'] = df['label'].map(label_to_idx)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Distribution train:\n{train_df['label'].value_counts()}")
