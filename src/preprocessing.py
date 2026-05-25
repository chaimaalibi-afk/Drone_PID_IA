# ============================================
# PREPROCESSING - DATASET AVANCÉ (600k échantillons)
# ============================================
print("="*80)
print(" PRETRAITEMENT DES DONNÉES POUR LSTM")
print("="*80)

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CHARGEMENT DU DATASET
# ============================================
print("\n1. CHARGEMENT DU DATASET")
print("-"*60)

# Adapter le chemin selon votre structure
file_path = '../data/drone_dataset_advanced.csv'  # ou 'data/drone_dataset_advanced.csv'

if not os.path.exists(file_path):
    # Essayer l'autre chemin
    file_path = r'C:\Users\pc lenovo\Documents\Data Science\Drone_PID_IA\data\drone_dataset_advanced.csv'
    
if not os.path.exists(file_path):
    print(f" Fichier non trouvé")
    exit(1)

df = pd.read_csv(file_path)
print(f" Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f" Colonnes: {list(df.columns)}")

# ============================================
# 2. DÉFINITION DES COLONNES
# ============================================
print("\n2. DÉFINITION DES COLONNES")
print("-"*60)

feature_cols = ['position_x', 'position_y', 'position_z', 
                'velocity_x', 'velocity_y', 'velocity_z']
target_cols = ['thrust', 'roll', 'pitch', 'yaw']

print(f" Features (entrées): {len(feature_cols)} colonnes")
print(f" Targets (sorties): {len(target_cols)} colonnes")

# ============================================
# 3. NETTOYAGE DE BASE
# ============================================
print("\n3. NETTOYAGE DE BASE")
print("-"*60)

# Suppression des doublons
initial_len = len(df)
df = df.drop_duplicates()
print(f"   Doublons supprimés: {initial_len - len(df)}")

# Vérification des valeurs manquantes
print(f"   Valeurs manquantes: {df.isnull().sum().sum()}")

if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print(f"   Lignes avec NaN supprimées")

# ============================================
# 4. PAS DE FILTRAGE SAVITZKY-GOLAY
# ============================================
print("\n4. FILTRAGE DES DONNÉES")
print("-"*60)
print("    Aucun filtrage appliqué (conservation du bruit)")
print("    Le bruit est conservé pour une meilleure robustesse du LSTM")

# ============================================
# 5. CRÉATION DES FEATURES TEMPORELLES
# ============================================
print("\n5. CRÉATION DES FEATURES TEMPORELLES")
print("-"*60)

def create_temporal_features(df, feature_cols, window_size=10, dt=0.02):
    """
    Crée des caractéristiques temporelles pour capturer la dynamique du système
    - Variables retardées (lags) : historique des états précédents
    - Moyenne mobile : tendance locale
    - Dérivée temporelle : accélération
    """
    df_temp = df.copy()
    original_cols = len(df_temp.columns)
    
    for col in feature_cols:
        # 1. Variables retardées (lags) - historique
        for lag in range(1, window_size + 1):
            df_temp[f'{col}_lag_{lag}'] = df_temp[col].shift(lag)
        
        # 2. Moyenne mobile - tendance locale
        df_temp[f'{col}_rolling_mean'] = df_temp[col].rolling(window=window_size).mean()
        
        # 3. Dérivée temporelle (accélération)
        df_temp[f'{col}_derivative'] = df_temp[col].diff() / dt
    
    # Supprimer les lignes avec NaN (créées par les décalages)
    before_drop = len(df_temp)
    df_temp = df_temp.dropna()
    after_drop = len(df_temp)
    
    print(f"   Fenêtre temporelle: {window_size} pas ({window_size * dt:.2f} secondes)")
    print(f"   Nouvelles colonnes créées: {len(df_temp.columns) - original_cols}")
    print(f"   Lignes supprimées (NaN): {before_drop - after_drop}")
    print(f"   Taille finale: {df_temp.shape[0]:,} lignes × {df_temp.shape[1]} colonnes")
    
    return df_temp

# Appliquer la création des features temporelles
window_size = 10  # 10 pas = 0.2 seconde d'historique
df_features = create_temporal_features(df, feature_cols, window_size=window_size, dt=0.02)

# Mettre à jour la liste des features (inclure les nouvelles colonnes)
new_feature_cols = [col for col in df_features.columns if col not in target_cols]
print(f"\n    Nouvelles features: {len(new_feature_cols)} colonnes")

# ============================================
# 6. SPLIT TEMPOREL (70/15/15)
# ============================================
print("\n6. SPLIT TEMPOREL (respect de l'ordre chronologique)")
print("-"*60)

def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Division temporelle des données (pas de mélange aléatoire)
    Respecte la causalité : on n'utilise pas le futur pour prédire le passé
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    print(f"   Train: {len(train):,} échantillons (t=0 à {train_end}) - {len(train)/n*100:.1f}%")
    print(f"   Val:   {len(val):,} échantillons (t={train_end} à {val_end}) - {len(val)/n*100:.1f}%")
    print(f"   Test:  {len(test):,} échantillons (t={val_end} à {n}) - {len(test)/n*100:.1f}%")
    
    return train, val, test

train, val, test = temporal_split(df_features, train_ratio=0.7, val_ratio=0.15)

# ============================================
# 7. SÉPARATION X (features) et y (targets)
# ============================================
print("\n7. SÉPARATION X (features) et y (targets)")
print("-"*60)

# Version originale (sans features temporelles)
X_train_orig = train[feature_cols].values
X_val_orig = val[feature_cols].values
X_test_orig = test[feature_cols].values

# Version enrichie (avec features temporelles)
X_train_enh = train[new_feature_cols].values
X_val_enh = val[new_feature_cols].values
X_test_enh = test[new_feature_cols].values

# Targets (commandes)
y_train = train[target_cols].values
y_val = val[target_cols].values
y_test = test[target_cols].values

print(f"   Version originale (sans features temporelles):")
print(f"     X_train: {X_train_orig.shape}")
print(f"     X_val:   {X_val_orig.shape}")
print(f"     X_test:  {X_test_orig.shape}")
print(f"\n   Version enrichie (avec features temporelles):")
print(f"     X_train: {X_train_enh.shape}")
print(f"     X_val:   {X_val_enh.shape}")
print(f"     X_test:  {X_test_enh.shape}")
print(f"\n   Targets (commandes):")
print(f"     y_train: {y_train.shape}")
print(f"     y_val:   {y_val.shape}")
print(f"     y_test:  {y_test.shape}")

# ============================================
# 8. NORMALISATION DES DONNÉES
# ============================================
print("\n8. NORMALISATION DES DONNÉES")
print("-"*60)

# Choix des scalers
# RobustScaler : moins sensible aux outliers (recommandé pour features avec perturbations)
# StandardScaler : pour les targets (distribution normale)
scaler_X_orig = RobustScaler()
scaler_X_enh = RobustScaler()
scaler_y = StandardScaler()

# Normalisation des features (version originale)
X_train_orig_norm = scaler_X_orig.fit_transform(X_train_orig)
X_val_orig_norm = scaler_X_orig.transform(X_val_orig)
X_test_orig_norm = scaler_X_orig.transform(X_test_orig)

# Normalisation des features (version enrichie)
X_train_enh_norm = scaler_X_enh.fit_transform(X_train_enh)
X_val_enh_norm = scaler_X_enh.transform(X_val_enh)
X_test_enh_norm = scaler_X_enh.transform(X_test_enh)

# Normalisation des targets
y_train_norm = scaler_y.fit_transform(y_train)
y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

print(f"    Scaler X (original): {type(scaler_X_orig).__name__}")
print(f"    Scaler X (enrichi): {type(scaler_X_enh).__name__}")
print(f"    Scaler y: {type(scaler_y).__name__}")

# ============================================
# 9. VÉRIFICATION DE LA NORMALISATION
# ============================================
print("\n9. VÉRIFICATION DE LA NORMALISATION")
print("-"*60)

print(f"\n Statistiques des features normalisées (X_train_orig):")
print(f"   mean: {X_train_orig_norm.mean():.6f} (attendue ≈ 0)")
print(f"   std:  {X_train_orig_norm.std():.6f} (attendue ≈ 1)")

print(f"\n Statistiques des targets normalisées (y_train):")
for i, name in enumerate(target_cols):
    print(f"   {name}: mean={y_train_norm[:, i].mean():.6f}, std={y_train_norm[:, i].std():.6f}")

# ============================================
# 10. SAUVEGARDE DES DONNÉES PRÉPARÉES
# ============================================
print("\n10. SAUVEGARDE DES DONNÉES PRÉPARÉES")
print("-"*60)

import joblib
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Sauvegarde des scalers
joblib.dump(scaler_X_orig, "models/scaler_X_original.pkl")
joblib.dump(scaler_X_enh, "models/scaler_X_enhanced.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# Sauvegarde des données sous format numpy
np.savez("data/prepared_data_advanced.npz",
         X_train_orig_norm=X_train_orig_norm,
         X_val_orig_norm=X_val_orig_norm,
         X_test_orig_norm=X_test_orig_norm,
         X_train_enh_norm=X_train_enh_norm,
         X_val_enh_norm=X_val_enh_norm,
         X_test_enh_norm=X_test_enh_norm,
         y_train_norm=y_train_norm,
         y_val_norm=y_val_norm,
         y_test_norm=y_test_norm,
         feature_cols=feature_cols,
         target_cols=target_cols,
         window_size=window_size)

print(" Scalers sauvegardés dans 'models/'")
print(" Données préparées sauvegardées dans 'data/prepared_data_advanced.npz'")

# ============================================
# 11. RÉCAPITULATIF FINAL
# ============================================
print("\n" + "="*80)
print(" RÉCAPITULATIF FINAL DU PREPROCESSING")
print("="*80)
print("\n PREPROCESSING TERMINÉ AVEC SUCCÈS !")
print(" Les données sont prêtes pour l'entraînement du LSTM !")