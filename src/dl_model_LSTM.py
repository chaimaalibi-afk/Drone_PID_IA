# ============================================
# DEEP LEARNING AVEC LSTM - DRONE CONTROL
# ============================================
print("="*80)
print(" ÉTAPE 6 : DEEP LEARNING AVEC LSTM")
print("="*80)

import numpy as np
import pandas as pd
import os
import time
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Création des dossiers
os.makedirs("resultats", exist_ok=True)
os.makedirs("models", exist_ok=True)

print(f" TensorFlow version: {tf.__version__}")
print(f" GPU disponible: {tf.config.list_physical_devices('GPU')}")

# ============================================
# 1. CHARGEMENT DES DONNÉES PRÉPARÉES
# ============================================
print("\n1. CHARGEMENT DES DONNÉES PRÉPARÉES")
print("-"*60)

# Charger les données du preprocessing
if os.path.exists("../data/prepared_data_advanced.npz"):
    file_path = "../data/prepared_data_advanced.npz"
elif os.path.exists("data/prepared_data_advanced.npz"):
    file_path = "data/prepared_data_advanced.npz"
else:
    print(" Fichier non trouvé!")
    exit(1)

data = np.load(file_path)
print(f" Fichier chargé: {file_path}")

# Données enrichies (feature temporels)
X_train_enh = data['X_train_enh_norm']   #  78 features
X_val_enh = data['X_val_enh_norm']
X_test_enh = data['X_test_enh_norm']

# Targets
y_train = data['y_train_norm']
y_val = data['y_val_norm']
y_test = data['y_test_norm']

# Récupérer les noms des colonnes
feature_cols = data['feature_cols'].tolist() if hasattr(data['feature_cols'], 'tolist') else list(data['feature_cols'])
target_cols = data['target_cols'].tolist() if hasattr(data['target_cols'], 'tolist') else list(data['target_cols'])

print(f" Données chargées:")
print(f"   X_train: {X_train_enh.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_val:   {X_val_enh.shape}")
print(f"   X_test:  {X_test_enh.shape}")

# ============================================
# 2. CRÉATION DES SÉQUENCES POUR LSTM
# ============================================
print("\n2. CRÉATION DES SÉQUENCES TEMPORELLES")
print("-"*60)

def create_sequences(X, y, seq_length=30):
    """Transforme les données en séquences pour LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Paramètres
#SEQ_LENGTH = 20  # 20 pas = 0.6 secondes d'historique
SEQ_LENGTH = 50  # Plus d'historique pour mieux capturer le roll

X_train_seq, y_train_seq = create_sequences(X_train_enh, y_train, SEQ_LENGTH)
X_val_seq, y_val_seq = create_sequences(X_val_enh, y_val, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_enh, y_test, SEQ_LENGTH)

print(f" Séquences créées:")
print(f"   Train: {X_train_seq.shape}")
print(f"   Val:   {X_val_seq.shape}")
print(f"   Test:  {X_test_seq.shape}")

# ============================================
# 3. ARCHITECTURE LSTM SIMPLIFIEE
# ============================================
def create_lstm_model(input_shape):
    """
    Modèle LSTM simplifié pour prédire les commandes PID
    Version plus rapide et moins régularisée
    """
    inputs = layers.Input(shape=input_shape)
    
    # Premier LSTM (plus petit)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = layers.BatchNormalization()(x)
    
    # Deuxième LSTM (sans recurrent_dropout)
    x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    
    # Couches fully connected (simplifiées)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    
    # Couche de sortie
    outputs = layers.Dense(4)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compilation avec learning rate plus élevé
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Créer le modèle (ces lignes sont ESSENTIELLES)
input_shape = (SEQ_LENGTH, X_train_enh.shape[1])
model = create_lstm_model(input_shape)

print(f" Modèle LSTM créé")
model.summary()

# ============================================
# 3. ARCHITECTURE LSTM OPTIMISÉE
# ============================================
#print("\n3. ARCHITECTURE LSTM")
#print("-"*60)

#def create_lstm_model(input_shape, num_units=[128, 64, 32], dropout_rate=0.3):
    #"""
    #Modèle LSTM pour prédire les commandes PID
    #"""
    #inputs = layers.Input(shape=input_shape)
    
    # Premier LSTM
    #x = layers.LSTM(num_units[0], return_sequences=True, 
     #               dropout=dropout_rate, recurrent_dropout=dropout_rate,
      #              kernel_regularizer=regularizers.l2(0.001))(inputs)
    #x = layers.BatchNormalization()(x)
    
    # Deuxième LSTM
    #x = layers.LSTM(num_units[1], return_sequences=True,
     #               dropout=dropout_rate, recurrent_dropout=dropout_rate,
      #              kernel_regularizer=regularizers.l2(0.001))(x)
    #x = layers.BatchNormalization()(x)
    
    # Troisième LSTM
    #x = layers.LSTM(num_units[2], return_sequences=False,
     #               dropout=dropout_rate, recurrent_dropout=dropout_rate,
      #              kernel_regularizer=regularizers.l2(0.001))(x)
    #x = layers.BatchNormalization()(x)
    
    # Couches fully connected
    #x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    #x = layers.Dropout(0.2)(x)
    #x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    #x = layers.Dropout(0.2)(x)
    
    # Couche de sortie
    #outputs = layers.Dense(4)(x)  # 4 sorties: thrust, roll, pitch, yaw
    
    #model = keras.Model(inputs=inputs, outputs=outputs)
    
    #return model

# Créer le modèle
#input_shape = (SEQ_LENGTH, X_train_orig.shape[1])
#model = create_lstm_model(input_shape)

#model.compile(
 #   optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
  #  loss='mse',
   # metrics=['mae']
#)

#print(f" Modèle LSTM créé")
#model.summary()

# ============================================
# 4. CALLBACKS POUR L'ENTRAÎNEMENT
# ============================================
print("\n4. CONFIGURATION DES CALLBACKS")
print("-"*60)

# Callback personnalisé pour afficher le R² à chaque époque
class R2Callback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        r2 = r2_score(self.y_val, y_pred)
        logs['val_r2'] = r2
        print(f" - val_r2: {r2:.4f}")

# Early stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Réduction du learning rate
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# Model checkpoint
checkpoint = callbacks.ModelCheckpoint(
    'models/best_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# R² callback
r2_callback = R2Callback(X_val_seq, y_val_seq)

print(f" Early stopping (patience=25)")
print(f" Reduce LR on plateau (patience=10)")
print(f" Model checkpoint")

# ============================================
# 5. ENTRAÎNEMENT DU MODÈLE
# ============================================
print("\n5. ENTRAÎNEMENT DU LSTM")
print("-"*60)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, reduce_lr, checkpoint, r2_callback],
    verbose=1
)

print(f"\n Entraînement terminé!")
print(f"   Meilleure époque: {len(history.history['loss'])}")
print(f"   Loss finale: {history.history['loss'][-1]:.6f}")
print(f"   Validation loss finale: {history.history['val_loss'][-1]:.6f}")

# ============================================
# 6. ÉVALUATION DU MODÈLE
# ============================================
print("\n6. ÉVALUATION DU MODÈLE")
print("-"*60)

# Utiliser le modèle déjà en mémoire (pas besoin de recharger)
best_model = model

# Prédictions sur test
y_pred = best_model.predict(X_test_seq)

# Métriques
mse = mean_squared_error(y_test_seq, y_pred)
mae = mean_absolute_error(y_test_seq, y_pred)
r2 = r2_score(y_test_seq, y_pred)

print(f"\n📊 PERFORMANCES GLOBALES:")
print(f"   R²:  {r2:.4f}")
print(f"   MSE: {mse:.4f}")
print(f"   MAE: {mae:.4f}")

# Métriques par commande
print(f"\n📊 PERFORMANCES PAR COMMANDE:")
print("-"*40)
for i, name in enumerate(target_cols):
    r2_i = r2_score(y_test_seq[:, i], y_pred[:, i])
    mae_i = mean_absolute_error(y_test_seq[:, i], y_pred[:, i])
    print(f"   {name:8} → R²: {r2_i:.4f}, MAE: {mae_i:.4f}")

# ============================================
# 7. COURBES D'APPRENTISSAGE
# ============================================
print("\n7. COURBES D'APPRENTISSAGE")
print("-"*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=1.5)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=1.5)
axes[0].set_title('Courbe d\'apprentissage - Loss (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# MAE
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=1.5)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=1.5)
axes[1].set_title('Courbe d\'apprentissage - MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_training_curves.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_training_curves.png")

# ============================================
# 8. PRÉDICTIONS vs RÉELLES
# ============================================
print("\n8. PRÉDICTIONS vs RÉELLES")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    ax.scatter(y_test_seq[:500, i], y_pred[:500, i], alpha=0.3, s=10)
    ax.plot([y_test_seq[:, i].min(), y_test_seq[:, i].max()], 
            [y_test_seq[:, i].min(), y_test_seq[:, i].max()], 
            'r--', linewidth=2, label='Prédiction parfaite')
    ax.set_xlabel(f'{name} (réel)')
    ax.set_ylabel(f'{name} (prédit)')
    ax.set_title(f'{name} - R²={r2_score(y_test_seq[:, i], y_pred[:, i]):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_predictions.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_predictions.png")

# ============================================
# 9. ANALYSE DES RÉSIDUS
# ============================================
print("\n9. ANALYSE DES RÉSIDUS")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    residuals = y_test_seq[:, i] - y_pred[:, i]
    ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax.axvline(x=np.mean(residuals), color='green', linestyle='--', 
               label=f'Moyenne: {np.mean(residuals):.4f}')
    ax.set_xlabel('Résidu')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'{name} - Distribution des résidus')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_residuals.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_residuals.png")

# ============================================
# 10. TESTS DE ROBUSTESSE
# ============================================
print("\n10. TESTS DE ROBUSTESSE")
print("-"*60)

def test_noise_robustness(model, X_test, y_test, noise_levels=[0, 0.05, 0.1, 0.2, 0.3]):
    """Teste la robustesse au bruit"""
    results = []
    for noise in noise_levels:
        X_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        y_pred = model.predict(X_noisy, verbose=0)
        r2 = r2_score(y_test, y_pred)
        results.append({'noise': noise, 'r2': r2})
        print(f"   Bruit {noise*100:.0f}% → R²: {r2:.4f}")
    return results

def test_perturbation_rejection(model, X_test, y_test, magnitudes=[0, 0.5, 1.0, 2.0]):
    """Teste le rejet de perturbations"""
    results = []
    mid = len(X_test) // 2
    for mag in magnitudes:
        X_pert = X_test.copy()
        X_pert[mid:mid+50] += mag
        y_pred = model.predict(X_pert, verbose=0)
        err_before = mean_squared_error(y_test[:mid], y_pred[:mid])
        err_after = mean_squared_error(y_test[mid:], y_pred[mid:])
        ratio = err_after / err_before if err_before > 0 else 1
        results.append({'magnitude': mag, 'ratio': ratio})
        print(f"   Magnitude {mag} → Ratio: {ratio:.3f}")
    return results

print("\n Test robustesse au bruit:")
noise_results = test_noise_robustness(best_model, X_test_seq, y_test_seq)

print("\n Test rejet de perturbation:")
pert_results = test_perturbation_rejection(best_model, X_test_seq, y_test_seq)

# Visualisation robustesse
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

noise_vals = [r['noise'] for r in noise_results]
r2_vals = [r['r2'] for r in noise_results]
axes[0].plot(noise_vals, r2_vals, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Niveau de bruit')
axes[0].set_ylabel('R²')
axes[0].set_title('Robustesse au bruit')
axes[0].grid(True, alpha=0.3)

mag_vals = [r['magnitude'] for r in pert_results]
ratio_vals = [r['ratio'] for r in pert_results]
axes[1].plot(mag_vals, ratio_vals, 's-', linewidth=2, markersize=8, color='red')
axes[1].axhline(y=1.0, color='green', linestyle='--')
axes[1].set_xlabel('Magnitude perturbation')
axes[1].set_ylabel('Ratio dégradation')
axes[1].set_title('Rejet de perturbations')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_robustness.png", dpi=150)
plt.show()

# ============================================
# 11. COMPARAISON PID vs LSTM
# ============================================
print("\n11. COMPARAISON PID vs LSTM")
print("-"*60)

print(f"""
PID théorique (optimal):
   - R² = 1.0000
   - MAE = 0.0000

Modèle LSTM:
   - R² = {r2:.4f}
   - MAE = {mae:.4f}

Écart par rapport au PID:
   - Différence R²: {1.0 - r2:.4f}
   - MAE supplémentaire: {mae:.4f}
""")

# ============================================
# 12. SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n12. SAUVEGARDE DES RÉSULTATS")
print("-"*60)

results = {
    'r2_global': float(r2),
    'mae_global': float(mae),
    'mse_global': float(mse),
    'per_command': {
        target_cols[i]: {
            'r2': float(r2_score(y_test_seq[:, i], y_pred[:, i])),
            'mae': float(mean_absolute_error(y_test_seq[:, i], y_pred[:, i]))
        } for i in range(len(target_cols))
    },
    'robustness': {
        'noise': noise_results,
        'perturbation': pert_results
    },
    'best_epoch': len(history.history['loss'])
}

with open('resultats/lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(" Résultats sauvegardés: resultats/lstm_results.json")
print(" Modèle sauvegardé: models/best_lstm_model.h5")

# ============================================
# 13. RAPPORT FINAL
# ============================================
print("\n" + "="*80)
print(" RAPPORT FINAL - LSTM DRONE CONTROL")
print("="*80)
print(f"""
Performances globales:
   - R² = {r2:.4f}
   - MAE = {mae:.4f}
   - MSE = {mse:.4f}

Performance par commande:
   - Thrust: R² = {results['per_command']['thrust']['r2']:.4f}
   - Roll:   R² = {results['per_command']['roll']['r2']:.4f}
   - Pitch:  R² = {results['per_command']['pitch']['r2']:.4f}
   - Yaw:    R² = {results['per_command']['yaw']['r2']:.4f}

Robustesse:
   - Bruit 10% → R² = {noise_results[2]['r2']:.4f}
   - Bruit 20% → R² = {noise_results[3]['r2']:.4f}

Fichiers générés:
   - models/best_lstm_model.h5
   - resultats/lstm_training_curves.png
   - resultats/lstm_predictions.png
   - resultats/lstm_residuals.png
   - resultats/lstm_robustness.png
   - resultats/lstm_results.json
""")
print("\n LSTM DRONE CONTROL TERMINÉ AVEC SUCCÈS !")