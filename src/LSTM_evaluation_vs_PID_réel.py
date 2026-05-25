# ============================================
# LSTM_EVALUATION.PY - ÉVALUATION DU MODÈLE LSTM ENTRAÎNÉ
# AVEC COMPARAISON RÉELLE LSTM vs PID
# ============================================
print("="*80)
print("ÉVALUATION DU MODÈLE LSTM - DRONE CONTROL")
print("="*80)

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Création des dossiers
os.makedirs("resultats", exist_ok=True)

print(f" TensorFlow version: {tf.__version__}")

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

# Utiliser la version ENRICHIE (avec features temporelles)
X_train_enh = data['X_train_enh_norm']
X_val_enh = data['X_val_enh_norm']
X_test_enh = data['X_test_enh_norm']

# Targets
y_train = data['y_train_norm']
y_val = data['y_val_norm']
y_test = data['y_test_norm']

# Récupérer les noms des colonnes
target_cols = data['target_cols'].tolist() if hasattr(data['target_cols'], 'tolist') else list(data['target_cols'])

print(f" Données chargées (version ENRICHIE avec features temporelles):")
print(f"   X_test_enh: {X_test_enh.shape}")
print(f"   y_test: {y_test.shape}")
print(f"   Commandes: {target_cols}")


scaler_X_enh = joblib.load('models/scaler_X_enhanced.pkl') if os.path.exists('models/scaler_X_enhanced.pkl') else None
scaler_y = joblib.load('models/scaler_y.pkl')

# ============================================
# 2. CRÉATION DES SÉQUENCES POUR LSTM
# ============================================
print("\n2. CRÉATION DES SÉQUENCES TEMPORELLES")
print("-"*60)

def create_sequences(X, y, seq_length=20):
    """Transforme les données en séquences pour LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 50

# Utiliser X_train_enh, X_val_enh, X_test_enh (version enrichie)
_, y_train_seq = create_sequences(X_train_enh, y_train, SEQ_LENGTH)
_, y_val_seq = create_sequences(X_val_enh, y_val, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_enh, y_test, SEQ_LENGTH)

print(f" Séquences créées (avec features temporelles):")
print(f"   X_test_seq: {X_test_seq.shape}")
print(f"   y_test_seq: {y_test_seq.shape}")

# ============================================
# 2.5 MODÈLE SÉPARÉ POUR LE THRUST (AJOUTER ICI)
# ============================================
print("\n2.5 MODÈLE LSTM SPÉCIALISÉ POUR LE THRUST")
print("-"*60)

def create_thrust_lstm_model(input_shape):
    """
    Modèle LSTM dédié uniquement à la prédiction du thrust
    """
    inputs = keras.layers.Input(shape=input_shape)
    
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)  # Une seule sortie pour thrust
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='mse',
        metrics=['mae']
    )
    return model

# Créer et entraîner le modèle thrust séparé
thrust_input_shape = (SEQ_LENGTH, X_train_orig.shape[1])
thrust_model = create_thrust_lstm_model(thrust_input_shape)

# Callbacks pour thrust
thrust_checkpoint = callbacks.ModelCheckpoint(
    'models/best_thrust_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Entraînement du modèle thrust
print(" Entraînement du modèle thrust séparé...")
thrust_history = thrust_model.fit(
    X_train_seq, y_train_seq[:, 0],  # Seule la colonne thrust
    validation_data=(X_val_seq, y_val_seq[:, 0]),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, reduce_lr, thrust_checkpoint],
    verbose=1
)

print(f" Modèle thrust entraîné avec succès!")

# ============================================
# 3. CHARGEMENT DU MODÈLE LSTM ENTRAÎNÉ
# ============================================
print("\n3. CHARGEMENT DU MODÈLE LSTM")
print("-"*60)

# Définir l'architecture (identique à l'entraînement)
def create_lstm_model(input_shape):
    """
    Modèle LSTM simplifié (identique à l'entraînement)
    """
    inputs = keras.layers.Input(shape=input_shape)
    
    # Premier LSTM
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    
    # Deuxième LSTM
    x = keras.layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Couches fully connected
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    
    # Couche de sortie
    outputs = keras.layers.Dense(4)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Créer le modèle avec la bonne forme
input_shape = (SEQ_LENGTH, X_test_enh.shape[1])
model = create_lstm_model(input_shape)

# Charger les poids sauvegardés
if os.path.exists('models/best_lstm_model.h5'):
    model.load_weights('models/best_lstm_model.h5')
    print(" Poids du modèle chargés: models/best_lstm_model.h5")
else:
    print(" Fichier du modèle non trouvé!")
    exit(1)

# ============================================
# 3.5 CHARGEMENT DU MODÈLE THRUST SÉPARÉ (AJOUTER ICI)
# ============================================
print("\n3.5 CHARGEMENT DU MODÈLE THRUST SÉPARÉ")
print("-"*60)

def create_thrust_lstm_model(input_shape):
    """
    Modèle LSTM pour thrust (identique à l'entraînement)
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), loss='mse')
    return model

# Charger le modèle thrust
thrust_model = create_thrust_lstm_model(input_shape)
if os.path.exists('models/best_thrust_lstm_model.h5'):
    thrust_model.load_weights('models/best_thrust_lstm_model.h5')
    print(" Modèle thrust chargé: models/best_thrust_lstm_model.h5")
else:
    print(" Modèle thrust non trouvé, utilisation du modèle principal")
    thrust_model = None

# ============================================
# 4. ÉVALUATION COMBINÉE (Thrust séparé + autres)
# ============================================
print("\n4. ÉVALUATION COMBINÉE LSTM")
print("-"*60)

# Prédictions combinées
y_pred_combined = np.zeros_like(y_test_seq)

if thrust_model is not None:
    # Thrust vient du modèle dédié
    y_pred_combined[:, 0] = thrust_model.predict(X_test_seq).flatten()
    # Roll, Pitch, Yaw viennent du modèle principal
    y_pred_combined[:, 1:] = model.predict(X_test_seq)[:, 1:]
else:
    # Fallback : tout vient du modèle principal
    y_pred_combined = model.predict(X_test_seq)

# Métriques globales
mse = mean_squared_error(y_test_seq, y_pred_combined)
mae = mean_absolute_error(y_test_seq, y_pred_combined)
r2 = r2_score(y_test_seq, y_pred_combined)

print(f"\n PERFORMANCES LSTM (AVEC MODÈLE THRUST DÉDIÉ):")
print(f"   R² = {r2:.4f}")
print(f"   MSE = {mse:.4f}")
print(f"   MAE = {mae:.4f}")

# Métriques par commande
print(f"\n PERFORMANCES PAR COMMANDE:")
print("-"*40)
for i, name in enumerate(target_cols):
    r2_i = r2_score(y_test_seq[:, i], y_pred_combined[:, i])
    mae_i = mean_absolute_error(y_test_seq[:, i], y_pred_combined[:, i])
    print(f"   {name:8} → R²: {r2_i:.4f}, MAE: {mae_i:.4f}")

# ============================================
# 4. ÉVALUATION DU MODÈLE LSTM
# ============================================
print("\n4. ÉVALUATION DU MODÈLE LSTM")
print("-"*60)

# Prédictions sur test
y_pred = model.predict(X_test_seq)

# Métriques globales LSTM
lstm_mse = mean_squared_error(y_test_seq, y_pred)
lstm_mae = mean_absolute_error(y_test_seq, y_pred)
lstm_r2 = r2_score(y_test_seq, y_pred)

print(f"\n PERFORMANCES LSTM:")
print(f"   R² = {lstm_r2:.4f}")
print(f"   MSE = {lstm_mse:.4f}")
print(f"   MAE = {lstm_mae:.4f}")

# Métriques par commande LSTM
print(f"\n PERFORMANCES LSTM PAR COMMANDE:")
print("-"*40)
for i, name in enumerate(target_cols):
    r2_i = r2_score(y_test_seq[:, i], y_pred[:, i])
    mae_i = mean_absolute_error(y_test_seq[:, i], y_pred[:, i])
    print(f"   {name:8} → R²: {r2_i:.4f}, MAE: {mae_i:.4f}")

# =================================================================
# 5. SIMULATION DU PID RÉEL (VERSION CORRIGÉE AVEC DÉNORMALISATION)
# =================================================================
print("\n5. SIMULATION DU PID RÉEL CORRIGÉ")
print("-"*60)

def simulate_real_pid_on_sequences_denormalized(X_test_seq, y_test_seq, scaler_X, scaler_y, dt=0.02):
    """
    Version CORRIGÉE : Dénormalise les données avant de les passer au PID
    """
    n_samples = len(X_test_seq)
    pid_commands = np.zeros((n_samples, 4))
    
    # Récupérer le dernier pas de chaque séquence (état actuel)
    X_last_step = X_test_seq[:, -1, :]  # Shape: (n_samples, n_features)
    
    # Dénormaliser les entrées pour que le PID travaille en valeurs physiques
    X_physical = scaler_X.inverse_transform(X_last_step)
    
    # Dénormaliser les cibles
    y_physical = scaler_y.inverse_transform(y_test_seq)
    
    # Paramètres PID (optimisés pour valeurs physiques)
    Kp_pos, Ki_pos, Kd_pos = 3.0, 0.05, 1.0   # Gains plus élevés pour le physique
    Kp_vel, Ki_vel, Kd_vel = 2.0, 0.03, 0.8
    masse = 1.38
    g = 9.81
    
    integral_pos = np.zeros(3)
    prev_error_pos = np.zeros(3)
    integral_vel = np.zeros(3)
    prev_error_vel = np.zeros(3)
    
    for t in range(n_samples):
        # Positions et vitesses physiques
        current_pos = X_physical[t, :3]
        current_vel = X_physical[t, 3:6]
        target = y_physical[t, :3]
        
        # PID position
        error_pos = target - current_pos
        integral_pos += error_pos * dt
        integral_pos = np.clip(integral_pos, -2.0, 2.0)
        derivative_pos = (error_pos - prev_error_pos) / dt if dt > 0 else 0
        pos_command = Kp_pos * error_pos + Ki_pos * integral_pos + Kd_pos * derivative_pos
        prev_error_pos = error_pos.copy()
        
        # PID vitesse (cible nulle pour stabilisation)
        target_vel = np.zeros(3)
        error_vel = target_vel - current_vel
        integral_vel += error_vel * dt
        integral_vel = np.clip(integral_vel, -2.0, 2.0)
        derivative_vel = (error_vel - prev_error_vel) / dt if dt > 0 else 0
        vel_command = Kp_vel * error_vel + Ki_vel * integral_vel + Kd_vel * derivative_vel
        prev_error_vel = error_vel.copy()
        
        # Commandes physiques
        thrust = np.clip(masse * g + pos_command[2] * masse, 8.0, 20.0)
        roll = np.clip(pos_command[1] * 0.08, -0.4, 0.4)
        pitch = np.clip(pos_command[0] * 0.08, -0.4, 0.4)
        yaw = np.clip(vel_command[0] * 0.5, -0.8, 0.8)
        
        # Normaliser les commandes PID pour comparer avec LSTM (même échelle)
        pid_commands[t] = scaler_y.transform([[thrust, roll, pitch, yaw]])[0]
    
    return pid_commands

# Exécuter la version corrigée du PID
print("Simulation du PID réel avec DÉNORMALISATION...")
pid_real_predictions = simulate_real_pid_on_sequences_denormalized(
    X_test_seq, y_test_seq, scaler_X_enh, scaler_y
)

# Évaluer le PID réel corrigé
pid_real_r2 = r2_score(y_test_seq, pid_real_predictions)
pid_real_mae = mean_absolute_error(y_test_seq, pid_real_predictions)

print(f"\n PERFORMANCES DU PID RÉEL (APRÈS DÉNORMALISATION):")
print(f"   R² = {pid_real_r2:.4f}")
print(f"   MAE = {pid_real_mae:.4f}")
# ============================================
# 5. SIMULATION DU PID RÉEL SUR LES MÊMES DONNÉES
# ============================================
print("\n5. SIMULATION DU PID RÉEL")
print("-"*60)

def simulate_real_pid_on_sequences(X_test_seq, y_test_seq, dt=0.02):
    """
    Simule un vrai PID sur les données de test (format séquences)
    """
    n_samples = len(X_test_seq)
    pid_commands = np.zeros((n_samples, 4))
    
    # Paramètres PID (identiques à la génération du dataset)
    Kp_pos, Ki_pos, Kd_pos = 0.5, 0.01, 0.3  # Plus petits car erreurs normalisées
    Kp_vel, Ki_vel, Kd_vel = 0.64, 0.01, 0.3
    masse = 1.38
    g = 9.81
    
    integral_pos = np.zeros(3)
    prev_error_pos = np.zeros(3)
    integral_vel = np.zeros(3)
    prev_error_vel = np.zeros(3)
    
    for t in range(n_samples):
        # Extraire position actuelle (dernier pas de la séquence)
        current_pos = X_test_seq[t, -1, :3]
        current_vel = X_test_seq[t, -1, 3:6]
        
        # Cible (depuis y_test_seq)
        target = y_test_seq[t, :3]
        
        # PID position
        error_pos = target - current_pos
        integral_pos += error_pos * dt
        integral_pos = np.clip(integral_pos, -2.0, 2.0)
        derivative_pos = (error_pos - prev_error_pos) / dt if dt > 0 else 0
        pos_command = Kp_pos * error_pos + Ki_pos * integral_pos + Kd_pos * derivative_pos
        prev_error_pos = error_pos.copy()
        
        # PID vitesse
        target_vel = np.zeros(3)
        error_vel = target_vel - current_vel
        integral_vel += error_vel * dt
        integral_vel = np.clip(integral_vel, -2.0, 2.0)
        derivative_vel = (error_vel - prev_error_vel) / dt if dt > 0 else 0
        vel_command = Kp_vel * error_vel + Ki_vel * integral_vel + Kd_vel * derivative_vel
        prev_error_vel = error_vel.copy()
        
        # Commandes physiques
        thrust = np.clip(masse * g + pos_command[2] * masse, 8.0, 20.0)
        roll = np.clip(pos_command[1] * 0.08, -0.4, 0.4)
        pitch = np.clip(pos_command[0] * 0.08, -0.4, 0.4)
        yaw = np.clip(vel_command[0] * 0.5, -0.8, 0.8)
        
        pid_commands[t] = [thrust, roll, pitch, yaw]
    
    return pid_commands

print(" Simulation du PID réel sur les données de test...")
pid_real_predictions = simulate_real_pid_on_sequences(X_test_seq, y_test_seq)

# Évaluer le PID réel
pid_real_mse = mean_squared_error(y_test_seq, pid_real_predictions)
pid_real_mae = mean_absolute_error(y_test_seq, pid_real_predictions)
pid_real_r2 = r2_score(y_test_seq, pid_real_predictions)

print(f"\n PERFORMANCES DU PID RÉEL:")
print(f"   R² = {pid_real_r2:.4f}")
print(f"   MSE = {pid_real_mse:.4f}")
print(f"   MAE = {pid_real_mae:.4f}")

# Métriques par commande PID réel
print(f"\n PERFORMANCES PID RÉEL PAR COMMANDE:")
print("-"*40)
for i, name in enumerate(target_cols):
    r2_i = r2_score(y_test_seq[:, i], pid_real_predictions[:, i])
    mae_i = mean_absolute_error(y_test_seq[:, i], pid_real_predictions[:, i])
    print(f"   {name:8} → R²: {r2_i:.4f}, MAE: {mae_i:.4f}")

# ============================================
# 6. COMPARAISON FINALE LSTM vs PID RÉEL
# ============================================
print("\n" + "="*60)
print(" COMPARAISON FINALE: LSTM vs PID RÉEL")
print("="*60)

#============= Tableau comparatif LSTM vs PID===============
print("\n" + "="*50)
print("TABLEAU COMPARATIF LSTM vs PID")
print("="*50)
print(f"LSTM (avec perturbations):  R² = {lstm_r2:.4f} | MAE = {lstm_mae:.4f}")
print(f"PID réel (sans pertur.):    R² = {pid_real_r2:.4f} | MAE = {pid_real_mae:.4f}")
print("="*50)

difference_r2 = lstm_r2 - pid_real_r2
difference_mae = lstm_mae - pid_real_mae

if difference_r2 > 0:
    print(f" LSTM est meilleur de {difference_r2:.4f} en R²")
    print(f"   (LSTM a appris à gérer les perturbations, contrairement au PID)")
else:
    print(f" PID est meilleur de {abs(difference_r2):.4f} en R²")
    print(f"   (Le PID reste performant, mais n'a pas été exposé aux perturbations)")

print(f"\n Analyse de l'écart:")
print(f"   R²:  {'LSTM' if difference_r2 > 0 else 'PID'} meilleur de {abs(difference_r2):.4f}")
print(f"   MAE: {'LSTM' if difference_mae < 0 else 'PID'} meilleur de {abs(difference_mae):.4f}")

# ============================================
# 7. VISUALISATION COMPARAISON LSTM vs PID
# ============================================
print("\n7. VISUALISATION COMPARAISON LSTM vs PID")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Échantillon pour la visualisation
    sample_size = min(500, len(y_test_seq))
    ax.plot(y_test_seq[:sample_size, i], label='Réel', linewidth=1.5, alpha=0.7)
    ax.plot(y_pred[:sample_size, i], label='LSTM', linewidth=1.5, alpha=0.7)
    ax.plot(pid_real_predictions[:sample_size, i], label='PID réel', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Pas de temps')
    ax.set_ylabel(f'{name}')
    ax.set_title(f'{name} - Comparaison LSTM vs PID réel')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_vs_pid_comparison.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_vs_pid_comparison.png")

# ============================================
# 8. COURBES PRÉDICTIONS vs RÉELLES (LSTM)
# ============================================
print("\n8. VISUALISATION PRÉDICTIONS LSTM vs RÉELLES")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, name in enumerate(target_cols):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    ax.scatter(y_test_seq[:500, i], y_pred[:500, i], alpha=0.3, s=10, label='LSTM')
    ax.plot([y_test_seq[:, i].min(), y_test_seq[:, i].max()], 
            [y_test_seq[:, i].min(), y_test_seq[:, i].max()], 
            'r--', linewidth=2, label='Prédiction parfaite')
    ax.set_xlabel(f'{name} (réel)')
    ax.set_ylabel(f'{name} (prédit)')
    ax.set_title(f'{name} - LSTM (R²={r2_score(y_test_seq[:, i], y_pred[:, i]):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_predictions_eval.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_predictions_eval.png")

# ============================================
# 9. ANALYSE DES RÉSIDUS (LSTM)
# ============================================
print("\n9. ANALYSE DES RÉSIDUS LSTM")
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
    ax.set_title(f'{name} - Distribution des résidus LSTM')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_residuals_eval.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_residuals_eval.png")

# ============================================
# 10. TESTS DE ROBUSTESSE LSTM
# ============================================
print("\n10. TESTS DE ROBUSTESSE LSTM")
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

print("\n Test robustesse au bruit LSTM:")
noise_results = test_noise_robustness(model, X_test_seq, y_test_seq)

print("\n Test rejet de perturbation LSTM:")
pert_results = test_perturbation_rejection(model, X_test_seq, y_test_seq)

# Visualisation robustesse
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

noise_vals = [r['noise'] for r in noise_results]
r2_vals = [r['r2'] for r in noise_results]
axes[0].plot(noise_vals, r2_vals, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Niveau de bruit')
axes[0].set_ylabel('R²')
axes[0].set_title('Robustesse au bruit - LSTM')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=pid_real_r2, color='red', linestyle='--', label=f'PID réel (R²={pid_real_r2:.3f})')
axes[0].legend()

mag_vals = [r['magnitude'] for r in pert_results]
ratio_vals = [r['ratio'] for r in pert_results]
axes[1].plot(mag_vals, ratio_vals, 's-', linewidth=2, markersize=8, color='red')
axes[1].axhline(y=1.0, color='green', linestyle='--', label='Performance nominale')
axes[1].set_xlabel('Magnitude perturbation')
axes[1].set_ylabel('Ratio dégradation')
axes[1].set_title('Rejet de perturbations - LSTM')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/lstm_robustness_eval.png", dpi=150)
plt.show()
print(" Graphique sauvegardé: resultats/lstm_robustness_eval.png")

# ============================================
# 11. SAUVEGARDE DES RÉSULTATS COMPLETS
# ============================================
print("\n11. SAUVEGARDE DES RÉSULTATS")
print("-"*60)

results = {
    'lstm': {
        'r2_global': float(lstm_r2),
        'mae_global': float(lstm_mae),
        'mse_global': float(lstm_mse),
        'per_command': {
            target_cols[i]: {
                'r2': float(r2_score(y_test_seq[:, i], y_pred[:, i])),
                'mae': float(mean_absolute_error(y_test_seq[:, i], y_pred[:, i]))
            } for i in range(len(target_cols))
        },
        'robustness': {
            'noise': noise_results,
            'perturbation': pert_results
        }
    },
    'pid_real': {
        'r2_global': float(pid_real_r2),
        'mae_global': float(pid_real_mae),
        'mse_global': float(pid_real_mse),
        'per_command': {
            target_cols[i]: {
                'r2': float(r2_score(y_test_seq[:, i], pid_real_predictions[:, i])),
                'mae': float(mean_absolute_error(y_test_seq[:, i], pid_real_predictions[:, i]))
            } for i in range(len(target_cols))
        }
    },
    'comparison': {
        'difference_r2': float(difference_r2),
        'difference_mae': float(difference_mae),
        'winner': 'LSTM' if difference_r2 > 0 else 'PID'
    }
}

with open('resultats/lstm_evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(" Résultats sauvegardés: resultats/lstm_evaluation_results.json")

# ============================================
# 12. RAPPORT FINAL
# ============================================
print("\n" + "="*80)
print(" RAPPORT FINAL - COMPARAISON LSTM vs PID RÉEL")
print("="*80)

# ===========COMPARAISON FINALE===========
print("\n" + "="*50)
print("CONCLUSION FINALE")
print("="*50)
print(f"LSTM: R² = {lstm_r2:.4f} | MAE = {lstm_mae:.4f}")
print(f"PID:  R² = {pid_real_r2:.4f} | MAE = {pid_real_mae:.4f}")
print(f"Écart R²: {abs(difference_r2):.4f} en faveur de {'LSTM' if difference_r2 > 0 else 'PID'}")
print(f"Verdict: {' LSTM surpasse le PID' if difference_r2 > 0 else ' PID reste supérieur'}")
print("="*50)

print("\n ÉVALUATION LSTM AVEC COMPARAISON PID RÉEL TERMINÉE AVEC SUCCÈS !")